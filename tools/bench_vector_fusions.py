import time
import torch
from pyramid_jit import PyramidJiT, QwenTextEncoder, SamplerConfig
from pyramid_jit.sampler import _text_kwargs, MomentumBuffer, adaptive_projected_guidance
from pyramid_jit.noise import StackedRandomGenerator
from pyramid_jit.attention import set_attention_backend
from pyramid_jit.fast import enable_fast_flags
from pyramid_jit.cuda_graph import DiTCUDAGraphRunner

weights = "/mnt/data/models/pyramid-jit"
qwen = "/mnt/data/models/Qwen3.5-4B"
prompt = "A simple red apple on a clean wooden table"
neg_prompt = "watermark, blur"

text_encoder = QwenTextEncoder(qwen, extraction_layers=(7, 15, 27), max_length=256, device="cuda", quantization="4bit")
text_cond = text_encoder([prompt])[0]
text_uncond = text_encoder([neg_prompt])[0]
cond = {k: v.to("cuda") for k, v in _text_kwargs(text_cond, 1, 256).items()}
uncond = {k: v.to("cuda") for k, v in _text_kwargs(text_uncond, 1, 256).items()}

def bench(model, use_cuda_graph=False, steps=25):
    sampler = SamplerConfig(sampling_steps=steps, height=512, width=512)
    gen = StackedRandomGenerator(device="cuda", seeds=[42], sm_count=sampler.noise_sm_count)
    x_t = sampler.noise_scale * gen.randn((1, 3, 1, 512, 512), dtype=torch.bfloat16, device="cuda")
    timesteps = torch.linspace(1.0, 0.0, steps + 1, device="cuda")
    t0_int = (timesteps[0] * sampler.num_timesteps).long().repeat(1)

    cond_graph = None
    uncond_graph = None
    if use_cuda_graph:
        cond_graph = DiTCUDAGraphRunner(model=model, x_t=x_t, t=t0_int, cond=cond, autocast_dtype=torch.bfloat16)
        uncond_graph = DiTCUDAGraphRunner(model=model, x_t=x_t, t=t0_int, cond=uncond, autocast_dtype=torch.bfloat16)
    
    # Warmup 2 steps
    with torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        for i in range(2):
            t = timesteps[i]
            t_int = (t * sampler.num_timesteps).long().repeat(1)
            if use_cuda_graph:
                _ = cond_graph(x_t=x_t, t=t_int, **cond)
            else:
                _ = model(x_t=x_t, t=t_int, **cond)
    torch.cuda.synchronize()

    t0 = time.time()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        mb = MomentumBuffer(momentum=sampler.apg_momentum)
        for i in range(steps):
            t = timesteps[i]
            t_next = timesteps[i+1]
            t_int = (t * sampler.num_timesteps).long().repeat(1)
            if use_cuda_graph:
                xc = cond_graph(x_t=x_t, t=t_int, **cond)
                xu = uncond_graph(x_t=x_t, t=t_int, **uncond)
            else:
                xc = model(x_t=x_t, t=t_int, **cond)
                xu = model(x_t=x_t, t=t_int, **uncond)
            vc = (x_t - xc) / t
            vu = (x_t - xu) / t
            vg = adaptive_projected_guidance(vc, vu, sampler.guidance_scale, mb, sampler.apg_eta, sampler.apg_rescale)
            x_t = x_t + (t_next - t) * vg
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    return elapsed, elapsed / steps * 1000

set_attention_backend("sdpa")
model = PyramidJiT.from_pretrained(weights, device="cuda", dtype=torch.bfloat16)

# 1. Baseline
t1, ms1 = bench(model, use_cuda_graph=False, steps=25)
print(f"1. Baseline Eager BF16:                {t1:.3f}s ({ms1:.1f} ms/step | {25/t1:.2f} it/s)")

# 2. Fused Triton Vector Ops
model.enable_fused_ops()
t2, ms2 = bench(model, use_cuda_graph=False, steps=25)
print(f"2. Fused Triton Vector Ops:             {t2:.3f}s ({ms2:.1f} ms/step | {25/t2:.2f} it/s) -> Speedup: {t1/t2:.2f}x")

# 3. Fused Ops + Fast Flags
enable_fast_flags()
t3, ms3 = bench(model, use_cuda_graph=False, steps=25)
print(f"3. Fused Ops + Fast Flags:              {t3:.3f}s ({ms3:.1f} ms/step | {25/t3:.2f} it/s) -> Speedup: {t1/t3:.2f}x")

# 4. Fused Ops + Fast Flags + Static CUDA Graph Replay
t4, ms4 = bench(model, use_cuda_graph=True, steps=25)
print(f"4. Fused Ops + Fast Flags + CUDA Graph: {t4:.3f}s ({ms4:.1f} ms/step | {25/t4:.2f} it/s) -> Speedup: {t1/t4:.2f}x")
