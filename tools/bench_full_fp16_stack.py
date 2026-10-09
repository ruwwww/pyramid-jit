import os
import time
import torch

from pyramid_jit import (
    PyramidJiT,
    QwenTextEncoder,
    SamplerConfig,
    generate,
)

PROMPT = (
    "A close-up portrait of a young white woman with vibrant, fiery red hair "
    "cascading over her shoulders in soft waves, framed from the shoulders up "
    "and centered against a softly blurred warm-toned background. Her fair, "
    "lightly freckled complexion sets off piercing green eyes and a subtle, "
    "closed-lipped smile. Soft natural light enters from the left of the frame, "
    "highlighting the texture of her hair and the curve of her cheek while leaving "
    "the right side in gentle shadow. A shallow depth of field renders the background "
    "into smooth, neutral bokeh."
)
NEG_PROMPT = "watermark, signature, logo, copyright, url, blurry, distorted"
SEED = 42
STEPS = 25

print("[*] Pre-encoding text with Qwen 4-bit...")
text_enc = QwenTextEncoder("/mnt/data/models/Qwen3.5-4B", quantization="4bit", device="cuda")

# Warmup Qwen
_ = text_enc([PROMPT])

sampler = SamplerConfig(sampling_steps=STEPS, height=512, width=512)

configs = [
    {
        "name": "1. Baseline BF16 (Eager PyTorch)",
        "dtype": torch.bfloat16,
        "fast_acc": False,
        "fused_ops": False,
        "cuda_graph": False,
    },
    {
        "name": "2. FP16 + Fast Accum (Eager)",
        "dtype": torch.float16,
        "fast_acc": True,
        "fused_ops": False,
        "cuda_graph": False,
    },
    {
        "name": "3. FP16 + Fast Accum + Triton Fused Ops",
        "dtype": torch.float16,
        "fast_acc": True,
        "fused_ops": True,
        "cuda_graph": False,
    },
    {
        "name": "4. FP16 + Fast Accum + Fused Ops + CUDA Graph",
        "dtype": torch.float16,
        "fast_acc": True,
        "fused_ops": True,
        "cuda_graph": True,
    },
]

results = []

for cfg in configs:
    print(f"\n==========================================")
    print(f"Testing: {cfg['name']}")
    print(f"==========================================")
    
    torch.backends.cuda.matmul.allow_fp16_accumulation = cfg["fast_acc"]
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    
    model = PyramidJiT.from_pretrained(
        "/mnt/data/models/pyramid-jit",
        dtype=cfg["dtype"],
        use_fused_ops=cfg["fused_ops"]
    ).to("cuda").eval()
    
    # Warmup 1 run (essential for CUDA Graph capture and Triton kernel JIT)
    _ = generate(
        model=model,
        text_encoder=text_enc,
        prompt=PROMPT,
        seeds=[SEED],
        sampler=sampler,
        negative_prompt=NEG_PROMPT,
        cuda_graph=cfg["cuda_graph"],
        quiet=True
    )
    torch.cuda.synchronize()
    
    # Benchmark runs
    times = []
    for run_i in range(3):
        t0 = time.perf_counter()
        _ = generate(
            model=model,
            text_encoder=text_enc,
            prompt=PROMPT,
            seeds=[SEED],
            sampler=sampler,
            negative_prompt=NEG_PROMPT,
            cuda_graph=cfg["cuda_graph"],
            quiet=True
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        
    avg_t = sum(times) / len(times)
    lat_step = (avg_t / STEPS) * 1000
    it_s = STEPS / avg_t
    print(f"Result: {avg_t:.3f}s total | {lat_step:.1f} ms/step | {it_s:.2f} it/s")
    
    results.append({
        "name": cfg["name"],
        "total_time": avg_t,
        "latency_step": lat_step,
        "throughput": it_s,
    })
    
    del model
    torch.cuda.empty_cache()

print("\n" + "="*70)
print("FINAL BENCHMARK SUMMARY (RTX 5060 Ti Blackwell, 512x512, 25 steps)")
print("="*70)
base_t = results[0]["total_time"]
for r in results:
    speedup = base_t / r["total_time"]
    print(f"{r['name']:<48} | {r['total_time']:.3f}s | {r['latency_step']:.1f} ms/step | {r['throughput']:.2f} it/s | {speedup:.2f}x")
