import argparse
import os
import time
from typing import List, Optional

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gradio as gr
import torch
import torch.amp as amp
from PIL import Image

from pyramid_jit import (
    PyramidJiT,
    QwenTextEncoder,
    SamplerConfig,
    active_attention_backend,
    set_attention_backend,
)
from pyramid_jit.attention import BACKENDS
from pyramid_jit.config import SamplerConfig
from pyramid_jit.noise import StackedRandomGenerator
from pyramid_jit.sampler import MomentumBuffer, _text_kwargs, adaptive_projected_guidance

# Global handles
MODEL: Optional[PyramidJiT] = None
TEXT_ENCODER: Optional[QwenTextEncoder] = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OFFLOAD_TEXT_ENCODER = True


def load_pipeline(
    weights_path: str = "/mnt/data/models/pyramid-jit",
    qwen_path: str = "/mnt/data/models/Qwen3.5-4B",
    attention_backend: str = "auto",
    quantization: Optional[str] = "4bit",
):
    global MODEL, TEXT_ENCODER
    print(f"[*] Setting attention backend: {attention_backend}")
    set_attention_backend(backend=attention_backend)
    print(f"[*] Active attention backend: {active_attention_backend()}")

    print(f"[*] Loading PyramidJiT from {weights_path}...")
    MODEL = PyramidJiT.from_pretrained(weights_dir=weights_path, device=DEVICE)

    print(f"[*] Loading QwenTextEncoder from {qwen_path} (quantization: {quantization} on {DEVICE})...")
    TEXT_ENCODER = QwenTextEncoder(
        model_path=qwen_path,
        extraction_layers=MODEL.config.text_layers,
        max_length=MODEL.config.text_len,
        device=torch.device(DEVICE),
        quantization=quantization,
    )
    print("[+] Pipeline successfully loaded and ready!")


@torch.no_grad()
def run_generation(
    prompt: str,
    negative_prompt: str,
    seed: int,
    sampling_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    native_noise: bool,
    progress=gr.Progress(track_tqdm=True),
):
    global MODEL, TEXT_ENCODER
    if MODEL is None or TEXT_ENCODER is None:
        raise gr.Error("Model is not loaded yet!")

    t0 = time.time()
    batch_size = 1
    seeds = [int(seed)]
    text_len = MODEL.text_len

    # 1. Text encoding on CPU
    t_text_0 = time.time()
    text_cond = TEXT_ENCODER([prompt])[0]
    text_uncond = TEXT_ENCODER([negative_prompt])[0]
    t_text = time.time() - t_text_0

    # 2. Package text conditioning onto CUDA
    cond = _text_kwargs(text=text_cond, batch_size=batch_size, text_len=text_len)
    uncond = _text_kwargs(text=text_uncond, batch_size=batch_size, text_len=text_len)
    cond = {k: v.to(DEVICE) for k, v in cond.items()}
    uncond = {k: v.to(DEVICE) for k, v in uncond.items()}

    # 3. Initial noise
    sampler = SamplerConfig(
        height=int(height),
        width=int(width),
        sampling_steps=int(sampling_steps),
        guidance_scale=float(guidance_scale),
        negative_prompt=negative_prompt,
        noise_sm_count=None if native_noise else SamplerConfig().noise_sm_count,
    )

    generator = StackedRandomGenerator(
        device=DEVICE, seeds=seeds, sm_count=sampler.noise_sm_count
    )
    x_t = sampler.noise_scale * generator.randn(
        (batch_size, 3, 1, sampler.height, sampler.width),
        dtype=torch.bfloat16,
        device=DEVICE,
    )

    timesteps = torch.linspace(1.0, 0.0, sampler.sampling_steps + 1, device=DEVICE)

    # 4. Diffusion sampling
    t_diff_0 = time.time()
    with amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        momentum_buffer = MomentumBuffer(momentum=sampler.apg_momentum)
        for i in range(sampler.sampling_steps):
            progress((i + 1) / sampler.sampling_steps, desc=f"Step {i+1}/{sampler.sampling_steps}")
            t = timesteps[i]
            t_int = (t * sampler.num_timesteps).long()
            t_batch = t_int.repeat(batch_size).to(DEVICE)

            x_pred_cond = MODEL(x_t=x_t, t=t_batch, **cond)
            x_pred_uncond = MODEL(x_t=x_t, t=t_batch, **uncond)
            v_cond = (x_t - x_pred_cond) / t
            v_uncond = (x_t - x_pred_uncond) / t

            v_guided = adaptive_projected_guidance(
                pred_cond=v_cond,
                pred_other=v_uncond,
                guidance_scale=sampler.guidance_scale,
                momentum_buffer=momentum_buffer,
                eta=sampler.apg_eta,
                rescale=sampler.apg_rescale,
            )

            dt = timesteps[i + 1] - t
            x_t = x_t + dt * v_guided

    t_diff = time.time() - t_diff_0
    total_time = time.time() - t0

    # 5. Extract output RGB uint8
    image_tensor = ((x_t[0] + 1) * 127.5).clamp(0, 255).to(torch.uint8)  # (3, 1, H, W)
    array = image_tensor[:, 0].permute(1, 2, 0).cpu().numpy()
    out_image = Image.fromarray(array)

    stats = (
        f"**Generation Complete!**\n"
        f"- Total time: {total_time:.2f}s (Text encode: {t_text:.2f}s | Diffusion: {t_diff:.2f}s)\n"
        f"- Resolution: {width}x{height} | Steps: {sampling_steps} | Seed: {seed}\n"
        f"- VRAM allocated: {torch.cuda.memory_allocated() / (1024**2):.1f} MB / "
        f"{torch.cuda.max_memory_allocated() / (1024**2):.1f} MB peak"
    )
    return out_image, stats


def build_ui():
    title = "Linum Pyramid-JiT (P-JiT) Web UI"
    description = (
        "P-JiT (Pyramid-JiT): 2.2B-parameter pixel-space diffusion transformer directly in RGB (no VAE), "
        "conditioned on Qwen3.5-4B multi-layer embeddings. Running on RTX 5060 Ti."
    )

    examples = [
        [
            "A close-up portrait of a young white woman with vibrant, fiery red hair cascading over her shoulders in soft waves, framed from the shoulders up and centered against a softly blurred warm-toned background. Her fair, lightly freckled complexion sets off piercing green eyes and a subtle, closed-lipped smile. Soft natural light enters from the left of the frame, highlighting the texture of her hair and the curve of her cheek while leaving the right side in gentle shadow. A shallow depth of field renders the background into smooth, neutral bokeh. Lights dangle out of focus on the left side of the frame.",
            "watermark, signature, logo, copyright, url",
            42,
            50,
            15.0,
            512,
            512,
            False,
        ],
        [
            "An oil painting of a 19th-century three-masted wooden ship battling a violent ocean storm at twilight. Colossal, foam-crested waves crash against the listing wooden hull, with sea spray whipping through tattered rigging and billowing sails. Dramatic breaks in the dark, churning purple storm clouds reveal slivers of golden orange twilight illumination hitting the water surface. Thick, expressive impasto brushstrokes, rich romanticism style reminiscent of Turner and Aivazovsky.",
            "watermark, signature, logo, copyright, url, modern, photograph",
            123,
            50,
            15.0,
            512,
            512,
            False,
        ],
        [
            "Cinematic photograph of a majestic Bengal tiger wading stealthily through a sun-dappled jungle river. Water ripples crystal-clear around its powerful shoulders, scattering brilliant golden caustic patterns across submerged river stones. Lush tropical foliage, emerald ferns, and misty morning sunbeams piercing the canopy overhead. Highly detailed wet fur texture, intense amber eyes focused forward.",
            "watermark, signature, logo, copyright, url",
            777,
            50,
            15.0,
            512,
            512,
            False,
        ],
    ]

    with gr.Blocks(title="Pyramid-JiT Web UI") as demo:
        gr.Markdown(f"# {title}\n{description}")
        with gr.Row():
            with gr.Column(scale=5):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the image in detail (dense descriptions around 60-80 words give best results)...",
                    lines=4,
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="watermark, signature, logo, copyright, url",
                    lines=2,
                )
                with gr.Row():
                    seed = gr.Number(label="Seed", value=42, precision=0)
                    randomize_seed = gr.Button("🎲 Randomize")

                with gr.Accordion("Advanced Settings", open=False):
                    with gr.Row():
                        sampling_steps = gr.Slider(
                            label="Sampling Steps", minimum=10, maximum=100, value=50, step=1
                        )
                        guidance_scale = gr.Slider(
                            label="Guidance Scale (APG)", minimum=1.0, maximum=30.0, value=15.0, step=0.5
                        )
                    with gr.Row():
                        width = gr.Slider(label="Width", minimum=256, maximum=1024, value=512, step=32)
                        height = gr.Slider(label="Height", minimum=256, maximum=1024, value=512, step=32)
                    native_noise = gr.Checkbox(
                        label="Native Noise (Use current GPU's noise generator instead of H100 SXM emulation)",
                        value=False,
                    )

                generate_btn = gr.Button("🎨 Generate Image", variant="primary", size="lg")

            with gr.Column(scale=5):
                output_image = gr.Image(label="Output Image", type="pil")
                stats_output = gr.Markdown()

        def randomize():
            import random
            return random.randint(0, 2**31 - 1)

        randomize_seed.click(fn=randomize, outputs=[seed])

        generate_btn.click(
            fn=run_generation,
            inputs=[
                prompt,
                negative_prompt,
                seed,
                sampling_steps,
                guidance_scale,
                height,
                width,
                native_noise,
            ],
            outputs=[output_image, stats_output],
        )

        gr.Examples(
            examples=examples,
            inputs=[
                prompt,
                negative_prompt,
                seed,
                sampling_steps,
                guidance_scale,
                height,
                width,
                native_noise,
            ],
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--weights", type=str, default="/mnt/data/models/pyramid-jit")
    parser.add_argument("--qwen", type=str, default="/mnt/data/models/Qwen3.5-4B")
    parser.add_argument("--backend", type=str, default="auto")
    parser.add_argument("--quantization", type=str, default="4bit", choices=["4bit", "8bit", "none"])
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    quant = None if args.quantization == "none" else args.quantization
    load_pipeline(
        weights_path=args.weights,
        qwen_path=args.qwen,
        attention_backend=args.backend,
        quantization=quant,
    )
    demo = build_ui()
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
