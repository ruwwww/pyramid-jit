import os
import time
import torch
from PIL import Image, ImageDraw, ImageFont

from pyramid_jit import (
    PyramidJiT,
    QwenTextEncoder,
    SamplerConfig,
    generate,
    set_attention_backend,
    active_attention_backend,
)
from pyramid_jit.fast import enable_fast_flags
from pyramid_jit.quant_convrot import quantize_model_convrot

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
HEIGHT = 512
WIDTH = 512

WEIGHTS_PATH = "/mnt/data/models/pyramid-jit"
QWEN_PATH = "/mnt/data/models/Qwen3.5-4B"
OUT_DIR = "/mnt/data/pyramid-jit/outputs"
os.makedirs(OUT_DIR, exist_ok=True)


def disable_fast_flags():
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if hasattr(torch.backends.cuda, "allow_fp16_bf16_reduction_math_sdp"):
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
    torch.backends.cudnn.benchmark = False


def main():
    print("[*] Loading QwenTextEncoder (4-bit NF4)...")
    text_encoder = QwenTextEncoder(
        model_path=QWEN_PATH,
        extraction_layers=(7, 15, 27),
        max_length=256,
        device=torch.device("cuda"),
        quantization="4bit",
    )

    sampler = SamplerConfig(
        height=HEIGHT,
        width=WIDTH,
        sampling_steps=STEPS,
        guidance_scale=15.0,
        negative_prompt=NEG_PROMPT,
        noise_sm_count=SamplerConfig().noise_sm_count,
    )

    configs = [
        {
            "name": "1. Baseline BF16 (SDPA)",
            "short": "baseline_bf16_sdpa",
            "fast": False,
            "backend": "sdpa",
            "convrot": False,
        },
        {
            "name": "2. Fast Flags (Half-Acc + SDPA)",
            "short": "fast_flags_sdpa",
            "fast": True,
            "backend": "sdpa",
            "convrot": False,
        },
        {
            "name": "3. SageAttention (BF16 + Sage)",
            "short": "sage_attention",
            "fast": False,
            "backend": "sage",
            "convrot": False,
        },
        {
            "name": "4. ConvRot INT8 + Sage + Fast",
            "short": "convrot_sage_fast",
            "fast": True,
            "backend": "sage",
            "convrot": True,
        },
    ]

    results = []

    for cfg in configs:
        print(f"\n==========================================")
        print(f"[*] Running Config: {cfg['name']}")
        print(f"==========================================")

        # 1. Setup flags & backend
        if cfg["fast"]:
            enable_fast_flags()
        else:
            disable_fast_flags()

        set_attention_backend(cfg["backend"])
        print(f"[*] Active backend: {active_attention_backend()}")

        # 2. Load fresh model
        model = PyramidJiT.from_pretrained(WEIGHTS_PATH, device="cuda", dtype=torch.bfloat16)
        if cfg["convrot"]:
            print("[*] Quantizing trunk to ConvRot INT8...")
            quantize_model_convrot(model)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # 3. Benchmark generation
        torch.cuda.synchronize()
        t0 = time.time()
        images = generate(
            model=model,
            text_encoder=text_encoder,
            prompt=PROMPT,
            seeds=[SEED],
            sampler=sampler,
            quiet=False,
        )
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)

        print(f"[+] Finished {cfg['name']}:")
        print(f"    Elapsed: {elapsed:.2f}s ({elapsed/STEPS*1000:.1f} ms/step)")
        print(f"    Peak VRAM: {peak_vram_mb:.1f} MB")

        # 4. Save individual image
        img_tensor = images[0]
        array = img_tensor[:, 0].permute(1, 2, 0).cpu().numpy()
        pil_img = Image.fromarray(array)
        img_path = os.path.join(OUT_DIR, f"{cfg['short']}_seed{SEED}.png")
        pil_img.save(img_path)

        results.append({
            "cfg": cfg,
            "img": pil_img,
            "elapsed": elapsed,
            "ms_step": elapsed / STEPS * 1000,
            "peak_vram": peak_vram_mb,
            "path": img_path,
        })

        # Cleanup model for next run
        del model
        torch.cuda.empty_cache()

    # 5. Build labeled comparison grid
    print("\n[*] Assembling 2x2 comparison grid...")
    grid_w = WIDTH * 2
    banner_h = 42
    cell_h = HEIGHT + banner_h
    grid_h = cell_h * 2

    grid_img = Image.new("RGB", (grid_w, grid_h), color=(18, 18, 20))
    draw = ImageDraw.Draw(grid_img)

    for i, res in enumerate(results):
        r = i // 2
        c = i % 2
        x_off = c * WIDTH
        y_off = r * cell_h

        # Paste image
        grid_img.paste(res["img"], (x_off, y_off + banner_h))

        # Draw banner background
        draw.rectangle(
            [(x_off, y_off), (x_off + WIDTH, y_off + banner_h)],
            fill=(26, 27, 32),
        )

        label_top = res["cfg"]["name"]
        label_sub = f"{res['elapsed']:.2f}s ({res['ms_step']:.1f} ms/step) | Peak VRAM: {res['peak_vram']:.0f} MB"

        draw.text((x_off + 12, y_off + 4), label_top, fill=(255, 255, 255))
        draw.text((x_off + 12, y_off + 22), label_sub, fill=(74, 222, 128))

    grid_path = os.path.join(OUT_DIR, "grid_comparison_same_seed_42.png")
    grid_img.save(grid_path, quality=95)
    print(f"[+] Comparison grid saved to {grid_path}!")


if __name__ == "__main__":
    main()
