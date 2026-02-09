"""
Test script for Wan2.2 I2V inference with quantized transformer.
"""

import torch
import numpy as np
from PIL import Image
from diffusers import WanImageToVideoPipeline, UniPCMultistepScheduler, AutoModel
from diffusers.utils import export_to_video

from nunchaku.models.transformers import NunchakuWanTransformer3DModel


def main():
    # 1. Load original pipeline with lora and fuse
    print("Loading original pipeline...")
    transformer_orig = AutoModel.from_pretrained(
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    )
    text_encoder = AutoModel.from_pretrained(
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16
    )
    vae = AutoModel.from_pretrained(
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        subfolder="vae",
        torch_dtype=torch.float16
    )

    pipe = WanImageToVideoPipeline.from_pretrained(
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        vae=vae,
        transformer=transformer_orig,
        text_encoder=text_encoder,
        torch_dtype=torch.bfloat16,
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)

    # Load and fuse lora for transformer (high_noise_model)
    print("Loading and fusing LoRA weights...")
    pipe.load_lora_weights(
        "lightx2v/Wan2.2-Lightning",
        weight_name="Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors",
        adapter_name="lightning",
    )

    # Load lora for transformer_2 (low_noise_model)
    kwargs = {"load_into_transformer_2": True}
    pipe.load_lora_weights(
        "lightx2v/Wan2.2-Lightning",
        weight_name="Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors",
        adapter_name="lightning_2",
        **kwargs
    )

    pipe.set_adapters(["lightning", "lightning_2"], adapter_weights=[1., 1.])
    pipe.fuse_lora(adapter_names=["lightning"], lora_scale=1., components=["transformer"])
    pipe.fuse_lora(adapter_names=["lightning_2"], lora_scale=1., components=["transformer_2"])
    pipe.unload_lora_weights()

    # 2. Replace transformer with quantized version
    # Note: Only transformer (high_noise_model fused) is quantized
    #       transformer_2 (low_noise_model fused) remains original
    print("Loading quantized transformer...")

    # Save transformer_2 reference before replacing
    transformer_2 = pipe.transformer_2
    transformer_2.to("cuda")

    # Replace transformer with quantized version
    del pipe.transformer
    torch.cuda.empty_cache()

    quantized_transformer = NunchakuWanTransformer3DModel.from_pretrained(
        "./wan2.2/wan2.2_merged.safetensors",
        device="cuda",
        torch_dtype=torch.bfloat16,
    )
    pipe.transformer = quantized_transformer
    pipe.transformer_2 = transformer_2
    pipe.enable_model_cpu_offload()
    # 3. Prepare image
    print("Preparing image...")
    image_path = "i2v_test_6_girl.png"
    image = Image.open(image_path).convert("RGB")

    max_area = 480 * 832
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    image = image.resize((width, height))
    print(f"Image size: {width}x{height}")

    # 4. Run inference
    print("Running inference...")
    prompt = "The camera captures the young woman as she holds a piece of paper with writing on it at chest level, her hands outstretched. With a playful spark in her eyes, she crumples the paper into a ball and brings it to her mouth, pretending to eat it. This whimsical gesture showcases her playful personality and charm, inviting viewers to join in on her fun and lighthearted energy."

    pipe.to("cuda")
    generator = torch.Generator(device='cuda').manual_seed(42)

    output = pipe(
        image=image,
        prompt=prompt,
        height=height,
        width=width,
        num_frames=81,
        guidance_scale=1,
        num_inference_steps=4,
        generator=generator,
    ).frames[0]

    # 5. Save output
    output_path = "wan_i2v_quantized_test.mp4"
    export_to_video(output, output_path)
    print(f"Video saved to: {output_path}")


if __name__ == "__main__":
    main()
