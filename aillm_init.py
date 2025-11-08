import os,torch
from pathlib import Path
from PIL import Image
import numpy as np
from controlnet_aux import MLSDdetector
from diffusers import (
    ControlNetModel,LCMScheduler,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

root_dir=os.getcwd()

def f_sdm_init(lora_path,controlnet_path,sd_model_path):
    processor = MLSDdetector.from_pretrained(lora_path)
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        sd_model_path, controlnet=controlnet, torch_dtype=torch.float16
    )
    # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # pipe.enable_xformers_memory_efficient_attention()
    # pipe.enable_model_cpu_offload()
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights("lcm-lora-sdv1-5")
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()
    return pipe,processor


def f_vlm_init(vl_model_path):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        vl_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(vl_pretrain_path, min_pixels=min_pixels, max_pixels=max_pixels)
    return model, processor



lora_path="/gemini/pretrain/aidesign/annotator_mlsd_large_512_fp32/muse/annotator_mlsd_large_512_fp32"
controlnet_path = "/gemini/pretrain/aidesign/control_v11p_sd15_mlsd/lllyasviel/control_v11p_sd15_mlsd"
# sd_model_path="/gemini/pretrain/aidesign/stable-diffusion-v1-5/AI-ModelScope/stable-diffusion-v1-5"
sd_model_path="/gemini/code/lcm_dreamer"
vl_model_path = "/gemini/pretrain/aidesign/qwen25vl3b/Qwen/Qwen2_5-VL-3B-Instruct"

aillm=f_sdm_init(lora_path,controlnet_path,sd_model_path)
model, processor = f_vlm_init(vl_model_path)

