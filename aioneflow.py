import cv2
import torch
from diffusers.utils import load_image
import os,torch
from pathlib import Path
from PIL import Image
import numpy as np
from controlnet_aux import MLSDdetector
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from onediff.infer_compiler import oneflow_compile

def f_aillm_init(lora_path,controlnet_path,sd_model_path):
    processor = MLSDdetector.from_pretrained(lora_path)
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        sd_model_path, controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()
    return pipe,processor


lora_path="/gemini/pretrain/aidesign/annotator_mlsd_large_512_fp32/muse/annotator_mlsd_large_512_fp32"
controlnet_path = "/gemini/pretrain/aidesign/control_v11p_sd15_mlsd/lllyasviel/control_v11p_sd15_mlsd"
sd_model_path="/gemini/pretrain/aidesign/stable-diffusion-v1-5/AI-ModelScope/stable-diffusion-v1-5"

pipe,processor=f_aillm_init(lora_path,controlnet_path,sd_model_path)

input_image="/gemini/code/aidesign/images/pic02.png"

# load an image
image = load_image(input_image)
image = np.array(image)

# generate image
generator = torch.manual_seed(666)
pipe.unet = oneflow_compile(pipe.unet)
pipe.vae.decoder = oneflow_compile(pipe.vae.decoder)
pipe.controlnet = oneflow_compile(pipe.controlnet)

prompt="""The floor plan shows a modern apartment with a clear layout. The living area is spacious, with ample space for movement. The kitchen is compact but functional, with essential appliances. Bedrooms are well-separated, providing privacy. The bathroom is efficiently designed with a walk-in shower. Overall, the design prioritizes functionality and comfort.
house_desc_style is : Modern Style, Ultra-minimalist,The floor plan shows a modern apartment with a clear layout. The living area is spacious, with ample space for movement. The kitchen is compact but functional, with essential appliances. Bedrooms are well-separated, providing privacy. The bathroom is efficiently designed with a walk-in shower. Overall, the design prioritizes functionality and comfort."""
negative_prompt="eyes,face,human,tree,grass,leaf,low quality，worst quality，bad anatomy，bad composition"
control_image = processor(image)
image_control=image_input.split(".")[0]+"_control.png"
control_image.save(image_control)