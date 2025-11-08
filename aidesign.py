import rembg,os,torch
from diffusers.utils import load_image
from PIL import Image
import numpy as np
from image_transformer import f_image_rotate
import time


os.environ["U2NET_HOME"]="/gemini/code/aidesign"


def remove_background(input_image,input_image_path):
    s1_time=time.time()
    rembg_session = rembg.new_session(model_name='isnet-general-use.onnx')
    input_image_rembg = rembg.remove(input_image, rembg_session)
    print("input_image_rembg saved in : {0}".format(input_image_path))
    input_image_rembg.save(input_image_path)
    print("rembg time is : {0}".format(time.time()-s1_time))
    return input_image_rembg


def f_design(aillm,image_input,house_class,house_desc,image_output=None):
    if not image_output:
        image_output_name=image_input.split(".")[-2]+"_output"
        image_output=image_input.replace(image_input.split(".")[-2],image_output_name)

    pipe,processor=aillm

    image = load_image(image_input)
    
    negative_prompt="any eyes,face,human,tree,grass,leaf,low quality，worst quality，bad anatomy，bad composition"

    prompt="real 3d interior design,high wall,every room with door,Furniture,best quality, 8K."
    prompt=prompt+house_desc if house_desc else prompt
    #prompt="real 3d interior design,high wall,every room with door. According to the characteristics of the part of the house, a reasonable furniture is generated"
    #prompt=prompt+house_desc if house_desc else prompt

    
    print("----------- prompt -----------")
    print(prompt)

    # image_rotate=f_image_rotate(np.array(image), theta=100, phi=0, gamma=0)

    control_image = processor(image)
    
    image_control=image_input.split(".")[0]+"_control.png"
    
    control_image.save(image_control)

    generator = torch.manual_seed(0)


    # image = pipe(prompt, negative_prompt=negative_prompt,num_inference_steps=30, generator=generator, image=control_image, controlnet_conditioning_scale=0.8).images[0]

    image = pipe(prompt, negative_prompt=negative_prompt,
             num_inference_steps=4, generator=generator, 
             image=control_image,
             controlnet_conditioning_scale=0.9,
            guidance_scale=1,
            cross_attention_kwargs={"scale": 1}).images[0]
    
    image_output=image_input.split(".")[0]+"_output.png"
    
    # f_image_rotate(np.array(image)).save(image_output)
    if house_class=="Floorplan":
        print("进行背景移除")
        # remove_background(image,image_output)
        image.save(image_output)
    else:
        print("不进行背景移除")
        image.save(image_output)
    return image_output