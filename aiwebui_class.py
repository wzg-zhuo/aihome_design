import os
os.chdir("/gemini/code/aidesign")
from aillm_init import aillm
from aihouse import f_house_class_desc
from aidesign import f_design
import gradio as gr
import numpy as np
from PIL import Image




# 设置上传和结果文件夹
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)



style_options = {
    "现代风格": ["简约", "黑白灰", "极简", "奶油", "轻奢"],
    "中式风格": ["传统中式", "新中式", "宋代中式"],
    "欧式风格": ["简欧", "北欧", "田园", "中古"],
    "美式风格": ["乡村田园", "复古", "简约", "轻奢", "中古"],
    "日式风格": ["传统", "原木", "侘寂"],
    "法式风格": ["轻法", "奶油", "田园", "复古"],
    "地中海风格": ["蓝白系", "黄褐系"],
    "东南亚风格": ["传统", "南洋复古"],
    "混搭风格": ["多种风格与色彩巧妙结合"]
}

style_translation = {
    # 主风格（一级分类）
    "现代风格": "Modern Style",
    "中式风格": "Chinese Style",
    "欧式风格": "European Style",
    "美式风格": "American Style",
    "日式风格": "Japanese Style",
    "法式风格": "French Style",
    "地中海风格": "Mediterranean Style",
    "东南亚风格": "Southeast Asian Style",
    "混搭风格": "Eclectic Style",

    # 子风格（二级分类）
    "简约": " Minimalist",
    "黑白灰": " Black & White & Gray",
    "极简": " Ultra-minimalist",
    "奶油": " Creamy",
    "轻奢": " Luxe",

    "传统中式": "Chinese Traditional",
    "新中式": "Neo-Chinese",
    "宋代中式": "Song Dynasty Chinese",

    "简欧": "European: Simplified",
    "北欧": "Scandinavian",
    "田园": "Countryside",
    "中古": "Mid-century",

    "乡村田园": "Countryside",
    "复古": "Vintage",

    "传统": "Japanese Traditional",
    "原木": "Natural Wood",
    "侘寂": "Wabi-sabi",

    "轻法": "Light",
    "蓝白系": "Blue & White",
    "黄褐系": "Yellow & Brown",

    "南洋复古": "South China Sea Vintage",

    "多种风格与色彩巧妙结合": "Blending multiple styles and colors"
}


def f_style_choices():
    style_choices = []
    style_dict_trans = {}
    for style_key, style_values in style_options.items():
        for style_value in style_values:
            style_kv = style_key + "," + style_value
            style_choices.append(style_kv)
            style_kv_en = style_translation.get(style_key) + "," + style_translation.get(style_value)
            style_dict_trans[style_kv] = style_kv_en
    return style_choices, style_dict_trans


style_choices, style_dict_trans = f_style_choices()


import time

def indoor_design_gen(image_input, style_input=None):
    if not style_input:
        style_input="现代风格,简约"
    style_input_en=style_dict_trans.get(style_input)
    image_output_path = image_input.replace("." + image_input.split(".")[-1], "_output.png") if image_input else None
    s1=time.time()
    house_class,house_desc=f_house_class_desc(image_input)
    s2=time.time()
    print("house_class is : {0}".format(house_class))
    print("house_desc is : {0}".format(house_desc))
    house_desc_style=style_input_en+","+house_desc
    print("house_desc_style is : {0}".format(house_desc_style))
    image_output=f_design(aillm,image_input,house_class,house_desc_style,image_output=image_output_path)
    s3=time.time()
    image_gen = Image.open(image_output)
    print("time of house class desc is : {0}".format(s2-s1))
    print("time of design is : {0}".format(s3-s2))
    print("cost time of all process is : {0}".format(s3-s1))
    return np.array(image_gen)



def reset_components():
    return [None, None, None]


theme=gr.themes.Soft(primary_hue="gray")

js_func = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

# result=emoji.emojize(u":thumbs_up: :beer:")


css="""[data-theme="dark"] {
    --bg-color: #1a1a1a;
    --text-color: #ffffff;
    /* 其他深色主题变量 */
}

[data-theme="light"] {
    --bg-color: #ffffff;
    --text-color: #1a1a1a;
    /* 其他浅色主题变量 */
}

body {
    background-color: var(--bg-color);
    color: var(--text-color);
    transition: background-color 0.6s ease, color 0.6s ease;
}

.theme-icon {
    transition: transform 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}"""

with gr.Blocks(title="houseDS 您的3D家装管家", css=css, js=js_func,theme=theme) as houseDS:

    gr.HTML("""<h1 align="center">{0} houseDS, 您的3D家装管家,专注生成装修设计方案 {1}</h1>""".format("🏠️","🏠️"))

    with gr.Row():
        style_input = gr.Radio(
            style_choices,
            label="请选择家装风格",
            interactive=True
        )
        image_input = gr.Image(type="filepath", label="上传您的户型图、房屋毛坯图")

    with gr.Row():
        image_output = gr.Image(type="filepath", label="您的家装方案")

    # 添加操作按钮行
    with gr.Row():
        submit_btn = gr.Button("生成方案")
        reset_btn = gr.Button("重新开始")

    # 事件绑定
    submit_btn.click(
        fn=indoor_design_gen,
        inputs=[image_input, style_input],
        outputs=[image_output]
    )

    reset_btn.click(
        fn=reset_components,
        outputs=[image_input, style_input, image_output]
    )

# 启动Gradio应用
houseDS.launch(server_name="0.0.0.0", server_port=6868, share=True)
