🎯 项目目标
houseDS - 3D家装管家：通过AI技术实现智能室内设计，用户上传户型图或房屋毛坯图，系统自动识别房间类型并生成相应风格的3D装修效果图。

🏗️ 项目结构
核心模块文件：
aiwebui.py - Web界面主程序

Gradio构建的用户界面

风格选择与中英文翻译

处理用户交互流程

aidesign.py - 设计生成核心

背景移除功能

ControlNet图像生成

家具布局设计

aihouse.py - 房屋识别与描述

房间类型分类（卧室、客厅、厨房等）

家具配置描述生成

视觉语言模型推理

aillm_init.py - AI模型初始化

Stable Diffusion + ControlNet初始化

Qwen视觉语言模型初始化

模型路径配置

image_transformer.py - 图像变换

3D旋转和透视变换

图像裁剪和边界处理

util.py - 工具函数

图像加载保存

角度转换等辅助功能

🔧 技术实现
AI模型栈：
视觉理解：Qwen2.5-VL-3B (多模态大模型)

图像生成：Stable Diffusion + ControlNet (MLSD直线检测)

加速优化：LCM-LoRA、xformers、CPU offload

背景移除：rembg + ISNet模型

核心流程：
图像上传 → 用户上传户型图/房间图

房间识别 → Qwen-VL分类房间类型并生成描述

风格选择 → 用户选择装修风格（现代、中式、欧式等）

设计生成 → ControlNet根据线稿生成装修效果

后处理 → 背景移除、图像优化

结果展示 → 返回3D装修效果图

特色功能：
多房间支持：15+种房间类型识别

风格体系：9大主风格 + 多种子风格

中英双语：完整的中英文风格翻译

实时生成：优化推理速度（4步LCM采样）

专业家具配置：根据不同房间类型推荐相应家具

🚀 部署配置
端口：6868

主题：深色主题

硬件要求：GPU（CUDA）、足够显存

模型路径：预设在/gemini/pretrain/目录
