# SuperCaption Qwen3-VL

**基于 Qwen3-VL 的图像和视频描述生成器**

具有 Web 界面的便携式应用程序，用于处理 Qwen3-VL 多模态模型。支持 Abliterated 模型，可处理任何内容，无审查限制。

[![Telegram](https://img.shields.io/badge/Telegram-NEURO--SOFT-blue?logo=telegram)](https://t.me/neuroport)
[![GitHub Stars](https://img.shields.io/github/stars/timoncool/qwen3-vl?style=social)](https://github.com/timoncool/qwen3-vl)

**[Русский](README.md) | [English](README_EN.md)**

---

## ⭐ 支持项目！

如果您喜欢 SuperCaption，**请在 GitHub 上给我们一个星标**！这有助于项目发展，并表明它对社区有用。

[![Star History Chart](https://api.star-history.com/svg?repos=timoncool/qwen3-vl&type=Date)](https://star-history.com/#timoncool/qwen3-vl&Date)

---

## 主要功能

### 📷 图像处理

| 功能 | 描述 |
|------|------|
| **图像描述** | 生成 50+ 种风格的描述：正式、创意、SEO、产品、社交媒体等 |
| **OCR** | 从任何图像识别文字 |
| **目标检测** | 带边界框的对象检测和定位 |
| **图像比较** | 多图像分析（前后对比、质量控制） |
| **批量处理** | 同时处理数百张图像 |

### 🎬 视频处理

| 功能 | 描述 |
|------|------|
| **视频分析** | 带时间戳的逐帧视频描述 |
| **动作检测** | 识别视频中的特定动作时刻 |
| **剪辑分析** | 评估转场、节奏、拍摄风格 |
| **批量视频处理** | 处理多个视频文件 |

### 🧠 智能功能

| 功能 | 描述 |
|------|------|
| **思维模式** | 用于复杂任务的思维链（Chain-of-Thought）推理 |
| **问题解决** | 逐步解决数学问题和逻辑问题 |
| **图表分析** | 从图表和可视化中提取数据 |
| **因果分析** | 理解事件序列 |

### 💾 导出和集成

| 功能 | 描述 |
|------|------|
| **TXT 导出** | 每张图像一个文件 |
| **JSON 导出** | 结构化格式的所有结果 |
| **CSV 导出** | Excel/Google Sheets 的表格格式 |
| **提示词预设** | 保存和加载常用提示词 |

---

## 描述类型（50+ 模板）

### 📝 基本描述
- **描述性（正式）** — 详细的正式描述
- **描述性（非正式）** — 友好的休闲描述
- **产品描述** — 用于电商和市场
- **SEO 描述** — 搜索引擎优化（最多 160 字符）
- **社交媒体文案** — Instagram/Twitter/TikTok 的吸引人文案

### 🎨 生成提示词
- **Stable Diffusion 提示词** — 在 SD 中重建图像的详细提示词
- **MidJourney 提示词** — MidJourney 格式提示词
- **Booru 标签** — Danbooru/Gelbooru 风格标签，逗号分隔
- **艺术评论分析** — 构图、风格、色彩、光线

### 📍 OCR 和文字识别
- **提取所有文字** — 所有单词、数字和符号的完整 OCR
- **带坐标的文字** — JSON 格式的文字 + 位置带 bbox
- **表格转 HTML** — 将表格转换为 HTML 标签
- **结构化 JSON** — 键值格式提取

### 🔀 图像比较
- **比较产品** — 分析产品之间的差异
- **前后对比** — 评估变化和改进
- **时间序列分析** — 从序列中得出趋势和预测
- **质量控制** — 缺陷检测，合格/不合格分类

### 📍 目标检测
- **检测对象及位置** — 带 bbox_2d 和标签的 JSON
- **视觉定位** — 带每个对象坐标的描述
- **查找并定位** — 搜索特定对象

### 🧠 分析模式
- **逐步数学** — 带详细步骤的问题解决
- **逻辑分析** — 结构化场景分解
- **因果分析** — 理解"发生了什么以及为什么"
- **仔细分析** — 回答前深入研究

### 📊 专业分析
- **图表分析** — 类型、坐标轴、趋势、结论
- **数据可视化** — 数值数据提取
- **医学图像** — 使用医学术语分析
- **技术图表** — 组件及其相互作用
- **文档提取** — JSON 格式的结构化数据
- **科学图像** — 科学现象描述

### 🎬 视频专用模式
- **事件时间线** — 带时间戳的时间顺序
- **动作检测** — 在视频中查找特定动作
- **长视频摘要** — 简要内容概述
- **剪辑分析** — 转场和风格评估

### 📚 教育类
- **解释概念** — 复杂主题的简单解释
- **教科书问题解决** — 逐步计算
- **历史分析** — 背景和意义
- **实验室设置** — 设备和程序描述

### 🎨 创意类
- **色彩分析** — 调色板、对比、和谐、情绪
- **建筑分析** — 风格、材料、文化意义
- **菜肴分析** — 作为厨师：配料、技术、摆盘
- **演示文稿/幻灯片** — 幻灯片内容和结构
- **工业安全** — 风险和建议

### 🎯 构图类
- **分层构图分析** — 背景、中景、前景
- **空间分析** — 布局、透视、对象关系
- **问题发现** — 什么有效，什么需要改进

---

## 截图

### OCR — 文字识别
![OCR](https://github.com/timoncool/qwen3-vl/blob/main/screenshots/01-ocr-text-recognition.png?raw=true)

### 图像描述
![Description](https://github.com/timoncool/qwen3-vl/blob/main/screenshots/02-image-description.png?raw=true)

### 视频分析
![Video](https://github.com/timoncool/qwen3-vl/blob/main/screenshots/03-video-analysis.png?raw=true)

### 批量处理
![Batch](https://github.com/timoncool/qwen3-vl/blob/main/screenshots/04-batch-processing.png?raw=true)

### 多图像比较
![Compare](https://github.com/timoncool/qwen3-vl/blob/main/screenshots/05-multi-image-compare.png?raw=true)

### 数学问题解决
![Math](https://github.com/timoncool/qwen3-vl/blob/main/screenshots/06-math-solver.png?raw=true)

### 目标检测
![Detection](https://github.com/timoncool/qwen3-vl/blob/main/screenshots/07-object-detection.png?raw=true)

---

## 可用模型

### Abliterated（无审查）— 推荐

| 模型 | 大小 | 显存 (4-bit) | 特点 |
|------|------|--------------|------|
| Huihui-Qwen3-VL-2B-Instruct-abliterated | 2B | ~2 GB | 快速，适合弱 GPU |
| Huihui-Qwen3-VL-2B-Thinking-abliterated | 2B | ~2 GB | 带推理模式 |
| Huihui-Qwen3-VL-4B-Instruct-abliterated | 4B | ~4 GB | 速度/质量平衡 |
| Huihui-Qwen3-VL-4B-Thinking-abliterated | 4B | ~4 GB | 带推理模式 |
| Huihui-Qwen3-VL-8B-Instruct-abliterated | 8B | ~6 GB | 高质量 |
| Huihui-Qwen3-VL-8B-Thinking-abliterated | 8B | ~6 GB | 带推理模式 |
| Huihui-Qwen3-VL-32B-Instruct-abliterated | 32B | ~20 GB | 最高质量 |
| Huihui-Qwen3-VL-32B-Thinking-abliterated | 32B | ~20 GB | 带推理模式 |

### 原版 Qwen（有审查）

| 模型 | 大小 | 显存 (4-bit) |
|------|------|--------------|
| Qwen3-VL-2B-Instruct | 2B | ~2 GB |
| Qwen3-VL-4B-Instruct | 4B | ~4 GB |
| Qwen3-VL-8B-Instruct | 8B | ~6 GB |

**Thinking 模型** 包含思维链模式 — 模型会"大声思考"，在最终答案之前展示推理过程。对复杂任务很有用。

---

## 安装

### Windows（推荐）

1. 下载并解压压缩包
2. 运行 `install.bat` 安装依赖
3. 运行 `run.bat` 启动应用程序

### 带自动更新启动

使用 `run_with_update.bat` 在每次启动时自动检查更新：

```
run_with_update.bat
```

脚本自动：
- 检查 git 仓库中的更新
- 下载新版本代码
- 启动应用程序

### 手动安装

```bash
# 克隆仓库
git clone https://github.com/timoncool/qwen3-vl.git
cd qwen3-vl

# 创建虚拟环境
python -m venv venv

# 激活 (Windows)
venv\Scripts\activate

# 激活 (Linux/Mac)
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 启动
python app.py
```

应用程序将在 `http://localhost:7860` 启动

---

## 额外描述选项

### 图像选项

| 选项 | 添加内容 |
|------|----------|
| 光照信息 | 光源描述及其效果 |
| 相机角度 | 拍摄角度和透视 |
| 水印 | 是否有水印 |
| JPEG 伪影 | 图像压缩质量 |
| 相机/照片详情 | 光圈、快门速度、ISO（如适用） |
| SFW/PG 级别 | 排除成人内容 |
| 美学质量 | 主观质量评价 |
| 构图风格 | 三分法则、对称、引导线 |
| 景深 | 焦点和背景模糊 |
| 仅关键元素 | 专注于最重要的 |

### 视频选项

| 选项 | 添加内容 |
|------|----------|
| 镜头运动 | 平移、缩放、静态 |
| 音频描述 | 音乐、语音、音效 |
| 情节/故事 | 事件发展 |
| 时间戳 | HH:MM:SS 格式的时间标记 |
| 剪辑风格 | 转场、节奏、效果 |
| 仅关键时刻 | 专注于重要事件 |

---

## 生成参数

| 参数 | 描述 | 建议 |
|------|------|------|
| **温度** | 创造力 | 0.1-0.3 用于精确任务，0.7-1.0 用于创意 |
| **最大令牌** | 响应长度 | 256-512 用于简短，1024+ 用于详细 |
| **Top-p** | 核采样 | 0.9 用于平衡，0.5 用于聚焦 |
| **Top-k** | 令牌限制 | 默认 50 |
| **种子** | 可重复性 | 特定数字用于重复，-1 用于随机 |

---

## 项目结构

```
qwen3-vl/
├── app.py              # 主应用程序（Gradio Web 界面）
├── install.bat         # Windows 安装程序
├── run.bat             # 应用程序启动器
├── run_with_update.bat # 带 git 自动更新的启动
├── requirements.txt    # Python 依赖
├── prompts/            # 提示词预设文件夹
├── temp/               # 临时文件
├── output/             # 导出结果
├── datasets/           # 训练数据集
├── screenshots/        # 界面截图
└── README.md
```

---

## 系统要求

### 最低配置
- **Python** 3.10+
- **CUDA** 兼容 GPU（NVIDIA）
- **显存**：4 GB（用于带 4-bit 量化的 2B 模型）
- **内存**：8 GB

### 推荐配置
- **显存**：8+ GB（用于 8B 模型）
- **内存**：16+ GB
- **SSD**：用于快速加载模型

---

## 故障排除

### CUDA 内存不足
- 使用较小的模型（2B 或 4B）
- 启用 4-bit 量化
- 关闭其他使用 GPU 的应用程序
- 减少 max_tokens

### 模型无法加载
- 检查网络连接
- 确保有足够的磁盘空间（模型为 2-20 GB）
- 模型缓存到 `~/.cache/huggingface/` 或本地 `models/`

### 生成缓慢
- 使用 4-bit 量化
- 选择较小的模型
- 减少视频的帧数

### 视频处理错误
- 确保安装了 ffprobe/ffmpeg
- 检查视频格式（支持 MP4、AVI、MOV、MKV）

---

## 致谢

**便携版：**
- [Nerual Dreming](https://t.me/nerual_dreming)
- [Slait](https://t.me/ruweb24)

**Telegram 频道：** [NEURO-SOFT](https://t.me/neuroport)

---

## 许可证

项目使用 [Qwen](https://github.com/QwenLM/Qwen2.5-VL) 模型，采用 Apache 2.0 许可证。

---

## ⭐ 别忘了给星标！

如果 SuperCaption 帮助了您 — 请在 GitHub 上给它一个 ⭐！

这是免费的，只需一秒钟，但真的能激励项目发展。

[![GitHub Repo stars](https://img.shields.io/github/stars/timoncool/qwen3-vl?style=for-the-badge&logo=github)](https://github.com/timoncool/qwen3-vl/stargazers)
