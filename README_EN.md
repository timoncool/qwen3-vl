# SuperCaption Qwen3-VL

**Photo and Video Description & Tag Generator based on Qwen3-VL**

Portable application with web interface for working with Qwen3-VL multimodal models. Supports Abliterated models for working with any content without censorship.

[![Telegram](https://img.shields.io/badge/Telegram-NEURO--SOFT-blue?logo=telegram)](https://t.me/neuroport)
[![GitHub Stars](https://img.shields.io/github/stars/timoncool/qwen3-vl?style=social)](https://github.com/timoncool/qwen3-vl)

**[Русский](README.md) | [中文](README_CN.md)**

---

## About Qwen3-VL Model

**Qwen3-VL** is a multimodal model from Alibaba Cloud capable of understanding images and videos. The model analyzes visual content and generates text descriptions.

**Important:** Qwen3-VL works only with visual information (images, video frames). The model **does not understand audio** — it cannot analyze music, speech, or sound effects.

Key model capabilities:
- Understanding images of any resolution
- Video analysis (frame-by-frame)
- OCR in 20+ languages
- Object Detection with coordinates
- Reasoning mode (Thinking) for complex tasks

Learn more: [Qwen2.5-VL on GitHub](https://github.com/QwenLM/Qwen2.5-VL)

---

## Main Features

### 📷 Image Processing

| Feature | Description |
|---------|-------------|
| **Image Description** | Generate descriptions in 50+ styles: formal, creative, SEO, product, social media, etc. |
| **OCR** | Text recognition from any images |
| **Object Detection** | Object detection and localization with bounding boxes |
| **Image Comparison** | Multi-image analysis (before/after, quality control) |
| **Batch Processing** | Process hundreds of images simultaneously |

### 🎬 Video Processing

| Feature | Description |
|---------|-------------|
| **Video Analysis** | Frame-by-frame video description with timestamps |
| **Action Detection** | Identify specific action moments in video |
| **Editing Analysis** | Evaluate transitions, pacing, filming style |
| **Batch Video Processing** | Process multiple video files |

### 🧠 Intelligent Features

| Feature | Description |
|---------|-------------|
| **Thinking Mode** | Chain-of-Thought reasoning for complex tasks |
| **Problem Solving** | Step-by-step math problems and logical questions |
| **Chart Analysis** | Data extraction from diagrams and visualizations |
| **Causal Analysis** | Understanding event sequences |

### 💾 Export and Integration

| Feature | Description |
|---------|-------------|
| **TXT Export** | One file per image |
| **JSON Export** | All results in structured format |
| **CSV Export** | Tabular format for Excel/Google Sheets |
| **Prompt Presets** | Save and load frequently used prompts |

---

## Description Types (50+ Templates)

### 📝 Basic Descriptions
- **Descriptive (Formal)** — detailed formal description
- **Descriptive (Informal)** — friendly casual description
- **Product Description** — for e-commerce and marketplaces
- **SEO Description** — search engine optimized (up to 160 characters)
- **Social Media Caption** — engaging caption for Instagram/Twitter/TikTok

### 🎨 Generation Prompts
- **Stable Diffusion Prompt** — detailed prompt to recreate image in SD
- **MidJourney Prompt** — MidJourney format prompt
- **Booru Tags** — Danbooru/Gelbooru style tags, comma-separated
- **Art Critic Analysis** — composition, style, color, lighting

### 📍 OCR and Text Recognition
- **Extract All Text** — full OCR of all words, numbers, and symbols
- **Text with Coordinates** — text + positions in JSON format with bbox
- **Table to HTML** — convert tables to HTML tags
- **Structured JSON** — extraction in key-value format

### 🔀 Image Comparison
- **Compare Products** — analyze differences between products
- **Before/After Comparison** — evaluate changes and improvements
- **Time-series Analysis** — trends and predictions from sequences
- **Quality Control** — defect detection, pass/fail sorting

### 📍 Object Detection
- **Detect Objects with Location** — JSON with bbox_2d and labels
- **Visual Grounding** — description with coordinates for each object
- **Find and Locate** — search for specific objects

### 🧠 Analytical Modes
- **Math Step-by-step** — problem solving with detailed steps
- **Logical Analysis** — structured scene breakdown
- **Causal Analysis** — understanding "what happened and why"
- **Careful Analysis** — deep study before answering

### 📊 Specialized Analyses
- **Chart Analysis** — type, axes, trends, conclusions
- **Data Visualization** — numerical data extraction
- **Medical Image** — analysis with medical terminology
- **Technical Diagram** — components and their interactions
- **Document Extraction** — structured data in JSON
- **Scientific Image** — scientific phenomena description

### 🎬 Video-specific Modes
- **Event Timeline** — chronology with timestamps
- **Action Detection** — find specific actions in video
- **Long Video Summary** — brief content overview
- **Editing Analysis** — transitions and style evaluation

### 📚 Educational
- **Explain Concept** — simple explanation of complex topics
- **Textbook Problem Solving** — step-by-step calculations
- **Historical Analysis** — context and significance
- **Lab Setup** — equipment and procedure description

### 🎨 Creative
- **Color Analysis** — palette, contrasts, harmony, mood
- **Architectural Analysis** — style, materials, cultural significance
- **Dish Analysis** — as a chef: ingredients, technique, presentation
- **Presentation/Slide** — slide content and structure
- **Industrial Safety** — risks and recommendations

### 🎯 Compositional
- **Layered Composition Analysis** — background, middle ground, foreground
- **Spatial Analysis** — layout, perspective, object relationships
- **Problem Finding** — what works, what to improve

---

## Screenshots

### OCR — Text Recognition
![OCR](https://github.com/timoncool/qwen3-vl/blob/main/screenshots/01-ocr-text-recognition.png?raw=true)

### Image Description
![Description](https://github.com/timoncool/qwen3-vl/blob/main/screenshots/02-image-description.png?raw=true)

### Video Analysis
![Video](https://github.com/timoncool/qwen3-vl/blob/main/screenshots/03-video-analysis.png?raw=true)

### Batch Processing
![Batch](https://github.com/timoncool/qwen3-vl/blob/main/screenshots/04-batch-processing.png?raw=true)

### Multi-image Comparison
![Compare](https://github.com/timoncool/qwen3-vl/blob/main/screenshots/05-multi-image-compare.png?raw=true)

### Math Problem Solving
![Math](https://github.com/timoncool/qwen3-vl/blob/main/screenshots/06-math-solver.png?raw=true)

### Object Detection
![Detection](https://github.com/timoncool/qwen3-vl/blob/main/screenshots/07-object-detection.png?raw=true)

### Model Selection during Installation
![Model Selection](https://github.com/timoncool/qwen3-vl/blob/main/screenshots/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%202025-11-26%20093555.png?raw=true)

---

## Available Models

### Abliterated (Uncensored) — Recommended

| Model | Size | VRAM (4-bit) | Features |
|-------|------|--------------|----------|
| Huihui-Qwen3-VL-2B-Instruct-abliterated | 2B | ~2 GB | Fast, for weak GPUs |
| Huihui-Qwen3-VL-2B-Thinking-abliterated | 2B | ~2 GB | With reasoning mode |
| Huihui-Qwen3-VL-4B-Instruct-abliterated | 4B | ~4 GB | Speed/quality balance |
| Huihui-Qwen3-VL-4B-Thinking-abliterated | 4B | ~4 GB | With reasoning mode |
| Huihui-Qwen3-VL-8B-Instruct-abliterated | 8B | ~6 GB | High quality |
| Huihui-Qwen3-VL-8B-Thinking-abliterated | 8B | ~6 GB | With reasoning mode |
| Huihui-Qwen3-VL-32B-Instruct-abliterated | 32B | ~20 GB | Maximum quality |
| Huihui-Qwen3-VL-32B-Thinking-abliterated | 32B | ~20 GB | With reasoning mode |

### Original Qwen (Censored)

| Model | Size | VRAM (4-bit) |
|-------|------|--------------|
| Qwen3-VL-2B-Instruct | 2B | ~2 GB |
| Qwen3-VL-4B-Instruct | 4B | ~4 GB |
| Qwen3-VL-8B-Instruct | 8B | ~6 GB |

**Thinking models** include Chain-of-Thought mode — the model "thinks aloud", showing reasoning before the final answer. Useful for complex tasks.

---

## Installation

### Windows (Recommended)

1. Download and extract the archive
2. Run `install.bat` to install dependencies
3. **Select model during installation:**
   - A list of available models with numbers will appear
   - Enter the model number (e.g., `1`) and press **Enter**
   - Press **Enter** again to confirm your selection
4. Run `run.bat` to launch the application

### Launch with Auto-update

Use `run_with_update.bat` for automatic update checking on each launch:

```
run_with_update.bat
```

The script automatically:
- Checks for updates in the git repository
- Downloads new code versions
- Launches the application

### Manual Installation

```bash
# Clone repository
git clone https://github.com/timoncool/qwen3-vl.git
cd qwen3-vl

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch
python app.py
```

Application will start at `http://localhost:7860`

---

## Project Structure

```
qwen3-vl/
├── app.py              # Main application (Gradio web interface)
├── install.bat         # Windows installer
├── run.bat             # Application launcher
├── run_with_update.bat # Launch with git auto-update
├── requirements.txt    # Python dependencies
├── prompts/            # Prompt presets folder
├── temp/               # Temporary files
├── output/             # Export results
├── datasets/           # Training datasets
├── screenshots/        # Interface screenshots
└── README.md
```

---

## Requirements

### Minimum
- **Git** — for auto-updates (download: [git-scm.com](https://git-scm.com/downloads))
- **Python** 3.10+ (embedded in portable version)
- **CUDA** compatible GPU (NVIDIA)
- **VRAM**: 4 GB (for 2B model with 4-bit quantization)
- **RAM**: 8 GB

### Recommended
- **VRAM**: 8+ GB (for 8B model)
- **RAM**: 16+ GB
- **SSD**: for fast model loading

---

## Troubleshooting

### CUDA out of memory
- Use a smaller model (2B or 4B)
- Enable 4-bit quantization
- Close other GPU-using applications
- Reduce max_tokens

### Model won't load
- Check internet connection
- Ensure sufficient disk space (models are 2-20 GB)
- Models cache to `~/.cache/huggingface/` or locally to `models/`

### Slow generation
- Use 4-bit quantization
- Choose smaller model
- Reduce number of frames for video

### Video processing errors
- Ensure ffprobe/ffmpeg is installed
- Check video format (MP4, AVI, MOV, MKV supported)

---

## Credits

**Original model:** [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) by Alibaba Cloud

**Portable version:**
- [Nerual Dreming](https://t.me/nerual_dreming) — founder of [ArtGeneration.me](https://artgeneration.me/), tech blogger, and neuro-evangelist.
- [Slait](https://t.me/ruweb24)

**Telegram channel:** [NEURO-SOFT](https://t.me/neuroport)

---

## License

Project uses [Qwen](https://github.com/QwenLM/Qwen2.5-VL) models under Apache 2.0 license.

---

## ⭐ Support the Project!

If SuperCaption helped you — give it a ⭐ on GitHub!

It's free and takes a second, but really motivates project development.

[![GitHub Repo stars](https://img.shields.io/github/stars/timoncool/qwen3-vl?style=for-the-badge&logo=github)](https://github.com/timoncool/qwen3-vl/stargazers)

[![Star History Chart](https://api.star-history.com/svg?repos=timoncool/qwen3-vl&type=Date)](https://star-history.com/#timoncool/qwen3-vl&Date)
