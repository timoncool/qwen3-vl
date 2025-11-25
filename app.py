import gradio as gr
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from PIL import Image
import random
import os
import warnings
from typing import List, Tuple, Optional, Generator
import requests
from io import BytesIO
import urllib.parse
import gc
import json
import csv
import shutil
from datetime import datetime
import time
import tempfile

# Optional: psutil for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Note: psutil not installed. RAM monitoring will be limited.")

# Suppress specific warnings
warnings.filterwarnings('ignore', message='.*meta device.*')

# Global flag for stopping generation
stop_generation_flag = False

# Base directory for portable app
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(SCRIPT_DIR, "temp")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
DATASETS_DIR = os.path.join(SCRIPT_DIR, "datasets")
PROMPTS_DIR = os.path.join(SCRIPT_DIR, "prompts")

# Create directories if they don't exist
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(PROMPTS_DIR, exist_ok=True)

# Extra options for description enhancement
EXTRA_OPTIONS = {
    "en": {
        "Include lighting info": "Include information about lighting.",
        "Include camera angle": "Include information about camera angle.",
        "Include watermark info": "Include information about whether there is a watermark or not.",
        "Include JPEG artifacts info": "Include information about whether there are JPEG artifacts or not.",
        "Include camera/photo details": "If it is a photo, include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.",
        "Keep it SFW/PG": "Do NOT include anything sexual; keep it PG.",
        "Don't mention resolution": "Do NOT mention the image's resolution.",
        "Include aesthetic quality": "Include information about the subjective aesthetic quality of the image from low to very high.",
        "Include composition style": "Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",
        "Don't mention text in image": "Do NOT mention any text that is in the image.",
        "Include depth of field": "Specify the depth of field and whether the background is in focus or blurred.",
        "Describe only key elements": "ONLY describe the most important elements of the image."
    },
    "ru": {
        "Добавить информацию об освещении": "Добавь информацию об освещении.",
        "Добавить ракурс камеры": "Добавь информацию о ракурсе камеры.",
        "Упомянуть водяной знак": "Укажи, есть ли водяной знак на изображении.",
        "Упомянуть JPEG артефакты": "Укажи, есть ли JPEG артефакты на изображении.",
        "Добавить детали камеры/фото": "Если это фотография, укажи информацию о камере, апертуре, выдержке, ISO и т.д.",
        "Сохранить SFW/PG рейтинг": "НЕ включай сексуальный контент, сохраняй рейтинг PG.",
        "Не упоминать разрешение": "НЕ упоминай разрешение изображения.",
        "Добавить эстетическую оценку": "Добавь информацию о субъективном эстетическом качестве от низкого до очень высокого.",
        "Добавить стиль композиции": "Добавь информацию о стиле композиции (ведущие линии, правило третей, симметрия).",
        "Не упоминать текст на изображении": "НЕ упоминай текст, который есть на изображении.",
        "Добавить глубину резкости": "Укажи глубину резкости и размытость фона.",
        "Описывать только ключевые элементы": "Описывай ТОЛЬКО самые важные элементы изображения."
    },
    "zh": {
        "包含光照信息": "包含光照信息。",
        "包含相机角度": "包含相机角度信息。",
        "提及水印": "说明图片是否有水印。",
        "提及JPEG伪影": "说明图片是否有JPEG伪影。",
        "包含相机/照片详情": "如果是照片，包含可能使用的相机信息，如光圈、快门速度、ISO等。",
        "保持SFW/PG级别": "不要包含任何性相关内容，保持PG级别。",
        "不要提及分辨率": "不要提及图片的分辨率。",
        "包含美学质量评价": "包含从低到非常高的主观美学质量评价。",
        "包含构图风格": "包含构图风格信息，如引导线、三分法则或对称性。",
        "不要提及图片中的文字": "不要提及图片中的任何文字。",
        "包含景深信息": "说明景深以及背景是否模糊。",
        "只描述关键元素": "只描述图片中最重要的元素。"
    }
}

def get_memory_info() -> str:
    """Get current memory usage information"""
    info_parts = []

    # GPU memory
    if torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            info_parts.append(f"GPU: {allocated:.1f}/{total:.1f} GB (reserved: {reserved:.1f} GB)")
        except:
            pass

    # RAM
    if PSUTIL_AVAILABLE:
        try:
            process = psutil.Process(os.getpid())
            ram_usage = process.memory_info().rss / (1024 ** 3)
            info_parts.append(f"RAM: {ram_usage:.1f} GB")
        except:
            pass

    return " | ".join(info_parts) if info_parts else "Memory info unavailable"

def load_prompt_presets() -> dict:
    """Load prompt presets from the prompts directory"""
    presets = {"None": ""}

    if not os.path.exists(PROMPTS_DIR):
        return presets

    for filename in os.listdir(PROMPTS_DIR):
        if filename.endswith('.txt'):
            preset_name = os.path.splitext(filename)[0]
            try:
                with open(os.path.join(PROMPTS_DIR, filename), 'r', encoding='utf-8') as f:
                    presets[preset_name] = f.read().strip()
            except:
                pass

    return presets

def save_prompt_preset(name: str, prompt: str) -> str:
    """Save a prompt preset to the prompts directory"""
    if not name or not name.strip():
        return "❌ Please provide a preset name"

    if not prompt or not prompt.strip():
        return "❌ Please provide a prompt to save"

    # Sanitize filename
    safe_name = "".join(c for c in name if c.isalnum() or c in "_ -").strip()
    if not safe_name:
        return "❌ Invalid preset name"

    try:
        os.makedirs(PROMPTS_DIR, exist_ok=True)
        filepath = os.path.join(PROMPTS_DIR, f"{safe_name}.txt")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(prompt.strip())
        return f"✅ Preset '{safe_name}' saved successfully!"
    except Exception as e:
        return f"❌ Error saving preset: {str(e)}"

def save_text_to_file(text: str, filename: str = "result.txt") -> str:
    """Save text to a temporary file and return the path"""
    temp_file = os.path.join(TEMP_DIR, filename)
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(text)
    return temp_file

def stop_generation():
    """Set the stop flag to True"""
    global stop_generation_flag
    stop_generation_flag = True
    return "🛑 Stopping generation..."

def reset_stop_flag():
    """Reset the stop flag"""
    global stop_generation_flag
    stop_generation_flag = False

# Custom CSS for beautiful UI
CUSTOM_CSS = """
/* Main container styling */
.gradio-container {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%) !important;
}

/* Card style blocks */
.card-style {
    background: rgba(255, 255, 255, 0.9) !important;
    border: 1px solid rgba(203, 213, 225, 0.5) !important;
    box-shadow: 0 8px 32px rgba(100, 116, 139, 0.1) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    margin-bottom: 1rem !important;
}

/* Main header with gradient */
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 20px;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
}

.main-header h1 {
    color: white !important;
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
}

.main-header p {
    color: rgba(255,255,255,0.9) !important;
    font-size: 1.1rem !important;
    margin: 0.5rem 0 0 0 !important;
}

.main-header a {
    color: white !important;
    text-decoration: underline;
}

/* Generate button styling */
.generate-btn {
    min-height: 50px !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
}

/* Status box */
.status-box {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%) !important;
    border: 1px solid #7dd3fc !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

/* Progress info */
.progress-info {
    background: #f0fdf4 !important;
    border: 1px solid #86efac !important;
    border-radius: 8px !important;
    padding: 0.75rem !important;
}

/* Dark mode support */
.dark .gradio-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
}

.dark .card-style {
    background: rgba(30, 41, 59, 0.9) !important;
    border: 1px solid rgba(51, 65, 85, 0.5) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
}

.dark .status-box {
    background: linear-gradient(135deg, #1e3a5f 0%, #1e293b 100%) !important;
    border: 1px solid #3b82f6 !important;
}

.dark .progress-info {
    background: #14532d !important;
    border: 1px solid #22c55e !important;
}

/* Accordion styling */
.settings-accordion {
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* Result variants */
.variant-box {
    border: 2px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.5rem;
    transition: all 0.2s ease;
}

.variant-box:hover {
    border-color: #667eea;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
}

/* Export buttons */
.export-btn {
    margin: 0.25rem !important;
}
"""

# Description types with prompts
DESCRIPTION_TYPES = {
    "en": {
        "Descriptive (Formal)": "Write a detailed and formal description of this image.",
        "Descriptive (Informal)": "Write a casual, friendly description of this image.",
        "Product Description": "Write a compelling product description for e-commerce based on this image.",
        "SEO Description": "Write an SEO-optimized description for this image, maximum 160 characters.",
        "Stable Diffusion Prompt": "Write a detailed Stable Diffusion prompt to recreate this image.",
        "MidJourney Prompt": "Write a MidJourney prompt to recreate this image.",
        "Booru Tags": "Write a list of Booru-style tags for this image, separated by commas.",
        "Art Critic Analysis": "Analyze this image like an art critic, discussing composition, style, color, lighting, and artistic elements.",
        "Social Media Caption": "Write an engaging social media caption for this image.",
        "Custom": ""
    },
    "ru": {
        "Описательный (формальный)": "Напиши детальное и формальное описание этого изображения на русском языке.",
        "Описательный (неформальный)": "Напиши непринуждённое, дружелюбное описание этого изображения на русском языке.",
        "Описание товара": "Напиши привлекательное описание товара для интернет-магазина на основе этого изображения на русском языке.",
        "SEO описание": "Напиши SEO-оптимизированное описание для этого изображения на русском языке, максимум 160 символов.",
        "Промпт Stable Diffusion": "Напиши детальный промпт для Stable Diffusion, чтобы воссоздать это изображение.",
        "Промпт MidJourney": "Напиши промпт для MidJourney, чтобы воссоздать это изображение.",
        "Теги Booru": "Напиши список тегов в стиле Booru для этого изображения, разделённых запятыми.",
        "Анализ искусствоведа": "Проанализируй это изображение как искусствовед, обсуждая композицию, стиль, цвет, освещение и художественные элементы на русском языке.",
        "Пост для соцсетей": "Напиши привлекательную подпись для социальных сетей к этому изображению на русском языке.",
        "Свой промпт": ""
    },
    "zh": {
        "描述性（正式）": "写一个详细正式的图像描述。",
        "描述性（非正式）": "写一个轻松友好的图像描述。",
        "产品描述": "根据这张图片为电商平台写一个吸引人的产品描述。",
        "SEO描述": "为这张图片写一个SEO优化的描述，最多160个字符。",
        "Stable Diffusion提示词": "写一个详细的Stable Diffusion提示词来重现这张图片。",
        "MidJourney提示词": "写一个MidJourney提示词来重现这张图片。",
        "Booru标签": "为这张图片写一个Booru风格的标签列表，用逗号分隔。",
        "艺术评论分析": "像艺术评论家一样分析这张图片，讨论构图、风格、色彩、光线和艺术元素。",
        "社交媒体文案": "为这张图片写一个吸引人的社交媒体文案。",
        "自定义": ""
    }
}

# Description lengths
DESCRIPTION_LENGTHS = {
    "en": {
        "Any": "",
        "Very Short (1-2 sentences)": "Keep it very short, 1-2 sentences maximum.",
        "Short (3-4 sentences)": "Keep it short, 3-4 sentences.",
        "Medium (1 paragraph)": "Write a medium-length description, about one paragraph.",
        "Long (2-3 paragraphs)": "Write a detailed description, 2-3 paragraphs.",
        "Very Long (comprehensive)": "Write a comprehensive and very detailed description."
    },
    "ru": {
        "Любая": "",
        "Очень короткая (1-2 предложения)": "Сделай очень коротко, максимум 1-2 предложения.",
        "Короткая (3-4 предложения)": "Сделай коротко, 3-4 предложения.",
        "Средняя (1 абзац)": "Напиши описание средней длины, около одного абзаца.",
        "Длинная (2-3 абзаца)": "Напиши детальное описание, 2-3 абзаца.",
        "Очень длинная (подробная)": "Напиши исчерпывающее и очень детальное описание."
    },
    "zh": {
        "任意": "",
        "非常短（1-2句）": "保持非常简短，最多1-2句。",
        "短（3-4句）": "保持简短，3-4句。",
        "中等（1段）": "写一个中等长度的描述，大约一段。",
        "长（2-3段）": "写一个详细的描述，2-3段。",
        "非常长（全面）": "写一个全面且非常详细的描述。"
    }
}

# Multi-language support
TRANSLATIONS = {
    "en": {
        "title": "Qwen VL Image Description Generator",
        "header": "🖼️ Image Description Generator based on Qwen Vision Language Models",
        "subtitle": "Upload an image and enter a prompt to generate a description using Qwen VL models.",
        "language": "Language",
        "language_info": "Select language",
        "model_selection": "Model Selection",
        "model_info": "Select a model for generating descriptions",
        "advanced_params": "⚙️ Advanced Parameters",
        "max_tokens": "Max New Tokens",
        "max_tokens_info": "Maximum number of tokens to generate",
        "temperature": "Temperature",
        "temperature_info": "Controls randomness of generation",
        "top_p": "Top-p (nucleus sampling)",
        "top_p_info": "Probability threshold for token sampling",
        "top_k": "Top-k",
        "top_k_info": "Number of most probable tokens to consider",
        "seed": "Seed",
        "seed_info": "Seed for reproducibility (-1 for random)",
        "random_seed_btn": "🎲 Random Seed",
        "single_processing": "📄 Single Processing",
        "batch_processing": "📚 Batch Processing",
        "upload_image": "Upload Image",
        "image_url": "Or enter Image URL",
        "image_url_placeholder": "https://example.com/image.jpg",
        "prompt": "Prompt",
        "prompt_placeholder": "For example: Create a product description for online store",
        "generate_btn": "🚀 Generate Description",
        "result": "Result",
        "upload_images": "Upload Images",
        "prompts_multiline": "Prompts (one per line)",
        "prompts_placeholder": "Create a product description for online store\nCreate SEO Description for product\n...",
        "prompts_info": "Specify one prompt for all images or one prompt per image",
        "process_batch_btn": "🚀 Process Batch",
        "results": "Results",
        "examples_title": "💡 Example Prompts:",
        "example_1": "Create a product description for online store",
        "example_2": "Create an SEO description for a product with a maximum of 160 characters.",
        "example_3": "Create an attractive product description for marketplace",
        "example_4": "Describe image in detail for product catalog",
        "error_no_image": "Please upload an image or provide an image URL",
        "error_no_prompt": "Please enter a prompt",
        "error_no_images": "Please upload images",
        "error_no_prompts": "Please enter prompts (one per line)",
        "error_prompt_mismatch": "Number of prompts ({}) does not match number of images ({}). Specify either one prompt for all images or one prompt per image.",
        "error_generation": "Error generating description: {}",
        "error_url_load": "Error loading image from URL: {}",
        "loading_model": "Loading model: {}",
        "model_loaded": "Model {} successfully loaded on {}",
        "image_label": "=== Image {}: {} ===",
        "prompt_label": "Prompt: {}",
        "result_label": "Result: {}",
        "model_size_warning": "⚠️ Note: Large models (8B+) may use CPU offloading if GPU memory is insufficient, which can slow down generation.",
        "quantization": "Quantization",
        "quantization_info": "Memory optimization (4-bit = ~75% less VRAM)",
        "quant_4bit": "4-bit (Recommended)",
        "quant_8bit": "8-bit (Better quality)",
        "quant_none": "None (Full precision)",
        "description_type": "Description Type",
        "description_type_info": "Select the type/style of description",
        "description_length": "Description Length",
        "description_length_info": "Select desired length",
        "num_variants": "Number of Variants",
        "num_variants_info": "Generate multiple description variants",
        "custom_prompt_override": "Custom Prompt (overrides type selection)",
        "custom_prompt_placeholder": "Enter your custom prompt here...",
        "status": "Status",
        "processing_time": "Processing time",
        "generating": "⏳ Generating...",
        "stop_btn": "🛑 Stop",
        "save_results": "💾 Save Results",
        "output_folder": "Output Folder Name",
        "output_folder_placeholder": "my_dataset",
        "export_format": "Export Format",
        "export_txt": "TXT (one file per image)",
        "export_json": "JSON (all results)",
        "export_csv": "CSV (table format)",
        "variant": "Variant",
        "copy_btn": "📋 Copy",
        "download_btn": "⬇️ Download",
        "generation_complete": "✅ Generation complete!",
        "seconds": "seconds",
        "processing_image": "Processing image",
        "of": "of",
        "extra_options": "Extra Options",
        "extra_options_info": "Additional modifiers for description",
        "character_name": "Character/Person Name",
        "character_name_placeholder": "e.g., John, Alice, or leave empty",
        "character_name_info": "If provided, will use this name instead of generic terms",
        "prompt_preset": "Prompt Preset",
        "prompt_preset_info": "Load a preset from prompts/ folder",
        "refresh_presets": "Refresh",
        "memory_usage": "Memory Usage",
        "download_result": "Download Result",
        "generation_stopped": "Generation stopped by user",
        "stopping": "Stopping..."
    },
    "ru": {
        "title": "Генератор описаний изображений Qwen VL",
        "header": "🖼️ Генератор описаний изображений на основе Qwen Vision Language Models",
        "subtitle": "Загрузите изображение и введите промт для генерации описания с помощью моделей Qwen VL.",
        "language": "Язык",
        "language_info": "Выберите язык",
        "model_selection": "Выбор модели",
        "model_info": "Выберите модель для генерации описаний",
        "advanced_params": "⚙️ Расширенные параметры",
        "max_tokens": "Макс. количество новых токенов",
        "max_tokens_info": "Максимальное количество токенов для генерации",
        "temperature": "Температура",
        "temperature_info": "Контролирует случайность генерации",
        "top_p": "Top-p (nucleus sampling)",
        "top_p_info": "Вероятностный порог для выборки токенов",
        "top_k": "Top-k",
        "top_k_info": "Количество наиболее вероятных токенов для рассмотрения",
        "seed": "Seed",
        "seed_info": "Seed для воспроизводимости (-1 для случайного)",
        "random_seed_btn": "🎲 Случайный seed",
        "single_processing": "📄 Одиночная обработка",
        "batch_processing": "📚 Пакетная обработка",
        "upload_image": "Загрузите изображение",
        "image_url": "Или введите URL изображения",
        "image_url_placeholder": "https://example.com/image.jpg",
        "prompt": "Промт",
        "prompt_placeholder": "Например: Создать описание товара для онлайн магазина",
        "generate_btn": "🚀 Генерировать описание",
        "result": "Результат",
        "upload_images": "Загрузите изображения",
        "prompts_multiline": "Промты (по одному на строку)",
        "prompts_placeholder": "Создать описание товара для онлайн магазина\nСоздать SEO-описание для товара\n...",
        "prompts_info": "Укажите один промт для всех изображений или по одному промту на каждое изображение",
        "process_batch_btn": "🚀 Обработать пакет",
        "results": "Результаты",
        "examples_title": "💡 Примеры промтов:",
        "example_1": "Создать описание товара ''  на русском языке",
        "example_2": "Создать SEO-описание для товара максимум 160 символов на русском языке",
        "example_3": "Создать привлекательное описание продукта для маркетплейса на русском языке",
        "example_4": "Детально описать изображение для каталога товаров на русском языке",
        "error_no_image": "Пожалуйста, загрузите изображение или укажите URL изображения",
        "error_no_prompt": "Пожалуйста, введите промт",
        "error_no_images": "Пожалуйста, загрузите изображения",
        "error_no_prompts": "Пожалуйста, введите промты (по одному на строку)",
        "error_prompt_mismatch": "Количество промтов ({}) не совпадает с количеством изображений ({}). Укажите либо один промт для всех изображений, либо по одному промту на каждое изображение.",
        "error_generation": "Ошибка при генерации описания: {}",
        "error_url_load": "Ошибка загрузки изображения по URL: {}",
        "loading_model": "Загрузка модели: {}",
        "model_loaded": "Модель {} успешно загружена на {}",
        "image_label": "=== Изображение {}: {} ===",
        "prompt_label": "Промт: {}",
        "result_label": "Результат: {}",
        "model_size_warning": "⚠️ Примечание: Большие модели (8B+) могут использовать выгрузку на CPU при недостатке памяти GPU, что может замедлить генерацию.",
        "quantization": "Квантизация",
        "quantization_info": "Оптимизация памяти (4-bit = ~75% меньше видеопамяти)",
        "quant_4bit": "4-bit (Рекомендуется)",
        "quant_8bit": "8-bit (Лучше качество)",
        "quant_none": "Нет (Полная точность)",
        "description_type": "Тип описания",
        "description_type_info": "Выберите тип/стиль описания",
        "description_length": "Длина описания",
        "description_length_info": "Выберите желаемую длину",
        "num_variants": "Количество вариантов",
        "num_variants_info": "Генерировать несколько вариантов описания",
        "custom_prompt_override": "Свой промпт (переопределяет выбор типа)",
        "custom_prompt_placeholder": "Введите свой промпт здесь...",
        "status": "Статус",
        "processing_time": "Время обработки",
        "generating": "⏳ Генерация...",
        "stop_btn": "🛑 Остановить",
        "save_results": "💾 Сохранить результаты",
        "output_folder": "Название папки",
        "output_folder_placeholder": "мой_датасет",
        "export_format": "Формат экспорта",
        "export_txt": "TXT (один файл на изображение)",
        "export_json": "JSON (все результаты)",
        "export_csv": "CSV (табличный формат)",
        "variant": "Вариант",
        "copy_btn": "📋 Копировать",
        "download_btn": "⬇️ Скачать",
        "generation_complete": "✅ Генерация завершена!",
        "seconds": "секунд",
        "processing_image": "Обработка изображения",
        "of": "из",
        "extra_options": "Дополнительные опции",
        "extra_options_info": "Дополнительные модификаторы описания",
        "character_name": "Имя персонажа/человека",
        "character_name_placeholder": "напр., Иван, Алиса, или оставьте пустым",
        "character_name_info": "Если указано, будет использоваться вместо общих терминов",
        "prompt_preset": "Пресет промпта",
        "prompt_preset_info": "Загрузить пресет из папки prompts/",
        "refresh_presets": "Обновить",
        "memory_usage": "Использование памяти",
        "download_result": "Скачать результат",
        "generation_stopped": "Генерация остановлена пользователем",
        "stopping": "Остановка..."
    },
    "zh": {
        "title": "Qwen VL 图像描述生成器",
        "header": "🖼️ 基于 Qwen Vision Language Models 的图像描述生成器",
        "subtitle": "上传图像并输入提示词，使用 Qwen VL 模型生成描述。",
        "language": "语言",
        "language_info": "选择语言",
        "model_selection": "模型选择",
        "model_info": "选择用于生成描述的模型",
        "advanced_params": "⚙️ 高级参数",
        "max_tokens": "最大新令牌数",
        "max_tokens_info": "生成的最大令牌数",
        "temperature": "温度",
        "temperature_info": "控制生成的随机性",
        "top_p": "Top-p（核采样）",
        "top_p_info": "令牌采样的概率阈值",
        "top_k": "Top-k",
        "top_k_info": "考虑的最可能令牌数",
        "seed": "随机种子",
        "seed_info": "用于可重现性的种子（-1 表示随机）",
        "random_seed_btn": "🎲 随机种子",
        "single_processing": "📄 单张处理",
        "batch_processing": "📚 批量处理",
        "upload_image": "上传图像",
        "image_url": "或输入图像URL",
        "image_url_placeholder": "https://example.com/image.jpg",
        "prompt": "提示词",
        "prompt_placeholder": "例如：为在线商店创建产品描述",
        "generate_btn": "🚀 生成描述",
        "result": "结果",
        "upload_images": "上传图像",
        "prompts_multiline": "提示词（每行一个）",
        "prompts_placeholder": "为在线商店创建产品描述\n为产品创建SEO描述\n...",
        "prompts_info": "为所有图像指定一个提示词，或为每个图像指定一个提示词",
        "process_batch_btn": "🚀 处理批次",
        "results": "结果",
        "examples_title": "💡 示例提示词：",
        "example_1": "为在线商店创建产品描述",
        "example_2": "为产品创建SEO描述最多 160 个字符",
        "example_3": "为市场创建有吸引力的产品描述",
        "example_4": "详细描述产品目录的图像",
        "error_no_image": "请上传图像或提供图像URL",
        "error_no_prompt": "请输入提示词",
        "error_no_images": "请上传图像",
        "error_no_prompts": "请输入提示词（每行一个）",
        "error_prompt_mismatch": "提示词数量（{}）与图像数量（{}）不匹配。请为所有图像指定一个提示词，或为每个图像指定一个提示词。",
        "error_generation": "生成描述时出错：{}",
        "error_url_load": "从URL加载图像时出错：{}",
        "loading_model": "正在加载模型：{}",
        "model_loaded": "模型 {} 已成功加载到 {}",
        "image_label": "=== 图像 {}: {} ===",
        "prompt_label": "提示词：{}",
        "result_label": "结果：{}",
        "model_size_warning": "⚠️ 注意：如果 GPU 内存不足，大型模型（8B+）可能会使用 CPU 卸载，这可能会减慢生成速度。",
        "quantization": "量化",
        "quantization_info": "内存优化（4位 = 减少约75%显存）",
        "quant_4bit": "4位（推荐）",
        "quant_8bit": "8位（更高质量）",
        "quant_none": "无（全精度）",
        "description_type": "描述类型",
        "description_type_info": "选择描述类型/风格",
        "description_length": "描述长度",
        "description_length_info": "选择所需长度",
        "num_variants": "变体数量",
        "num_variants_info": "生成多个描述变体",
        "custom_prompt_override": "自定义提示词（覆盖类型选择）",
        "custom_prompt_placeholder": "在此输入您的自定义提示词...",
        "status": "状态",
        "processing_time": "处理时间",
        "generating": "⏳ 生成中...",
        "stop_btn": "🛑 停止",
        "save_results": "💾 保存结果",
        "output_folder": "输出文件夹名称",
        "output_folder_placeholder": "我的数据集",
        "export_format": "导出格式",
        "export_txt": "TXT（每张图片一个文件）",
        "export_json": "JSON（所有结果）",
        "export_csv": "CSV（表格格式）",
        "variant": "变体",
        "copy_btn": "📋 复制",
        "download_btn": "⬇️ 下载",
        "generation_complete": "✅ 生成完成！",
        "seconds": "秒",
        "processing_image": "处理图像",
        "of": "/",
        "extra_options": "额外选项",
        "extra_options_info": "描述的额外修饰符",
        "character_name": "角色/人物名称",
        "character_name_placeholder": "例如：小明、小红，或留空",
        "character_name_info": "如果提供，将使用此名称代替通用术语",
        "prompt_preset": "提示词预设",
        "prompt_preset_info": "从prompts/文件夹加载预设",
        "refresh_presets": "刷新",
        "memory_usage": "内存使用",
        "download_result": "下载结果",
        "generation_stopped": "用户停止了生成",
        "stopping": "停止中..."
    }
}

# Default language
current_language = "en"

def get_text(key: str) -> str:
    """Get translated text for the current language"""
    return TRANSLATIONS[current_language].get(key, key)

def get_description_types() -> list:
    """Get description types for current language"""
    return list(DESCRIPTION_TYPES.get(current_language, DESCRIPTION_TYPES["en"]).keys())

def get_description_lengths() -> list:
    """Get description lengths for current language"""
    return list(DESCRIPTION_LENGTHS.get(current_language, DESCRIPTION_LENGTHS["en"]).keys())

def get_extra_options() -> list:
    """Get extra options for current language"""
    return list(EXTRA_OPTIONS.get(current_language, EXTRA_OPTIONS["en"]).keys())

def get_extra_option_prompt(option: str) -> str:
    """Get the prompt text for an extra option"""
    options_dict = EXTRA_OPTIONS.get(current_language, EXTRA_OPTIONS["en"])
    return options_dict.get(option, "")

def build_prompt(
    description_type: str,
    description_length: str,
    custom_prompt: str,
    base_prompt: str = "",
    extra_options: list = None,
    character_name: str = ""
) -> str:
    """Build the final prompt based on type, length, custom input, extra options and character name"""
    # If custom prompt is provided, use it (but still add character name if present)
    if custom_prompt and custom_prompt.strip():
        final_prompt = custom_prompt.strip()
    else:
        # Get type prompt
        types_dict = DESCRIPTION_TYPES.get(current_language, DESCRIPTION_TYPES["en"])
        type_prompt = types_dict.get(description_type, "")

        # If type is Custom, use base prompt
        if not type_prompt:
            type_prompt = base_prompt if base_prompt else "Describe this image."

        # Get length modifier
        lengths_dict = DESCRIPTION_LENGTHS.get(current_language, DESCRIPTION_LENGTHS["en"])
        length_modifier = lengths_dict.get(description_length, "")

        # Combine type and length
        if length_modifier:
            final_prompt = f"{type_prompt} {length_modifier}"
        else:
            final_prompt = type_prompt

    # Add character name instruction if provided
    if character_name and character_name.strip():
        name = character_name.strip()
        if current_language == "ru":
            final_prompt += f" Если на изображении есть человек или персонаж, используй имя '{name}' вместо общих терминов."
        elif current_language == "zh":
            final_prompt += f" 如果图片中有人或角色，请使用名字'{name}'而不是通用术语。"
        else:
            final_prompt += f" If there is a person or character in the image, use the name '{name}' instead of generic terms."

    # Add extra options
    if extra_options:
        for option in extra_options:
            option_prompt = get_extra_option_prompt(option)
            if option_prompt:
                final_prompt += f" {option_prompt}"

    return final_prompt

def create_output_folder(folder_name: str = None) -> str:
    """Create and return path to output folder"""
    if folder_name and folder_name.strip():
        # Sanitize folder name
        safe_name = "".join(c for c in folder_name if c.isalnum() or c in "_ -").strip()
        if not safe_name:
            safe_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        safe_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    folder_path = os.path.join(DATASETS_DIR, safe_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def save_results_txt(results: List[dict], output_folder: str) -> List[str]:
    """Save results as individual TXT files"""
    saved_files = []
    for result in results:
        filename = os.path.splitext(os.path.basename(result.get("image_path", "image")))[0]
        txt_path = os.path.join(output_folder, f"{filename}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            if isinstance(result.get("descriptions"), list):
                for i, desc in enumerate(result["descriptions"], 1):
                    f.write(f"=== Variant {i} ===\n{desc}\n\n")
            else:
                f.write(result.get("description", ""))
        saved_files.append(txt_path)

        # Copy image if exists
        if result.get("image_path") and os.path.exists(result["image_path"]):
            img_dest = os.path.join(output_folder, os.path.basename(result["image_path"]))
            if not os.path.exists(img_dest):
                shutil.copy2(result["image_path"], img_dest)

    return saved_files

def save_results_json(results: List[dict], output_folder: str) -> str:
    """Save all results as a single JSON file"""
    json_path = os.path.join(output_folder, "results.json")
    export_data = []
    for result in results:
        export_data.append({
            "image": os.path.basename(result.get("image_path", "")),
            "prompt": result.get("prompt", ""),
            "descriptions": result.get("descriptions", [result.get("description", "")]),
            "timestamp": datetime.now().isoformat()
        })

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)

    return json_path

def save_results_csv(results: List[dict], output_folder: str) -> str:
    """Save all results as a CSV file"""
    csv_path = os.path.join(output_folder, "results.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Prompt", "Description", "Variant"])

        for result in results:
            image_name = os.path.basename(result.get("image_path", ""))
            prompt = result.get("prompt", "")
            descriptions = result.get("descriptions", [result.get("description", "")])

            for i, desc in enumerate(descriptions, 1):
                writer.writerow([image_name, prompt, desc, i])

    return csv_path

class ImageDescriptionGenerator:
    def __init__(self):
        self.model = None
        self.processor = None
        self.current_model_name = None
        self.current_quantization = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_name: str, quantization: str = "4-bit"):
        """Загрузка модели с квантизацией BitsAndBytes"""
        # Проверяем нужна ли перезагрузка
        if (self.current_model_name == model_name and
            self.current_quantization == quantization and
            self.model is not None):
            return

        print(get_text("loading_model").format(model_name))
        print(f"Quantization: {quantization}")

        # Предупреждение о больших моделях
        if "8B" in model_name or "32B" in model_name or "30B" in model_name:
            print(get_text("model_size_warning"))

        # Освобождаем память от предыдущей модели
        if self.model is not None:
            del self.model
            del self.processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        # Настройка квантизации BitsAndBytes
        bnb_config = None
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        if quantization == "4-bit" and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            print("Using 4-bit quantization (NF4 + double quant) - ~75% VRAM reduction")
        elif quantization == "8-bit" and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            print("Using 8-bit quantization - ~50% VRAM reduction")
        else:
            print("Using full precision (bfloat16/float32)")

        # Загружаем новую модель с подавлением предупреждений
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            load_kwargs = {
                "device_map": "auto",
                "trust_remote_code": True,
            }

            if bnb_config is not None:
                load_kwargs["quantization_config"] = bnb_config
            else:
                load_kwargs["torch_dtype"] = dtype

            # SDPA (Scaled Dot Product Attention) - встроен в PyTorch 2.0+
            # Быстрый и работает на Windows без доп. зависимостей
            if torch.cuda.is_available():
                load_kwargs["attn_implementation"] = "sdpa"
                print("Using SDPA attention (PyTorch native, fast)")

            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                **load_kwargs
            )
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        self.current_model_name = model_name
        self.current_quantization = quantization
        print(get_text("model_loaded").format(model_name, self.device))
    
    def generate_description(
        self,
        image_path: str,
        prompt: str,
        model_name: str,
        quantization: str = "4-bit",
        max_new_tokens: int = 1024,
        temperature: float = 0.6,
        top_p: float = 0.9,
        top_k: int = 50,
        seed: int = -1
    ) -> str:
        """Генерация описания для одного изображения"""
        try:
            # Загружаем модель если необходимо
            self.load_model(model_name, quantization)
            
            # Устанавливаем seed если указан
            if seed != -1:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
            
            # Подготавливаем сообщения для модели
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            # Подготавливаем текст для модели
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Обрабатываем изображение и текст
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            # Генерируем ответ (inference_mode быстрее чем no_grad)
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True if temperature > 0 else False,
                    use_cache=True,  # KV кэширование для ускорения
                )
            
            # Декодируем результат
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            return output_text[0]
            
        except Exception as e:
            return get_text("error_generation").format(str(e))

# Создаем глобальный экземпляр генератора
generator = ImageDescriptionGenerator()

def load_image_from_url(url: str) -> Optional[str]:
    """Load image from URL and save to temporary file"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Load image from response content
        image = Image.open(BytesIO(response.content))
        
        # Save to temporary file
        temp_path = os.path.join(TEMP_DIR, "temp_url_image.jpg")
        image.save(temp_path)
        
        return temp_path
    except Exception as e:
        raise Exception(get_text("error_url_load").format(str(e)))

def process_single_image(
    image,
    video,
    image_url: str,
    description_type: str,
    description_length: str,
    custom_prompt: str,
    extra_options: list,
    character_name: str,
    num_variants: int,
    model_name: str,
    quantization: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    progress=gr.Progress(track_tqdm=True)
) -> Generator:
    """Обработка одного изображения/видео с генерацией нескольких вариантов"""
    global stop_generation_flag
    reset_stop_flag()
    start_time = time.time()

    # Check if we have either uploaded image/video or URL
    if image is None and video is None and not image_url.strip():
        yield get_text("error_no_image"), "", [], None
        return

    # Build prompt from type, length, custom, extra options and character name
    final_prompt = build_prompt(
        description_type, description_length, custom_prompt,
        extra_options=extra_options or [],
        character_name=character_name or ""
    )
    if not final_prompt.strip():
        yield get_text("error_no_prompt"), "", [], None
        return

    temp_path = None
    num_variants = int(num_variants) if num_variants and str(num_variants).strip() else 1

    try:
        # Priority: URL > video > image
        if image_url and image_url.strip():
            yield f"⏳ {get_text('generating')} (loading from URL...)", final_prompt, [], None
            image_path = load_image_from_url(image_url.strip())
            temp_path = image_path
        elif video is not None:
            # Video uploaded - use it directly
            image_path = video
        elif hasattr(image, 'shape'):
            # Сохраняем временное изображение если это numpy array
            temp_path = os.path.join(TEMP_DIR, "temp_image.jpg")
            Image.fromarray(image).save(temp_path)
            image_path = temp_path
        else:
            image_path = image

        results = []
        for i in range(num_variants):
            # Check stop flag
            if stop_generation_flag:
                elapsed_time = time.time() - start_time
                yield f"🛑 {get_text('generation_stopped')} ({get_text('processing_time')}: {elapsed_time:.1f} {get_text('seconds')})", final_prompt, results, None
                return

            variant_seed = seed if seed == -1 else seed + i
            memory_info = get_memory_info()
            status_msg = f"⏳ {get_text('generating')} ({get_text('variant')} {i+1}/{num_variants}) | {memory_info}"
            yield status_msg, final_prompt, results, None

            result = generator.generate_description(
                image_path=image_path,
                prompt=final_prompt,
                model_name=model_name,
                quantization=quantization,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=variant_seed
            )
            results.append(result)

        # Calculate processing time
        elapsed_time = time.time() - start_time
        memory_info = get_memory_info()
        final_status = f"{get_text('generation_complete')} ({get_text('processing_time')}: {elapsed_time:.1f} {get_text('seconds')}) | {memory_info}"

        # Prepare download file
        download_path = None
        if results:
            all_text = "\n\n".join([f"=== Variant {i+1} ===\n{r}" for i, r in enumerate(results)])
            download_path = save_text_to_file(all_text, f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

        yield final_status, final_prompt, results, download_path

    except Exception as e:
        yield f"❌ Error: {str(e)}", final_prompt, [], None
    finally:
        # Удаляем временный файл
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

        # Clean up GPU memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def process_batch_images(
    files: List,
    description_type: str,
    description_length: str,
    custom_prompt: str,
    extra_options: list,
    character_name: str,
    num_variants: int,
    output_folder_name: str,
    export_formats: List[str],
    model_name: str,
    quantization: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    progress=gr.Progress(track_tqdm=True)
) -> Generator:
    """Обработка пакета изображений с прогресс-баром и экспортом"""
    global stop_generation_flag
    reset_stop_flag()
    start_time = time.time()

    if not files:
        yield get_text("error_no_images"), "", None
        return

    # Build prompt from type, length, custom, extra options and character name
    final_prompt = build_prompt(
        description_type, description_length, custom_prompt,
        extra_options=extra_options or [],
        character_name=character_name or ""
    )
    if not final_prompt.strip():
        yield get_text("error_no_prompts"), "", None
        return

    num_variants = int(num_variants)
    total_files = len(files)
    all_results = []
    output_lines = []

    # Create output folder
    output_folder = create_output_folder(output_folder_name)

    for idx, file in enumerate(progress.tqdm(files, desc="Processing images")):
        # Check stop flag
        if stop_generation_flag:
            elapsed_time = time.time() - start_time
            final_status = f"🛑 {get_text('generation_stopped')}\n"
            final_status += f"📊 {idx} {get_text('of')} {total_files} images processed in {elapsed_time:.1f} {get_text('seconds')}"
            yield final_status, "\n".join(output_lines), None
            return

        image_path = file.name if hasattr(file, 'name') else file
        filename = os.path.basename(image_path)

        # Status update with memory info
        memory_info = get_memory_info()
        status_msg = f"⏳ {get_text('processing_image')} {idx + 1} {get_text('of')} {total_files}: {filename} | {memory_info}"
        yield status_msg, "\n".join(output_lines), None

        descriptions = []
        for v in range(num_variants):
            # Check stop flag between variants
            if stop_generation_flag:
                break

            variant_seed = seed if seed == -1 else seed + idx * num_variants + v

            result = generator.generate_description(
                image_path=image_path,
                prompt=final_prompt,
                model_name=model_name,
                quantization=quantization,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=variant_seed
            )
            descriptions.append(result)

        if not descriptions:
            continue

        # Store result
        all_results.append({
            "image_path": image_path,
            "prompt": final_prompt,
            "descriptions": descriptions,
            "description": descriptions[0] if descriptions else ""
        })

        # Add to output
        output_lines.append(f"{'='*50}")
        output_lines.append(get_text("image_label").format(idx + 1, filename))
        output_lines.append(get_text("prompt_label").format(final_prompt))

        for v_idx, desc in enumerate(descriptions, 1):
            if num_variants > 1:
                output_lines.append(f"\n--- {get_text('variant')} {v_idx} ---")
            output_lines.append(f"{desc}\n")

        yield status_msg, "\n".join(output_lines), None

    # Save results in selected formats
    saved_paths = []
    if export_formats and all_results:
        if "TXT" in export_formats:
            txt_files = save_results_txt(all_results, output_folder)
            saved_paths.extend(txt_files)

        if "JSON" in export_formats:
            json_path = save_results_json(all_results, output_folder)
            saved_paths.append(json_path)

        if "CSV" in export_formats:
            csv_path = save_results_csv(all_results, output_folder)
            saved_paths.append(csv_path)

    # Calculate total time
    elapsed_time = time.time() - start_time
    memory_info = get_memory_info()

    # Final status
    final_status = f"{get_text('generation_complete')}\n"
    final_status += f"📊 {total_files} images processed in {elapsed_time:.1f} {get_text('seconds')} | {memory_info}\n"
    if saved_paths:
        final_status += f"💾 Results saved to: {output_folder}"

    # Prepare download file for batch results
    download_path = None
    if all_results:
        download_path = save_text_to_file("\n".join(output_lines), f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    yield final_status, "\n".join(output_lines), download_path

    # Clean up GPU memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def random_seed() -> int:
    """Генерация случайного seed"""
    return random.randint(0, 2**32 - 1)

def update_examples():
    return [
        [get_text("example_1")],
        [get_text("example_2")],
        [get_text("example_3")],
        [get_text("example_4")]
    ]

def create_interface():
    """Create Gradio interface with current language and beautiful styling"""
    with gr.Blocks(
        title=get_text("title"),
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="purple", neutral_hue="slate"),
        css=CUSTOM_CSS
    ) as demo:
        # Beautiful gradient header with credits
        gr.HTML("""
        <div class="main-header">
            <h1>🖼️ Qwen VL Image Description Generator PRO</h1>
            <p>Advanced AI-powered image description with multiple styles and export options</p>
            <p style="font-size: 0.9rem; margin-top: 0.75rem; opacity: 0.85;">
                Developed by <a href="#">NeuralSoft</a> |
                Powered by <a href="https://huggingface.co/Qwen" target="_blank">Qwen Vision Models</a> |
                <a href="https://github.com/timoncool/qwen3-vl" target="_blank">GitHub</a>
            </p>
        </div>
        """)

        # Header that will be updated for language
        header_md = gr.Markdown(f"""
        {get_text("subtitle")}
        """)
        
        # Общие настройки - модель, квантизация и язык
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=[
                    # Abliterated models (без цензуры) - рекомендуемые
                    ("2B Instruct Abliterated (~1.2GB 4-bit)", "huihui-ai/Huihui-Qwen3-VL-2B-Instruct-abliterated"),
                    ("2B Thinking Abliterated (~1.2GB 4-bit)", "huihui-ai/Huihui-Qwen3-VL-2B-Thinking-abliterated"),
                    ("4B Instruct Abliterated (~2.5GB 4-bit)", "huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated"),
                    ("4B Thinking Abliterated (~2.5GB 4-bit)", "huihui-ai/Huihui-Qwen3-VL-4B-Thinking-abliterated"),
                    ("8B Instruct Abliterated (~5GB 4-bit)", "huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated"),
                    ("8B Thinking Abliterated (~5GB 4-bit)", "huihui-ai/Huihui-Qwen3-VL-8B-Thinking-abliterated"),
                    ("30B-A3B MoE Instruct Abliterated (~18GB 4-bit)", "huihui-ai/Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated"),
                    ("30B-A3B MoE Thinking Abliterated (~18GB 4-bit)", "huihui-ai/Huihui-Qwen3-VL-30B-A3B-Thinking-abliterated"),
                    ("32B Instruct Abliterated (~18GB 4-bit)", "huihui-ai/Huihui-Qwen3-VL-32B-Instruct-abliterated"),
                    ("32B Thinking Abliterated (~18GB 4-bit)", "huihui-ai/Huihui-Qwen3-VL-32B-Thinking-abliterated"),
                    # Original Qwen models
                    ("Qwen 2B Instruct (~1.2GB 4-bit)", "Qwen/Qwen3-VL-2B-Instruct"),
                    ("Qwen 4B Instruct (~2.5GB 4-bit)", "Qwen/Qwen3-VL-4B-Instruct"),
                    ("Qwen 8B Instruct (~5GB 4-bit)", "Qwen/Qwen3-VL-8B-Instruct"),
                ],
                value="huihui-ai/Huihui-Qwen3-VL-2B-Instruct-abliterated",
                label=get_text("model_selection"),
                info=get_text("model_info"),
                scale=3
            )
            quantization_dropdown = gr.Dropdown(
                choices=[
                    ("4-bit (Recommended)", "4-bit"),
                    ("8-bit (Better quality)", "8-bit"),
                    ("None (Full precision)", "none"),
                ],
                value="4-bit",
                label=get_text("quantization"),
                info=get_text("quantization_info"),
                scale=1
            )
            language_dropdown = gr.Dropdown(
                choices=[("English", "en"), ("Русский", "ru"), ("中文", "zh")],
                value=current_language,
                label=get_text("language"),
                info=get_text("language_info"),
                scale=1
            )
        
        # Расширенные параметры
        advanced_accordion = gr.Accordion(get_text("advanced_params"), open=False)
        with advanced_accordion:
            with gr.Row():
                max_tokens_slider = gr.Slider(
                    minimum=1,
                    maximum=4096,
                    value=1024,
                    step=1,
                    label=get_text("max_tokens"),
                    info=get_text("max_tokens_info")
                )
                temperature_slider = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.6,
                    step=0.1,
                    label=get_text("temperature"),
                    info=get_text("temperature_info")
                )
            
            with gr.Row():
                top_p_slider = gr.Slider(
                    minimum=0.05,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label=get_text("top_p"),
                    info=get_text("top_p_info")
                )
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=1000,
                    value=50,
                    step=1,
                    label=get_text("top_k"),
                    info=get_text("top_k_info")
                )
            
            with gr.Row():
                seed_number = gr.Number(
                    value=-1,
                    label=get_text("seed"),
                    info=get_text("seed_info"),
                    precision=0
                )
                random_seed_btn = gr.Button(get_text("random_seed_btn"), size="sm")
        
        # Вкладки для одиночной и пакетной обработки
        tabs = gr.Tabs()
        with tabs:
            # Вкладка одиночной обработки
            single_tab = gr.TabItem(get_text("single_processing"))
            with single_tab:
                with gr.Row():
                    with gr.Column(scale=1, elem_classes="card-style"):
                        gr.Markdown("### 📷 Input")
                        
                        # Tabs for Image and Video
                        with gr.Tabs():
                            with gr.TabItem("🖼️ Image"):
                                single_image = gr.Image(
                                    type="numpy",
                                    label=get_text("upload_image"),
                                    height=300
                                )
                                single_image_url = gr.Textbox(
                                    label=get_text("image_url"),
                                    placeholder=get_text("image_url_placeholder"),
                                    lines=1
                                )
                            with gr.TabItem("🎥 Video"):
                                single_video = gr.Video(
                                    label="Upload Video",
                                    height=300
                                )

                        gr.Markdown("### 📝 Description Settings")

                        # Prompt preset dropdown with save option
                        with gr.Row():
                            single_preset = gr.Dropdown(
                                choices=list(load_prompt_presets().keys()),
                                value="None",
                                label=get_text("prompt_preset"),
                                info=get_text("prompt_preset_info"),
                                scale=3
                            )
                            single_refresh_presets = gr.Button(
                                get_text("refresh_presets"),
                                size="sm",
                                scale=1
                            )
                        
                        # Save preset accordion
                        with gr.Accordion("💾 Save Preset", open=False):
                            with gr.Row():
                                single_save_preset_name = gr.Textbox(
                                    label="Preset Name",
                                    placeholder="my_preset",
                                    scale=2
                                )
                                single_save_preset_btn = gr.Button(
                                    "Save",
                                    size="sm",
                                    scale=1
                                )
                            single_save_preset_status = gr.Markdown("")

                        single_desc_type = gr.Dropdown(
                            choices=get_description_types(),
                            value=get_description_types()[0],
                            label=get_text("description_type"),
                            info=get_text("description_type_info")
                        )
                        single_desc_length = gr.Dropdown(
                            choices=get_description_lengths(),
                            value=get_description_lengths()[0],
                            label=get_text("description_length"),
                            info=get_text("description_length_info")
                        )
                        single_num_variants = gr.Slider(
                            minimum=1,
                            maximum=5,
                            value=1,
                            step=1,
                            label=get_text("num_variants"),
                            info=get_text("num_variants_info")
                        )

                        # Character name field
                        single_character_name = gr.Textbox(
                            label=get_text("character_name"),
                            placeholder=get_text("character_name_placeholder"),
                            info=get_text("character_name_info"),
                            lines=1
                        )

                        # Extra options
                        with gr.Accordion(get_text("extra_options"), open=False):
                            single_extra_options = gr.CheckboxGroup(
                                choices=get_extra_options(),
                                value=[],
                                label="",
                                info=get_text("extra_options_info")
                            )

                        with gr.Accordion(get_text("custom_prompt_override"), open=False):
                            single_custom_prompt = gr.Textbox(
                                placeholder=get_text("custom_prompt_placeholder"),
                                lines=3,
                                label=""
                            )

                        with gr.Row():
                            single_submit_btn = gr.Button(
                                get_text("generate_btn"),
                                variant="primary",
                                elem_classes="generate-btn",
                                scale=3
                            )
                            single_stop_btn = gr.Button(
                                get_text("stop_btn"),
                                variant="stop",
                                scale=1
                            )

                    with gr.Column(scale=1, elem_classes="card-style"):
                        gr.Markdown("### 📊 Results")
                        single_status = gr.Textbox(
                            label=get_text("status"),
                            interactive=False,
                            elem_classes="status-box"
                        )
                        single_prompt_used = gr.Textbox(
                            label="Prompt Used",
                            interactive=False,
                            lines=2
                        )

                        # Multiple variant outputs
                        single_outputs = []
                        for i in range(5):
                            with gr.Group(visible=(i == 0)) as variant_group:
                                variant_output = gr.Textbox(
                                    label=f"{get_text('variant')} {i+1}" if i > 0 else get_text("result"),
                                    lines=8,
                                    show_copy_button=True
                                )
                                single_outputs.append((variant_group, variant_output))

                        # Download result file
                        single_download = gr.File(
                            label=get_text("download_result"),
                            visible=True
                        )

                # Кликабельные примеры промтов
                examples_title = gr.Markdown(f"### {get_text('examples_title')}")

            # Вкладка пакетной обработки
            batch_tab = gr.TabItem(get_text("batch_processing"))
            with batch_tab:
                with gr.Row():
                    with gr.Column(scale=1, elem_classes="card-style"):
                        gr.Markdown("### 📁 Input Files")
                        batch_images = gr.File(
                            file_count="multiple",
                            label=get_text("upload_images"),
                            file_types=["image"]
                        )

                        gr.Markdown("### 📝 Description Settings")

                        # Prompt preset dropdown
                        with gr.Row():
                            batch_preset = gr.Dropdown(
                                choices=list(load_prompt_presets().keys()),
                                value="None",
                                label=get_text("prompt_preset"),
                                info=get_text("prompt_preset_info"),
                                scale=4
                            )
                            batch_refresh_presets = gr.Button(
                                get_text("refresh_presets"),
                                size="sm",
                                scale=1
                            )

                        batch_desc_type = gr.Dropdown(
                            choices=get_description_types(),
                            value=get_description_types()[0],
                            label=get_text("description_type"),
                            info=get_text("description_type_info")
                        )
                        batch_desc_length = gr.Dropdown(
                            choices=get_description_lengths(),
                            value=get_description_lengths()[0],
                            label=get_text("description_length"),
                            info=get_text("description_length_info")
                        )
                        batch_num_variants = gr.Slider(
                            minimum=1,
                            maximum=3,
                            value=1,
                            step=1,
                            label=get_text("num_variants"),
                            info=get_text("num_variants_info")
                        )

                        # Character name field
                        batch_character_name = gr.Textbox(
                            label=get_text("character_name"),
                            placeholder=get_text("character_name_placeholder"),
                            info=get_text("character_name_info"),
                            lines=1
                        )

                        # Extra options
                        with gr.Accordion(get_text("extra_options"), open=False):
                            batch_extra_options = gr.CheckboxGroup(
                                choices=get_extra_options(),
                                value=[],
                                label="",
                                info=get_text("extra_options_info")
                            )

                        with gr.Accordion(get_text("custom_prompt_override"), open=False):
                            batch_custom_prompt = gr.Textbox(
                                placeholder=get_text("custom_prompt_placeholder"),
                                lines=3,
                                label=""
                            )

                        gr.Markdown("### 💾 Export Settings")
                        batch_output_folder = gr.Textbox(
                            label=get_text("output_folder"),
                            placeholder=get_text("output_folder_placeholder"),
                            value=""
                        )
                        batch_export_formats = gr.CheckboxGroup(
                            choices=["TXT", "JSON", "CSV"],
                            value=["TXT"],
                            label=get_text("export_format")
                        )

                        with gr.Row():
                            batch_submit_btn = gr.Button(
                                get_text("process_batch_btn"),
                                variant="primary",
                                elem_classes="generate-btn",
                                scale=3
                            )
                            batch_stop_btn = gr.Button(
                                get_text("stop_btn"),
                                variant="stop",
                                scale=1
                            )

                    with gr.Column(scale=1, elem_classes="card-style"):
                        gr.Markdown("### 📊 Results")
                        batch_status = gr.Textbox(
                            label=get_text("status"),
                            interactive=False,
                            elem_classes="status-box"
                        )
                        batch_output = gr.Textbox(
                            label=get_text("results"),
                            lines=20,
                            show_copy_button=True
                        )

                        # Download result file
                        batch_download = gr.File(
                            label=get_text("download_result"),
                            visible=True
                        )
        
        # Обработчики событий
        def change_language(lang):
            global current_language
            current_language = lang

            # Return updated text for key components
            return [
                f"{get_text('subtitle')}",  # header_md
                gr.update(label=get_text("model_selection"), info=get_text("model_info")),  # model_dropdown
                gr.update(label=get_text("quantization"), info=get_text("quantization_info")),  # quantization_dropdown
                gr.update(label=get_text("language"), info=get_text("language_info")),  # language_dropdown
                gr.update(label=get_text("advanced_params")),  # advanced_accordion
                gr.update(value=get_text("generate_btn")),  # single_submit_btn
                gr.update(value=get_text("process_batch_btn")),  # batch_submit_btn
                gr.update(choices=get_description_types(), value=get_description_types()[0]),  # single_desc_type
                gr.update(choices=get_description_lengths(), value=get_description_lengths()[0]),  # single_desc_length
                gr.update(choices=get_description_types(), value=get_description_types()[0]),  # batch_desc_type
                gr.update(choices=get_description_lengths(), value=get_description_lengths()[0]),  # batch_desc_length
                gr.update(choices=get_extra_options(), value=[]),  # single_extra_options
                gr.update(choices=get_extra_options(), value=[]),  # batch_extra_options
                gr.update(value=get_text("stop_btn")),  # single_stop_btn
                gr.update(value=get_text("stop_btn")),  # batch_stop_btn
            ]

        language_dropdown.change(
            fn=change_language,
            inputs=language_dropdown,
            outputs=[
                header_md,
                model_dropdown,
                quantization_dropdown,
                language_dropdown,
                advanced_accordion,
                single_submit_btn,
                batch_submit_btn,
                single_desc_type,
                single_desc_length,
                batch_desc_type,
                batch_desc_length,
                single_extra_options,
                batch_extra_options,
                single_stop_btn,
                batch_stop_btn,
            ]
        )

        # Stop button handlers
        single_stop_btn.click(
            fn=stop_generation,
            outputs=single_status
        )

        batch_stop_btn.click(
            fn=stop_generation,
            outputs=batch_status
        )

        # Preset refresh handlers
        def refresh_presets():
            presets = load_prompt_presets()
            return gr.update(choices=list(presets.keys()), value="None")

        single_refresh_presets.click(
            fn=refresh_presets,
            outputs=single_preset
        )

        batch_refresh_presets.click(
            fn=refresh_presets,
            outputs=batch_preset
        )

        # Preset selection handlers (load preset text into custom prompt)
        def load_preset(preset_name):
            presets = load_prompt_presets()
            if preset_name and preset_name != "None":
                return presets.get(preset_name, "")
            return ""

        single_preset.change(
            fn=load_preset,
            inputs=single_preset,
            outputs=single_custom_prompt
        )

        batch_preset.change(
            fn=load_preset,
            inputs=batch_preset,
            outputs=batch_custom_prompt
        )

        random_seed_btn.click(
            fn=random_seed,
            outputs=seed_number
        )

        # Update variant visibility based on slider
        def update_variant_visibility(num_variants):
            updates = []
            for i in range(5):
                updates.append(gr.update(visible=(i < num_variants)))
            return updates

        single_num_variants.change(
            fn=update_variant_visibility,
            inputs=[single_num_variants],
            outputs=[group for group, _ in single_outputs]
        )

        # Single image processing with button lock
        def process_single_wrapper(image, video, image_url, desc_type, desc_length, custom_prompt,
                                   extra_options, character_name, num_variants,
                                   model_name, quantization, max_tokens, temperature, top_p, top_k, seed):
            # Disable button at start
            yield gr.update(value=get_text("generating"), interactive=False), "", "", *[gr.update(value="") for _ in range(5)], None

            results = []
            download_path = None

            # Process and yield results
            for status, prompt_used, results, download_path in process_single_image(
                image, video, image_url, desc_type, desc_length, custom_prompt,
                extra_options, character_name, num_variants,
                model_name, quantization, max_tokens, temperature, top_p, top_k, seed
            ):
                # Prepare outputs for each variant box
                variant_outputs = []
                for i in range(5):
                    if i < len(results):
                        variant_outputs.append(gr.update(value=results[i]))
                    else:
                        variant_outputs.append(gr.update(value=""))

                yield gr.update(value=get_text("generating"), interactive=False), status, prompt_used, *variant_outputs, download_path

            # Re-enable button at end
            final_outputs = []
            for i in range(5):
                if i < len(results):
                    final_outputs.append(gr.update(value=results[i]))
                else:
                    final_outputs.append(gr.update(value=""))

            yield gr.update(value=get_text("generate_btn"), interactive=True), status, prompt_used, *final_outputs, download_path

        single_submit_btn.click(
            fn=process_single_wrapper,
            inputs=[
                single_image,
                single_video,
                single_image_url,
                single_desc_type,
                single_desc_length,
                single_custom_prompt,
                single_extra_options,
                single_character_name,
                single_num_variants,
                model_dropdown,
                quantization_dropdown,
                max_tokens_slider,
                temperature_slider,
                top_p_slider,
                top_k_slider,
                seed_number
            ],
            outputs=[single_submit_btn, single_status, single_prompt_used] + [output for _, output in single_outputs] + [single_download]
        )
        
        # Preset handlers
        def refresh_presets_list():
            presets = load_prompt_presets()
            return gr.update(choices=list(presets.keys()), value="None")
        
        single_refresh_presets.click(
            fn=refresh_presets_list,
            outputs=single_preset
        )
        
        def load_preset(preset_name):
            if preset_name and preset_name != "None":
                presets = load_prompt_presets()
                return presets.get(preset_name, "")
            return ""
        
        single_preset.change(
            fn=load_preset,
            inputs=single_preset,
            outputs=single_custom_prompt
        )
        
        def save_preset(name, prompt):
            if not name:
                return "❌ Please enter a preset name"
            msg = save_prompt_preset(name, prompt)
            new_presets = list(load_prompt_presets().keys())
            return msg, gr.update(choices=new_presets, value=name if "successfully" in msg else None)
        
        single_save_preset_btn.click(
            fn=save_preset,
            inputs=[single_save_preset_name, single_custom_prompt],
            outputs=[single_save_preset_status, single_preset]
        )

        # Batch processing with button lock
        def process_batch_wrapper(files, desc_type, desc_length, custom_prompt,
                                  extra_options, character_name, num_variants,
                                  output_folder, export_formats, model_name, quantization,
                                  max_tokens, temperature, top_p, top_k, seed):
            # Disable button at start
            yield gr.update(value=get_text("generating"), interactive=False), "", "", None

            download_path = None

            # Process and yield results
            for status, output_text, download_path in process_batch_images(
                files, desc_type, desc_length, custom_prompt,
                extra_options, character_name, num_variants,
                output_folder, export_formats, model_name, quantization,
                max_tokens, temperature, top_p, top_k, seed
            ):
                yield gr.update(value=get_text("generating"), interactive=False), status, output_text, download_path

            # Re-enable button at end
            yield gr.update(value=get_text("process_batch_btn"), interactive=True), status, output_text, download_path

        batch_submit_btn.click(
            fn=process_batch_wrapper,
            inputs=[
                batch_images,
                batch_desc_type,
                batch_desc_length,
                batch_custom_prompt,
                batch_extra_options,
                batch_character_name,
                batch_num_variants,
                batch_output_folder,
                batch_export_formats,
                model_dropdown,
                quantization_dropdown,
                max_tokens_slider,
                temperature_slider,
                top_p_slider,
                top_k_slider,
                seed_number
            ],
            outputs=[batch_submit_btn, batch_status, batch_output, batch_download]
        )

        return demo

# Создаем интерфейс Gradio
demo = create_interface()

if __name__ == "__main__":
    # Enable queue for progress bar support
    demo.queue(max_size=20, default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=None,
        share=False,
        show_error=True,
        inbrowser=True
    )
