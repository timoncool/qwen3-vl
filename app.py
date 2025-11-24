import gradio as gr
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from PIL import Image
import random
import os
import warnings
from typing import List, Tuple, Optional
import requests
from io import BytesIO
import urllib.parse
import gc

# Suppress specific warnings
warnings.filterwarnings('ignore', message='.*meta device.*')

# Base directory for portable app
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(SCRIPT_DIR, "temp")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Create directories if they don't exist
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        "quant_none": "None (Full precision)"
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
        "prompts_placeholder": "Создать описание товара для онлайн магазина\nСоздать SEO Description для товара\n...",
        "prompts_info": "Укажите один промт для всех изображений или по одному промту на каждое изображение",
        "process_batch_btn": "🚀 Обработать пакет",
        "results": "Результаты",
        "examples_title": "💡 Примеры промтов:",
        "example_1": "Создать описание товара ''  на русском языке",
        "example_2": "Создать SEO Description для товара максимум 160 символов на русском языке",
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
        "quantization_info": "Оптимизация памяти (4-bit = ~75% меньше VRAM)",
        "quant_4bit": "4-bit (Рекомендуется)",
        "quant_8bit": "8-bit (Лучше качество)",
        "quant_none": "Нет (Полная точность)"
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
        "quant_none": "无（全精度）"
    }
}

# Default language
current_language = "en"

def get_text(key: str) -> str:
    """Get translated text for the current language"""
    return TRANSLATIONS[current_language].get(key, key)

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
    image_url: str,
    prompt: str,
    model_name: str,
    quantization: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int
) -> str:
    """Обработка одного изображения"""
    # Check if we have either uploaded image or URL
    if image is None and not image_url.strip():
        return get_text("error_no_image")

    if not prompt.strip():
        return get_text("error_no_prompt")

    temp_path = None

    try:
        # Priority: URL over uploaded image (if both provided, URL takes precedence)
        if image_url.strip():
            image_path = load_image_from_url(image_url.strip())
            temp_path = image_path
        elif hasattr(image, 'shape'):
            # Сохраняем временное изображение если это numpy array
            temp_path = os.path.join(TEMP_DIR, "temp_image.jpg")
            Image.fromarray(image).save(temp_path)
            image_path = temp_path
        else:
            image_path = image

        result = generator.generate_description(
            image_path=image_path,
            prompt=prompt,
            model_name=model_name,
            quantization=quantization,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed
        )

        return result

    except Exception as e:
        return str(e)
    finally:
        # Удаляем временный файл
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

def process_batch_images(
    files: List,
    prompts_text: str,
    model_name: str,
    quantization: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int
) -> str:
    """Обработка пакета изображений"""
    if not files:
        return get_text("error_no_images")

    if not prompts_text.strip():
        return get_text("error_no_prompts")

    # Разбиваем промты по строкам
    prompts = [p.strip() for p in prompts_text.split('\n') if p.strip()]

    if len(prompts) == 1:
        # Если один промт, используем его для всех изображений
        prompts = prompts * len(files)
    elif len(prompts) != len(files):
        return get_text("error_prompt_mismatch").format(len(prompts), len(files))

    results = []
    for idx, (file, prompt) in enumerate(zip(files, prompts), 1):
        image_path = file.name if hasattr(file, 'name') else file
        result = generator.generate_description(
            image_path=image_path,
            prompt=prompt,
            model_name=model_name,
            quantization=quantization,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed if seed == -1 else seed + idx - 1  # Разный seed для каждого изображения
        )
        results.append(get_text("image_label").format(idx, os.path.basename(image_path)) + "\n")
        results.append(get_text("prompt_label").format(prompt) + "\n")
        results.append(get_text("result_label").format(result) + "\n\n")

    return "".join(results)

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
    """Create Gradio interface with current language"""
    with gr.Blocks(title=get_text("title"), theme=gr.themes.Soft()) as demo:
        # Header that will be updated
        header_md = gr.Markdown(f"""
        # {get_text("header")}
        
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
                    with gr.Column(scale=1):
                        single_image = gr.Image(
                            type="numpy",
                            label=get_text("upload_image"),
                            height=350
                        )
                        single_image_url = gr.Textbox(
                            label=get_text("image_url"),
                            placeholder=get_text("image_url_placeholder"),
                            lines=1
                        )
                        single_prompt = gr.Textbox(
                            label=get_text("prompt"),
                            placeholder=get_text("prompt_placeholder"),
                            lines=3
                        )
                        single_submit_btn = gr.Button(get_text("generate_btn"), variant="primary")
                    
                    with gr.Column(scale=1):
                        single_output = gr.Textbox(
                            label=get_text("result"),
                            lines=15,
                            show_copy_button=True
                        )
                
                # Кликабельные примеры промтов
                examples_title = gr.Markdown(f"### {get_text('examples_title')}")
                single_examples = gr.Dataset(
                    components=[single_prompt],
                    samples=update_examples(),
                    type="values"
                )
            
            # Вкладка пакетной обработки
            batch_tab = gr.TabItem(get_text("batch_processing"))
            with batch_tab:
                with gr.Row():
                    with gr.Column(scale=1):
                        batch_images = gr.File(
                            file_count="multiple",
                            label=get_text("upload_images"),
                            file_types=["image"]
                        )
                        batch_prompts = gr.Textbox(
                            label=get_text("prompts_multiline"),
                            placeholder=get_text("prompts_placeholder"),
                            lines=5,
                            info=get_text("prompts_info")
                        )
                        batch_submit_btn = gr.Button(get_text("process_batch_btn"), variant="primary")
                    
                    with gr.Column(scale=1):
                        batch_output = gr.Textbox(
                            label=get_text("results"),
                            lines=20,
                            show_copy_button=True
                        )
        
        # Обработчики событий
        def change_language(lang):
            global current_language
            current_language = lang

            # Return updated text for all components
            return [
                f"""
        # {get_text("header")}

        {get_text("subtitle")}
        """,  # header_md
                gr.update(label=get_text("model_selection"), info=get_text("model_info")),  # model_dropdown
                gr.update(label=get_text("quantization"), info=get_text("quantization_info")),  # quantization_dropdown
                gr.update(label=get_text("language"), info=get_text("language_info")),  # language_dropdown
                gr.update(label=get_text("advanced_params")),  # advanced_accordion
                gr.update(label=get_text("max_tokens"), info=get_text("max_tokens_info")),  # max_tokens_slider
                gr.update(label=get_text("temperature"), info=get_text("temperature_info")),  # temperature_slider
                gr.update(label=get_text("top_p"), info=get_text("top_p_info")),  # top_p_slider
                gr.update(label=get_text("top_k"), info=get_text("top_k_info")),  # top_k_slider
                gr.update(label=get_text("seed"), info=get_text("seed_info")),  # seed_number
                gr.update(value=get_text("random_seed_btn")),  # random_seed_btn
                gr.update(label=get_text("single_processing")),  # single_tab
                gr.update(label=get_text("upload_image")),  # single_image
                gr.update(label=get_text("image_url"), placeholder=get_text("image_url_placeholder")),  # single_image_url
                gr.update(label=get_text("prompt"), placeholder=get_text("prompt_placeholder")),  # single_prompt
                gr.update(value=get_text("generate_btn")),  # single_submit_btn
                gr.update(label=get_text("result")),  # single_output
                f"### {get_text('examples_title')}",  # examples_title
                gr.update(samples=update_examples()),  # single_examples
                gr.update(label=get_text("batch_processing")),  # batch_tab
                gr.update(label=get_text("upload_images")),  # batch_images
                gr.update(label=get_text("prompts_multiline"), placeholder=get_text("prompts_placeholder"), info=get_text("prompts_info")),  # batch_prompts
                gr.update(value=get_text("process_batch_btn")),  # batch_submit_btn
                gr.update(label=get_text("results")),  # batch_output
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
                max_tokens_slider,
                temperature_slider,
                top_p_slider,
                top_k_slider,
                seed_number,
                random_seed_btn,
                single_tab,
                single_image,
                single_image_url,
                single_prompt,
                single_submit_btn,
                single_output,
                examples_title,
                single_examples,
                batch_tab,
                batch_images,
                batch_prompts,
                batch_submit_btn,
                batch_output,
            ]
        )

        single_examples.click(
            fn=lambda x: x[0] if x else "",
            inputs=[single_examples],
            outputs=[single_prompt]
        )
        
        random_seed_btn.click(
            fn=random_seed,
            outputs=seed_number
        )
        
        single_submit_btn.click(
            fn=process_single_image,
            inputs=[
                single_image,
                single_image_url,
                single_prompt,
                model_dropdown,
                quantization_dropdown,
                max_tokens_slider,
                temperature_slider,
                top_p_slider,
                top_k_slider,
                seed_number
            ],
            outputs=single_output
        )

        batch_submit_btn.click(
            fn=process_batch_images,
            inputs=[
                batch_images,
                batch_prompts,
                model_dropdown,
                quantization_dropdown,
                max_tokens_slider,
                temperature_slider,
                top_p_slider,
                top_k_slider,
                seed_number
            ],
            outputs=batch_output
        )
        
        return demo

# Создаем интерфейс Gradio
demo = create_interface()

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True
    )
