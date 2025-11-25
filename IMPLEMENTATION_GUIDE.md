# Qwen3-VL Multimodal Implementation Guide

This guide shows exactly how to add advanced multimodal features to the current application.

---

## Current vs. Proposed Architecture

### Current Message Structure (Single Media)

```python
# Current implementation (line 1162-1170 in app.py)
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",  # OR "video" - mutually exclusive
                "image": "path/to/single_image.jpg"
            },
            {
                "type": "text",
                "text": "Describe this image."
            },
        ],
    }
]
```

### Proposed Message Structure (Multiple Media)

```python
# Proposed: Support multiple images, videos, and interleaved text
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Compare these products:"},
            {"type": "image", "image": "product1.jpg"},
            {"type": "image", "image": "product2.jpg"},
            {"type": "text", "text": "And this demonstration:"},
            {"type": "video", "video": "demo.mp4"},
            {"type": "text", "text": "Which is better for beginners?"},
        ],
    }
]
```

---

## Implementation Plan

### Step 1: Update Backend (app.py)

#### 1.1 Modify `_prepare_inputs` Method

**Location**: Line ~1126 in `/home/user/qwen3-vl/app.py`

**Current signature:**
```python
def _prepare_inputs(
    self,
    media_path: str,  # Single path
    prompt: str,
    model_name: str,
    quantization: str,
    seed: int,
    is_video: bool = False
):
```

**New signature:**
```python
def _prepare_inputs(
    self,
    media_items: List[Dict[str, str]],  # List of {type, path}
    prompt: str,
    model_name: str,
    quantization: str,
    seed: int,
):
```

**New implementation:**
```python
def _prepare_inputs(
    self,
    media_items: List[Dict[str, str]],
    prompt: str,
    model_name: str,
    quantization: str,
    seed: int,
):
    """
    Prepare inputs for generation with support for multiple media items.

    Args:
        media_items: List of dicts with 'type' and 'path' keys
                    Example: [
                        {'type': 'image', 'path': 'img1.jpg'},
                        {'type': 'video', 'path': 'vid.mp4'},
                    ]
        prompt: Final text prompt
        model_name: Model to use
        quantization: Quantization setting
        seed: Random seed
    """
    # Load model if necessary
    self.load_model(model_name, quantization)

    if self.model is None or self.processor is None:
        raise Exception("Model not loaded")

    # Set seed
    if seed != -1:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    # Build content array
    content = []

    # Add all media items with optional labels
    for i, item in enumerate(media_items):
        media_type = item['type']
        media_path = item['path']

        # Add label if multiple items
        if len(media_items) > 1:
            label = item.get('label', f"{media_type.capitalize()} {i+1}")
            content.append({
                "type": "text",
                "text": f"{label}:"
            })

        # Add media
        if media_type == "video":
            content.append({
                "type": "video",
                "video": media_path,
            })
        else:  # image
            content.append({
                "type": "image",
                "image": media_path,
            })

    # Add final text prompt
    content.append({
        "type": "text",
        "text": prompt
    })

    # Create messages
    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    # Process with existing pipeline
    text = self.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = self.processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(self.device)
    return inputs
```

#### 1.2 Update `generate_description` Method

**Location**: Line ~1189 in app.py

**Change calls from:**
```python
inputs = self._prepare_inputs(
    image_path, prompt, model_name, quantization, seed, is_video
)
```

**To:**
```python
media_items = [{'type': 'image' if not is_video else 'video', 'path': image_path}]
inputs = self._prepare_inputs(
    media_items, prompt, model_name, quantization, seed
)
```

#### 1.3 Update `generate_description_stream` Method

Same change as 1.2 for the streaming version (line ~1236).

#### 1.4 Create New Method for Multi-Media

Add this new method to `ImageDescriptionGenerator` class:

```python
def generate_multi_media_description(
    self,
    media_items: List[Dict[str, str]],
    prompt: str,
    model_name: str,
    quantization: str = "4-bit",
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.8,
    top_k: int = 20,
    seed: int = -1,
) -> str:
    """
    Generate description for multiple images and/or videos.

    Args:
        media_items: List of {'type': 'image'|'video', 'path': '...', 'label': '...'}
        prompt: Text prompt
        model_name: Model to use
        ...other standard parameters...

    Returns:
        Generated description
    """
    try:
        inputs = self._prepare_inputs(
            media_items, prompt, model_name, quantization, seed
        )

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=temperature > 0,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return output_text.strip()

    except Exception as e:
        return f"Error: {str(e)}"
```

---

### Step 2: Update UI (Gradio Components)

#### 2.1 Add Multi-Image Tab

**Location**: Around line 1750 in app.py (in the Gradio interface section)

**Add new tab:**
```python
with gr.TabItem("🖼️ Множественные изображения"):
    with gr.Row():
        with gr.Column(scale=1):
            multi_images = gr.File(
                label="Загрузить несколько изображений",
                file_count="multiple",
                file_types=["image"],
                type="filepath"
            )
            multi_videos = gr.File(
                label="Загрузить видео (опционально)",
                file_count="multiple",
                file_types=["video"],
                type="filepath"
            )

            # Interleaved text option
            multi_use_labels = gr.Checkbox(
                label="Добавить автоматические метки",
                value=True,
                info="Автоматически добавить 'Изображение 1:', 'Изображение 2:' и т.д."
            )

            # Custom labels
            multi_custom_labels = gr.Textbox(
                label="Пользовательские метки (опционально)",
                placeholder="Одна метка на строку, по порядку загруженных файлов",
                lines=5,
                info="Например:\nПродукт A\nПродукт B\nПродукт C"
            )

            multi_prompt = gr.Textbox(
                label="Промпт",
                placeholder="Например: Сравните эти изображения и опишите их различия",
                lines=3
            )

            with gr.Accordion("⚙️ Расширенные параметры", open=False):
                multi_max_tokens = gr.Slider(
                    minimum=128,
                    maximum=4096,
                    value=1024,
                    step=128,
                    label="Макс. токены"
                )
                multi_temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Температура"
                )
                multi_seed = gr.Number(
                    label="Seed",
                    value=-1,
                    precision=0
                )

            multi_submit_btn = gr.Button(
                "🚀 Генерировать описание",
                variant="primary"
            )

        with gr.Column(scale=1):
            multi_status = gr.Textbox(
                label="Статус",
                lines=2
            )
            multi_output = gr.Textbox(
                label="Результат",
                lines=20
            )

            # Token usage info
            multi_token_info = gr.Markdown(
                "**Информация о токенах:**\n"
                "- Изображения: ~0\n"
                "- Видео: ~0\n"
                "- Всего: ~0"
            )
```

#### 2.2 Create Processing Function

```python
def process_multi_media(
    images,
    videos,
    use_labels,
    custom_labels,
    prompt,
    model_name,
    quantization,
    max_tokens,
    temperature,
    seed
):
    """Process multiple images and videos"""
    try:
        # Validate inputs
        if not images and not videos:
            return "Ошибка: Загрузите хотя бы одно изображение или видео", "", ""

        if not prompt:
            return "Ошибка: Введите промпт", "", ""

        # Build media items list
        media_items = []

        # Parse custom labels if provided
        labels = []
        if custom_labels:
            labels = [l.strip() for l in custom_labels.split('\n') if l.strip()]

        # Add images
        if images:
            for i, img_path in enumerate(images):
                item = {
                    'type': 'image',
                    'path': img_path
                }
                if use_labels:
                    if i < len(labels):
                        item['label'] = labels[i]
                    else:
                        item['label'] = f"Изображение {i+1}"
                media_items.append(item)

        # Add videos
        if videos:
            start_idx = len(media_items)
            for i, vid_path in enumerate(videos):
                item = {
                    'type': 'video',
                    'path': vid_path
                }
                if use_labels:
                    label_idx = start_idx + i
                    if label_idx < len(labels):
                        item['label'] = labels[label_idx]
                    else:
                        item['label'] = f"Видео {i+1}"
                media_items.append(item)

        # Calculate approximate tokens
        num_images = len([m for m in media_items if m['type'] == 'image'])
        num_videos = len([m for m in media_items if m['type'] == 'video'])
        approx_image_tokens = num_images * 1000  # ~1K per image
        approx_video_tokens = num_videos * 5000  # ~5K per video
        total_tokens = approx_image_tokens + approx_video_tokens

        token_info = (
            f"**Информация о токенах:**\n"
            f"- Изображения ({num_images}): ~{approx_image_tokens:,}\n"
            f"- Видео ({num_videos}): ~{approx_video_tokens:,}\n"
            f"- Всего (примерно): ~{total_tokens:,}\n\n"
            f"{'⚠️ Внимание: Много токенов! Может быть медленно.' if total_tokens > 20000 else '✅ Приемлемое количество токенов.'}"
        )

        status = f"Обработка {len(media_items)} элементов..."

        # Generate
        result = generator.generate_multi_media_description(
            media_items=media_items,
            prompt=prompt,
            model_name=model_name,
            quantization=quantization,
            max_new_tokens=max_tokens,
            temperature=temperature,
            seed=seed
        )

        final_status = f"✅ Готово! Обработано {len(media_items)} элементов."

        return final_status, result, token_info

    except Exception as e:
        return f"❌ Ошибка: {str(e)}", "", ""
```

#### 2.3 Connect UI to Function

```python
multi_submit_btn.click(
    fn=process_multi_media,
    inputs=[
        multi_images,
        multi_videos,
        multi_use_labels,
        multi_custom_labels,
        multi_prompt,
        model_dropdown,
        quantization_dropdown,
        multi_max_tokens,
        multi_temperature,
        multi_seed
    ],
    outputs=[
        multi_status,
        multi_output,
        multi_token_info
    ]
)
```

---

### Step 3: Testing

#### Test Case 1: Two Images
```
Images: cat.jpg, dog.jpg
Prompt: "Compare these two animals"
Expected: Comparison of cat and dog
```

#### Test Case 2: Five Product Images
```
Images: product1.jpg, product2.jpg, product3.jpg, product4.jpg, product5.jpg
Custom Labels: "Apple iPhone\nSamsung Galaxy\nGoogle Pixel\nXiaomi\nOnePlus"
Prompt: "Create a comparison table with: design, camera, price range"
Expected: Structured comparison table
```

#### Test Case 3: Images + Video
```
Images: screenshot1.jpg, screenshot2.jpg
Videos: demo.mp4
Prompt: "Does the video demonstrate the features shown in screenshots?"
Expected: Analysis comparing screenshots to video
```

#### Test Case 4: Document Pages
```
Images: page1.jpg, page2.jpg, page3.jpg, page4.jpg, page5.jpg
Prompt: "Summarize this document"
Expected: Multi-page document summary
```

---

## Advanced Features

### Feature 1: Interleaved Content Builder

Create a UI where users can build complex prompts:

```python
with gr.TabItem("🎨 Конструктор промпта"):
    # State to hold content items
    content_items = gr.State([])

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Добавить элемент:")

            add_type = gr.Radio(
                ["Текст", "Изображение", "Видео"],
                value="Текст",
                label="Тип элемента"
            )

            add_text = gr.Textbox(
                label="Текст",
                visible=True
            )
            add_image = gr.Image(
                label="Изображение",
                type="filepath",
                visible=False
            )
            add_video = gr.Video(
                label="Видео",
                visible=False
            )

            add_btn = gr.Button("➕ Добавить элемент")

            gr.Markdown("### Текущая структура:")
            content_preview = gr.JSON(
                label="Элементы промпта",
                value=[]
            )

            clear_btn = gr.Button("🗑️ Очистить все")
            generate_btn = gr.Button("🚀 Генерировать", variant="primary")

        with gr.Column(scale=1):
            builder_output = gr.Textbox(
                label="Результат",
                lines=25
            )

    # Toggle visibility based on type
    def update_inputs(choice):
        return (
            gr.update(visible=(choice == "Текст")),
            gr.update(visible=(choice == "Изображение")),
            gr.update(visible=(choice == "Видео"))
        )

    add_type.change(
        fn=update_inputs,
        inputs=[add_type],
        outputs=[add_text, add_image, add_video]
    )
```

### Feature 2: Multi-Turn Conversation

Add conversation history:

```python
with gr.TabItem("💬 Диалог"):
    conversation_history = gr.State([])

    with gr.Row():
        with gr.Column(scale=1):
            conv_image = gr.Image(
                label="Добавить изображение (опционально)",
                type="filepath"
            )
            conv_message = gr.Textbox(
                label="Ваше сообщение",
                lines=3
            )
            conv_send_btn = gr.Button("📤 Отправить", variant="primary")
            conv_clear_btn = gr.Button("🗑️ Очистить историю")

        with gr.Column(scale=2):
            conv_chatbot = gr.Chatbot(
                label="Разговор",
                height=600
            )

    def send_message(image, message, history):
        # Add user message to history
        # Generate response
        # Update chatbot display
        pass

    conv_send_btn.click(
        fn=send_message,
        inputs=[conv_image, conv_message, conversation_history],
        outputs=[conv_chatbot, conversation_history]
    )
```

### Feature 3: Token Budget Monitor

Real-time token calculation:

```python
def calculate_tokens(images, videos):
    """Calculate approximate token usage"""
    if images is None:
        num_images = 0
    else:
        num_images = len(images) if isinstance(images, list) else 1

    if videos is None:
        num_videos = 0
    else:
        num_videos = len(videos) if isinstance(videos, list) else 1

    # Approximate calculations
    image_tokens = num_images * 1000  # ~1K per image
    video_tokens = num_videos * 5000  # ~5K per video
    total = image_tokens + video_tokens

    # Warning levels
    if total > 50000:
        level = "🔴 КРИТИЧНО"
        warning = "Очень много токенов! Высокий риск OOM."
    elif total > 20000:
        level = "🟡 ВНИМАНИЕ"
        warning = "Много токенов. Генерация может быть медленной."
    else:
        level = "🟢 ОК"
        warning = "Приемлемое количество токенов."

    return f"""
**Использование токенов:**

- Изображения: {num_images} × 1000 = ~{image_tokens:,} токенов
- Видео: {num_videos} × 5000 = ~{video_tokens:,} токенов
- **Всего: ~{total:,} токенов**

**Статус: {level}**
{warning}

*Лимит контекста: 256,000 токенов*
    """

# Connect to inputs
multi_images.change(
    fn=calculate_tokens,
    inputs=[multi_images, multi_videos],
    outputs=[multi_token_info]
)

multi_videos.change(
    fn=calculate_tokens,
    inputs=[multi_images, multi_videos],
    outputs=[multi_token_info]
)
```

---

## Error Handling

### Handle OOM Errors

```python
def generate_with_fallback(media_items, prompt, model_name, quantization, **kwargs):
    """Generate with automatic fallback on OOM"""
    try:
        return generator.generate_multi_media_description(
            media_items, prompt, model_name, quantization, **kwargs
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            # Try with reduced resolution
            print("⚠️ OOM detected, reducing resolution...")

            # Clear cache
            torch.cuda.empty_cache()

            # Retry with reduced settings
            try:
                return generator.generate_multi_media_description(
                    media_items, prompt, model_name, "4-bit",  # Force 4-bit
                    max_new_tokens=512,  # Reduce tokens
                    **{k: v for k, v in kwargs.items() if k != 'max_new_tokens'}
                )
            except RuntimeError as e2:
                return f"❌ CUDA OOM Error даже после оптимизации. Рекомендации:\n1. Уменьшите количество изображений\n2. Используйте модель меньшего размера (2B)\n3. Закройте другие программы\n\nОшибка: {str(e2)}"
        else:
            return f"❌ Ошибка: {str(e)}"
    except Exception as e:
        return f"❌ Непредвиденная ошибка: {str(e)}"
```

---

## Performance Benchmarks

Expected performance on different hardware:

### RTX 3060 (12GB VRAM)
- **2B model (4-bit)**: 5 images + 1 video → ~5-10 seconds
- **8B model (4-bit)**: 3 images + 1 video → ~10-15 seconds

### RTX 4090 (24GB VRAM)
- **8B model (4-bit)**: 10 images + 2 videos → ~15-20 seconds
- **32B model (4-bit)**: 5 images + 1 video → ~20-30 seconds

### CPU Only (Not Recommended)
- **2B model**: 2 images → ~60-120 seconds
- Not viable for production use

---

## Deployment Checklist

Before deploying multi-image features:

- [ ] Backend methods updated
- [ ] UI components added
- [ ] Error handling implemented
- [ ] Token calculation working
- [ ] Test with 2 images (baseline)
- [ ] Test with 5 images (moderate)
- [ ] Test with 10 images (stress)
- [ ] Test image + video mix
- [ ] Test custom labels
- [ ] Test on target hardware
- [ ] Document new features in README
- [ ] Update help text/tooltips
- [ ] Add usage examples
- [ ] Monitor GPU memory in production

---

## Next Steps

1. **Implement basic multi-image** (Priority 1)
   - Update `_prepare_inputs`
   - Add multi-image UI tab
   - Test with 2-5 images

2. **Add image + video mixing** (Priority 2)
   - Extend multi-image to support videos
   - Test combinations

3. **Build content constructor** (Priority 3)
   - Create interleaved content builder UI
   - Allow arbitrary ordering

4. **Add conversation mode** (Priority 4)
   - Implement history management
   - Create chat interface

---

**Implementation Time Estimates:**

- **Basic multi-image**: 4-6 hours
- **Image + video mixing**: 2-3 hours
- **Content builder**: 6-8 hours
- **Conversation mode**: 8-12 hours

**Total**: ~20-30 hours for complete multimodal support

---

**Created**: 2025-11-25
**Target codebase**: `/home/user/qwen3-vl/app.py`
**Required dependencies**: Already installed (no new dependencies needed)
