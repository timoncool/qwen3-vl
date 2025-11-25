# Advanced Prompting Techniques for Qwen VL Models

A comprehensive guide to prompt engineering for Qwen2-VL, Qwen2.5-VL, and Qwen3-VL vision-language models.

## Table of Contents
- [Overview](#overview)
- [Special Tokens and Format](#special-tokens-and-format)
- [Best Prompt Patterns for Vision Tasks](#best-prompt-patterns-for-vision-tasks)
- [System Prompts Examples](#system-prompts-examples)
- [Few-Shot Prompting for Vision](#few-shot-prompting-for-vision)
- [Chain-of-Thought for Vision](#chain-of-thought-for-vision)
- [Prompt Engineering Best Practices](#prompt-engineering-best-practices)
- [Task-Specific Examples](#task-specific-examples)
- [Parameters Optimization](#parameters-optimization)
- [Community Examples](#community-examples)

---

## Overview

Qwen VL models (Qwen2-VL, Qwen2.5-VL, Qwen3-VL) are state-of-the-art vision-language models that support:
- **Dynamic Resolution**: Process images of any resolution (optimal: 480×480 to 2560×2560)
- **Video Understanding**: Comprehend videos over 1 hour long with temporal reasoning
- **Multi-Image Processing**: Handle multiple images in single context (32K context length)
- **Visual Grounding**: Output bounding boxes with coordinates normalized to [0, 1000)
- **Multilingual Support**: English, Chinese, Russian, Japanese, Korean, Arabic, and more

---

## Special Tokens and Format

### Vision Delimiters
```
<|vision_start|>  - Marks start of visual content
<|vision_end|>    - Marks end of visual content
<|image_pad|>     - Placeholder for image tokens
```

### Dialogue Markers (ChatML Format)
```
<|im_start|>      - Start of message
<|im_end|>        - End of message
```

### Grounding Tokens
```
<|box_start|>         - Start of bounding box coordinates
<|box_end|>           - End of bounding box coordinates
<|object_ref_start|>  - Start of object reference
<|object_ref_end|>    - End of object reference
<|extra_0|>           - Start of important phrase wrapper
<|extra_1|>           - End of important phrase wrapper
<|extra_2|>           - Start of bounding box wrapper
<|extra_3|>           - End of bounding box wrapper
```

### ChatML Format Structure
```
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>
Describe this image.
<|im_end|>
<|im_start|>assistant
[Model response]
<|im_end|>
```

### Message Format (API/Code)
```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "path/to/image.jpg"},
            {"type": "text", "text": "Your question here"}
        ]
    }
]
```

---

## Best Prompt Patterns for Vision Tasks

### 1. Task-Specific Instructions

**Pattern**: Start with clear task definition
- Detection: "Detect all [objects] in the image..."
- Counting: "Count how many [objects] are in..."
- Classification: "Identify and classify..."
- Description: "Describe this image in detail..."

**Examples**:
```
"Detect all objects in the image and return their locations and labels
(car, truck, bus, cycle, bike, etc.) in the form of coordinates"

"Count the number of people in this image and describe their activities"

"Identify the type of document and extract all relevant information"
```

### 2. Specificity and Detail

**Pattern**: Narrow focus by specifying objects or attributes
```
"Describe the red car in the center of the image"

"Identify all birds in the image and describe their species,
colors, and positions"

"Extract text from the invoice, specifically: invoice number,
date, total amount, and vendor name"
```

### 3. Output Format Specification

**Pattern**: Define desired output structure
```
"Describe this image. Output your answer in JSON format with keys:
main_subject, colors, mood, objects"

"Extract all text from this image and provide the position of each
text region in [x1, y1, x2, y2] format"

"List all detected objects in the format:
object_name: [x1, y1, x2, y2], confidence"
```

### 4. Context-Aware Prompting

**Pattern**: Provide context for better understanding
```
"This is a medical X-ray image. Identify any abnormalities and
describe their location"

"This is a product photo for an e-commerce listing. Create a detailed,
attractive description highlighting key features and selling points"

"This is a technical diagram. Explain the workflow and relationships
between components"
```

### 5. Multi-Step Instructions

**Pattern**: Break complex tasks into steps
```
"First, identify all people in the image. Then, describe what each
person is doing. Finally, explain how they are interacting with
each other"

"Analyze this chart: 1) Identify the chart type, 2) Extract the title
and axis labels, 3) Summarize the key trends shown"
```

---

## System Prompts Examples

### General Assistant
```
You are a helpful vision-language assistant specialized in analyzing
images and videos accurately and thoroughly.
```

### Specialized Roles

**For Chart Analysis**:
```
You are a Vision Language Model specialized in interpreting visual data
from chart images. Your task is to analyze the provided chart image and
respond to queries with concise answers, usually a single word, number,
or short phrase. Focus on delivering accurate, succinct answers based
on the visual information.
```

**For OCR and Document Processing**:
```
You are an expert at extracting text and structured information from
documents. Accurately transcribe all text while maintaining the original
layout and structure. Output information in the requested format without
omissions or fabrications.
```

**For Product Description Generation**:
```
You are a professional e-commerce content writer. Create compelling,
detailed product descriptions that highlight key features, benefits,
and selling points. Use vivid, descriptive language that appeals to
potential customers.
```

**For Visual Grounding Tasks**:
```
You are a precise object detection and localization specialist. Identify
objects in images and provide accurate bounding box coordinates. Use the
format: object_name [x1, y1, x2, y2] where coordinates are normalized
to [0, 1000).
```

**For Video Analysis**:
```
You are an expert in video understanding and temporal reasoning. Analyze
video content by understanding the sequence of events, causality, changes
over time, and relationships between different moments in the video.
```

**Important Note**: For general conversation scenarios that don't involve
specific agent tool calls, it's recommended NOT to set a System Message.
Instead, pass instructions through the User Message for optimal performance.

---

## Few-Shot Prompting for Vision

Few-shot prompting helps the model understand the desired output format and style through examples.

### Pattern 1: Direct Examples

```
Analyze these examples first:

Example 1:
Image: [dog playing with ball]
Output: A golden retriever joyfully playing with a red ball in a
grassy backyard on a sunny afternoon.

Example 2:
Image: [cat sleeping]
Output: A tabby cat peacefully sleeping on a soft cushion near a
sunlit window.

Now analyze this image following the same descriptive style:
[Your target image]
```

### Pattern 2: Task-Specific Few-Shot

**Object Detection with Examples**:
```
I will show you how to detect and label objects:

Example 1: "car [100, 200, 300, 400], person [450, 100, 550, 350]"
Example 2: "tree [50, 50, 200, 500], building [600, 100, 900, 600]"

Now detect all objects in this image using the same format:
```

### Pattern 3: Multi-Image Few-Shot

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Here are examples of good captions:"},
            {"type": "image", "image": "example1.jpg"},
            {"type": "text", "text": "Caption: A serene mountain landscape..."},
            {"type": "image", "image": "example2.jpg"},
            {"type": "text", "text": "Caption: An urban street scene..."},
            {"type": "text", "text": "Now caption this image in the same style:"},
            {"type": "image", "image": "target.jpg"}
        ]
    }
]
```

### Pattern 4: Few-Shot for Domain Alignment

For specialized domains, provide examples that help the model understand the domain-specific terminology and format:

```
These are examples of medical image analysis:

Example 1: "Chest X-ray showing clear lung fields, normal cardiac
silhouette, no pleural effusion."

Example 2: "CT scan revealing a 2cm mass in the right upper lobe,
suspicious for malignancy."

Now analyze this medical image:
```

### Community Implementation

A recent autonomous driving project used Qwen2.5-VL-72B with prompts embedding:
- Coordinate systems
- Spatial reasoning rules
- Role-playing
- Chain-of-Thought/Tree-of-Thought reasoning
- Few-shot examples tailored to each question type

---

## Chain-of-Thought for Vision

Chain-of-Thought (CoT) prompting helps models reason step-by-step, especially for complex visual reasoning tasks.

### Basic CoT Pattern

```
"Think carefully step by step."
```

This simple addition significantly improves performance on complex tasks.

### Explicit Step-by-Step

```
"Analyze this image step by step:
1. First, identify the main subject
2. Then, describe the background and context
3. Next, note any important details or objects
4. Finally, explain the overall scene and atmosphere"
```

### Mathematical Reasoning with Vision

```
"How much would I pay if I want to order two Salmon Burgers and three
Meat Lover's Pizzas? Think carefully step by step.

Step 1: Find the price of Salmon Burger
Step 2: Calculate cost for 2 Salmon Burgers
Step 3: Find the price of Meat Lover's Pizza
Step 4: Calculate cost for 3 pizzas
Step 5: Add both amounts for total"
```

### Visual Problem Decomposition

```
"Analyze this complex scene by breaking it down:
- First, what is the setting and time of day?
- Second, who are the people and what are they doing?
- Third, what objects are present and how are they being used?
- Fourth, what relationships exist between elements?
- Finally, what is the overall narrative or purpose of this scene?"
```

### Causal Chain-of-Thought

```
"Examine this sequence of images and explain:
1. What happened first?
2. What caused the change?
3. What was the effect?
4. What will likely happen next?

Provide reasoning for each step."
```

### Compositional CoT for Complex Scenes

For complex multi-object scenes:
```
"Let's analyze this systematically:

Step 1: Spatial Layout
- Describe the overall arrangement of objects
- Identify foreground, middle ground, and background

Step 2: Object Identification
- List all major objects and their attributes
- Note colors, sizes, and conditions

Step 3: Relationships
- How do objects relate to each other?
- Are there any interactions or dependencies?

Step 4: Context and Purpose
- What is the purpose or function of this scene?
- What story does it tell?"
```

### Qwen3-VL Thinking Model

Qwen3-VL-8B-Thinking is specially optimized for STEM and math reasoning:
- Notices fine details
- Breaks down problems step by step
- Analyzes cause and effect
- Gives logical, evidence-based answers

---

## Prompt Engineering Best Practices

### 1. Clarity and Specificity

**Good**:
```
"Describe this image in detail with grounding"
```

**Better**:
```
"Create a detailed description of this image. For each significant
object or person, provide its location using bounding boxes. Include
information about colors, spatial relationships, and actions."
```

### 2. Task Definition First

Always start with a clear task definition:
- What needs to be detected/identified/extracted?
- What format should the output take?
- What level of detail is required?

### 3. Context-Aware Replacements

For video processing, the model can automatically understand context:
```
# The model will understand this is a video, not an image
"Describe this video in detail"

# You can also be explicit
"Analyze this video sequence and describe the temporal progression
of events"
```

### 4. Leverage Model Strengths

**Dynamic Resolution**: No need to resize images
```
"Analyze this high-resolution architectural drawing"
# Model handles any resolution automatically
```

**Long Context**: Use for multi-image tasks
```
"Compare these 10 product images and identify which one best matches
the description: [description]"
```

**Temporal Understanding**: For videos
```
"In this 30-minute video, identify all instances where [specific
action] occurs and provide timestamps"
```

### 5. Grounding for Precision

For tasks requiring precise localization:
```
"Describe image in details with grounding"
# This special phrase triggers full grounding capabilities
```

### 6. Multi-Turn Conversations

Maintain context for follow-up questions:
```python
# First turn
query = "What objects are in this image?"
response, history = model.chat(tokenizer, query=query, history=None)

# Follow-up turn (maintains context)
query = "Which object is the largest?"
response, history = model.chat(tokenizer, query=query, history=history)
```

### 7. Character/Name Consistency

When you need consistent naming:
```
"If there is a person or character in the image, use the name 'Sarah'
instead of generic terms like 'woman' or 'person'."
```

### 8. Language Consistency

Model supports multilingual inputs seamlessly - maintain language consistency or mix as needed:
```
# English
"Describe this image in detail"

# Chinese
"详细描述这张图片"

# Mixed (if needed)
"Describe this image. 请用中文输出。"
```

### 9. Avoid Ambiguity

**Avoid**:
```
"Tell me about this"
```

**Better**:
```
"Describe the main subject, background elements, colors, mood, and
any text visible in this image"
```

### 10. Video-Specific Best Practices

**FPS Parameter**: When using pre-extracted frames
```python
# Inform model of frame rate for better temporal understanding
messages = [...video frames...]
# Use fps parameter to indicate time intervals
```

**Temporal Queries**:
```
"At what timestamp does the person enter the room?"
"How long does the action last?"
"What is the sequence of events in chronological order?"
```

---

## Task-Specific Examples

### Image Captioning

**Basic**:
```
"Describe this image."
```

**Detailed**:
```
"Create a comprehensive description of this image, including the main
subject, setting, atmosphere, colors, and any notable details."
```

**With Grounding** (triggers bounding boxes):
```
"Describe image in details with grounding"
"Generate the caption in English with grounding:"
```

**Style-Specific**:
```
"Create a vivid, poetic description of this landscape image"
"Generate a technical, precise description of this product image"
```

### Object Detection

**Basic Detection**:
```
"Detect all objects in this image"
```

**Specific Categories**:
```
"Detect all vehicles in the image and return their locations and
labels (car, truck, bus, motorcycle) in the form of coordinates"
```

**With Counting**:
```
"Count and locate all people in this image. For each person, provide
their position and describe what they are doing"
```

### Visual Question Answering

**Simple Questions**:
```
"What color is the car?"
"How many people are in the image?"
```

**Complex Questions**:
```
"Based on the photo, which floor is the Department of Otorhinolaryngology on?"
"What is the relationship between the two people in the foreground?"
```

**Multi-Step Questions**:
```
"Who is the director of the movie shown in this poster, and what other
famous films have they directed?"
```

### OCR and Document Understanding

**Basic OCR**:
```
"Extract all text from this image"
"Read all texts in the image, output in lines"
```

**Structured Extraction**:
```
"Spotting all the text in the image with line-level, and output in
JSON format"
```

**Table Extraction**:
```
"Convert this table into HTML format using <tr> and <td> tags,
accurately representing merged cells and the layout from top-left
to bottom-right"
```

**Specific Field Extraction**:
```
"Extract the invoice number, train number, departure station, arrival
station, departure date and time, seat number, seat class, ticket price,
ID card number, and passenger name from the train ticket image.
Accurately extract the key information without omissions or fabrications"
```

**Document Type + Extraction**:
```
"Identify the document type, then extract all key-value pairs and
present them in a structured format"
```

### Multi-Image Comparison

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "image1.jpg"},
            {"type": "image", "image": "image2.jpg"},
            {"type": "text", "text": "Compare these two images. What are the similarities and differences?"}
        ]
    }
]
```

**Specific Comparison**:
```
"Compare the products in these two images. Which one appears to be
better quality and why?"

"These are before and after photos. Describe what changed between them"
```

### Video Understanding

**Basic Video Description**:
```
"Describe what happens in this video"
```

**Temporal Analysis**:
```
"Describe the sequence of events in this video chronologically,
noting key moments and transitions"
```

**Specific Event Detection**:
```
"Identify all instances where a person enters or exits the frame,
providing approximate timestamps"
```

**Long Video Understanding**:
```
"This is a 45-minute lecture video. Summarize the main topics covered
and key points made by the speaker"
```

### Visual Grounding

**General Grounding**:
```
"请给我框出图中上海环球金融中心和东方明珠"
(Frame the Shanghai World Financial Center and Oriental Pearl Tower)
```

**Referential Grounding**:
```
"Point out the eyes on the giraffe"
"Locate all stop signs in the image"
```

**Grounded Captioning**: Each sentence will have important phrases and bounding boxes marked:
```
"Generate a detailed grounded caption for this image"
# Output includes <|extra_0|>phrase<|extra_1|> and
# <|extra_2|>bbox<|extra_3|> markers
```

### E-Commerce Product Descriptions

**Structured Description**:
```
"Create a product description for this item with the following structure:
- Title (compelling, SEO-friendly)
- Key Features (3-5 bullet points)
- Detailed Description (2-3 paragraphs)
- Specifications (extracted from visible information)"
```

**Target Audience Specific**:
```
"Describe this product for an online marketplace targeting budget-conscious
shoppers. Emphasize value and practical features"
```

**Multilingual**:
```
"Создать описание тарелки арт:123 для онлайн магазина"
(Create a description of plate art:123 for an online store)
```

### Scientific/Technical Images

**Diagram Analysis**:
```
"This is a technical diagram. Identify all components, explain their
functions, and describe how they interact with each other"
```

**Medical Imaging**:
```
"Analyze this medical image. Identify any visible structures and note
any abnormalities. Use medical terminology where appropriate"
```

**Data Visualization**:
```
"Analyze this chart: identify the type, extract data points, describe
trends, and provide key insights"
```

---

## Parameters Optimization

### Default/Recommended Settings

Qwen2-VL's official default parameters (from generation_config.json):

```python
{
    "temperature": 0.01,    # Very low - highly deterministic
    "top_p": 0.001,         # Very restrictive
    "top_k": 1,             # Greedy decoding
    "max_new_tokens": 512   # Default response length
}
```

These conservative settings are optimized for **precision and consistency** rather than creative variation.

### Parameter Guidelines

**Temperature** (0.0 - 2.0):
- **0.01-0.3**: Highly deterministic, best for:
  - OCR and text extraction
  - Object detection
  - Factual descriptions
  - Document parsing

- **0.4-0.7**: Balanced, good for:
  - General image description
  - Product descriptions
  - Visual question answering

- **0.8-1.2**: More creative, suitable for:
  - Artistic descriptions
  - Creative writing based on images
  - Exploratory analysis

**Top-p** (0.0 - 1.0):
- **0.001-0.1**: Very focused, deterministic
- **0.3-0.7**: Balanced diversity
- **0.8-0.95**: More diverse outputs

**Top-k** (1 - 100+):
- **1-5**: Greedy, consistent
- **10-30**: Moderate diversity
- **40-100**: High diversity

**Max New Tokens** (1 - 4096):
- **128-512**: Short to medium responses
- **512-1024**: Detailed descriptions
- **1024-2048**: Very comprehensive analysis
- **2048-4096**: Extremely detailed, multi-part responses

### Task-Specific Recommendations

**OCR/Text Extraction**:
```python
temperature = 0.01
top_p = 0.001
top_k = 1
```

**Product Descriptions**:
```python
temperature = 0.6
top_p = 0.7
top_k = 20
```

**Creative Captions**:
```python
temperature = 0.8
top_p = 0.85
top_k = 40
```

**Object Detection**:
```python
temperature = 0.01
top_p = 0.01
top_k = 1
```

**General Q&A**:
```python
temperature = 0.3
top_p = 0.5
top_k = 10
```

### Important Notes

1. **Temperature Control Issues**: Some users report that temperature changes don't affect output when using vLLM. This may be an implementation-specific issue.

2. **Seed for Reproducibility**: Use a fixed seed value for consistent results across runs.

3. **Resolution Optimization**: Object localization is most robust between 480×480 and 2560×2560. Outside this range, accuracy may decrease.

---

## Community Examples

### Official Repository Examples

**Qwen-VL Tutorial** (github.com/QwenLM/Qwen-VL):
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# Basic query
query = tokenizer.from_list_format([
    {'image': 'path/to/image.jpg'},
    {'text': 'Describe this image'}
])
response, history = model.chat(tokenizer, query=query, history=None)

# Follow-up question
query = tokenizer.from_list_format([
    {'text': 'What colors are dominant?'}
])
response, history = model.chat(tokenizer, query=query, history=history)
```

### Visual Grounding Example

```python
query = tokenizer.from_list_format([
    {'image': 'shanghai.jpg'},
    {'text': '请给我框出图中上海环球金融中心和东方明珠'}
])
response, history = model.chat(tokenizer, query=query, history=None)

# Visualize bounding boxes
image = tokenizer.draw_bbox_on_latest_picture(response, history)
image.save('output_with_boxes.jpg')

# Clean text (remove markup)
import re
clean_text = re.sub(r'<ref>(.*?)</ref>(?:<box>.*?</box>)*', r'\1', response)
```

### Multi-Image Analysis

```python
query = tokenizer.from_list_format([
    {'image': 'chongqing.jpg'},
    {'image': 'beijing.jpg'},
    {'text': '上面两张图片分别是哪两个城市？请对它们进行对比。'}
])
response, history = model.chat(tokenizer, query=query, history=None)
```

### Mathematical Reasoning

```python
query = tokenizer.from_list_format([
    {'image': 'menu.jpg'},
    {'text': 'How much would I pay if I want to order two Salmon Burgers and three Meat Lover\'s Pizzas? Think carefully step by step.'}
])
response, history = model.chat(tokenizer, query=query, history=None)
```

### Invoice Extraction (Practical Application)

From Qwen2-VL invoice extraction guides:
```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "invoice.jpg"},
            {
                "type": "text",
                "text": "Extract all data from this invoice without missing anything. Structure the output as JSON with keys: invoice_number, date, vendor, items (array), subtotal, tax, total"
            }
        ]
    }
]
```

### ComfyUI Integration

The ComfyUI_Qwen2_5-VL-Instruct project demonstrates:
- Text-based queries
- Video queries (frame-by-frame)
- Single-image queries
- Multi-image batch processing

### Autonomous Driving Application

Research implementation with Qwen2.5-VL-72B:
- Embedded coordinate systems
- Spatial reasoning rules
- Role-playing prompts
- CoT/ToT reasoning
- Task-specific few-shot examples
- Dynamic inference parameters per question type

### Document Understanding Projects

**RolmOCR** (Built on Qwen 2.5 VL):
- State-of-the-art OCR performance
- Advanced document understanding
- Table detection and extraction
- Formula recognition
- Multi-language support

### Community Insights

**HuggingFace Discussions**:
- Multiple images in PDFs: Process sequentially while maintaining context
- Grounding with multiple boxes: Use specific prompt "Describe image in details with grounding"
- Temperature issues: Be aware of potential framework-specific implementation differences

---

## Advanced Techniques Summary

### 1. Iterative Refinement

For complex tasks, use multi-turn conversations to refine understanding:
```
Turn 1: "What objects are in this image?"
Turn 2: "Focus on the objects in the foreground"
Turn 3: "Provide precise coordinates for the red car"
```

### 2. Compositional Reasoning

Break down complex scenes into manageable parts:
```
"Analyze this scene in layers:
Layer 1: Background elements and setting
Layer 2: Middle-ground objects and their arrangement
Layer 3: Foreground subjects and their activities
Layer 4: Overall composition and visual flow"
```

### 3. Cross-Modal Reasoning

Combine visual and textual understanding:
```
"This image shows a product. Based on the visual appearance and any
visible text, identify the brand, model, key features, and target market"
```

### 4. Temporal Chaining (Videos)

For video understanding, chain temporal observations:
```
"Track the main subject through this video:
- Initial position and state (0-10s)
- Actions and movements (10-30s)
- Interactions with environment (30-45s)
- Final position and outcome (45s-end)"
```

### 5. Meta-Prompting

Instruct the model about how to approach the task:
```
"Before answering, first observe the image carefully. Then, organize
your thoughts about what's important. Finally, provide a structured
response covering: subject, context, details, and interpretation"
```

---

## References and Resources

### Official Documentation
- [Qwen Official Documentation](https://qwen.readthedocs.io/)
- [Qwen GitHub - Qwen-VL](https://github.com/QwenLM/Qwen-VL)
- [Qwen GitHub - Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- [Qwen3-VL Tutorial](https://github.com/QwenLM/Qwen-VL/blob/master/TUTORIAL.md)

### Model Pages
- [Qwen2-VL-7B-Instruct on HuggingFace](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
- [Qwen2.5-VL-7B-Instruct on HuggingFace](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [Qwen3-VL-8B-Instruct on HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)

### Papers
- [Qwen-VL Paper (arXiv:2308.12966)](https://ar5iv.labs.arxiv.org/html/2308.12966)
- [Qwen2-VL Paper (arXiv:2409.12191)](https://arxiv.org/html/2409.12191v1)
- [Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)

### Deployment Guides
- [Qwen2.5-VL Usage Guide - vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen2.5-VL.html)
- [Qwen3-VL Usage Guide - vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html)
- [Alibaba Cloud Model Studio - Vision](https://www.alibabacloud.com/help/en/model-studio/vision)

### Community Resources
- [Using Qwen-VL Practical Guide (Medium)](https://medium.com/@shivashishbhardwaj/using-qwen-vl-for-vision-language-tasks-a-practical-guide-ba19b6e86d7e)
- [Video Understanding with Qwen2-VL (Medium)](https://medium.com/@tenyks_blogger/qwen2-vl-expert-vision-language-model-for-video-understanding-db5da45560f3)
- [Object Detection with Qwen 2.5 (PyImageSearch)](https://pyimagesearch.com/2025/06/09/object-detection-and-visual-grounding-with-qwen-2-5/)
- [Fine-Tuning Qwen2.5 VL Guide (F22 Labs)](https://www.f22labs.com/blogs/complete-guide-to-fine-tuning-qwen-2-5-vl-model/)

### Specialized Applications
- [Qwen-VL OCR Documentation (Alibaba Cloud)](https://www.alibabacloud.com/help/en/model-studio/qwen-vl-ocr)
- [Qwen2-VL Invoice Data Extraction Guide](https://aihorizonforecast.substack.com/p/qwen2-vl-hands-on-guides-for-invoice)
- [RolmOCR - Advanced OCR Built on Qwen 2.5 VL](https://www.marktechpost.com/2025/04/05/reducto-ai-released-rolmocr-a-sota-ocr-model-built-on-qwen-2-5-vl-fully-open-source-and-apache-2-0-licensed-for-advanced-document-understanding/)

### Research Papers
- [Compositional Chain-of-Thought Prompting for Large Multimodal Models](https://openaccess.thecvf.com/content/CVPR2024/papers/Mitra_Compositional_Chain-of-Thought_Prompting_for_Large_Multimodal_Models_CVPR_2024_paper.pdf)
- [Enhancing VLMs for Autonomous Driving (arXiv:2510.24152)](https://arxiv.org/html/2510.24152v1)
- [Improved GUI Grounding via Iterative Narrowing (arXiv:2411.13591)](https://arxiv.org/html/2411.13591v2)

---

## Quick Reference Card

### Most Effective Prompts by Task

| Task | Recommended Prompt |
|------|-------------------|
| **Detailed Captioning** | "Describe image in details with grounding" |
| **Object Detection** | "Detect all [objects] and return locations in coordinate form" |
| **OCR** | "Extract all text from this image, output in lines" |
| **Table Extraction** | "Convert this table into HTML format using <tr> and <td> tags" |
| **Video Summary** | "Describe the sequence of events chronologically" |
| **Comparison** | "Compare these images and identify similarities and differences" |
| **Math Problems** | "Calculate [task]. Think carefully step by step." |
| **Product Description** | "Create a detailed, attractive product description highlighting key features" |

### Key Parameters by Use Case

| Use Case | Temperature | Top-p | Top-k |
|----------|------------|-------|-------|
| **OCR/Extraction** | 0.01 | 0.001 | 1 |
| **Object Detection** | 0.01 | 0.01 | 1 |
| **General Description** | 0.3-0.6 | 0.5-0.7 | 10-20 |
| **Creative Writing** | 0.8-1.0 | 0.8-0.9 | 40+ |
| **Product Descriptions** | 0.6 | 0.7 | 20 |

---

## Conclusion

Effective prompting for Qwen VL models requires:
1. **Clear task definition** at the start of prompts
2. **Specific details** about what to detect/extract/describe
3. **Output format specification** when structure matters
4. **Step-by-step reasoning** for complex tasks
5. **Context awareness** for specialized domains
6. **Appropriate parameter tuning** for the task
7. **Multi-turn conversations** for iterative refinement
8. **Few-shot examples** for consistent output format

The models excel at:
- Dynamic resolution handling (no need to resize)
- Long videos (1+ hours) with temporal reasoning
- Multi-image contexts (32K context length)
- Visual grounding with precise coordinates
- Multilingual understanding
- Chain-of-thought reasoning for complex scenarios

Experiment with these techniques and adjust based on your specific use case for optimal results!
