# Qwen2-VL & Qwen3-VL Video Processing Capabilities - Deep Research

## Table of Contents
1. [How to Feed Video to the Model](#how-to-feed-video-to-the-model)
2. [Supported Video Formats](#supported-video-formats)
3. [FPS Limitations and Requirements](#fps-limitations-and-requirements)
4. [Frame Sampling Strategies](#frame-sampling-strategies)
5. [Maximum Video Length/Duration](#maximum-video-lengthduration)
6. [Video Preprocessing Requirements](#video-preprocessing-requirements)
7. [Code Examples for Video Input](#code-examples-for-video-input)
8. [FPS Errors and Limitations](#fps-errors-and-limitations)
9. [Advanced Features](#advanced-features)

---

## How to Feed Video to the Model

Qwen2-VL and Qwen3-VL support **three methods** for video input:

### 1. **Video File Path (Local)**
```python
{
    "type": "video",
    "video": "file:///path/to/video1.mp4",
    "fps": 2.0,
    "max_pixels": 360 * 420
}
```

### 2. **Video URL (Remote)**
```python
{
    "type": "video",
    "video": "https://example.com/video.mp4"
}
```

### 3. **Image Frame Sequence**
```python
{
    "type": "video",
    "video": [
        "file:///path/to/frame1.jpg",
        "file:///path/to/frame2.jpg",
        "file:///path/to/frame3.jpg",
        "file:///path/to/frame4.jpg"
    ]
}
```

**Important Notes:**
- Video URLs: torchvision ≥0.19.0 supports HTTP/HTTPS; decord supports HTTP only
- Can override backend using: `FORCE_QWENVL_VIDEO_READER=torchvision` or `FORCE_QWENVL_VIDEO_READER=decord`
- File size limit: **150MB maximum**

---

## Supported Video Formats

### File Formats
- **MP4** (most commonly used)
- **AVI**
- **MKV**
- **MOV**
- **FLV**
- **WMV**

### Video Backends
The system supports multiple backends with automatic fallback:
1. **Decord** (primary, if available)
2. **Torchcodec** (fallback if Decord unavailable)
3. **Torchvision** (final fallback, supports file://, http://, https://)

### Codec Requirements
No specific codec requirements documented. The models work with standard codecs supported by the backend libraries.

---

## FPS Limitations and Requirements

### Default Settings
- **Default FPS**: 2.0 frames per second
- **FPS Range**: (0.1, 10.0)
- **Default Behavior**: `do_sample_frames=True` allows custom FPS

### FPS Constants (from vision_process.py)
```python
FPS = 2.0                 # Default frames per second
FPS_MIN_FRAMES = 4        # Minimum frame extraction
FPS_MAX_FRAMES = 768      # Maximum frame limit (hardcoded)
FRAME_FACTOR = 2          # Frames must be divisible by this
```

### FPS Calculation Formula
```python
nframes = total_frames / video_fps * fps
nframes = min(max(nframes, min_frames), max_frames)
sample_fps = nframes / max(total_frames, 1e-6) * video_fps
```

### Important Limitations
- **Actual FPS may differ** from user-specified FPS due to token constraints
- FPS is clamped between `min_frames` and `max_frames`
- Result must be divisible by `FRAME_FACTOR` (2)
- For videos >5 FPS, frames may overlap in time due to `tokens_per_second=2` in model config

### Recommended Usage
- **High-speed motion**: Use higher FPS (4-10)
- **Static or long videos**: Use lower FPS (0.5-2)
- **Default use cases**: FPS=2.0 is optimal

---

## Frame Sampling Strategies

### Method 1: Using FPS Parameter
```python
{
    "type": "video",
    "video": "file:///path/to/video.mp4",
    "fps": 2.0,              # Extract 2 frames per second
    "min_frames": 4,         # Optional: minimum frames
    "max_frames": 768        # Optional: maximum frames
}
```

**How it works:**
- System calculates frames based on video's native FPS
- Linear interpolation: `torch.linspace(0, total_frames - 1, nframes).round().long()`
- Respects min/max frame boundaries

### Method 2: Using nframes Parameter
```python
{
    "type": "video",
    "video": "file:///path/to/video.mp4",
    "nframes": 100           # Extract exactly 100 frames (if possible)
}
```

**Important:** Cannot use both `fps` and `nframes` simultaneously.

### Method 3: Image List with Fixed FRAME_FACTOR
When using image lists as video:
- Frame count controlled by `FRAME_FACTOR = 2`
- No direct FPS control
- Number of images determines frame count

### Frame Constraints
- **Minimum frames**: `ceil_by_factor(FPS_MIN_FRAMES, FRAME_FACTOR)` = 4
- **Maximum frames**: `floor_by_factor(min(FPS_MAX_FRAMES, total_frames), FRAME_FACTOR)` = 768
- **Divisibility**: All frame counts rounded to multiples of FRAME_FACTOR (2)

### Token Budget Control
```python
VIDEO_MIN_TOKEN_NUM = 128
VIDEO_MAX_TOKEN_NUM = 768
```
Frames are resized to keep tokens within this range.

---

## Maximum Video Length/Duration

### Duration Capabilities

#### Qwen2-VL
- Can process videos **over 20 minutes**
- Suitable for video-based Q&A, dialog, and content creation

#### Qwen2.5-VL
- Can comprehend videos **over 1 hour**
- Enhanced ability to pinpoint relevant video segments
- Supports event detection and timeline analysis

#### Qwen3-VL
- Supports videos up to **several hours** with full recall
- Native 256K context, expandable to 1M tokens
- Handles "hours-long video" according to documentation

### Frame Limit Constraint
**Critical Limitation:** Maximum 768 frames extracted from any video

**Impact on long videos:**
- 2-hour video: ~1 frame every 9.4 seconds
- Reduces temporal resolution for event detection
- Makes detailed timeline analysis difficult

**Solution (as of October 2025):**
- Environment variable `QWEN_VL_MAX_FRAMES` added (PR #853)
- Allows users to override the default 768 frame limit
- Maintains backward compatibility

### Context Window
- Default: 32,768 tokens
- Can extend to 64K+ for long videos using YaRN technique
- For very long inputs, modify `max_position_embeddings` directly

### Video Duration Limits (Alibaba Cloud API)
- Minimum: 2 seconds
- Maximum: 10 minutes
- Note: This is API-specific, not a model limitation

---

## Video Preprocessing Requirements

### Normalization Parameters
```python
# Same for both Qwen2-VL and Qwen3-VL
IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]
```

### Rescaling
```python
RESCALE_FACTOR = 1/255  # 0.00392156862745098
```

### Resolution and Resizing
- **Qwen2.5-VL**: Dimensions rounded to nearest multiple of **28**
- **Qwen3-VL**: Dimensions rounded to nearest multiple of **32**
- **Patch size**: 14
- **Temporal patch size**: 2
- **Merge size**: 2

### Token Constraints
```python
# Image constraints
IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384

# Video constraints
VIDEO_MIN_TOKEN_NUM = 128
VIDEO_MAX_TOKEN_NUM = 768
```

### Smart Resize Function
The `smart_resize()` function:
- Uses **BICUBIC interpolation** with antialiasing
- Maintains aspect ratio
- Ensures token count stays within limits
- Resizes to multiples of patch size

### Processor Configuration Example
```python
processor.video_processor.size = {
    "longest_edge": 16384 * 32 * 32,  # max pixels across all frames
    "shortest_edge": 256 * 32 * 32     # min pixel budget
}
```

### total_pixels Parameter
```python
# Recommended: below 24576 * 32 * 32 to avoid excessive sequences
{
    "type": "video",
    "video": "file:///path/to/video.mp4",
    "total_pixels": 20480 * 28 * 28,  # Total across ALL frames
    "min_pixels": 16 * 28 * 28,
    "fps": 2.0
}
```

**Formula:** For video shape T×H×W, product T×H×W ≤ total_pixels

---

## Code Examples for Video Input

### Basic Video Processing (Qwen2.5-VL)
```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Load model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# Define messages with video
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4",
                "fps": 2.0
            },
            {"type": "text", "text": "Describe this video."}
        ]
    }
]

# Prepare inputs
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
)
inputs = inputs.to("cuda")

# Generate
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)
print(output_text)
```

### Video with Custom FPS and Resolution
```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "file:///path/to/video1.mp4",
                "fps": 4.0,                      # Custom FPS
                "max_pixels": 360 * 420,         # Resolution limit
                "resized_height": 280,           # Target height
                "resized_width": 280             # Target width
            },
            {"type": "text", "text": "Describe this video."}
        ]
    }
]
```

### Video with Metadata (Qwen3-VL)
```python
from qwen_vl_utils import process_vision_info

# Get actual sampling metadata
images, videos, video_kwargs = process_vision_info(
    messages,
    image_patch_size=16,
    return_video_kwargs=True,
    return_video_metadata=True
)

# Check actual sampled FPS
actual_fps = video_kwargs.get('fps', 'N/A')
print(f"Requested FPS: 2.0, Actual sampled FPS: {actual_fps}")

# Process with metadata
inputs = processor(
    text=[text],
    images=images,
    videos=videos,
    padding=True,
    return_tensors="pt",
    **video_kwargs
)
```

### Video Segment Processing (Time Range)
```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "file:///path/to/long_video.mp4",
                "video_start": 10.0,  # Start at 10 seconds
                "video_end": 30.0,    # End at 30 seconds
                "fps": 2.0
            },
            {"type": "text", "text": "What happens between 10 and 30 seconds?"}
        ]
    }
]
```

### Multiple Videos in One Message
```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "file:///path/to/video1.mp4",
                "fps": 2.0
            },
            {
                "type": "video",
                "video": "file:///path/to/video2.mp4",
                "fps": 2.0
            },
            {"type": "text", "text": "Compare these two videos."}
        ]
    }
]
```

### Video as Image Frame Sequence
```python
import os

frame_paths = [f"file:///{os.path.abspath(f)}" for f in sorted(os.listdir("frames/"))]

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": frame_paths[:100]  # First 100 frames
            },
            {"type": "text", "text": "Describe this video."}
        ]
    }
]
```

### Streaming Generation
```python
from transformers import TextIteratorStreamer
from threading import Thread

streamer = TextIteratorStreamer(
    processor.tokenizer,
    skip_prompt=True,
    skip_special_tokens=True
)

generation_kwargs = dict(
    inputs,
    streamer=streamer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9
)

thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

for text in streamer:
    print(text, end="", flush=True)

thread.join()
```

### Video with Token Budget Control
```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "file:///path/to/video.mp4",
                "total_pixels": 20480 * 28 * 28,  # Control total tokens
                "min_pixels": 16 * 28 * 28,
                "fps": 3.0
            },
            {"type": "text", "text": "Describe this video."}
        ]
    }
]
```

### Complete Working Example (app.py style)
```python
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

class VideoProcessor:
    def __init__(self, model_name="Qwen/Qwen3-VL-8B-Instruct"):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model(model_name)

    def load_model(self, model_name):
        print(f"Loading model: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        print(f"Model loaded on {self.device}")

    def process_video(self, video_path, prompt, fps=2.0, max_new_tokens=512):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "fps": fps
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Prepare inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)

        # Generate
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                use_cache=True
            )

        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return output[0]

# Usage
processor = VideoProcessor()
result = processor.process_video(
    video_path="file:///path/to/video.mp4",
    prompt="Describe this video in detail.",
    fps=2.0
)
print(result)
```

---

## FPS Errors and Limitations

### Common FPS-Related Errors

#### 1. **KeyError: 'video_fps'**
**Cause:** Missing video metadata during processing
**Solution:**
- Update `qwen-vl-utils` to version ≥0.0.11
- Ensure video backend (torchvision/decord) is properly installed
- Verify video file is not corrupted

#### 2. **FPS Parameter Not Being Respected**
**Issue:** Setting `fps=8.0` but logs show native video FPS (29.97)

**Explanation:**
- The logged FPS is from torchvision's output (original video FPS)
- Actual frame sampling happens downstream in `vision_process.py`
- To verify actual sampling, use:
```python
images, videos, video_kwargs = process_vision_info(
    messages, return_video_kwargs=True
)
actual_fps = video_kwargs['fps']
print(f"Actual sampled FPS: {actual_fps}")
```

**Status:** Issue resolved in newer versions

#### 3. **Frame Overlap at High FPS (≥5)**
**Issue:** At FPS ≥5, frames have same time index

**Cause:**
- Model config has `tokens_per_second=2`
- Frames overlap when sampling rate exceeds this

**Example:**
- Request: 12 frames at fps=5
- Actual: 9 frames due to overlap prevention

**Workaround:**
- Keep FPS ≤4 for distinct timestamps
- Use `nframes` parameter instead of `fps`

#### 4. **FRAME_FACTOR Limitation with Image Lists**
**Issue:** Cannot control FPS when video is an image list

**Cause:**
- Frame count controlled by hardcoded `FRAME_FACTOR=2`
- No FPS parameter support for image lists

**Workaround:**
- Pre-sample images before passing to model
- Control frame count by number of images provided

#### 5. **ValueError: nframes out of range**
**Error:** `ValueError: nframes must be in [FRAME_FACTOR, total_frames]`

**Cause:**
- Calculated nframes falls outside valid range
- Usually due to very short videos or extreme FPS values

**Solution:**
```python
# Check constraints before processing
min_allowed = FRAME_FACTOR  # 2
max_allowed = min(FPS_MAX_FRAMES, total_frames)  # min(768, total_frames)

if calculated_nframes < min_allowed or calculated_nframes > max_allowed:
    # Adjust fps or use nframes parameter
    pass
```

#### 6. **Maximum Frame Limit (768) Exceeded**
**Issue:** Long videos (2+ hours) only get ~1 frame per 9.4 seconds

**Impact:**
- Reduced temporal resolution
- Missing fine-grained events
- Timeline analysis difficulties

**Solution (New):**
```bash
# Set environment variable to override limit
export QWEN_VL_MAX_FRAMES=1536

# In Python
import os
os.environ['QWEN_VL_MAX_FRAMES'] = '1536'
```

**Note:** Feature added in PR #853 (October 2025)

### FPS Calculation Issues

#### Understanding the Formula
```python
# Step 1: Calculate desired frames
nframes = total_frames / video_fps * fps

# Step 2: Apply min/max constraints
min_frames = ceil_by_factor(FPS_MIN_FRAMES, FRAME_FACTOR)  # 4
max_frames = floor_by_factor(min(FPS_MAX_FRAMES, total_frames), FRAME_FACTOR)  # ≤768

# Step 3: Clamp and round
nframes = min(max(nframes, min_frames), max_frames)
nframes = round_to_multiple_of(nframes, FRAME_FACTOR)  # Must be even

# Step 4: Calculate actual sample FPS
sample_fps = nframes / max(total_frames, 1e-6) * video_fps
```

#### Why FPS Differs from Request
**Token constraint system:**
- Video must stay within 128-768 tokens
- Frame count adjusted to meet token budget
- Actual FPS recalculated based on final frame count

**Example:**
```
Request: fps=4.0 for 3600-frame video at 30fps
Calculated: nframes = 3600/30*4 = 480 frames
After constraints: nframes = 480 (within limits)
Actual FPS: 480/3600*30 = 4.0 ✓

Request: fps=10.0 for same video
Calculated: nframes = 3600/30*10 = 1200 frames
After constraints: nframes = 768 (capped at FPS_MAX_FRAMES)
Actual FPS: 768/3600*30 = 6.4 (reduced from 10.0!)
```

### Troubleshooting Checklist

1. **Check Dependencies**
```bash
pip install --upgrade qwen-vl-utils transformers torch torchvision
pip install decord  # Optional but recommended
```

2. **Verify Video File**
```python
import torchvision
video_reader = torchvision.io.VideoReader(video_path)
metadata = video_reader.get_metadata()
print(f"FPS: {metadata['video']['fps'][0]}")
print(f"Duration: {metadata['video']['duration'][0]}")
print(f"Frames: {metadata['video']['num_frames'][0]}")
```

3. **Check Actual Sampling**
```python
images, videos, video_kwargs = process_vision_info(
    messages, return_video_kwargs=True
)
print(f"Requested FPS: {requested_fps}")
print(f"Actual FPS: {video_kwargs.get('fps', 'N/A')}")
print(f"Frames extracted: {len(videos[0]) if videos else 0}")
```

4. **Monitor Memory**
```python
import torch
print(f"GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"GPU cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
```

5. **Enable Debug Logging**
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qwen_vl_utils")
logger.setLevel(logging.DEBUG)
```

---

## Advanced Features

### Video Grounding / Temporal Localization
Qwen2.5-VL supports pinpointing specific moments in videos:

```python
prompt = """
When does the person start running?
Provide timestamps in format HH:MM:SS.
"""

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "file:///path/to/video.mp4",
                "fps": 2.0
            },
            {"type": "text", "text": prompt}
        ]
    }
]

# Model can respond with timestamps like: "00:01:23 - 00:01:45"
```

**Key Features:**
- Absolute time alignment via mRoPE
- Temporal sequence and speed understanding
- Supports multiple timestamp formats (seconds, HH:MM:SS, HMSF)

### Multi-Video Comparison
```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "file:///video1.mp4",
                "fps": 2.0
            },
            {
                "type": "video",
                "video": "file:///video2.mp4",
                "fps": 2.0
            },
            {"type": "text", "text": "Compare the actions in these two videos."}
        ]
    }
]
```

### Mixed Image and Video Input
```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "file:///reference.jpg"
            },
            {
                "type": "video",
                "video": "file:///video.mp4",
                "fps": 2.0
            },
            {"type": "text", "text": "Does this video match the reference image?"}
        ]
    }
]
```

### Video OCR
Qwen3-VL has enhanced video OCR capabilities:
```python
prompt = "Extract all text visible in this video with timestamps."
```

### Environment Variables
```bash
# Override video backend
export FORCE_QWENVL_VIDEO_READER=torchvision  # or 'decord'

# Override maximum frame limit (NEW)
export QWEN_VL_MAX_FRAMES=1536

# Hugging Face cache
export HF_HOME=/path/to/cache
```

---

## Performance Optimization Tips

1. **Reduce FPS for long videos**
   - 2-hour video: use fps=0.5 or fps=1.0
   - Short clips: use fps=4.0 or higher

2. **Use total_pixels to control memory**
   ```python
   "total_pixels": 20480 * 28 * 28  # Adjust based on GPU memory
   ```

3. **Process video segments instead of full video**
   ```python
   "video_start": start_time,
   "video_end": end_time
   ```

4. **Use quantization for large models**
   ```python
   from transformers import BitsAndBytesConfig

   bnb_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_compute_dtype=torch.float16
   )

   model = Qwen3VLForConditionalGeneration.from_pretrained(
       model_name,
       quantization_config=bnb_config,
       device_map="auto"
   )
   ```

5. **Enable KV cache for faster generation**
   ```python
   model.generate(**inputs, use_cache=True)
   ```

---

## Summary of Key Constraints

| Parameter | Qwen2-VL | Qwen2.5-VL | Qwen3-VL |
|-----------|----------|------------|----------|
| **Default FPS** | 2.0 | 2.0 | 2.0 |
| **FPS Range** | 0.1 - 10.0 | 0.1 - 10.0 | 0.1 - 10.0 |
| **Min Frames** | 4 | 4 | 4 |
| **Max Frames** | 768 | 768* | 768* |
| **Frame Factor** | 2 | 2 | 2 |
| **Max Video Duration** | 20+ min | 1+ hour | Several hours |
| **Context Window** | 32K | 32K (ext. 64K) | 256K (ext. 1M) |
| **Video Token Min** | 128 | 128 | 128 |
| **Video Token Max** | 768 | 768 | 768 |
| **File Size Limit** | 150MB | 150MB | 150MB |
| **Patch Size** | 14 | 14 | 14 |
| **Rounding Multiple** | 28 | 28 | 32 |

\* Configurable via `QWEN_VL_MAX_FRAMES` environment variable (Oct 2025+)

---

## References

- [Qwen3-VL GitHub Repository](https://github.com/QwenLM/Qwen3-VL)
- [Qwen2.5-VL HuggingFace Model Card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [Video FPS Issue #1329](https://github.com/QwenLM/Qwen3-VL/issues/1329)
- [Frame Limit Issue #852](https://github.com/QwenLM/Qwen2.5-VL/issues/852)
- [vision_process.py Source Code](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py)
- [DeepWiki: Image and Video Processing](https://deepwiki.com/QwenLM/Qwen2.5-VL/3.1-image-and-video-processing)
- [Alibaba Cloud Model Studio Documentation](https://www.alibabacloud.com/help/en/model-studio/vision)
- [vLLM Qwen3-VL Usage Guide](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html)

---

**Document Generated:** 2025-11-25
**Research Version:** Deep Analysis of Qwen2-VL, Qwen2.5-VL, and Qwen3-VL
**Repository:** /home/user/qwen3-vl
