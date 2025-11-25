"""
Qwen3-VL Multimodal Examples
Demonstrates various multimodal capabilities not yet implemented in the main app.

These examples show how to use:
1. Multiple images in a single prompt
2. Image + video combinations
3. Interleaved text and images
4. Complex multimodal scenarios

Note: These are example patterns. To run them, you need to:
- Have the required model loaded
- Provide actual image/video file paths
- Have sufficient GPU memory
"""

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from typing import List, Dict, Any


class MultimodalQwenVL:
    """Enhanced Qwen3-VL wrapper with advanced multimodal support"""

    def __init__(self, model_name: str = "Qwen/Qwen3-VL-8B-Instruct", quantization: str = "4-bit"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model(quantization)

    def load_model(self, quantization: str = "4-bit"):
        """Load model with quantization"""
        print(f"Loading {self.model_name} with {quantization} quantization...")

        if quantization == "4-bit":
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        print("Model loaded successfully!")

    def generate(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
    ) -> str:
        """Generate response from messages"""
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process vision info
        image_inputs, video_inputs = process_vision_info(messages)

        # Prepare inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Generate
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=temperature > 0,
            )

        # Decode
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


# ============================================================================
# Example 1: Multiple Images in Single Prompt
# ============================================================================

def example_multiple_images():
    """Process multiple images in a single prompt"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Multiple Images in Single Prompt")
    print("="*70)

    image_paths = [
        "/path/to/image1.jpg",
        "/path/to/image2.jpg",
        "/path/to/image3.jpg",
    ]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_paths[0]},
                {"type": "image", "image": image_paths[1]},
                {"type": "image", "image": image_paths[2]},
                {"type": "text", "text": "Compare these three images and describe their similarities and differences."}
            ]
        }
    ]

    print("\nPrompt structure:")
    print(f"- {len(image_paths)} images")
    print("- 1 text prompt")
    print("\nExpected output: Comparative analysis of all three images")

    # To run:
    # model = MultimodalQwenVL()
    # response = model.generate(messages)
    # print(f"\nResponse: {response}")


# ============================================================================
# Example 2: Interleaved Text and Images
# ============================================================================

def example_interleaved_content():
    """Process interleaved text labels and images"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Interleaved Text and Images")
    print("="*70)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "I'm showing you a process in 3 steps:"},
                {"type": "text", "text": "Step 1 - Initial setup:"},
                {"type": "image", "image": "/path/to/step1.jpg"},
                {"type": "text", "text": "Step 2 - Processing:"},
                {"type": "image", "image": "/path/to/step2.jpg"},
                {"type": "text", "text": "Step 3 - Final result:"},
                {"type": "image", "image": "/path/to/step3.jpg"},
                {"type": "text", "text": "Explain what happened in each step and the overall process."}
            ]
        }
    ]

    print("\nPrompt structure:")
    print("- Interleaved text labels")
    print("- 3 images with context")
    print("- Final question")
    print("\nExpected output: Step-by-step process explanation")


# ============================================================================
# Example 3: Image + Video Combination
# ============================================================================

def example_image_and_video():
    """Process both images and videos in one prompt"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Image + Video Combination")
    print("="*70)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here are reference screenshots:"},
                {"type": "image", "image": "/path/to/screenshot1.jpg"},
                {"type": "image", "image": "/path/to/screenshot2.jpg"},
                {"type": "text", "text": "And here's the demonstration video:"},
                {"type": "video", "video": "/path/to/demo.mp4"},
                {"type": "text", "text": "Does the video match the reference screenshots? Identify any discrepancies."}
            ]
        }
    ]

    print("\nPrompt structure:")
    print("- 2 reference images")
    print("- 1 demonstration video")
    print("- Comparative analysis question")
    print("\nExpected output: Comparison between screenshots and video")


# ============================================================================
# Example 4: Document Analysis with Multiple Pages
# ============================================================================

def example_document_analysis():
    """Analyze multi-page document"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Multi-Page Document Analysis")
    print("="*70)

    document_pages = [
        "/path/to/contract_page1.jpg",
        "/path/to/contract_page2.jpg",
        "/path/to/contract_page3.jpg",
        "/path/to/contract_page4.jpg",
    ]

    content = [{"type": "text", "text": "I have a 4-page contract document:"}]

    # Add all pages with labels
    for i, page in enumerate(document_pages, 1):
        content.append({"type": "text", "text": f"Page {i}:"})
        content.append({"type": "image", "image": page})

    # Add analysis question
    content.append({
        "type": "text",
        "text": "Please:\n1. Summarize the key terms\n2. Identify the parties involved\n3. Note any unusual clauses\n4. Highlight important dates or deadlines"
    })

    messages = [{"role": "user", "content": content}]

    print("\nPrompt structure:")
    print(f"- {len(document_pages)} pages")
    print("- Labeled by page number")
    print("- Structured analysis request")
    print("\nExpected output: Comprehensive document analysis")


# ============================================================================
# Example 5: Product Comparison
# ============================================================================

def example_product_comparison():
    """Compare multiple products from images"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Product Comparison")
    print("="*70)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Compare these three laptops:"},
                {"type": "text", "text": "Laptop A:"},
                {"type": "image", "image": "/path/to/laptop_a_front.jpg"},
                {"type": "image", "image": "/path/to/laptop_a_specs.jpg"},
                {"type": "text", "text": "Laptop B:"},
                {"type": "image", "image": "/path/to/laptop_b_front.jpg"},
                {"type": "image", "image": "/path/to/laptop_b_specs.jpg"},
                {"type": "text", "text": "Laptop C:"},
                {"type": "image", "image": "/path/to/laptop_c_front.jpg"},
                {"type": "image", "image": "/path/to/laptop_c_specs.jpg"},
                {"type": "text", "text": "Create a comparison table with: design, specifications, ports, and estimated price range."}
            ]
        }
    ]

    print("\nPrompt structure:")
    print("- 3 products")
    print("- 2 images per product (design + specs)")
    print("- Structured output request")
    print("\nExpected output: Comparison table")


# ============================================================================
# Example 6: Quality Control with Reference Examples
# ============================================================================

def example_quality_control():
    """Quality control using reference examples"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Quality Control with References")
    print("="*70)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "These are GOOD quality examples:"},
                {"type": "image", "image": "/path/to/good_example1.jpg"},
                {"type": "image", "image": "/path/to/good_example2.jpg"},
                {"type": "image", "image": "/path/to/good_example3.jpg"},
                {"type": "text", "text": "These are DEFECTIVE examples:"},
                {"type": "image", "image": "/path/to/defect_example1.jpg"},
                {"type": "image", "image": "/path/to/defect_example2.jpg"},
                {"type": "text", "text": "Now analyze this production video:"},
                {"type": "video", "video": "/path/to/production_line.mp4"},
                {"type": "text", "text": "Identify any items in the video that match the defective examples. Provide timestamps."}
            ]
        }
    ]

    print("\nPrompt structure:")
    print("- 3 good reference images")
    print("- 2 defect reference images")
    print("- 1 production video to analyze")
    print("- Detection task with timestamps")
    print("\nExpected output: Defect detection report with timestamps")


# ============================================================================
# Example 7: Multi-Turn Conversation
# ============================================================================

def example_multi_turn_conversation():
    """Demonstrate multi-turn conversation with context"""
    print("\n" + "="*70)
    print("EXAMPLE 7: Multi-Turn Conversation")
    print("="*70)

    # First turn
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/path/to/circuit_diagram.jpg"},
                {"type": "text", "text": "What type of circuit is this?"}
            ]
        }
    ]

    print("\nTurn 1:")
    print("- 1 circuit diagram")
    print("- Question about circuit type")

    # Simulate response (in real usage, get from model)
    simulated_response = "This is a full-bridge rectifier circuit used to convert AC to DC..."

    # Add assistant response to conversation
    messages.append({
        "role": "assistant",
        "content": [
            {"type": "text", "text": simulated_response}
        ]
    })

    # Second turn with new image
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": "Now look at this modified version:"},
            {"type": "image", "image": "/path/to/modified_circuit.jpg"},
            {"type": "text", "text": "What changed compared to the first diagram and why would someone make this modification?"}
        ]
    })

    print("\nTurn 2:")
    print("- Reference to previous image (context maintained)")
    print("- New modified circuit image")
    print("- Comparative analysis question")
    print("\nExpected output: Analysis of changes with reasoning")

    # To run full conversation:
    # model = MultimodalQwenVL()
    # Turn 1
    # response1 = model.generate(messages[:1])
    # messages.append({"role": "assistant", "content": [{"type": "text", "text": response1}]})
    # Turn 2
    # messages.append({"role": "user", "content": [...]})
    # response2 = model.generate(messages)


# ============================================================================
# Example 8: Code Generation from UI Screenshots
# ============================================================================

def example_code_generation_from_ui():
    """Generate code from multiple UI screenshots"""
    print("\n" + "="*70)
    print("EXAMPLE 8: Code Generation from UI")
    print("="*70)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Generate React components for this e-commerce flow:"},
                {"type": "text", "text": "1. Landing Page:"},
                {"type": "image", "image": "/path/to/landing_page.jpg"},
                {"type": "text", "text": "2. Product List:"},
                {"type": "image", "image": "/path/to/product_list.jpg"},
                {"type": "text", "text": "3. Product Detail:"},
                {"type": "image", "image": "/path/to/product_detail.jpg"},
                {"type": "text", "text": "4. Shopping Cart:"},
                {"type": "image", "image": "/path/to/cart.jpg"},
                {"type": "text", "text": "5. Checkout Flow (video):"},
                {"type": "video", "video": "/path/to/checkout_demo.mp4"},
                {"type": "text", "text": "Generate:\n- React component structure\n- Tailwind CSS styling\n- State management setup\n- Routing configuration"}
            ]
        }
    ]

    print("\nPrompt structure:")
    print("- 4 UI screenshots")
    print("- 1 interaction video")
    print("- Structured code generation request")
    print("\nExpected output: Complete React application code")


# ============================================================================
# Example 9: Time-Series Analysis
# ============================================================================

def example_time_series():
    """Analyze changes over time from multiple images"""
    print("\n" + "="*70)
    print("EXAMPLE 9: Time-Series Analysis")
    print("="*70)

    timestamps = [
        ("2024-01-01 08:00", "/path/to/site_2024_01_01_08.jpg"),
        ("2024-01-01 12:00", "/path/to/site_2024_01_01_12.jpg"),
        ("2024-01-01 16:00", "/path/to/site_2024_01_01_16.jpg"),
        ("2024-01-02 08:00", "/path/to/site_2024_01_02_08.jpg"),
        ("2024-01-02 16:00", "/path/to/site_2024_01_02_16.jpg"),
    ]

    content = [{"type": "text", "text": "Construction site progress over 2 days:"}]

    for timestamp, image_path in timestamps:
        content.append({"type": "text", "text": f"{timestamp}:"})
        content.append({"type": "image", "image": image_path})

    content.append({
        "type": "text",
        "text": "Analyze:\n1. What progress was made?\n2. Identify any delays or issues\n3. Estimate completion percentage\n4. Note any safety concerns"
    })

    messages = [{"role": "user", "content": content}]

    print("\nPrompt structure:")
    print(f"- {len(timestamps)} timestamped images")
    print("- Progress tracking request")
    print("\nExpected output: Temporal analysis with progress tracking")


# ============================================================================
# Example 10: Medical Image Analysis
# ============================================================================

def example_medical_imaging():
    """Analyze medical images with different modalities"""
    print("\n" + "="*70)
    print("EXAMPLE 10: Medical Image Analysis")
    print("="*70)
    print("⚠️  Note: For demonstration purposes only. Not for actual medical diagnosis.")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Patient imaging study:"},
                {"type": "text", "text": "X-Ray (Front):"},
                {"type": "image", "image": "/path/to/xray_front.jpg"},
                {"type": "text", "text": "X-Ray (Side):"},
                {"type": "image", "image": "/path/to/xray_side.jpg"},
                {"type": "text", "text": "CT Scan slices:"},
                {"type": "image", "image": "/path/to/ct_slice1.jpg"},
                {"type": "image", "image": "/path/to/ct_slice2.jpg"},
                {"type": "image", "image": "/path/to/ct_slice3.jpg"},
                {"type": "text", "text": "Ultrasound video:"},
                {"type": "video", "video": "/path/to/ultrasound.mp4"},
                {"type": "text", "text": "Compare findings across all imaging modalities and note any areas requiring attention."}
            ]
        }
    ]

    print("\nPrompt structure:")
    print("- 2 X-ray images (different angles)")
    print("- 3 CT scan slices")
    print("- 1 ultrasound video")
    print("- Cross-modality analysis request")
    print("\nExpected output: Comprehensive imaging analysis")
    print("⚠️  DISCLAIMER: For educational purposes only!")


# ============================================================================
# Main Demo Runner
# ============================================================================

def run_all_examples():
    """Run all example demonstrations"""
    print("\n" + "="*70)
    print("QWEN3-VL MULTIMODAL CAPABILITIES - EXAMPLES")
    print("="*70)
    print("\nThese examples demonstrate advanced multimodal capabilities")
    print("that are NOT currently implemented in the main app.py")
    print("\nTo actually run these examples:")
    print("1. Uncomment the model initialization lines")
    print("2. Replace /path/to/... with actual file paths")
    print("3. Ensure you have sufficient GPU memory")
    print("="*70)

    example_multiple_images()
    example_interleaved_content()
    example_image_and_video()
    example_document_analysis()
    example_product_comparison()
    example_quality_control()
    example_multi_turn_conversation()
    example_code_generation_from_ui()
    example_time_series()
    example_medical_imaging()

    print("\n" + "="*70)
    print("EXAMPLES COMPLETE")
    print("="*70)
    print("\nFor more information, see MULTIMODAL_CAPABILITIES.md")


if __name__ == "__main__":
    run_all_examples()
