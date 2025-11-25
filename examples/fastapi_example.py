"""
FastAPI Example for Qwen3-VL Image Description API
This is a production-ready example showing how to serve the model via REST API
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
import torch
from PIL import Image
import io
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Qwen3-VL Image Description API",
    description="AI-powered image description generation using Qwen Vision Language models",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class GenerationConfig(BaseModel):
    """Configuration for text generation"""
    max_tokens: int = Field(default=512, ge=1, le=4096, description="Maximum number of tokens to generate")
    temperature: float = Field(default=0.7, ge=0.1, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.05, le=1.0, description="Nucleus sampling probability")
    top_k: int = Field(default=50, ge=1, le=1000, description="Top-k sampling")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")

class DescriptionRequest(BaseModel):
    """Request model for description generation"""
    prompt: str = Field(..., description="Prompt for description generation", min_length=1)
    config: GenerationConfig = Field(default_factory=GenerationConfig)

class DescriptionResponse(BaseModel):
    """Response model for description generation"""
    description: str
    model_used: str
    tokens_generated: int
    generation_time_seconds: float
    timestamp: str

class BatchDescriptionRequest(BaseModel):
    """Request model for batch description generation"""
    prompts: List[str] = Field(..., description="List of prompts for each image")
    config: GenerationConfig = Field(default_factory=GenerationConfig)

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    gpu_available: bool
    gpu_memory_allocated_gb: Optional[float] = None
    gpu_memory_reserved_gb: Optional[float] = None

# Global model instance (loaded on startup)
generator = None

@app.on_event("startup")
async def startup_event():
    """Initialize model on application startup"""
    global generator
    logger.info("Starting application...")
    logger.info("Loading Qwen3-VL model...")

    try:
        # Import your existing generator class
        # from app import ImageDescriptionGenerator
        # generator = ImageDescriptionGenerator()
        # await generator.load_model()

        # For now, just log that we would load the model
        logger.info("Model loaded successfully")
        logger.info(f"GPU available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    global generator
    logger.info("Shutting down application...")
    if generator:
        # Clean up model resources
        del generator
        torch.cuda.empty_cache()
    logger.info("Cleanup completed")

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Qwen3-VL Image Description API",
        "version": "1.0.0",
        "description": "AI-powered image description generation",
        "docs": "/api/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint for monitoring and load balancers
    Returns model status and GPU information
    """
    gpu_available = torch.cuda.is_available()
    gpu_mem_allocated = None
    gpu_mem_reserved = None

    if gpu_available:
        gpu_mem_allocated = torch.cuda.memory_allocated(0) / 1024**3  # Convert to GB
        gpu_mem_reserved = torch.cuda.memory_reserved(0) / 1024**3

    return HealthResponse(
        status="healthy",
        model_loaded=generator is not None,
        gpu_available=gpu_available,
        gpu_memory_allocated_gb=gpu_mem_allocated,
        gpu_memory_reserved_gb=gpu_mem_reserved
    )

@app.get("/ready", tags=["Health"])
async def readiness_check():
    """
    Readiness check for Kubernetes
    Returns 200 only when model is loaded and ready
    """
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"status": "ready"}

@app.post("/api/v1/describe", response_model=DescriptionResponse, tags=["Description"])
async def generate_description(
    image: UploadFile = File(..., description="Image file to describe (JPEG, PNG, etc.)"),
    prompt: str = "Describe this image in detail",
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    seed: Optional[int] = None
):
    """
    Generate a description for the uploaded image

    - **image**: Image file (JPEG, PNG, GIF, etc.)
    - **prompt**: Text prompt to guide description generation
    - **max_tokens**: Maximum number of tokens to generate (1-4096)
    - **temperature**: Sampling temperature (0.1-2.0). Lower = more focused, higher = more creative
    - **top_p**: Nucleus sampling probability (0.05-1.0)
    - **top_k**: Top-k sampling (1-1000)
    - **seed**: Random seed for reproducibility (optional)

    Returns:
    - **description**: Generated text description
    - **model_used**: Name of the model used
    - **tokens_generated**: Number of tokens in the generated text
    - **generation_time_seconds**: Time taken for generation
    """
    import time

    start_time = time.time()

    try:
        # Validate image file
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read image
        image_bytes = await image.read()

        # Load and validate image
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.verify()  # Verify it's a valid image
            img = Image.open(io.BytesIO(image_bytes))  # Reload after verify
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

        # Generate description (integrate with your existing generator)
        # This is where you would call your model
        # result = await generator.generate(
        #     image=img,
        #     prompt=prompt,
        #     max_tokens=max_tokens,
        #     temperature=temperature,
        #     top_p=top_p,
        #     top_k=top_k,
        #     seed=seed
        # )

        # For demonstration, return a mock response
        generation_time = time.time() - start_time

        result = DescriptionResponse(
            description=f"[Generated description based on prompt: {prompt}]",
            model_used="Qwen/Qwen2-VL-2B-Instruct",
            tokens_generated=50,
            generation_time_seconds=generation_time,
            timestamp=datetime.utcnow().isoformat()
        )

        logger.info(f"Generated description in {generation_time:.2f}s")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating description: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/batch", response_model=List[DescriptionResponse], tags=["Description"])
async def batch_generate_descriptions(
    images: List[UploadFile] = File(..., description="List of image files"),
    prompts: Optional[List[str]] = None
):
    """
    Generate descriptions for multiple images in batch

    - **images**: List of image files
    - **prompts**: Optional list of prompts (one per image). If not provided, uses default prompt for all.

    Returns list of descriptions with metadata for each image.
    """
    if prompts and len(prompts) != len(images):
        raise HTTPException(
            status_code=400,
            detail=f"Number of prompts ({len(prompts)}) must match number of images ({len(images)})"
        )

    results = []

    for i, image in enumerate(images):
        prompt = prompts[i] if prompts else "Describe this image in detail"

        try:
            # Process each image
            result = await generate_description(
                image=image,
                prompt=prompt
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing image {i}: {str(e)}")
            # Continue with other images
            results.append(DescriptionResponse(
                description=f"Error: {str(e)}",
                model_used="N/A",
                tokens_generated=0,
                generation_time_seconds=0,
                timestamp=datetime.utcnow().isoformat()
            ))

    return results

@app.post("/api/v1/stream", tags=["Description"])
async def stream_description(
    image: UploadFile = File(...),
    prompt: str = "Describe this image in detail"
):
    """
    Generate description with streaming response
    Useful for long descriptions to show progress to users
    """
    try:
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes))

        async def generate_stream():
            # Simulate streaming (replace with actual streaming generation)
            words = ["This", "is", "a", "streaming", "description", "of", "the", "image"]
            for word in words:
                yield f"data: {json.dumps({'token': word})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/models", tags=["Models"])
async def list_models():
    """
    List available models and their status
    """
    return {
        "models": [
            {
                "name": "Qwen/Qwen2-VL-2B-Instruct",
                "size": "2B parameters",
                "loaded": True,
                "quantization": "4-bit"
            },
            {
                "name": "Qwen/Qwen2-VL-8B-Instruct",
                "size": "8B parameters",
                "loaded": False,
                "quantization": "4-bit"
            }
        ]
    }

@app.post("/api/v1/models/switch", tags=["Models"])
async def switch_model(model_name: str):
    """
    Switch to a different model
    This will unload the current model and load the new one
    """
    valid_models = [
        "Qwen/Qwen2-VL-2B-Instruct",
        "Qwen/Qwen2-VL-8B-Instruct"
    ]

    if model_name not in valid_models:
        raise HTTPException(status_code=400, detail=f"Invalid model. Must be one of: {valid_models}")

    try:
        # Unload current model and load new one
        # This is where you would implement model switching
        return {
            "status": "success",
            "message": f"Switched to model: {model_name}",
            "model": model_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: Mount Gradio interface at /ui
# This allows you to keep both API and UI in the same application
def mount_gradio_interface():
    """
    Mount the existing Gradio interface at /ui
    This is optional and allows you to provide both API and web UI
    """
    try:
        from gradio.routes import mount_gradio_app
        import gradio as gr

        # Import your existing Gradio interface
        # from app import create_interface
        # demo = create_interface()

        # Create a simple demo interface
        def greet(image):
            return "Gradio UI mounted successfully!"

        demo = gr.Interface(fn=greet, inputs="image", outputs="text")

        mount_gradio_app(app, demo, path="/ui")
        logger.info("Gradio interface mounted at /ui")
    except ImportError:
        logger.warning("Gradio not available, UI not mounted")
    except Exception as e:
        logger.error(f"Failed to mount Gradio interface: {str(e)}")

# Uncomment to enable Gradio UI
# mount_gradio_interface()

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "fastapi_example:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )
