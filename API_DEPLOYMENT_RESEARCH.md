# API Integrations and Deployment Research

## Project Overview
This document provides comprehensive research on API integrations and deployment options for the Qwen3-VL image description generator application.

**Current Stack:**
- Python-based Gradio web application
- Qwen Vision Language models (2B and 8B)
- GPU-accelerated inference with PyTorch
- Local file processing with batch capabilities

---

## 1. REST API Implementation Options

### 1.1 FastAPI (Recommended)
FastAPI is the industry standard for ML model deployment in 2025, offering high performance and production-ready features.

#### Key Features:
- **Async Support:** Built-in asynchronous request handling for better responsiveness under load
- **Auto Documentation:** Automatic OpenAPI (Swagger) and ReDoc documentation
- **Type Safety:** Pydantic models for request/response validation
- **Performance:** Based on Starlette and Uvicorn, matching NodeJS performance
- **Easy Integration:** Native support for mounting Gradio apps

#### Implementation Example:
```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import base64
from PIL import Image
import io

app = FastAPI(
    title="Qwen3-VL Image Description API",
    description="AI-powered image description generation",
    version="1.0.0"
)

class GenerationConfig(BaseModel):
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)
    top_p: float = Field(default=0.9, ge=0.05, le=1.0)
    top_k: int = Field(default=50, ge=1, le=1000)
    seed: Optional[int] = None

class DescriptionRequest(BaseModel):
    prompt: str = Field(..., description="Description prompt")
    config: Optional[GenerationConfig] = GenerationConfig()

class DescriptionResponse(BaseModel):
    description: str
    model_used: str
    tokens_generated: int
    generation_time: float

@app.post("/api/v1/describe", response_model=DescriptionResponse)
async def generate_description(
    image: UploadFile = File(..., description="Image file"),
    request: DescriptionRequest
):
    """
    Generate description for uploaded image
    """
    try:
        # Load image
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes))

        # Generate description (integrate with existing generator)
        result = await process_image(img, request.prompt, request.config)

        return DescriptionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/batch", response_model=List[DescriptionResponse])
async def batch_generate_descriptions(
    images: List[UploadFile] = File(...),
    prompts: List[str]
):
    """
    Batch process multiple images
    """
    # Implementation here
    pass

# Mount Gradio interface (optional - keep existing UI)
from gradio.routes import mount_gradio_app
import gradio as gr

# Your existing Gradio interface
demo = create_interface()
mount_gradio_app(app, demo, path="/ui")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
```

#### Production Configuration:
```python
# gunicorn_conf.py
import multiprocessing

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = 1  # Single worker for GPU models to avoid memory issues
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
timeout = 120
keepalive = 5

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
```

### 1.2 Flask (Alternative)
While FastAPI is recommended, Flask is a simpler option for basic APIs:

```python
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

@app.route('/api/v1/describe', methods=['POST'])
def describe_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    prompt = request.form.get('prompt', '')

    # Process image
    result = generator.generate(file, prompt)

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

---

## 2. Docker Deployment

### 2.1 GPU-Enabled Dockerfile
```dockerfile
# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install Python 3.11
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for models and outputs
RUN mkdir -p /app/models /app/output /app/datasets /app/prompts

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models

# Expose ports
EXPOSE 7860 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5m --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run application
CMD ["python3", "app.py"]
```

### 2.2 Multi-Stage Build (Production)
```dockerfile
# Stage 1: Build environment
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS builder

WORKDIR /build
COPY requirements.txt .

RUN apt-get update && apt-get install -y python3.11 python3-pip
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime environment
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.11 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy application
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

EXPOSE 7860 8000

CMD ["python3", "app.py"]
```

### 2.3 Docker Compose Configuration
```yaml
version: '3.8'

services:
  qwen3-vl-app:
    build:
      context: .
      dockerfile: Dockerfile
    image: qwen3-vl:latest
    container_name: qwen3-vl-app

    # GPU support
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

    ports:
      - "7860:7860"  # Gradio UI
      - "8000:8000"  # FastAPI

    volumes:
      - ./models:/app/models
      - ./output:/app/output
      - ./datasets:/app/datasets
      - ./prompts:/app/prompts

    environment:
      - CUDA_VISIBLE_DEVICES=0
      - HF_HOME=/app/models
      - TRANSFORMERS_CACHE=/app/models
      - GRADIO_SERVER_NAME=0.0.0.0

    restart: unless-stopped

    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7860/"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 5m

  # Optional: NGINX reverse proxy
  nginx:
    image: nginx:alpine
    container_name: qwen3-vl-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - qwen3-vl-app
    restart: unless-stopped
```

### 2.4 NGINX Configuration
```nginx
upstream qwen3_vl {
    server qwen3-vl-app:7860 max_fails=3 fail_timeout=30s;
}

upstream qwen3_vl_api {
    server qwen3-vl-app:8000 max_fails=3 fail_timeout=30s;
}

# Rate limiting
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

server {
    listen 80;
    server_name your-domain.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL configuration
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Gradio UI
    location / {
        proxy_pass http://qwen3_vl;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts for long-running inference
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }

    # API endpoints with rate limiting
    location /api/ {
        limit_req zone=api_limit burst=20 nodelay;

        proxy_pass http://qwen3_vl_api;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Larger timeouts for ML inference
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }

    # Health check
    location /health {
        proxy_pass http://qwen3_vl_api/health;
        access_log off;
    }
}
```

---

## 3. Cloud Deployment Options

### 3.1 AWS Deployment

#### Option A: AWS ECS with EC2 (GPU Support)
**Best for:** Production workloads requiring GPU acceleration

**Architecture:**
- ECS Cluster with GPU-enabled EC2 instances (g4dn, g5, p3, or p4d)
- Application Load Balancer for traffic distribution
- Auto Scaling based on CPU/GPU utilization
- CloudWatch for monitoring

**Infrastructure as Code (Terraform):**
```hcl
# main.tf
provider "aws" {
  region = "us-east-1"
}

# ECR Repository
resource "aws_ecr_repository" "qwen3_vl" {
  name                 = "qwen3-vl"
  image_tag_mutability = "MUTABLE"
}

# ECS Cluster
resource "aws_ecs_cluster" "qwen3_vl" {
  name = "qwen3-vl-cluster"
}

# ECS Task Definition
resource "aws_ecs_task_definition" "qwen3_vl" {
  family                   = "qwen3-vl-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["EC2"]
  cpu                      = "4096"
  memory                   = "16384"

  container_definitions = jsonencode([{
    name  = "qwen3-vl"
    image = "${aws_ecr_repository.qwen3_vl.repository_url}:latest"

    portMappings = [{
      containerPort = 7860
      hostPort      = 7860
      protocol      = "tcp"
    }]

    resourceRequirements = [{
      type  = "GPU"
      value = "1"
    }]

    environment = [
      { name = "HF_HOME", value = "/app/models" },
      { name = "CUDA_VISIBLE_DEVICES", value = "0" }
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = "/ecs/qwen3-vl"
        "awslogs-region"        = "us-east-1"
        "awslogs-stream-prefix" = "ecs"
      }
    }
  }])
}

# Application Load Balancer
resource "aws_lb" "qwen3_vl" {
  name               = "qwen3-vl-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = var.public_subnet_ids
}

# Target Group
resource "aws_lb_target_group" "qwen3_vl" {
  name        = "qwen3-vl-tg"
  port        = 7860
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"

  health_check {
    enabled             = true
    path                = "/health"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 30
    interval            = 60
  }
}

# ECS Service
resource "aws_ecs_service" "qwen3_vl" {
  name            = "qwen3-vl-service"
  cluster         = aws_ecs_cluster.qwen3_vl.id
  task_definition = aws_ecs_task_definition.qwen3_vl.arn
  desired_count   = 1
  launch_type     = "EC2"

  load_balancer {
    target_group_arn = aws_lb_target_group.qwen3_vl.arn
    container_name   = "qwen3-vl"
    container_port   = 7860
  }

  network_configuration {
    subnets         = var.private_subnet_ids
    security_groups = [aws_security_group.ecs_tasks.id]
  }
}
```

**Deployment Steps:**
```bash
# Build and push Docker image
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker build -t qwen3-vl .
docker tag qwen3-vl:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/qwen3-vl:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/qwen3-vl:latest

# Deploy with Terraform
terraform init
terraform plan
terraform apply
```

#### Option B: AWS SageMaker
**Best for:** Managed ML inference with auto-scaling

```python
# sagemaker_deploy.py
import sagemaker
from sagemaker.pytorch import PyTorchModel

role = "arn:aws:iam::account-id:role/SageMakerRole"

pytorch_model = PyTorchModel(
    model_data='s3://bucket/model.tar.gz',
    role=role,
    framework_version='2.0.0',
    py_version='py310',
    entry_point='inference.py',
    source_dir='./src'
)

predictor = pytorch_model.deploy(
    instance_type='ml.g5.xlarge',
    initial_instance_count=1,
    endpoint_name='qwen3-vl-endpoint'
)
```

#### Option C: AWS Lambda (CPU-only, small models)
**Best for:** Serverless, sporadic workloads, small models only

**Limitations:**
- No GPU support
- 15-minute timeout
- 10GB memory limit
- 250MB unzipped deployment package

**Not recommended for this application due to GPU requirements**

### 3.2 Google Cloud Platform (GCP) Deployment

#### Option A: GKE (Google Kubernetes Engine) with GPUs
**Best for:** Kubernetes-native deployments with GPU support

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qwen3-vl
spec:
  replicas: 2
  selector:
    matchLabels:
      app: qwen3-vl
  template:
    metadata:
      labels:
        app: qwen3-vl
    spec:
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-t4
      containers:
      - name: qwen3-vl
        image: gcr.io/project-id/qwen3-vl:latest
        ports:
        - containerPort: 7860
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        volumeMounts:
        - name: model-cache
          mountPath: /app/models
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: qwen3-vl-service
spec:
  type: LoadBalancer
  selector:
    app: qwen3-vl
  ports:
  - port: 80
    targetPort: 7860
```

#### Option B: Cloud Run with GPU (Recommended for 2025)
**Best for:** Serverless with GPU support, pay-per-second billing

**Key Features (2025):**
- NVIDIA L4 GPU support
- Auto-scaling to zero
- Per-second billing
- Fully managed

```yaml
# cloud-run-gpu.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: qwen3-vl
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "0"
        autoscaling.knative.dev/maxScale: "5"
        run.googleapis.com/cpu-throttling: "false"
    spec:
      containerConcurrency: 1
      containers:
      - image: gcr.io/project-id/qwen3-vl:latest
        ports:
        - containerPort: 7860
        resources:
          limits:
            nvidia.com/gpu: "1"
            memory: "16Gi"
            cpu: "4"
        env:
        - name: GRADIO_SERVER_NAME
          value: "0.0.0.0"
```

**Deployment:**
```bash
# Build and push
gcloud builds submit --tag gcr.io/project-id/qwen3-vl

# Deploy with GPU
gcloud run deploy qwen3-vl \
  --image gcr.io/project-id/qwen3-vl \
  --platform managed \
  --region us-central1 \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --memory 16Gi \
  --cpu 4 \
  --timeout 300 \
  --concurrency 1 \
  --min-instances 0 \
  --max-instances 5
```

#### Option C: Vertex AI
**Best for:** Managed ML platform with MLOps features

```python
# vertex_deploy.py
from google.cloud import aiplatform

aiplatform.init(project='project-id', location='us-central1')

model = aiplatform.Model.upload(
    display_name='qwen3-vl',
    artifact_uri='gs://bucket/model/',
    serving_container_image_uri='gcr.io/project-id/qwen3-vl:latest'
)

endpoint = model.deploy(
    machine_type='n1-standard-4',
    accelerator_type='NVIDIA_TESLA_T4',
    accelerator_count=1,
    min_replica_count=1,
    max_replica_count=5
)
```

### 3.3 Azure Deployment

#### Option A: Azure Kubernetes Service (AKS) with GPU
```bash
# Create AKS cluster with GPU support
az aks create \
  --resource-group qwen3-vl-rg \
  --name qwen3-vl-cluster \
  --node-count 1 \
  --node-vm-size Standard_NC6s_v3 \
  --enable-cluster-autoscaler \
  --min-count 1 \
  --max-count 3 \
  --generate-ssh-keys
```

#### Option B: Azure Container Instances (Limited GPU)
```bash
az container create \
  --resource-group qwen3-vl-rg \
  --name qwen3-vl-instance \
  --image myregistry.azurecr.io/qwen3-vl:latest \
  --gpu-count 1 \
  --gpu-sku V100 \
  --cpu 4 \
  --memory 16 \
  --ports 7860 8000
```

#### Option C: Azure Machine Learning
```python
# azure_ml_deploy.py
from azureml.core import Workspace, Model
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.model import InferenceConfig

ws = Workspace.from_config()

model = Model.register(
    workspace=ws,
    model_path='./model',
    model_name='qwen3-vl'
)

inference_config = InferenceConfig(
    entry_script='score.py',
    environment=env
)

deployment_config = AciWebservice.deploy_configuration(
    cpu_cores=4,
    memory_gb=16,
    gpu_cores=1
)

service = Model.deploy(
    workspace=ws,
    name='qwen3-vl-service',
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config
)
```

### 3.4 Cloud Platform Comparison

| Feature | AWS ECS | GCP Cloud Run | Azure AKS | AWS SageMaker |
|---------|---------|---------------|-----------|---------------|
| GPU Support | Yes (EC2) | Yes (L4) | Yes | Yes |
| Serverless | No | Yes | No | Serverless option |
| Auto-scaling | Yes | Excellent | Yes | Yes |
| Cold Start | N/A | Fast | N/A | Moderate |
| Pricing | Pay for instances | Pay per second | Pay for nodes | Pay per hour |
| Complexity | Moderate | Low | High | Low |
| Best For | Consistent workload | Variable workload | Complex deployments | ML-focused |

---

## 4. Serverless Options

### 4.1 Platform Comparison (2025)

| Platform | GPU Support | Max Timeout | Memory Limit | Best Use Case |
|----------|-------------|-------------|--------------|---------------|
| AWS Lambda | No | 15 min | 10 GB | Not suitable |
| GCP Cloud Run | Yes (L4) | 60 min | 32 GB | **Recommended** |
| Azure Functions | Limited | 10 min | 4 GB | Not suitable |
| Modal | Yes | Unlimited | Unlimited | Development |
| Banana.dev | Yes | Varies | Varies | ML inference |

### 4.2 Google Cloud Run with GPU (Best Option)

**Why Cloud Run wins in 2025:**
- Native NVIDIA L4 GPU support
- Pay-per-second billing (only pay when processing)
- Auto-scales to zero (cost-effective for variable workloads)
- 60-minute timeout (sufficient for inference)
- Fully managed infrastructure

**Example Implementation:**
```python
# app_cloud_run.py
import os
from fastapi import FastAPI
from gradio.routes import mount_gradio_app

# Your existing application
app = FastAPI()

# ... API endpoints ...

# Mount Gradio
demo = create_interface()
mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

### 4.3 Alternative: Modal.com
**Excellent for development and prototyping**

```python
# modal_deploy.py
import modal

stub = modal.Stub("qwen3-vl")

image = modal.Image.debian_slim().pip_install(
    "torch", "transformers", "gradio", "qwen-vl-utils"
).run_commands(
    "pip install flash-attn --no-build-isolation"
)

@stub.function(
    image=image,
    gpu="A10G",
    memory=16384,
    timeout=600
)
def generate_description(image_bytes, prompt):
    # Your inference code
    pass

@stub.asgi_app()
def app():
    import gradio as gr
    # Your Gradio interface
    return demo
```

---

## 5. Load Balancing Strategies

### 5.1 Traditional vs ML-Aware Load Balancing

**Problem with Traditional Load Balancing:**
- Round-robin and least-connections are "model-blind"
- Don't account for GPU memory usage
- Can't optimize for inference queue depth
- May route requests to overloaded instances

### 5.2 Gateway API Inference Extension (Kubernetes - 2025)

**Introduced in 2025 specifically for ML workloads**

Key features:
- Model-aware routing based on real-time metrics
- Queue length optimization
- GPU memory utilization tracking
- Support for multi-model deployments
- A/B testing and canary deployments

**Implementation with NGINX Gateway Fabric:**
```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: InferencePool
metadata:
  name: qwen3-vl-pool
spec:
  targetRef:
    group: ""
    kind: Service
    name: qwen3-vl-service
  endpointPicker:
    type: QueueLength
    config:
      metricEndpoint: "/metrics"
      queueLengthThreshold: 10
  modelCapabilities:
    - name: qwen3-vl-2b
      accelerator: nvidia-l4
      memoryGB: 8
    - name: qwen3-vl-8b
      accelerator: nvidia-a10g
      memoryGB: 16
```

### 5.3 NGINX Configuration for ML Inference

```nginx
upstream qwen3_vl_pool {
    # Use least connections instead of round-robin
    least_conn;

    # Server definitions with health checks
    server backend1:7860 max_fails=3 fail_timeout=30s;
    server backend2:7860 max_fails=3 fail_timeout=30s;
    server backend3:7860 max_fails=3 fail_timeout=30s;

    # Keepalive connections
    keepalive 32;
}

# Session affinity for stateful inference
map $cookie_session_id $sticky_backend {
    default "";
    ~^(.+)$ "$1";
}

server {
    listen 80;

    location / {
        # Sticky sessions based on cookie
        hash $cookie_session_id consistent;

        proxy_pass http://qwen3_vl_pool;

        # WebSocket support for Gradio
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Extended timeouts for ML inference
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
        proxy_buffering off;

        # Request buffering
        client_max_body_size 50M;
        client_body_buffer_size 50M;
    }
}
```

### 5.4 Kubernetes Horizontal Pod Autoscaler (HPA)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: qwen3-vl-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: qwen3-vl
  minReplicas: 1
  maxReplicas: 10
  metrics:
  # CPU-based scaling
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  # Memory-based scaling
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  # Custom metric: inference queue length
  - type: Pods
    pods:
      metric:
        name: inference_queue_length
      target:
        type: AverageValue
        averageValue: "5"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 1
        periodSeconds: 60
```

### 5.5 Load Balancing Best Practices for ML

1. **Session Affinity:** Use sticky sessions to route users to the same instance
2. **Queue-Based Routing:** Route to instances with shortest queues
3. **Model-Aware:** Consider which models are loaded on each instance
4. **GPU Memory:** Monitor and route based on available GPU memory
5. **Graceful Shutdown:** Allow in-flight requests to complete before scaling down
6. **Health Checks:** Verify model is loaded and GPU is available

---

## 6. Production-Ready Examples

### 6.1 Complete FastAPI + Docker + Kubernetes Setup

**Directory Structure:**
```
qwen3-vl-production/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py     # API endpoints
│   │   └── models.py     # Pydantic models
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py     # Configuration
│   │   └── inference.py  # Model inference logic
│   └── gradio_app.py     # Gradio interface
├── kubernetes/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── hpa.yaml
│   └── configmap.yaml
├── docker/
│   ├── Dockerfile
│   ├── Dockerfile.dev
│   └── docker-compose.yml
├── tests/
│   ├── test_api.py
│   └── test_inference.py
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

**main.py:**
```python
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel
import uvicorn
import logging
from typing import List, Optional
import time

from .core.inference import ImageDescriptionGenerator
from .api.models import DescriptionRequest, DescriptionResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
REQUEST_COUNT = Counter('request_count', 'Total API requests', ['endpoint', 'status'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration', ['endpoint'])
INFERENCE_COUNT = Counter('inference_count', 'Total inferences', ['model'])
INFERENCE_DURATION = Histogram('inference_duration_seconds', 'Inference duration', ['model'])

app = FastAPI(
    title="Qwen3-VL Image Description API",
    description="AI-powered image description generation using Qwen Vision Language models",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
generator = None

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global generator
    logger.info("Loading model...")
    generator = ImageDescriptionGenerator()
    await generator.load_model()
    logger.info("Model loaded successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global generator
    if generator:
        await generator.cleanup()
    logger.info("Application shutdown")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if generator is None or not generator.is_ready():
        raise HTTPException(status_code=503, detail="Model not ready")
    return {
        "status": "healthy",
        "model_loaded": True,
        "gpu_available": generator.gpu_available
    }

@app.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes"""
    if generator is None:
        raise HTTPException(status_code=503, detail="Not ready")
    return {"status": "ready"}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.post("/api/v1/describe", response_model=DescriptionResponse)
async def generate_description(
    image: UploadFile = File(...),
    prompt: str = "Describe this image in detail",
    max_tokens: int = 512,
    temperature: float = 0.7
):
    """
    Generate description for uploaded image

    - **image**: Image file (JPEG, PNG)
    - **prompt**: Description prompt
    - **max_tokens**: Maximum tokens to generate
    - **temperature**: Sampling temperature
    """
    start_time = time.time()

    try:
        # Read image
        image_bytes = await image.read()

        # Generate description
        result = await generator.generate(
            image_bytes=image_bytes,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

        # Metrics
        duration = time.time() - start_time
        REQUEST_COUNT.labels(endpoint='/api/v1/describe', status='success').inc()
        REQUEST_DURATION.labels(endpoint='/api/v1/describe').observe(duration)
        INFERENCE_COUNT.labels(model=result['model_used']).inc()
        INFERENCE_DURATION.labels(model=result['model_used']).observe(result['inference_time'])

        return DescriptionResponse(**result)

    except Exception as e:
        REQUEST_COUNT.labels(endpoint='/api/v1/describe', status='error').inc()
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/batch", response_model=List[DescriptionResponse])
async def batch_generate(
    images: List[UploadFile] = File(...),
    prompts: Optional[List[str]] = None
):
    """Batch process multiple images"""
    if prompts and len(prompts) != len(images):
        raise HTTPException(
            status_code=400,
            detail="Number of prompts must match number of images"
        )

    results = []
    for i, image in enumerate(images):
        image_bytes = await image.read()
        prompt = prompts[i] if prompts else "Describe this image in detail"

        result = await generator.generate(
            image_bytes=image_bytes,
            prompt=prompt
        )
        results.append(DescriptionResponse(**result))

    return results

# Mount Gradio interface (optional)
try:
    from gradio.routes import mount_gradio_app
    from .gradio_app import create_interface

    demo = create_interface()
    mount_gradio_app(app, demo, path="/ui")
    logger.info("Gradio UI mounted at /ui")
except ImportError:
    logger.warning("Gradio not available, UI will not be mounted")

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_level="info"
    )
```

**config.py:**
```python
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Qwen3-VL API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Model
    MODEL_NAME: str = "Qwen/Qwen2-VL-2B-Instruct"
    MODEL_CACHE_DIR: str = "/app/models"
    DEVICE: str = "cuda"
    QUANTIZATION: str = "4-bit"

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    MAX_WORKERS: int = 1
    REQUEST_TIMEOUT: int = 300

    # Inference
    MAX_BATCH_SIZE: int = 4
    MAX_TOKENS: int = 512
    TEMPERATURE: float = 0.7

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
```

**kubernetes/deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qwen3-vl
  labels:
    app: qwen3-vl
    version: v1
spec:
  replicas: 2
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: qwen3-vl
  template:
    metadata:
      labels:
        app: qwen3-vl
        version: v1
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      # Node selector for GPU nodes
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-t4

      # Anti-affinity to spread pods across nodes
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - qwen3-vl
              topologyKey: kubernetes.io/hostname

      # Init container to download models
      initContainers:
      - name: model-downloader
        image: python:3.11-slim
        command: ["/bin/sh", "-c"]
        args:
          - |
            pip install huggingface_hub
            python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2-VL-2B-Instruct', cache_dir='/models')"
        volumeMounts:
        - name: model-cache
          mountPath: /models

      containers:
      - name: qwen3-vl
        image: gcr.io/project-id/qwen3-vl:latest
        imagePullPolicy: Always

        ports:
        - name: http
          containerPort: 8000
          protocol: TCP

        env:
        - name: MODEL_CACHE_DIR
          value: "/models"
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: TRANSFORMERS_CACHE
          value: "/models"

        resources:
          requests:
            cpu: "2"
            memory: "8Gi"
            nvidia.com/gpu: "1"
          limits:
            cpu: "4"
            memory: "16Gi"
            nvidia.com/gpu: "1"

        # Probes
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 300
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3

        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3

        volumeMounts:
        - name: model-cache
          mountPath: /models
        - name: output
          mountPath: /app/output

      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
      - name: output
        emptyDir: {}

      # Graceful shutdown
      terminationGracePeriodSeconds: 60
```

### 6.2 Monitoring and Observability

**Prometheus + Grafana Setup:**

```yaml
# prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s

    scrape_configs:
    - job_name: 'qwen3-vl'
      kubernetes_sd_configs:
      - role: pod
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__
```

**Custom Metrics:**
```python
from prometheus_client import Counter, Histogram, Gauge, Info

# Request metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

# Model metrics
model_inference_duration = Histogram(
    'model_inference_duration_seconds',
    'Model inference duration',
    ['model_name', 'quantization']
)

model_load_time = Gauge(
    'model_load_time_seconds',
    'Time taken to load model'
)

active_requests = Gauge(
    'active_requests',
    'Number of active requests'
)

# GPU metrics
gpu_memory_used = Gauge(
    'gpu_memory_used_bytes',
    'GPU memory used',
    ['gpu_id']
)

gpu_utilization = Gauge(
    'gpu_utilization_percent',
    'GPU utilization',
    ['gpu_id']
)
```

### 6.3 CI/CD Pipeline (GitHub Actions)

```yaml
# .github/workflows/deploy.yml
name: Build and Deploy

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

env:
  PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
  GKE_CLUSTER: qwen3-vl-cluster
  GKE_ZONE: us-central1-a
  IMAGE: qwen3-vl

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run tests
      run: |
        pytest tests/ --cov=app --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push'

    steps:
    - uses: actions/checkout@v3

    - name: Set up Cloud SDK
      uses: google-github-actions/setup-gcloud@v1
      with:
        service_account_key: ${{ secrets.GCP_SA_KEY }}
        project_id: ${{ secrets.GCP_PROJECT_ID }}

    - name: Configure Docker
      run: gcloud auth configure-docker

    - name: Build Docker image
      run: |
        docker build -t gcr.io/$PROJECT_ID/$IMAGE:$GITHUB_SHA \
                     -t gcr.io/$PROJECT_ID/$IMAGE:latest .

    - name: Push Docker image
      run: |
        docker push gcr.io/$PROJECT_ID/$IMAGE:$GITHUB_SHA
        docker push gcr.io/$PROJECT_ID/$IMAGE:latest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v3

    - name: Set up Cloud SDK
      uses: google-github-actions/setup-gcloud@v1
      with:
        service_account_key: ${{ secrets.GCP_SA_KEY }}
        project_id: ${{ secrets.GCP_PROJECT_ID }}

    - name: Get GKE credentials
      run: |
        gcloud container clusters get-credentials $GKE_CLUSTER --zone $GKE_ZONE

    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/qwen3-vl \
          qwen3-vl=gcr.io/$PROJECT_ID/$IMAGE:$GITHUB_SHA
        kubectl rollout status deployment/qwen3-vl
```

---

## 7. Advanced Model Serving Frameworks

### 7.1 KServe
**Best for:** Kubernetes-native deployments with advanced features

**Features:**
- Multi-model serving
- A/B testing and canary deployments
- Model versioning and rollback
- Auto-scaling based on metrics
- Integration with monitoring tools

**Example:**
```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: qwen3-vl
spec:
  predictor:
    containers:
    - name: kserve-container
      image: gcr.io/project-id/qwen3-vl:latest
      resources:
        limits:
          nvidia.com/gpu: 1
          memory: 16Gi
        requests:
          nvidia.com/gpu: 1
          memory: 16Gi
  canaryTrafficPercent: 10
```

### 7.2 Ray Serve
**Best for:** Model composition and many-model serving

**Features:**
- Framework-agnostic
- Model composition
- Independent auto-scaling per model
- Python-native API

**Example:**
```python
from ray import serve
import ray

ray.init()
serve.start()

@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    max_concurrent_queries=5
)
class Qwen3VLModel:
    def __init__(self):
        self.generator = ImageDescriptionGenerator()

    async def __call__(self, request):
        image = await request.body()
        result = self.generator.generate(image)
        return result

Qwen3VLModel.deploy()
```

### 7.3 NVIDIA Triton Inference Server
**Best for:** High-performance GPU inference

**Features:**
- Multi-framework support
- Dynamic batching
- Model ensemble
- GPU optimization

**Example config.pbtxt:**
```
name: "qwen3_vl"
platform: "pytorch_libtorch"
max_batch_size: 8
input [
  {
    name: "input_image"
    data_type: TYPE_UINT8
    dims: [ -1, -1, 3 ]
  }
]
output [
  {
    name: "output_text"
    data_type: TYPE_STRING
    dims: [ -1 ]
  }
]
instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
dynamic_batching {
  preferred_batch_size: [ 4, 8 ]
  max_queue_delay_microseconds: 100000
}
```

---

## 8. Cost Optimization Strategies

### 8.1 Cloud Cost Comparison (Estimated Monthly)

| Deployment Option | GPU Type | Monthly Cost | Best For |
|-------------------|----------|--------------|----------|
| AWS EC2 g4dn.xlarge | T4 | ~$400 | Consistent workload |
| GCP Cloud Run (L4) | L4 | $50-500 | Variable workload |
| Azure NC6s_v3 | V100 | ~$1,200 | High performance |
| Modal (on-demand) | A10G | $100-300 | Development |

### 8.2 Cost Optimization Tips

1. **Use Spot Instances:** Save up to 70% with preemptible/spot instances
2. **Auto-scaling:** Scale to zero during idle periods
3. **Model Quantization:** 4-bit models use less GPU memory
4. **Batch Processing:** Process multiple images together
5. **CDN Caching:** Cache common results
6. **Regional Selection:** Choose cheaper regions when possible

---

## 9. Security Best Practices

### 9.1 API Security

```python
from fastapi import Security, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

@app.post("/api/v1/describe")
async def generate_description(
    image: UploadFile,
    token: dict = Depends(verify_token)
):
    # Your code here
    pass
```

### 9.2 Rate Limiting

```python
from fastapi import Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/api/v1/describe")
@limiter.limit("10/minute")
async def generate_description(request: Request, image: UploadFile):
    # Your code here
    pass
```

### 9.3 Input Validation

```python
from fastapi import UploadFile, HTTPException
from PIL import Image
import io

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

async def validate_image(file: UploadFile):
    # Check file extension
    if not file.filename.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
        raise HTTPException(400, "Invalid file type")

    # Check file size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large")

    # Validate image
    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()
    except:
        raise HTTPException(400, "Invalid image file")

    await file.seek(0)
    return file
```

---

## 10. Recommended Architecture

### For Production Deployment:

**Option A: High Availability (Kubernetes + GPU)**
```
Internet → Cloud Load Balancer → Ingress Controller → NGINX → FastAPI Pods (GPU)
                                                              → Gradio Pods (GPU)
```

**Components:**
- GKE or EKS cluster with GPU node pools
- NGINX Ingress with rate limiting
- FastAPI for REST API
- Gradio for web UI
- Prometheus + Grafana for monitoring
- Horizontal Pod Autoscaler

**Option B: Serverless (Cost-Optimized)**
```
Internet → Cloud Load Balancer → GCP Cloud Run (GPU) → Model Storage
```

**Components:**
- Google Cloud Run with L4 GPUs
- FastAPI + Gradio in single container
- Cloud Storage for model cache
- Auto-scaling to zero
- Pay-per-second billing

### Recommended: Hybrid Approach
- **API:** FastAPI on Cloud Run for auto-scaling
- **UI:** Gradio on separate Cloud Run instance
- **Storage:** Cloud Storage for models and outputs
- **Monitoring:** Cloud Monitoring + custom metrics
- **CI/CD:** Cloud Build or GitHub Actions

---

## 11. Next Steps

1. **Choose Deployment Platform:**
   - Variable workload → GCP Cloud Run
   - Consistent workload → ECS with EC2
   - Complex requirements → Kubernetes

2. **Implement API Layer:**
   - FastAPI for REST endpoints
   - Keep Gradio for UI (optional)
   - Add authentication and rate limiting

3. **Create Docker Image:**
   - Use multi-stage builds
   - Optimize for GPU
   - Include health checks

4. **Set Up CI/CD:**
   - Automated testing
   - Container builds
   - Deployment pipeline

5. **Add Monitoring:**
   - Prometheus metrics
   - Logging
   - Alerting

6. **Load Testing:**
   - Test with realistic workloads
   - Optimize based on results

---

## Resources

### Official Documentation
- [FastAPI](https://fastapi.tiangolo.com/)
- [Gradio Deployment Guide](https://www.gradio.app/guides/deploying-gradio-with-docker)
- [Google Cloud Run GPU](https://cloud.google.com/run/docs/configuring/services/gpu)
- [AWS ECS GPU](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/managed-instances-gpu.html)
- [Kubernetes Gateway API Inference Extension](https://kubernetes.io/blog/2025/06/05/introducing-gateway-api-inference-extension/)

### Tools and Frameworks
- [KServe](https://kserve.github.io/)
- [Ray Serve](https://docs.ray.io/en/latest/serve/)
- [NVIDIA Triton](https://github.com/triton-inference-server/)
- [NGINX Gateway Fabric](https://docs.nginx.com/nginx-gateway-fabric/)

### Tutorials and Examples
- [MLOps with FastAPI and MLflow](https://liviaerxin.github.io/blog/end-to-end-ml-deployment)
- [FastAPI ML Skeleton](https://github.com/eightBEC/fastapi-ml-skeleton)
- [Hugging Face Docker Deployment](https://www.runpod.io/articles/guides/deploy-hugging-face-docker)

---

## Conclusion

For the Qwen3-VL project, I recommend:

1. **API Framework:** FastAPI for production-ready REST API
2. **Deployment:** Google Cloud Run with GPU support (most cost-effective for variable workloads)
3. **Alternative:** Kubernetes with GPU nodes for high-availability requirements
4. **Load Balancing:** NGINX with Gateway API Inference Extension for ML-aware routing
5. **Monitoring:** Prometheus + Grafana for observability

This setup provides:
- Scalability (auto-scale based on demand)
- Cost-efficiency (pay only for actual usage)
- Production-ready (health checks, monitoring, logging)
- Developer-friendly (easy deployment and maintenance)
