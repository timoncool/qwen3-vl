# Deployment Examples for Qwen3-VL

This directory contains production-ready deployment examples for the Qwen3-VL image description application.

## Contents

1. **fastapi_example.py** - Production-ready FastAPI REST API implementation
2. **Dockerfile.production** - Multi-stage Docker build for GPU-enabled deployment
3. **docker-compose.yml** - Docker Compose configuration with NGINX and monitoring
4. **nginx.conf** - Production NGINX configuration with load balancing
5. **kubernetes-deployment.yaml** - Complete Kubernetes deployment with GPU support
6. **github-actions-deploy.yml** - CI/CD pipeline for automated deployment

## Quick Start

### Local Development with Docker

```bash
# Build the image
docker build -f examples/Dockerfile.production -t qwen3-vl:latest ..

# Run with GPU support
docker run --gpus all -p 8000:8000 -p 7860:7860 qwen3-vl:latest

# Or use Docker Compose (includes NGINX)
cd examples
docker-compose up -d
```

### Deploy to Kubernetes

```bash
# Apply all configurations
kubectl apply -f examples/kubernetes-deployment.yaml

# Check deployment status
kubectl get pods -n qwen3-vl
kubectl get services -n qwen3-vl
kubectl get ingress -n qwen3-vl

# View logs
kubectl logs -f deployment/qwen3-vl -n qwen3-vl
```

### Deploy to Google Cloud Run (Serverless with GPU)

```bash
# Build and push image
gcloud builds submit --tag gcr.io/PROJECT_ID/qwen3-vl

# Deploy with GPU
gcloud run deploy qwen3-vl \
  --image gcr.io/PROJECT_ID/qwen3-vl \
  --platform managed \
  --region us-central1 \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --memory 16Gi \
  --cpu 4 \
  --timeout 300 \
  --allow-unauthenticated
```

## FastAPI REST API

The FastAPI example provides a production-ready REST API with the following endpoints:

### Endpoints

- `GET /` - API information
- `GET /health` - Health check with GPU status
- `GET /ready` - Readiness check for Kubernetes
- `POST /api/v1/describe` - Generate description for single image
- `POST /api/v1/batch` - Batch process multiple images
- `POST /api/v1/stream` - Streaming description generation
- `GET /api/v1/models` - List available models
- `POST /api/v1/models/switch` - Switch to different model
- `GET /api/docs` - Interactive API documentation (Swagger UI)
- `GET /api/redoc` - Alternative API documentation (ReDoc)

### Usage Examples

**Python:**
```python
import requests

# Single image description
with open('image.jpg', 'rb') as f:
    files = {'image': f}
    data = {'prompt': 'Describe this image in detail'}
    response = requests.post('http://localhost:8000/api/v1/describe', files=files, data=data)
    print(response.json())
```

**cURL:**
```bash
curl -X POST "http://localhost:8000/api/v1/describe" \
  -F "image=@image.jpg" \
  -F "prompt=Describe this image"
```

**JavaScript:**
```javascript
const formData = new FormData();
formData.append('image', file);
formData.append('prompt', 'Describe this image');

fetch('http://localhost:8000/api/v1/describe', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

## Docker Deployment

### Building the Image

The production Dockerfile uses multi-stage builds for optimal size:

```bash
# Build production image
docker build -f examples/Dockerfile.production -t qwen3-vl:latest .

# Run with GPU
docker run --gpus all -p 8000:8000 qwen3-vl:latest
```

### Docker Compose

The docker-compose.yml includes:
- Main application with GPU support
- NGINX reverse proxy
- Prometheus monitoring (optional)
- Grafana dashboards (optional)

```bash
# Start all services
docker-compose -f examples/docker-compose.yml up -d

# Start with monitoring
docker-compose -f examples/docker-compose.yml --profile monitoring up -d

# View logs
docker-compose -f examples/docker-compose.yml logs -f

# Stop services
docker-compose -f examples/docker-compose.yml down
```

## Kubernetes Deployment

The Kubernetes configuration includes:
- Deployment with GPU support
- Horizontal Pod Autoscaler
- Ingress with NGINX
- ConfigMap for configuration
- PersistentVolumeClaim for model cache
- ServiceMonitor for Prometheus

### Prerequisites

1. Kubernetes cluster with GPU nodes
2. NVIDIA device plugin installed
3. NGINX Ingress Controller

```bash
# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# Install NGINX Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/cloud/deploy.yaml
```

### Deployment

```bash
# Deploy application
kubectl apply -f examples/kubernetes-deployment.yaml

# Check status
kubectl get all -n qwen3-vl

# Scale deployment
kubectl scale deployment qwen3-vl --replicas=3 -n qwen3-vl

# Update image
kubectl set image deployment/qwen3-vl qwen3-vl=gcr.io/project/qwen3-vl:v2 -n qwen3-vl

# Rollback if needed
kubectl rollout undo deployment/qwen3-vl -n qwen3-vl
```

## CI/CD with GitHub Actions

The GitHub Actions workflow provides:
- Automated testing
- Docker image building and pushing
- Security scanning with Trivy
- Deployment to staging and production
- Automatic rollback on failure

### Setup

1. Add GitHub Secrets:
   - `GCP_PROJECT_ID` - Your GCP project ID
   - `GCP_SA_KEY` - Service account key JSON

2. Configure environment URLs in workflow

3. Push to trigger deployment:
   ```bash
   git push origin develop  # Deploy to staging
   git push origin main     # Deploy to production
   ```

## Cloud Deployment Options

### Google Cloud Platform

**Cloud Run (Recommended for variable workloads):**
- Serverless with GPU support (L4)
- Auto-scaling to zero
- Pay-per-second billing

**GKE (Recommended for consistent workloads):**
- Full Kubernetes control
- Multiple GPU types available
- Advanced networking and scaling

### Amazon Web Services

**ECS with EC2:**
- GPU instances (g4dn, g5, p3, p4d)
- Auto Scaling Groups
- Application Load Balancer

**SageMaker:**
- Managed inference endpoints
- Built-in monitoring
- A/B testing support

### Microsoft Azure

**AKS:**
- GPU-enabled node pools
- Azure Monitor integration
- Azure Container Registry

**Azure ML:**
- Managed endpoints
- AutoML capabilities
- MLOps tools

## Monitoring and Observability

### Prometheus Metrics

The application exposes Prometheus metrics at `/metrics`:
- Request count and duration
- Inference count and duration
- GPU memory usage
- Active requests

### Grafana Dashboards

Import the provided dashboards for visualization:
- API performance metrics
- Model inference statistics
- GPU utilization
- System resources

### Logging

Structured JSON logging is enabled by default:
```python
logger.info("Generated description", extra={
    "model": "qwen3-vl-2b",
    "duration": 1.23,
    "tokens": 50
})
```

## Security Best Practices

1. **Use non-root user** in containers
2. **Enable HTTPS** in production
3. **Implement authentication** for API endpoints
4. **Set rate limits** to prevent abuse
5. **Scan images** for vulnerabilities
6. **Use secrets management** for sensitive data
7. **Enable network policies** in Kubernetes
8. **Regularly update** dependencies

## Performance Optimization

### Model Loading
- Pre-download models in init containers
- Use persistent volumes for model cache
- Consider model quantization (4-bit, 8-bit)

### Request Handling
- Enable batch processing for multiple images
- Use request queuing for high load
- Implement caching for repeated requests

### GPU Utilization
- Monitor GPU memory usage
- Set appropriate resource limits
- Use dynamic batching when possible

## Troubleshooting

### Common Issues

**GPU not detected:**
```bash
# Check NVIDIA drivers
nvidia-smi

# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check Kubernetes GPU resources
kubectl describe nodes | grep nvidia.com/gpu
```

**Model loading timeout:**
- Increase `initialDelaySeconds` in probes
- Pre-download models in init container
- Use faster storage (SSD, NVMe)

**Out of memory:**
- Reduce batch size
- Use smaller model (2B instead of 8B)
- Enable model quantization
- Increase GPU memory limit

## Support and Resources

For more information, refer to:
- [Main README](../README.md)
- [API Deployment Research](../API_DEPLOYMENT_RESEARCH.md)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Docker Documentation](https://docs.docker.com/)

## License

This project uses the Qwen models under Apache 2.0 license.
