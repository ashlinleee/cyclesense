# CycleSense Deployment Guide

This guide covers the complete deployment setup for CycleSense using GitHub Actions, Docker, and Render.

## Architecture

```
┌─────────────────┐
│   GitHub Actions │
│   (CI/CD Pipeline)│
└────────┬────────┘
         │
         ├─────────────────────────────┐
         │                             │
         ▼                             ▼
┌─────────────────┐          ┌─────────────────┐
│  Backend API    │          │  Frontend UI    │
│  (FastAPI)      │◄─────────│  (Streamlit)    │
│  Port: 8000     │  API     │  Port: 8501     │
└─────────────────┘  Calls   └─────────────────┘
         │
         ▼
┌─────────────────┐
│   ML Model      │
│   Artifacts     │
└─────────────────┘
```

## Prerequisites

### 1. GitHub Secrets Configuration

Add the following secrets to your GitHub repository (Settings → Secrets and variables → Actions):

```
RENDER_API_KEY=your_render_api_key
RENDER_BACKEND_SERVICE_ID=your_backend_service_id  
RENDER_FRONTEND_SERVICE_ID=your_frontend_service_id
RENDER_BACKEND_WEBHOOK=your_backend_webhook_url
RENDER_FRONTEND_WEBHOOK=your_frontend_webhook_url
API_URL=https://cyclesense-api.onrender.com
FRONTEND_URL=https://cyclesense-frontend.onrender.com
```

### 2. Render Setup

1. Create two web services on Render:
   - **Backend Service**: `cyclesense-api` (FastAPI)
   - **Frontend Service**: `cyclesense-frontend` (Streamlit)

2. Configure each service:
   - Connect GitHub repository
   - Set build context to `cyclesense/`
   - Use respective Dockerfiles
   - Configure environment variables

3. Get service IDs and webhooks from Render dashboard

## Local Development

### Using Docker Compose

```bash
# Build and start all services
docker-compose up --build

# Access services
# Backend: http://localhost:8000
# Frontend: http://localhost:8501
# API Docs: http://localhost:8000/docs
```

### Individual Services

```bash
# Backend only
cd cyclesense
docker build -t cyclesense-backend -f Dockerfile .
docker run -p 8000:8000 cyclesense-backend

# Frontend only
cd cyclesense  
docker build -t cyclesense-frontend -f Dockerfile.streamlit --build-arg API_URL=http://localhost:8000 .
docker run -p 8501:8501 -e API_URL=http://localhost:8000 cyclesense-frontend
```

## CI/CD Pipeline

### Workflow Stages

1. **Test**: Run unit tests and data validation
2. **Build Backend**: Build and push Docker image to GitHub Container Registry
3. **Build Frontend**: Build and push Docker image to GitHub Container Registry  
4. **Deploy Backend**: Deploy backend to Render using webhook
5. **Deploy Frontend**: Deploy frontend to Render using webhook
6. **Health Check**: Verify both services are operational

### Pipeline Triggers

- **Push to main**: Full deployment pipeline
- **Push to develop**: Test and build only
- **Pull requests**: Test only

## Service Connection

### Frontend-Backend Communication

The Streamlit frontend connects to the FastAPI backend via:

```python
# In streamlit_app.py
API_URL = os.getenv("API_URL", "http://localhost:8000")
```

### Environment Variables

**Backend (render.yaml):**
```yaml
envVars:
  - key: API_URL
    value: https://cyclesense-api.onrender.com
```

**Frontend (render.yaml):**
```yaml
envVars:
  - key: API_URL
    value: https://cyclesense-api.onrender.com
```

## Monitoring

### Health Checks

**Backend:**
```bash
curl https://cyclesense-api.onrender.com/health
```

**Frontend:**
```bash
curl https://cyclesense-frontend.onrender.com/_stcore/health
```

### API Endpoints

- `GET /health` - Health check
- `GET /model-info` - Model metadata
- `POST /predict` - Make predictions
- `GET /docs` - API documentation

## Troubleshooting

### Common Issues

1. **Frontend can't connect to backend**
   - Check API_URL environment variable
   - Verify backend service is running
   - Check CORS configuration

2. **Docker build failures**
   - Check Dockerfile syntax
   - Verify requirements.txt is present
   - Check build context path

3. **Deployment failures**
   - Verify GitHub secrets are set correctly
   - Check Render service webhooks
   - Review GitHub Actions logs

### Debug Mode

Enable debug logging by setting:

```bash
export DEBUG=true
docker-compose up
```

## Scaling

### Backend Scaling

- Increase `numInstances` in render.yaml
- Add load balancing for multiple instances
- Consider database connection pooling

### Frontend Scaling

- Streamlit is stateful, so each user needs their own instance
- Use Render's scaling options based on traffic
- Consider session management for production

## Security

### Best Practices

1. Never commit secrets to repository
2. Use GitHub Secrets for sensitive data
3. Enable HTTPS in production
4. Implement rate limiting on API
5. Add authentication for production use

### CORS Configuration

The FastAPI backend includes CORS middleware:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Backup and Recovery

### Data Backup

- Model artifacts are stored in `artifacts/` directory
- Use Render's disk persistence for data files
- Implement regular backup cycles

### Rollback

```bash
# Rollback to previous deployment
render rollback cyclesense-api
render rollback cyclesense-frontend
```

## Cost Optimization

### Render Free Tier

- Both services fit in free tier limits
- Monitor usage to avoid overages
- Implement caching to reduce API calls

### GitHub Actions

- Use caching for Docker builds
- Optimize test execution time
- Limit workflow runs to necessary branches

## Support

For issues or questions:
1. Check GitHub Actions logs
2. Review Render service logs
3. Check Docker container logs
4. Verify environment variables
5. Test services locally first
