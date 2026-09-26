# CycleSense

CycleSense is an explainable machine-learning system that learns from a user's profile and previous menstrual-cycle history to estimate their next cycle length and analyze patterns influencing that prediction.

## 🚀 Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/ashlinleee/cyclesense.git
cd cyclesense

# Using Docker Compose (recommended)
docker-compose up --build

# Access services
# Frontend: http://localhost:8501
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r cyclesense/requirements.txt

# Train the model
cd cyclesense
python src/train_fast.py

# Start backend
uvicorn api.main:app --reload --port 8000

# Start frontend (in another terminal)
streamlit run ui/streamlit_app.py --server.port 8501
```

## 📋 Features

- **Cycle Prediction**: ML-based next cycle length prediction
- **Pattern Analysis**: Historical cycle pattern insights
- **Model Explainability**: SHAP values and feature importance
- **Interactive Dashboard**: Streamlit-based user interface
- **REST API**: FastAPI backend for model serving

## 🏗️ Architecture

```
┌─────────────────┐
│   Streamlit UI  │
│   (Frontend)    │
└────────┬────────┘
         │ HTTP/REST
         ▼
┌─────────────────┐
│   FastAPI       │
│   (Backend)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ML Model      │
│   (Scikit-learn)│
└─────────────────┘
```

## 🧪 Testing

```bash
# Run tests
cd cyclesense
pytest tests/ -v

# Run data validation
python src/data.py

# Run feature engineering tests
python tests/test_features.py
```

## 🚢 Deployment

### Automated Deployment

The project uses GitHub Actions for CI/CD:

- **Test**: Runs on every push and PR
- **Build**: Creates Docker images and pushes to GitHub Container Registry
- **Deploy**: Deploys to Render web services
- **Health Check**: Verifies deployment success

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

### Manual Deployment

```bash
# Build backend image
docker build -t cyclesense-backend -f cyclesense/Dockerfile cyclesense/

# Build frontend image
docker build -t cyclesense-frontend -f cyclesense/Dockerfile.streamlit --build-arg API_URL=https://your-api.com cyclesense/

# Deploy to your preferred platform
```

## 📊 Model Performance

- **Algorithm**: Histogram-based Gradient Boosting
- **MAE**: ≈ 1.7 days
- **RMSE**: ≈ 2.18 days
- **R²**: ≈ 0.306

## ⚠️ Medical Disclaimer

CycleSense provides data-driven cycle estimates for educational demonstration and pattern exploration. Predictions are estimates and should not be used for diagnosis, contraception, fertility planning, or medical decisions.

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python 3.9
- **Frontend**: Streamlit
- **ML**: Scikit-learn, MLflow
- **Data**: Pandas, NumPy
- **Deployment**: Docker, GitHub Actions, Render
- **Testing**: Pytest

## 📁 Project Structure

```
cyclesense/
├── cyclesense/
│   ├── api/              # FastAPI backend
│   ├── src/              # ML pipeline code
│   ├── ui/               # Streamlit frontend
│   ├── data/             # Data files
│   ├── artifacts/        # Model artifacts
│   ├── tests/            # Test files
│   └── requirements.txt  # Python dependencies
├── .github/              # GitHub Actions workflows
├── docker-compose.yml    # Local development
├── DEPLOYMENT.md         # Deployment guide
└── README.md            # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📄 License

This project is for educational purposes. See LICENSE file for details.

## 🙏 Acknowledgments

- Built for educational demonstration of ML pipeline best practices
- Uses synthetic health data for learning purposes
- Inspired by real-world health-tech applications
