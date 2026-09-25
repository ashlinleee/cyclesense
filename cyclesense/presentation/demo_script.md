# CycleSense Demo Script

**Duration: 5-7 minutes**

---

## Slide 1: Problem (30 seconds)

**Presenter:** "Many people track their menstrual cycles, but simple fixed-calendar assumptions don't capture individual patterns. Cycle lengths vary based on lifestyle, stress, health factors, and personal history."

**Key Points:**
- Cycle lengths vary individually
- Fixed calendars don't work for everyone
- Need personalized pattern recognition

---

## Slide 2: Product Introduction (30 seconds)

**Presenter:** "Introducing CycleSense - an explainable machine-learning system that learns from user profiles and cycle history to estimate next cycle length and analyze patterns."

**Key Points:**
- Educational ML prototype
- Uses profile + historical data
- Predicts next cycle length
- Analyzes influencing patterns

---

## Slide 3: Dataset (45 seconds)

**Presenter:** "We have two datasets: 2,000 user profiles with demographic and lifestyle information, and 17,976 cycle logs with detailed measurements. The datasets are joined by user_id to create a comprehensive feature set."

**Key Points:**
- 2,000 user profiles (demographics, lifestyle)
- 17,976 cycle logs (detailed measurements)
- Joined by user_id
- 34 total features after integration

---

## Slide 4: Feature Engineering (1 minute)

**Presenter:** "The most important part is leakage-safe feature engineering. We create historical features like lag features, rolling statistics, and cycle changes. We also engineer cross-source features like stress deltas and sleep deltas, and date features for seasonal patterns."

**Key Points:**
- **Historical Features:** Lag features (1,2,3), rolling mean/std (3,5 cycles), cycle changes
- **Profile Features:** Age, BMI, lifestyle factors (always available)
- **Cross-Source:** Stress delta, sleep delta, interaction terms
- **Date Features:** Month, quarter, cyclical representations

**Why These?**
- Only information available at prediction time
- Captures individual patterns
- No future data leakage

---

## Slide 5: Leakage Prevention (45 seconds)

**Presenter:** "Data leakage is critical. We categorize features into SAFE (available at prediction time), CONDITIONAL (timing-dependent), and EXCLUDED (not available). We use a strict feature set for production - only information realistically available when making predictions."

**Key Points:**
- **SAFE:** 14 features (profile, current cycle)
- **CONDITIONAL:** 10 features (depends on timing)
- **EXCLUDED:** 6 features (leakage risk)
- **Strict vs Extended feature sets**
- Never use future cycles to predict past

---

## Slide 6: Experiments (1 minute)

**Presenter:** "We ran multiple experiments tracked with MLflow. Baselines: global mean (1.94 days MAE) and previous cycle (2.30 days MAE). ML models: Linear, Ridge, Random Forest, Gradient Boosting. Gradient Boosting performed best with 1.70 days MAE - 26% improvement over previous cycle baseline."

**Key Points:**
- **Baselines:** Mean (1.94 MAE), Previous cycle (2.30 MAE)
- **ML Models:** Linear (1.74), Ridge (1.74), RF (1.73), GB (1.70)
- **Best Model:** Gradient Boosting (1.70 MAE)
- **Improvement:** +26% over previous cycle baseline
- All tracked in MLflow

---

## Slide 7: Product Demo (2 minutes)

**Presenter:** "Let me show you the live application. I'll enter user profile information: age 28, BMI 22.8, lifestyle factors. Current cycle: 28 days, previous cycles. Click Predict."

**Demo Steps:**
1. Enter profile: age 28, BMI 22.8, good diet, 5-6 days exercise
2. Enter current cycle: 28 days, moderate pain, normal stress
3. Enter historical cycles: 28, 29, 27, 28, 30
4. Click "Predict Next Cycle"
5. Show prediction: "28.4 days - ML Estimate"
6. Show model version and performance metrics
7. Show feature importance: PCOS diagnosis, rolling means, BMI

**Key Points:**
- User-friendly interface
- Real-time prediction
- Model explainability
- Medical disclaimer visible

---

## Slide 8: MLOps Pipeline (1 minute)

**Presenter:** "This isn't just a model - it's a complete MLOps system. DVC versions the data, MLflow tracks experiments and manages the model registry, FastAPI serves predictions, Streamlit provides the UI, Docker containers everything, and GitHub Actions automates CI/CD to Docker Hub."

**Key Points:**
- **DVC:** Data versioning and reproducibility
- **MLflow:** Experiment tracking, model registry
- **FastAPI:** Production inference service
- **Streamlit:** User interface
- **Docker:** Containerization
- **GitHub Actions:** Automated CI/CD to Docker Hub

---

## Slide 9: Closing (30 seconds)

**Presenter:** "CycleSense demonstrates how a raw dataset becomes a reproducible, versioned, explainable ML service. It's not medical advice - it's an educational demonstration of end-to-end machine learning engineering."

**Key Points:**
- Complete ML/MLOps pipeline
- Leakage-safe feature engineering
- Explainable predictions
- Production-ready architecture
- Educational demonstration

---

## Demo Backup (if internet/Docker Hub fails)

**Instructions:**
1. Start locally: `docker build -t cyclesense . && docker run -p 8000:8000 -p 8501:8501 cyclesense`
2. Use local Docker container for demo
3. Screenshots prepared as backup
4. Video recording of local demo available

---

## Key Questions to Anticipate

**Q: Why MAE instead of accuracy?**
A: MAE expresses error directly in days - intuitive for stakeholders. "MAE of 1.7 days" means predictions differ from observed by 1.7 days on average.

**Q: Why not random train/test split?**
A: Multiple cycles belong to same user. Random splitting would leak information. GroupShuffleSplit keeps all cycles of a user together.

**Q: What is data leakage?**
A: Using information not available at prediction time. We audit features and only use those realistically available when predicting the next cycle.

**Q: Why Gradient Boosting?**
A: Best performance (1.70 MAE), handles non-linear relationships, robust to outliers, good with mixed feature types.

**Q: Why separate API and UI?**
A: Separation of concerns. API handles ML inference, UI handles visualization. Production systems separate backend/frontend.

**Q: How would you improve with real data?**
A: More sophisticated features, external factors (medications, health conditions), uncertainty estimation, real-time updates, user feedback loop.

---

## Technical Details (if asked)

**Model Performance:**
- MAE: 1.70 days
- RMSE: 2.18 days  
- R²: 0.306
- Training time: 0.45s

**Top Features:**
1. PCOS diagnosis (0.250 importance)
2. Rolling mean last 5 cycles (0.126)
3. Rolling mean last 3 cycles (0.022)
4. BMI (0.013)
5. Birth control use (0.008)

**Data Quality:**
- 2,000 users, 17,976 cycles
- Mean cycles per user: 8.99
- Cycle length: 27.8 ± 2.4 days
- 11.13% missing prev_cycle_length (expected for first cycles)

**Architecture:**
- Python 3.9, scikit-learn, pandas
- MLflow for tracking, DVC for data
- FastAPI backend, Streamlit frontend
- Docker containerization
- GitHub Actions CI/CD
