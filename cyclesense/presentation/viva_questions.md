# CycleSense Viva Questions

**30+ likely questions with concise answers**

---

## Dataset & Problem

**Q1: Why this dataset?**
A: Educational dataset with realistic menstrual health data - 2,000 user profiles and 17,976 cycle logs. Demonstrates joining relational data, handling temporal sequences, and working with mixed feature types (numerical, categorical, time-series).

**Q2: Why regression?**
A: Predicting next cycle length is a continuous numerical outcome (days). Regression is appropriate for predicting quantities. Classification would require arbitrary binning of cycle lengths.

**Q3: Why predict next cycle length?**
A: Practical prediction problem - people want to know when their next cycle will start. Uses historical patterns to forecast future behavior. More actionable than current cycle reconstruction.

**Q4: Why MAE?**
A: MAE expresses error directly in days - intuitive interpretation. "MAE of 1.7 days" means predictions differ from observed next-cycle length by 1.7 days on average. More stakeholder-friendly than RMSE or R².

**Q5: Why not accuracy?**
A: Accuracy is for classification. This is regression - we predict continuous values (days), not classes. MAE/RMSE/R² are appropriate regression metrics.

---

## Data Splitting

**Q6: Why not random train/test split?**
A: Multiple cycles belong to the same user. Random splitting would leak information - a user's cycles could appear in both train and test, inflating performance. We use GroupShuffleSplit to keep all cycles of a user together.

**Q7: What is data leakage?**
A: Using information not available at prediction time. Example: using "prepared_before_period" (response to current cycle) to predict the next cycle. This creates artificially good performance that doesn't generalize.

**Q8: Why GroupKFold/GroupShuffleSplit?**
A: Prevents user-level data leakage. Ensures all cycles of a user are in either train or test, not both. More realistic evaluation where model must predict for unseen users.

**Q9: Why chronological ordering within groups?**
A: Temporal integrity - past cycles should predict future cycles, not vice versa. We sort by user_id and start_date before creating lag/rolling features.

---

## Feature Engineering

**Q10: Why rolling features?**
A: Capture individual patterns over time. Rolling mean shows typical cycle length, rolling std shows variability. More informative than single historical values.

**Q11: Why three-cycle rolling average?**
A: Balance between capturing recent patterns and having enough data. Three cycles is a reasonable window for menstrual cycles (approx. 3 months). Tested empirically.

**Q12: Why median imputation?**
A: Robust to outliers compared to mean. Cycle length data may have extreme values (medical conditions, irregularities). Median provides central tendency without skew.

**Q13: Why one-hot encoding?**
A: Categorical variables (diet_quality, exercise_frequency) have no inherent order. One-hot creates binary features without imposing arbitrary numerical relationships. Handles unknown categories gracefully.

**Q14: Why scaling?**
A: Linear models (Ridge, Lasso) optimize coefficients sensitive to feature scale. StandardScaler normalizes features to same scale. Tree models (RF, GB) don't need scaling - split decisions are scale-invariant.

**Q15: Why doesn't Random Forest need scaling?**
A: Tree splits find optimal thresholds for each feature independently. Feature scale doesn't affect where splits occur. Different from distance-based models (KNN) or coefficient-based models (linear regression).

**Q16: Why use Pipeline?**
A: Ensures training and production use identical preprocessing. Prevents training-production divergence. Reproducible feature transformations. Best practice for ML systems.

**Q17: Why ColumnTransformer?**
A. Apply different preprocessing to different feature types. Numerical features get imputation + scaling, categorical get imputation + one-hot. Clean separation of concerns.

---

## MLOps

**Q18: Why DVC instead of Git?**
A: Git versions code, DVC versions large datasets. Git would bloat repository with CSV files. DVC tracks which exact dataset version produced a training pipeline. Enables data reproducibility.

**Q19: Why MLflow?**
A. Centralized experiment tracking. Logs hyperparameters, metrics, artifacts. Model comparison and selection. Model registry for versioning. Standard tool for ML experiment management.

**Q20: Why Model Registry?**
A. Version control for models. Promotion workflow (staging → production). Quality gates before deployment. Audit trail of model deployments. Production ML best practice.

**Q21: Why FastAPI?**
A. Production-grade async web framework. Automatic API documentation (Swagger). Type validation with Pydantic. Fast performance. Industry standard for ML APIs.

**Q22: Why Streamlit?**
A. Rapid UI development for ML apps. Built-in widgets and visualizations. Python-only - no frontend expertise needed. Perfect for ML demos and dashboards.

**Q23: Why shouldn't Streamlit load model.joblib directly?**
A. Separation of concerns. Streamlit should call API, not load model. API handles ML inference, UI handles visualization. Enables scaling, API reuse, proper architecture.

**Q24: Why Docker?**
A. Consistent environment across dev/prod. Isolates dependencies. Simplifies deployment. Reproducible builds. Standard for containerized applications.

**Q25: Why GitHub Actions?**
A. Automated CI/CD. Test on every push. Build Docker images automatically. Push to Docker Hub. Free for public repositories. Integrates with GitHub ecosystem.

**Q26: Why Docker Hub?**
A. Public container registry. Easy distribution of Docker images. Free for public repositories. Standard registry for Docker containers. Enables pull-and-run deployment.

---

## Monitoring & Drift

**Q27: What is data drift?**
A. Change in statistical properties of input data over time. Example: user demographics shift, measurement practices change. Doesn't automatically mean model failure, but signals need for investigation.

**Q28: What is concept drift?**
A. Change in relationship between features and target over time. Example: new medical treatment changes cycle patterns. Model performance degrades even if data distribution stable.

**Q29: Why KS-test?**
A. Kolmogorov-Smirnov test compares two distributions. Quantifies if reference and current data come from same distribution. Statistical basis for drift detection. Standard for numerical feature drift.

**Q30: Why use SHAP/permutation importance?**
A. Global model explanation. Permutation importance measures feature impact on performance. SHAP explains individual predictions. Both provide model interpretability - critical for healthcare applications.

**Q31: Why didn't you use PCA?**
A. Evaluated PCA but rejected it. Dimensionality reduction reduced interpretability. Performance improvement insufficient to justify complexity. Stakeholders need interpretable features, not abstract components.

---

## Project Specifics

**Q32: What are the project's limitations?**
A: Educational dataset (not real patient data). Not medical advice. Simplified features. No real-time updates. Limited external factors. Prototype, not production system.

**Q33: How would you improve with real-world data?**
A: More sophisticated features (medications, health conditions). Uncertainty estimation (prediction intervals). Real-time model updates. User feedback loop. Integration with health data sources. Clinical validation.

**Q34: What's the best model and why?**
A: Gradient Boosting (1.70 MAE). Best performance among tested models. Handles non-linear relationships. Robust to outliers. Good with mixed feature types. Reasonable training time.

**Q35: What features are most important?**
A: PCOS diagnosis (strongest predictor), rolling means (historical patterns), BMI, birth control use. These make clinical sense - medical conditions and history strongly influence cycle patterns.

**Q36: How did you handle missing values?**
A: Structural missingness (first cycles have no previous length) left as NaN. Others imputed with median (numerical) or most frequent (categorical). Documented all decisions in data profile.

**Q37: What's the improvement over baseline?**
A: 26% improvement over previous cycle baseline (2.30 → 1.70 MAE). 10% improvement over global mean baseline (1.94 → 1.70 MAE). Demonstrates ML adds value over simple heuristics.

---

## Engineering Decisions

**Q38: Why this technology stack?**
A: Python (ML ecosystem), scikit-learn (modeling), pandas (data), MLflow (tracking), DVC (data), FastAPI (API), Streamlit (UI), Docker (deployment). Standard, well-supported tools for ML engineering.

**Q39: How do you ensure reproducibility?**
A: Random seeds set. DVC versions data. MLflow tracks experiments. Docker packages environment. Parameterized pipeline. All steps scripted and version-controlled.

**Q40: What would you do differently with more time?**
A. Add uncertainty estimation. Implement more sophisticated features. Add real-time monitoring. Deploy to cloud (AWS/GCP). Add user authentication. Conduct user testing. Integrate with health data APIs.

---

## Statistical Concepts

**Q41: Explain R².**
A. Coefficient of determination. Proportion of variance in target explained by features. R² of 0.306 means model explains 30.6% of variance in next cycle length. Not perfect, but better than baseline.

**Q42: What does negative R² mean?**
A. Model performs worse than predicting the mean. Indicates model is not learning useful patterns. Happened with Lasso when it eliminated all features (regularization too strong).

**Q43: Why 80-20 train-test split?**
A. Standard practice. 80% for training (enough data to learn patterns), 20% for testing (unbiased evaluation). Could be adjusted based on dataset size.

**Q44: How do you handle class imbalance?**
A. Not applicable - this is regression, not classification. For classification would use stratified sampling, class weights, or resampling techniques.

---

## Business Context

**Q45: How would you deploy this in production?**
A. Containerize with Docker. Deploy to cloud (AWS EC2/GCP Compute). Use load balancer for scaling. Set up monitoring (Prometheus/Grafana). Implement logging and alerting. Use blue-green deployment for updates.

**Q46: How would you monitor model performance?**
A. Track prediction metrics (MAE, latency). Monitor data drift with statistical tests. Set up alerts for performance degradation. Log prediction distribution. Regular evaluation on held-out test set.

**Q47: How would you handle model retraining?**
A. Automated retraining pipeline. Quality gates before promotion. A/B testing with champion model. Schedule based on drift detection or time intervals. Manual approval for production deployment.

**Q48: What are the ethical considerations?**
A. Medical disclaimer critical. Not diagnostic - educational only. Privacy of health data. Bias in training data. Transparency about limitations. User consent for data use.

---

## Quick Response Template

**If you don't know the answer:**
1. Acknowledge the question
2. Relate to what you do know
3. Explain how you would find out
4. Propose a reasonable approach

**Example:** "I haven't implemented that specific feature, but based on the project architecture, I would approach it by [reasonable approach]. In a production setting, I would research best practices and likely use [standard tool/method]."
