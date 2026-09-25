"""
MLflow Model Registry and Promotion Workflow for CycleSense.
Implements model registration, versioning, and promotion based on quality criteria.
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import logging
import json
from pathlib import Path

from src.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, MODEL_REGISTRY_NAME, ARTIFACTS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelPromoter:
    """Handle model registration and promotion in MLflow."""
    
    def __init__(self):
        self.client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        self.model_name = MODEL_REGISTRY_NAME
    
    def create_registered_model(self):
        """Create registered model if it doesn't exist."""
        try:
            self.client.create_registered_model(self.model_name)
            logger.info(f"Created registered model: {self.model_name}")
        except Exception as e:
            if "already exists" in str(e):
                logger.info(f"Registered model {self.model_name} already exists")
            else:
                raise e
    
    def get_best_run(self, experiment_name=MLFLOW_EXPERIMENT_NAME):
        """Get the best run from the experiment based on MAE."""
        experiment = self.client.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            raise ValueError(f"Experiment {experiment_name} not found")
        
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.mae ASC"]
        )
        
        if not runs:
            raise ValueError("No runs found in experiment")
        
        best_run = runs[0]
        logger.info(f"Best run: {best_run.info.run_id}")
        logger.info(f"Best MAE: {best_run.data.metrics.get('mae', 'N/A')}")
        
        return best_run
    
    def register_model(self, run_id, model_name="model"):
        """Register a model from a run."""
        model_uri = f"runs:/{run_id}/{model_name}"
        
        # Create model version
        model_version = self.client.create_model_version(
            name=self.model_name,
            source=model_uri,
            run_id=run_id
        )
        
        logger.info(f"Registered model version: {model_version.version}")
        return model_version
    
    def promote_to_staging(self, version):
        """Promote a model version to staging stage."""
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=version,
            stage="Staging",
            archive_existing_versions=False
        )
        logger.info(f"Promoted model version {version} to staging stage")
    
    def promote_to_production(self, version):
        """Promote a model version to production stage."""
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=version,
            stage="Production",
            archive_existing_versions=False
        )
        logger.info(f"Promoted model version {version} to production stage")
    
    def get_production_model(self):
        """Get the current production model."""
        model_versions = self.client.get_latest_versions(
            self.model_name,
            stages=["Production"]
        )
        
        if model_versions:
            production = model_versions[0]
            logger.info(f"Current production model: version {production.version}")
            return production
        else:
            logger.info("No production model found")
            return None
    
    def compare_staging_vs_production(self, staging_version):
        """Compare staging model with current production."""
        production = self.get_production_model()
        
        if production is None:
            logger.info("No production model to compare with. Staging will become production.")
            return True
        
        # Get run info for both models
        staging_run = self.client.get_model_version(self.model_name, staging_version)
        production_run = self.client.get_model_version(self.model_name, production.version)
        
        staging_metrics = self.client.get_run(staging_run.run_id).data.metrics
        production_metrics = self.client.get_run(production_run.run_id).data.metrics
        
        logger.info(f"Staging MAE: {staging_metrics.get('mae', 'N/A')}")
        logger.info(f"Production MAE: {production_metrics.get('mae', 'N/A')}")
        
        # Compare based on MAE (lower is better)
        staging_mae = staging_metrics.get('mae', float('inf'))
        production_mae = production_metrics.get('mae', float('inf'))
        
        improvement = (production_mae - staging_mae) / production_mae * 100
        logger.info(f"Improvement: {improvement:+.2f}%")
        
        # Promote if staging is better or within 1% threshold
        if staging_mae <= production_mae * 1.01:
            logger.info("Staging model is better or equivalent. Promotion recommended.")
            return True
        else:
            logger.info("Staging model is worse. Promotion not recommended.")
            return False


def quality_gate_check(run_info):
    """
    Check if a model meets quality criteria for promotion.
    
    Args:
        run_info: MLflow run info
        
    Returns:
        bool: True if model passes quality gate
    """
    metrics = run_info.data.metrics
    
    # Quality criteria
    mae_threshold = 2.5  # Maximum acceptable MAE (days)
    r2_threshold = 0.2   # Minimum acceptable R²
    
    mae = metrics.get('mae', float('inf'))
    r2 = metrics.get('r2', -float('inf'))
    
    logger.info(f"Quality Gate Check:")
    logger.info(f"  MAE: {mae:.3f} (threshold: {mae_threshold})")
    logger.info(f"  R²: {r2:.3f} (threshold: {r2_threshold})")
    
    if mae > mae_threshold:
        logger.warning(f"MAE {mae:.3f} exceeds threshold {mae_threshold}")
        return False
    
    if r2 < r2_threshold:
        logger.warning(f"R² {r2:.3f} below threshold {r2_threshold}")
        return False
    
    logger.info("Quality gate passed")
    return True


def auto_promote_workflow():
    """
    Automatic promotion workflow:
    1. Get best run from experiment
    2. Check quality gate
    3. Register as staging
    4. Compare with production
    5. Promote to production if better
    """
    logger.info("=== AUTO PROMOTION WORKFLOW ===")
    
    promoter = ModelPromoter()
    
    # Create registered model if needed
    promoter.create_registered_model()
    
    # Get best run
    best_run = promoter.get_best_run()
    
    # Check quality gate
    if not quality_gate_check(best_run):
        logger.error("Model failed quality gate. Promotion aborted.")
        return False
    
    # Register model
    model_version = promoter.register_model(best_run.info.run_id)
    
    # Promote to staging
    promoter.promote_to_staging(model_version.version)
    
    # Compare with production and promote if better
    if promoter.compare_staging_vs_production(model_version.version):
        promoter.promote_to_production(model_version.version)
        logger.info("Model promoted to production!")
        return True
    else:
        logger.info("Model remains in staging (not better than production)")
        return False


def manual_promote_workflow(run_id):
    """
    Manual promotion workflow for a specific run.
    
    Args:
        run_id: MLflow run ID to promote
    """
    logger.info(f"=== MANUAL PROMOTION WORKFLOW FOR RUN {run_id} ===")
    
    promoter = ModelPromoter()
    
    # Create registered model if needed
    promoter.create_registered_model()
    
    # Get run info
    run = promoter.client.get_run(run_id)
    
    # Check quality gate
    if not quality_gate_check(run):
        logger.error("Model failed quality gate. Promotion aborted.")
        return False
    
    # Register model
    model_version = promoter.register_model(run_id)
    
    # Promote to staging
    promoter.promote_to_staging(model_version.version)
    
    # Promote to production (manual override)
    promoter.promote_to_production(model_version.version)
    logger.info("Model manually promoted to production!")
    
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Manual promotion with run ID
        run_id = sys.argv[1]
        manual_promote_workflow(run_id)
    else:
        # Auto promotion workflow
        auto_promote_workflow()
