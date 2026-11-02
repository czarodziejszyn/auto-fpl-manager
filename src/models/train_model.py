import pandas as pd
import logging
import joblib
from pathlib import Path
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self):
        self.processed_dir = Path("data/processed")
        self.models_dir = Path("models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.features_path = self.processed_dir / "features_train.parquet"

    def train(self):
        logger.info("Loading processed features...")
        if not self.features_path.exists():
            logger.error(
                "Features not found. Run feature_engineering.py first.")
            return

        df = pd.read_parquet(self.features_path)

        features = [
            "now_cost",
            "minutes_rolling_3",
            "expected_goal_involvements_rolling_3",
            "ict_index_rolling_3",
        ]
        target = "target_points_next_gw"

        logger.info("Splitting data chronologically (time-series split)...")
        split_idx = int(len(df) * 0.8)

        X_train = df[features].iloc[:split_idx]
        y_train = df[target].iloc[:split_idx]
        X_test = df[features].iloc[split_idx:]
        y_test = df[target].iloc[split_idx:]

        logger.info("Training LightGBM on {len(X_train)} samples...")
        model = LGBMRegressor(
            n_estimators=100, learning_rate=0.05, random_state=42, max_depth=5
        )
        model.fit(X_train, y_train)

        logger.info("Evaluating model...")
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)

        logger.info(f"Model evaluation -> mean absolute error (MAE): {
            mae:.2f} points per player")

        model_path = self.models_dir / "lgbm_fpl_model.joblib"
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()
