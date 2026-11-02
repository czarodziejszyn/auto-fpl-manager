import pandas as pd
import logging
import joblib
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelPredictor:
    def __init__(self):
        self.inference_path = Path("data/processed/features_inference.parquet")
        self.model_path = Path("models/lgbm_fpl_model.joblib")

    def predict_current_gw(self) -> pd.DataFrame:
        logger.info("Loading inference data and model artifacts...")
        if not self.inference_path.exists():
            logger.error("Missing data or model artifact.")
            return pd.DataFrame()

        df = pd.read_parquet(self.inference_path)
        model = joblib.load(self.model_path)

        features = [
            "now_cost",
            "minutes_rolling_3",
            "expected_goal_involvements_rolling_3",
            "ict_index_rolling_3",
        ]

        logger.info("Generating xP (expected points) predictions...")
        df["expected_points"] = model.predict(df[features])

        df["name"] = df["first_name"] + " " + df["second_name"]
        df = df.rename(columns={"element_type": "position", "now_cost": "cost"})

        df_optimizer = df[
            ["player_id", "name", "team", "position", "cost", "expected_points"]
        ]

        df_optimizer.loc[:, "expected_points"] = df_optimizer["expected_points"].clip(
            lower=0.0
        )

        return df_optimizer
