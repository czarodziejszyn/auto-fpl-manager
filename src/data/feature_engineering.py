import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    def __init__(self):
        self.raw_dir = Path("data/raw")
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def create_features(self):
        logger.info("Loading raw data...")
        try:
            df_players = pd.read_parquet(self.raw_dir / "players_raw.parquet")
            df_gw = pd.read_parquet(self.raw_dir / "gameweeks_raw.parquet")
        except FileNotFoundError as e:
            logger.error(f"Raw data not found. Run ETL pipeline first. {e}")
            return

        cols_to_float = [
            "expected_goals",
            "expected_assists",
            "expected_goal_involvments",
            "ict_index",
        ]
        for col in cols_to_float:
            if col in df_gw.columns:
                df_gw[col] = df_gw[col].astype(float)

        logger.info(
            "Sorting data by player and fixture for time-series operations...")
        df_gw = df_gw.sort_values(by=["player_id", "round"])

        logger.info("Calculating rolling features (3-match window)...")

        rolling_cols = [
            "total_points",
            "minutes",
            "expected_goal_involvements",
            "ict_index",
        ]

        for col in rolling_cols:
            df_gw[f"{col}_rolling_3"] = (
                df_gw.groupby("player_id")[col]
                .shift(1)
                .rolling(window=3, min_periods=1)
                .mean()
            )

        logger.info("Creating target variable (next gw points)...")
        df_gw["target_points_next_gw"] = df_gw.groupby("player_id")[
            "total_points"
        ].shift(-1)

        df_model = df_gw.dropna(subset=["target_points_next_gw"])

        logger.info(
            "Merging with static player data (positions, costs, teams)...")

        df_final = df_model.merge(
            df_players[
                ["id", "first_name", "second_name",
                    "element_type", "now_cost", "team"]
            ],
            left_on="player_id",
            right_on="id",
            how="inner",
        ).drop(columns=["id"])

        output_path = self.processed_dir / "features.parquet"
        df_final.to_parquet(output_path, engine="pyarrow", index=False)
        logger.info(f"Feature dataset saved to {
            output_path} with {len(df_final)} rows.")


if __name__ == "__main__":
    fe = FeatureEngineer()
    fe.create_features()
