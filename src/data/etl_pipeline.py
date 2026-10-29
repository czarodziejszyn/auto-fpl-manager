import pandas as pd
import logging
from pathlib import Path
from src.data.api_client import FPLClient

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FPLExtractor:
    def __init__(self):
        self.client = FPLClient()
        self.raw_dir = Path("data/raw")
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def extract_and_save_players(self):
        logger.info("Starting ETL process for players...")
        raw_data = self.client.get_bootstrap_statistic()
        players_data = raw_data.get("elements", [])
        if not players_data:
            logger.error("No player data found!")
            return

        df = pd.DataFrame(players_data)
        logger.info(f"Loaded {len(df)} players into DataFrame.")

        columns_of_interest = [
            "id",
            "first_name",
            "second_name",
            "team",
            "element_type",
            "now_cost",
            "total_points",
            "minutes",
            "goals_scored",
            "assists",
            "clean_sheets",
            "influence",
            "creativity",
            "threat",
            "ixt_index",
        ]

        existing_cols = [
            col for col in columns_of_interest if col in df.columns]
        df = df[existing_cols]

        if "now_cost" in df.columns:
            df["now_cost"] = df["now_cost"] / 10.0

        output_path = self.raw_dir / "players_raw.parquet"
        df.to_parquet(output_path, engine="pyarrow", index=False)
        logger.info(f"Data saved succesfully to {output_path}")


if __name__ == "__main__":
    extractor = FPLExtractor()
    extractor.extract_and_save_players()
