import logging
from src.data.etl_pipeline import FPLExtractor
from src.data.feature_engineering import FeatureEngineer
from src.models.train_model import ModelTrainer
from src.models.predict_model import ModelPredictor
from src.optimization.squad_optimizer import SquadOptimizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_pipeline():
    logger.info("=== STARTING AUTO-FPL MANAGER PIPELINE ===")

    extractor = FPLExtractor()
    player_ids = extractor.extract_and_save_players()
    if player_ids:
        extractor.extract_gameweek_history(player_ids)

    fe = FeatureEngineer()
    fe.create_features()

    trainer = ModelTrainer()
    trainer.train()

    predictor = ModelPredictor()
    df_predictions = predictor.predict_current_gw()

    if df_predictions.empty:
        logger.error("Pipeline stopped. No predictions generated.")
        return

    optimizer = SquadOptimizer(df_predictions, budget=100.0)
    best_squad = optimizer.optimize()

    if not best_squad.empty:
        logger.info("\n=== OPTIMAL SQAD FOR NEXT GAMEWEEK ===")
        print(
            best_squad[["name", "position", "cost", "expected_points"]].to_string(
                index=False
            )
        )

        total_cost = best_squad["cost"].sum()
        total_xp = best_squad["expected_points"].sum()
        print("-" * 50)
        print(f"Total budget used: {total_cost}M / 100.0M")
        print(f"Total expected points (xP): {total_xp:.2f}")

    logger.info("=== PIPELINE FINISHED ===")


if __name__ == "__main__":
    run_pipeline()
