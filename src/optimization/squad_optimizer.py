import pandas as pd
import pulp
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SquadOptimizer:
    def __init__(self, df_players: pd.DataFrame, budget: float = 100.0):
        """
        df_players must have columns:
        ['id', 'name', 'team', 'position', 'cost', 'expected_points']
        """
        self.df = df_players.reset_index(drop=True)
        self.budget = budget

    def optimize(self) -> pd.DataFrame:
        logger.info(
            "Initializing Mixed-Integer Linear Programming (MILP) solver...")

        prob = pulp.LpProblem("AutoFPL_Optimization", pulp.LpMaximize)

        player_vars = {}
        for i in self.df.index:
            player_vars[i] = prob.add_variable(
                f"player_{i}", cat=pulp.LpBinary)

        prob += pulp.lpSum(
            [self.df.loc[i, "expected_points"] * player_vars[i]
                for i in self.df.index]
        )

        prob += (
            pulp.lpSum([self.df.loc[i, "cost"] * player_vars[i]
                       for i in self.df.index])
            <= self.budget
        )

        prob += pulp.lpSum([player_vars[i] for i in self.df.index]) == 15

        prob += (
            pulp.lpSum(
                [
                    player_vars[i]
                    for i in self.df.index
                    if self.df.loc[i, "position"] == 1
                ]
            )
            == 2
        )
        prob += (
            pulp.lpSum(
                [
                    player_vars[i]
                    for i in self.df.index
                    if self.df.loc[i, "position"] == 1
                ]
            )
            == 5
        )
        prob += (
            pulp.lpSum(
                [
                    player_vars[i]
                    for i in self.df.index
                    if self.df.loc[i, "position"] == 1
                ]
            )
            == 5
        )
        prob += (
            pulp.lpSum(
                [
                    player_vars[i]
                    for i in self.df.index
                    if self.df.loc[i, "position"] == 1
                ]
            )
            == 3
        )

        teams = self.df["team"].unique()
        for team_id in teams:
            prob += (
                pulp.lpSum(
                    [
                        player_vars[i]
                        for i in self.df.index
                        if self.df.loc[i, "team"] == team_id
                    ]
                )
                <= 3
            )

        logger.info("Solving the optimization problem...")
        prob.solve()

        if pulp.LpStatus[prob.status] != "Optimal":
            logger.error("Solver could not find an optimal solution!")
            return pd.DataFrame()

        selected_indices = [
            i for i in self.df.index if player_vars[i].varValue == 1.0]
        squad_df = self.df.loc[selected_indices].sort_values(
            by=["position", "expected_points"], ascending=[True, False]
        )

        total_cost = squad_df["cost"].sum()
        total_xp = squad_df["expected_points"].sum()
        logger.info(f"Optimal squad generated. Total cost: {
            total_cost}M, total expected points: {total_xp:.2f}")

        return squad_df


if __name__ == "__main__":
    logger.info("Running standalone test with mock data...")
    mock_data = {
        "id": range(1, 21),
        "name": [f"Player_{i}" for i in range(1, 21)],
        "team": [1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10],
        "position": [1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 1, 2, 3, 4, 4],
        "cost": [
            4.0,
            5.0,
            6.0,
            4.5,
            5.5,
            7.0,
            4.0,
            10.0,
            8.0,
            6.5,
            5.0,
            4.5,
            12.0,
            8.0,
            6.0,
            4.5,
            5.0,
            7.0,
            6.5,
            5.5,
        ],
        "expected_points": [
            3.1,
            4.2,
            5.0,
            3.5,
            4.8,
            6.0,
            2.5,
            8.5,
            7.0,
            5.2,
            4.0,
            3.8,
            9.5,
            6.5,
            5.0,
            3.6,
            4.1,
            5.5,
            4.8,
            4.2,
        ],
    }
    df_mock = pd.DataFrame(mock_data)

    optimizer = SquadOptimizer(df_mock, budget=100.0)
    best_squad = optimizer.optimize()
    print("\n=== OPTIMAL SQUAD ===")
    print(best_squad[["name", "position", "cost", "expected_points"]])
