import pandas as pd
import logging
from ortools.linear_solver import pywraplp

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

        solver = pywraplp.Solver.CreateSolver("SCIP")
        if not solver:
            logger.error("Could not SCIP solver.")
            return pd.DataFrame()

        player_vars = {}
        for i in self.df.index:
            player_vars[i] = solver.IntVar(0, 1, f"player_{i}")

        objective = solver.Objective()
        for i in self.df.index:
            objective.SetCoefficient(
                player_vars[i], float(self.df.loc[i, "expected_points"])
            )
        objective.SetMaximization()

        # Budget
        budget_constraint = solver.Constraint(
            0.0, float(self.budget), "Budget")
        for i in self.df.index:
            budget_constraint.SetCoefficient(
                player_vars[i], float(self.df.loc[i, "cost"])
            )

        # 15 players team
        total_players = solver.Constraint(15.0, 15.0, "Total_players")
        for i in self.df.index:
            total_players.SetCoefficient(player_vars[i], 1.0)

        # 2gk, 5def, 5mid, 3fwd
        req_positions = {1: 2, 2: 5, 3: 5, 4: 3}
        for pos, count in req_positions.items():
            pos_constraint = solver.Constraint(
                float(count), float(count), f"Position_{pos}"
            )
            for i in self.df.index:
                if self.df.loc[i, "position"] == pos:
                    pos_constraint.SetCoefficient(player_vars[i], 1.0)

        # 3 players per club
        teams = self.df["team"].unique()
        for team_id in teams:
            team_constraint = solver.Constraint(0.0, 3.0, f"Team_{team_id}")
            for i in self.df.index:
                if self.df.loc[i, "team"] == team_id:
                    team_constraint.SetCoefficient(player_vars[i], 1.0)

        logger.info("Solving the MILP problem...")
        status = solver.Solve()

        if status != pywraplp.Solver.OPTIMAL:
            logger.error(
                f"Solver failed to find an optimal solution! Status: {status}")
            return pd.DataFrame()

        selected_indices = [
            i for i in self.df.index if player_vars[i].solution_value() > 0.5
        ]
        squad_df = self.df.loc[selected_indices].sort_values(
            by=["position", "expected_points"], ascending=[True, False]
        )

        total_cost = squad_df["cost"].sum()
        total_xp = squad_df["expected_points"].sum()
        logger.info(f"Optimal squad generated. Cost: {
                    total_cost}M, xP: {total_xp:.2f}")

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
    # Bezpieczne wyświetlanie wyników, które chroni przed KeyError z pustej tabeli
    if not best_squad.empty:
        print(
            best_squad[["name", "position", "cost", "expected_points"]].to_string(
                index=False
            )
        )
    else:
        print("Failed to generate a valid squad.")
