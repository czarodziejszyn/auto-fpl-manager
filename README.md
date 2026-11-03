# Auto-FPL Manager

An automated MLOps pipeline for predicting Fantasy Premier League (FPL) player performance and optimizing squad selection using Machine Learning and Mixed-Integer Linear Programming (MILP).

## Architecture

1. **ETL:** Automated extraction of live FPL API data into Parquet format.
2. **Feature Engineering:** Time-series processing with shifted rolling windows to prevent data leakage.
3. **Machine Learning:** LightGBM regressor predicting expected points (xP).
4. **Optimization:** Google OR-Tools (SCIP) solver selecting the optimal squad under budget and formation constraints.
5. **CI/CD:** GitHub Actions workflow for weekly autonomous execution and continuous training.

## Requirements

* Docker
* Docker Compose

## Usage

To run the full pipeline in an isolated container:

```bash
docker compose up --build
```
The latest generated squad can be found in the `reports/latest_squad.md` file.

## Known Limitations & Future Roadmap

The current version serves as a Minimum Viable Product (MVP) and solves a static Knapsack Problem. Planned architectural upgrades include:

* **Multi-Period Optimization:** Transitioning from single-gameweek optimization to a multi-gameweek horizon, factoring in free transfers and point hits.
* **Starting XI vs Bench:** Optimizing the active 11 players while minimizing the cost of the 4 bench players, rather than maximizing the total points of all 15.
* **Captaincy & Chips:** Adding objective function multipliers for captains and logic for chips (Wildcard, Bench Boost).
* **Advanced xP Modeling:** Integrating Fixture Difficulty Rating (FDR) and opponent defensive statistics into the LightGBM feature set.
