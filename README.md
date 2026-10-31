# AutoFPL-Manager

An end-to-end autonomous MLOps agent that predicts Fantasy Premier League (FPL) player performance and mathematically optimizes squad selection using Mixed-Integer Linear Programming (MILP).

## System Architecture

This project goes beyond basic data science tutorials by implementing a fully automated, production-ready pipeline:

1. **Data Extraction (ETL):** Robust FPL API client with exponential backoff (`tenacity`) and automated conversion of nested JSON payloads into columnar `Parquet` storage.
2. **Feature Engineering:** Time-series processing with shifted rolling windows to prevent data leakage.
3. **Machine Learning:** `LightGBM` regressor trained on chronological splits to predict expected points (xP) for the upcoming Gameweek.
4. **Operations Research:** A MILP optimization engine built with `PuLP` and the COIN-OR CBC solver. It selects the absolute best 15-man squad while respecting hard constraints (100.0M budget, max 3 players per team, valid formations).

## Tech Stack

* **Language:** Python 3.14
* **Package Management:** `uv`
* **Data Processing:** `pandas`, `pyarrow` (Parquet)
* **Machine Learning:** `LightGBM`, `scikit-learn`
* **Optimization:** `PuLP`, COIN-OR CBC Solver
* **Infrastructure:** Docker, Docker Compose

## Getting Started (Recommended: Docker)

To avoid issues with installing system-level mathematical solvers (COIN-OR CBC) on different operating systems, this project is fully containerized.

### Prerequisites
* Docker and Docker Compose installed on your machine.

### Installation & Execution

1. Clone the repository:
```bash
git clone [https://github.com/czarodziejszyn/auto-fpl-manager.git](https://github.com/czarodziejszyn/auto-fpl-manager.git)
cd AutoFPL-Manager
```

2. Build and run the system:
```bash
docker-compose up --build
```
*This command will build the isolated environment, install the system-level CBC solver, sync Python dependencies using `uv`, and execute the optimization pipeline.*

## Local Development 

If you prefer to run the project locally without Docker, ensure you have the `coinor-cbc` solver installed directly on your OS (e.g., `sudo apt-get install coinor-cbc` on Debian/Ubuntu).

1. Install dependencies using `uv`:
```bash
uv sync
```

2. Run the components step-by-step:
```bash
# 1. Extract raw data
uv run python -m src.data.etl_pipeline

# 2. Process time-series features
uv run python -m src.data.feature_engineering

# 3. Train the LightGBM model
uv run python -m src.models.train_model

# 4. Run the MILP optimization engine
uv run python -m src.optimization.squad_optimizer
```
