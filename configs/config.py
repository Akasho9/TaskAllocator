"""
Configuration settings for the Task Allocation System.

Authors: Group-27
    - Nikita Dhiman (M25AI1001)(G24AIT001)
    - Hensi Solanki (M25AI1090)(G24AIT115)
    - Akash Singh (M25AI1043)(G24AIT054)
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.absolute()

DATA_CONFIG = {
    "developers_path": BASE_DIR / "data" / "developers.csv",
    "contributions_path": BASE_DIR / "data" / "contributions.csv",
    "technologies_path": BASE_DIR / "data" / "technologies.csv",
    "projects_path": BASE_DIR / "data" / "projects.csv",
    "tasks_path": BASE_DIR / "data" / "tasks.csv",
}

MODEL_CONFIG = {
    "embedding_dim": 128,
    "hidden_dim": 256,
    "output_dim": 64,
    "num_layers": 3,
    "dropout_rate": 0.2,
    "learning_rate": 0.001,
    "num_epochs": 100,
    "batch_size": 32,
    "validation_split": 0.2,
    "early_stopping_patience": 15,
}

GRAPH_CONFIG = {
    "node_types": ["developer", "file", "technology", "project"],
    "edge_types": [
        ("developer", "contributes_to", "file"),
        ("developer", "masters", "technology"),
        ("developer", "works_on", "project"),
        ("file", "uses", "technology"),
        ("project", "requires", "technology"),
    ],
    "aggregation_method": "mean",
    "neighbor_sample_size": 10,
}

ALLOCATION_CONFIG = {
    "top_k_candidates": 3,
    "workload_balance_weight": 0.3,
    "expertise_match_weight": 0.5,
    "availability_weight": 0.2,
    "max_tasks_per_developer": 5,
}

LOG_CONFIG = {
    "log_dir": BASE_DIR / "logs",
    "log_level": "INFO",
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
}

OUTPUT_CONFIG = {
    "results_dir": BASE_DIR / "results",
    "model_checkpoint_dir": BASE_DIR / "checkpoints",
    "visualization_dir": BASE_DIR / "visualizations",
}

PROFICIENCY_MAPPING = {
    "Expert": 1.0,
    "Advanced": 0.75,
    "Intermediate": 0.5,
    "Beginner": 0.25,
}

PRIORITY_WEIGHTS = {
    "Critical": 1.0,
    "High": 0.8,
    "Medium": 0.6,
    "Low": 0.4,
}

COMPLEXITY_MAPPING = {
    "Expert": 3,
    "Advanced": 2,
    "Intermediate": 1,
}
