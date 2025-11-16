"""
GNN-based Task Allocation Pipeline Runner.

Authors: Group-27
    - Nikita Dhiman (M25AI1001)(G24AIT001)
    - Hensi Solanki (M25AI1090)(G24AIT115)
    - Akash Singh (M25AI1043)(G24AIT054)
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from src.task_allocator.task_allocator_gnn import (
    GNNDeveloperExpertiseModeler,
    GNNTaskAllocator
)


def setup_logger(log_config: Dict[str, Any]) -> logging.Logger:
    """Configure logging system for the task allocation pipeline."""
    log_dir = log_config['log_dir']
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'task_allocation_gnn_{timestamp}.log'

    logger = logging.getLogger('TaskAllocatorGNN')
    logger.setLevel(getattr(logging, log_config['log_level']))

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        log_config['log_format'],
        datefmt=log_config['date_format']
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("="*80)
    logger.info("GNN-Based Task Allocation System Initialized")
    logger.info("="*80)

    return logger


def run(data_config: Dict[str, Any], model_config: Dict[str, Any],
        allocation_config: Dict[str, Any], log_config: Dict[str, Any]) -> None:
    """Execute the complete GNN-based task allocation pipeline."""
    logger = setup_logger(log_config)

    logger.info("Phase 1: Data Loading and Validation")
    logger.info("-" * 80)

    expertise_modeler = GNNDeveloperExpertiseModeler(data_config, logger)

    developers_df, contributions_df, technologies_df, projects_df, tasks_df = (
        expertise_modeler.load_data()
    )

    logger.info("")
    logger.info("Phase 2: Heterogeneous Graph Construction")
    logger.info("-" * 80)

    expertise_modeler.construct_heterogeneous_graph(
        developers_df,
        contributions_df,
        technologies_df,
        projects_df
    )

    logger.info("")
    logger.info("Phase 3: GNN Model Training and Embedding Generation")
    logger.info("-" * 80)

    expertise_modeler.prepare_technology_mapping(technologies_df)
    expertise_modeler.train_gnn_model(developers_df, technologies_df, tasks_df)

    logger.info("")
    logger.info("Phase 4: GNN-Based Task Allocation Processing")
    logger.info("-" * 80)

    allocator = GNNTaskAllocator(allocation_config, logger, expertise_modeler)
    allocations = allocator.allocate_all_tasks(tasks_df)

    logger.info("")
    logger.info("Phase 5: Results Generation and Reporting")
    logger.info("-" * 80)

    allocation_report = allocator.generate_allocation_report()

    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = output_dir / f'allocation_report_gnn_{timestamp}.csv'

    allocation_report.to_csv(report_path, index=False)
    logger.info(f"GNN-based allocation report saved to: {report_path}")

    logger.info("")
    logger.info("="*80)
    logger.info("GNN-Based Task Allocation Pipeline Completed Successfully")
    logger.info("="*80)

    logger.info("")
    logger.info("Allocation Summary:")
    logger.info(f"  Total Tasks Processed: {len(tasks_df)}")
    logger.info(f"  Successful Allocations: {len(allocations)}")
    logger.info(f"  Average Confidence Score: {allocation_report['Confidence Score'].mean():.3f}")
    logger.info(f"  Tasks by Priority:")
    for priority in ['Critical', 'High', 'Medium', 'Low']:
        count = len(allocation_report[allocation_report['Priority'] == priority])
        logger.info(f"    {priority}: {count}")

    logger.info("")
    logger.info("Top Allocated Developers:")
    dev_counts = allocation_report['Developer Name'].value_counts()
    for dev_name, count in dev_counts.head(5).items():
        logger.info(f"  {dev_name}: {count} tasks")


def main():
    """Entry point for the GNN-based task allocation runner."""
    from configs.config import (
        DATA_CONFIG,
        MODEL_CONFIG,
        ALLOCATION_CONFIG,
        LOG_CONFIG
    )

    run(DATA_CONFIG, MODEL_CONFIG, ALLOCATION_CONFIG, LOG_CONFIG)


if __name__ == "__main__":
    main()
