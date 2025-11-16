"""
GNN-Based Task Allocator System - Main Entry Point.

This version uses real Graph Neural Networks with PyTorch Geometric
for learning developer expertise through message passing.

Authors: Group-27
    - Nikita Dhiman (M25AI1001)(G24AIT001)
    - Hensi Solanki (M25AI1090)(G24AIT115)
    - Akash Singh (M25AI1043)(G24AIT054)
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from runner.task_allocator_gnn import main as run_allocation_pipeline


def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║                                                                ║")
    print("║      GNN-Based Developer Expertise Task Allocator              ║")
    print("║                                                                ║")
    print("║                      Group-27                                  ║")
    print("║     Nikita Dhiman   |   Hensi Solanki   |   Akash Singh       ║")
    print("║     M25AI1001       |   M25AI1090       |   M25AI1043         ║")
    print("║     G24AIT001       |   G24AIT115       |   G24AIT054         ║")
    print("║                                                                ║")
    print("║              Using PyTorch Geometric GNN Models                ║")
    print("║                                                                ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()

    run_allocation_pipeline()


if __name__ == "__main__":
    main()
