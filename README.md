# GNN-Based Task Allocation System

## Team Information - Group 27

| Name | Roll Number | Enrollment Number |
|------|-------------|-------------------|
| Nikita Dhiman | M25AI1001 | G24AIT001 |
| Hensi Solanki | M25AI1090 | G24AIT115 |
| Akash Singh | M25AI1043 | G24AIT054 |

**Course:** SDE - Major Project  
**Academic Year:** 2024-2026

---

## Project Overview

An intelligent task allocation system that uses Graph Neural Networks (GNN) to automatically assign software development tasks to team members. The system builds a heterogeneous graph of developers, technologies, and contributions, then applies deep learning to learn optimal task-developer matches based on skills, expertise, and workload.

**Key Features:**
- Heterogeneous graph construction with 4 node types (developers, files, technologies, projects)
- 3-layer GraphSAGE GNN model with message passing
- Automatic task allocation with confidence scoring
- Comprehensive visualization dashboard
- Production-ready logging system

---

## Installation & Setup

### Prerequisites
- Python 3.12 or higher
- pip package manager

### Installation Steps
```bash
# Clone the repository
git clone https://github.com/Akasho9/TaskAllocator.git
cd TaskAllocator

# Create virtual environment
python3 -m venv TaskAllocator
source TaskAllocator/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### 1. Run Complete Pipeline
Executes graph construction, GNN training, task allocation, and visualization generation.
```bash
python main_gnn.py
```

**Output Location:**
- Allocation Report: `results/allocation_report_gnn_TIMESTAMP.csv`
- Execution Logs: `logs/task_allocation_gnn_TIMESTAMP.log`
- Visualizations: `visualizations/*.png`

**Expected Runtime:** 8-10 seconds

### 2. Generate Visualizations Only
Creates all charts and graphs from existing allocation results.
```bash
python app.py
```

**Output:** 5 visualization files in `visualizations/` directory

### 3. Interactive Demo
Step-by-step walkthrough of the system with live data display.
```bash
python demo.py
```

**Features:** Color-coded sections, progress indicators, system statistics

---

## Quality Attributes

### 1. Correctness

**Definition:** The system produces accurate and valid task allocations based on developer expertise and task requirements.

**Evidence:**
- 100% allocation success rate (30 out of 30 tasks successfully allocated)
- Average confidence score: 52.2% (above baseline random assignment: 33.3%)
- Zero allocation errors or exceptions during execution
- All task-developer matches respect workload constraints (max 5 tasks per developer)

**Verification:**
```bash
# Check allocation results
cat results/allocation_report_gnn_*.csv | wc -l

# Verify no errors in logs
grep -i "error" logs/task_allocation_gnn_*.log
```

**Result Location:** `results/allocation_report_gnn_*.csv`

---

### 2. Efficiency

**Definition:** The system executes quickly with minimal resource consumption, suitable for real-time deployment.

**Evidence:**
- Complete pipeline execution: 8-10 seconds
- Graph construction: <0.5 seconds (68 nodes, 140 edges)
- GNN training: 3-5 seconds (20 epochs with early stopping)
- Task allocation: <0.5 seconds (30 tasks, 10 developers)
- Memory footprint: <500 MB peak usage

**Verification:**
```bash
# Measure execution time
time python main_gnn.py
```

**Log Evidence:** Timestamps in `logs/task_allocation_gnn_*.log` show sub-second processing for each phase

---

### 3. Maintainability

**Definition:** The codebase is well-structured, documented, and easy to understand, modify, and extend.

**Evidence:**
- Modular architecture: 6 distinct modules with single responsibilities
- Comprehensive docstrings: 100% function and class documentation coverage
- Clean code practices: No try-catch blocks, meaningful variable names
- Consistent code style: Following PEP 8 guidelines
- Configuration centralization: All parameters in `configs/config.py`

**Verification:**
```bash
# Count lines of documentation
grep -r '"""' src/ | wc -l

# Check code organization
tree src/

# Verify no empty exception handlers
grep -r "except.*pass" src/
```

**Code Structure:**
```
src/task_allocator/
├── libs/              # Graph building and feature encoding
├── models/            # GNN architecture and training
└── task_allocator_gnn.py  # Main allocation logic
```

---

## Technology Stack

- **Python 3.12:** Core programming language
- **PyTorch 2.2.0+:** Deep learning framework
- **PyTorch Geometric 2.5.0+:** Graph neural network library
- **Pandas 2.1.4+:** Data processing
- **Matplotlib/Seaborn:** Visualization
- **NetworkX 3.2.1+:** Graph algorithms

---

## Project Structure
```
TaskAllocator/
├── data/                  # Input CSV files
├── src/                   # Source code modules
├── runner/                # Execution pipeline
├── configs/               # Configuration files
├── results/               # Allocation reports (generated)
├── logs/                  # Execution logs (generated)
├── visualizations/        # Charts and graphs (generated)
├── main_gnn.py           # Main entry point
├── app.py                # Visualization generator
├── demo.py               # Interactive demo
└── requirements.txt      # Python dependencies
```

---

## Results Summary

- **Tasks Allocated:** 30/30 (100%)
- **Average Confidence:** 52.2%
- **Workload Balance:** Perfect (3 tasks per developer)
- **Priority Distribution:** 7 Critical, 12 High, 10 Medium, 1 Low
- **Execution Time:** 8-10 seconds
- **Memory Usage:** <500 MB

---

## Contact Information

For queries or collaboration:
- Nikita Dhiman: nikita.dhiman@example.com
- Hensi Solanki: hensi.solanki@example.com
- Akash Singh: akash.singh@example.com

**GitHub Repository:** https://github.com/Akasho9/TaskAllocator

---

**Date:** 16th November 2025
