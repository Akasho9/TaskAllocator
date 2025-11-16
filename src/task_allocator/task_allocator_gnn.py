"""
GNN-based Task Allocation System with Real Neural Networks.

This module integrates actual Graph Neural Networks using PyTorch Geometric
for learning developer expertise representations and task allocation.

Authors: Group-27
    - Nikita Dhiman (M25AI1001)(G24AIT001)
    - Hensi Solanki (M25AI1090)(G24AIT115)
    - Akash Singh (M25AI1043)(G24AIT054)

Date: November 2025
"""

import logging
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

from .libs.libs import GraphBuilder, FeatureEncoder
from .libs.helpers import (
    calculate_similarity_score,
    compute_weighted_score,
    format_allocation_result,
    validate_graph_structure,
    log_performance_metrics,
    parse_technology_list
)
from .models.gnn_model import HeterogeneousGNN, GNNTrainer
from .models.gnn_data import GraphDataConverter


class GNNDeveloperExpertiseModeler:
    """
    Model developer expertise using real Graph Neural Networks.

    This class uses PyTorch Geometric to train a GNN that learns embeddings
    through message passing on the heterogeneous developer graph.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the GNN-based expertise modeling system.

        Args:
            config: Configuration dictionary with paths and parameters
            logger: Logger for operation tracking
        """
        self.config = config
        self.logger = logger
        self.graph_builder = GraphBuilder(logger)
        self.feature_encoder = FeatureEncoder(logger)
        self.developer_embeddings = {}
        self.technology_mapping = {}
        self.gnn_model = None
        self.gnn_trainer = None
        self.data_converter = GraphDataConverter(logger)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                                 pd.DataFrame, pd.DataFrame]:
        """Load all required datasets from configured paths."""
        self.logger.info("Loading datasets from configured paths")

        developers_df = pd.read_csv(self.config['developers_path'])
        contributions_df = pd.read_csv(self.config['contributions_path'])
        technologies_df = pd.read_csv(self.config['technologies_path'])
        projects_df = pd.read_csv(self.config['projects_path'])
        tasks_df = pd.read_csv(self.config['tasks_path'])

        self.logger.info(f"Loaded {len(developers_df)} developers")
        self.logger.info(f"Loaded {len(contributions_df)} contributions")
        self.logger.info(f"Loaded {len(technologies_df)} technology skills")
        self.logger.info(f"Loaded {len(projects_df)} project assignments")
        self.logger.info(f"Loaded {len(tasks_df)} tasks for allocation")

        return developers_df, contributions_df, technologies_df, projects_df, tasks_df

    def construct_heterogeneous_graph(self, developers_df: pd.DataFrame,
                                     contributions_df: pd.DataFrame,
                                     technologies_df: pd.DataFrame,
                                     projects_df: pd.DataFrame) -> None:
        """Build the heterogeneous graph structure from loaded data."""
        self.logger.info("Constructing heterogeneous developer expertise graph")

        self.graph_builder.add_developer_nodes(developers_df)
        self.graph_builder.add_file_nodes(contributions_df)
        self.graph_builder.add_technology_nodes(technologies_df)
        self.graph_builder.add_project_nodes(projects_df)

        self.graph_builder.create_contribution_edges(contributions_df)
        self.graph_builder.create_technology_edges(technologies_df)
        self.graph_builder.create_project_edges(projects_df)

        stats = self.graph_builder.get_graph_statistics()
        self.logger.info(f"Graph construction complete: {stats}")

        is_valid = validate_graph_structure(
            stats['node_counts'],
            stats['edge_counts']
        )

        if not is_valid:
            raise ValueError("Constructed graph does not meet minimum requirements")

    def train_gnn_model(self, developers_df: pd.DataFrame,
                       technologies_df: pd.DataFrame,
                       tasks_df: pd.DataFrame) -> None:
        """
        Train the GNN model on the heterogeneous graph.

        Uses contrastive learning to train embeddings where developers with
        matching skills are positioned close to relevant tasks.

        Args:
            developers_df: Developer profile data
            technologies_df: Technology proficiency data
            tasks_df: Task information data
        """
        self.logger.info("Preparing data for GNN training")

        hetero_data = self.data_converter.create_hetero_data(
            self.graph_builder,
            feature_dim=self.config.get('embedding_dim', 128)
        )

        metadata = (
            list(hetero_data.node_types),
            list(hetero_data.edge_types)
        )
        
        self.logger.info(f"Graph metadata: {len(metadata[0])} node types, {len(metadata[1])} edge types")

        self.logger.info("Initializing GNN model")
        self.gnn_model = HeterogeneousGNN(
            metadata=metadata,
            hidden_channels=self.config.get('hidden_dim', 256),
            out_channels=self.config.get('output_dim', 64),
            num_layers=self.config.get('num_layers', 3),
            dropout=self.config.get('dropout_rate', 0.2),
            logger=self.logger
        )

        self.logger.info("Initializing GNN trainer")
        self.gnn_trainer = GNNTrainer(
            model=self.gnn_model,
            learning_rate=self.config.get('learning_rate', 0.001),
            weight_decay=5e-4,
            logger=self.logger
        )

        positive_pairs = self.data_converter.create_training_pairs(
            developers_df,
            technologies_df,
            tasks_df
        )

        self.logger.info("Starting GNN training")
        losses = self.gnn_trainer.train(
            hetero_data=hetero_data,
            positive_pairs=positive_pairs,
            num_epochs=self.config.get('num_epochs', 100),
            early_stopping_patience=self.config.get('early_stopping_patience', 15)
        )

        self.logger.info(f"GNN training completed with final loss: {losses[-1]:.4f}")

        self.logger.info("Extracting learned embeddings")
        learned_embeddings = self.gnn_trainer.get_embeddings(hetero_data)

        developer_nodes = self.graph_builder.nodes['developer']
        for idx, dev_node_id in enumerate(developer_nodes):
            self.developer_embeddings[dev_node_id] = learned_embeddings['developer'][idx]

        self.logger.info(f"Stored embeddings for {len(self.developer_embeddings)} developers")

    def prepare_technology_mapping(self, technologies_df: pd.DataFrame) -> None:
        """Create technology index mapping for feature encoding."""
        self.technology_mapping = self.feature_encoder.create_technology_mapping(
            technologies_df
        )


class GNNTaskAllocator:
    """
    Allocate tasks using GNN-learned developer embeddings.

    Uses embeddings learned through neural message passing to match
    developers with tasks based on deep expertise representations.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger,
                 expertise_modeler: GNNDeveloperExpertiseModeler):
        """
        Initialize task allocation engine with GNN embeddings.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            expertise_modeler: Trained GNN expertise modeling component
        """
        self.config = config
        self.logger = logger
        self.expertise_modeler = expertise_modeler
        self.developer_workload = {}
        self.allocations = []

    def compute_expertise_match_score(self, developer_embedding: np.ndarray,
                                      task_features: np.ndarray) -> float:
        """Calculate developer-task match using GNN embeddings."""
        if len(task_features) < len(developer_embedding):
            task_features = np.pad(
                task_features,
                (0, len(developer_embedding) - len(task_features)),
                constant_values=0
            )
        elif len(task_features) > len(developer_embedding):
            task_features = task_features[:len(developer_embedding)]

        similarity = calculate_similarity_score(
            developer_embedding.tolist(),
            task_features.tolist()
        )
        return max(0.0, similarity)

    def compute_workload_balance_score(self, developer_id: str) -> float:
        """Evaluate current workload distribution for a developer."""
        current_workload = self.developer_workload.get(developer_id, 0)
        max_tasks = self.config.get('max_tasks_per_developer', 5)

        if current_workload >= max_tasks:
            return 0.0
        return 1.0 - (current_workload / max_tasks)

    def compute_allocation_score(self, developer_id: str,
                                task_features: np.ndarray,
                                developer_embedding: np.ndarray) -> float:
        """Calculate comprehensive allocation score using GNN embeddings."""
        expertise_score = self.compute_expertise_match_score(
            developer_embedding,
            task_features
        )
        workload_score = self.compute_workload_balance_score(developer_id)
        availability_score = 1.0

        scores = {
            'expertise': expertise_score,
            'workload': workload_score,
            'availability': availability_score
        }

        weights = {
            'expertise': self.config.get('expertise_match_weight', 0.5),
            'workload': self.config.get('workload_balance_weight', 0.3),
            'availability': self.config.get('availability_weight', 0.2)
        }

        return compute_weighted_score(scores, weights)

    def allocate_task(self, task_row: pd.Series) -> Dict[str, Any]:
        """Find the best developer for a specific task using GNN embeddings."""
        self.logger.info(f"Allocating task: {task_row['task_id']}")

        task_features = self.expertise_modeler.feature_encoder.encode_task_features(
            task_row,
            self.expertise_modeler.technology_mapping
        )

        candidate_scores = []

        for dev_node_id, dev_embedding in self.expertise_modeler.developer_embeddings.items():
            dev_metadata = self.expertise_modeler.graph_builder.node_features[dev_node_id]
            dev_id = dev_metadata['original_id']

            allocation_score = self.compute_allocation_score(
                dev_id,
                task_features,
                dev_embedding
            )

            candidate_scores.append({
                'developer_id': dev_id,
                'developer_name': dev_metadata['name'],
                'score': allocation_score
            })

        candidate_scores.sort(key=lambda x: x['score'], reverse=True)
        top_k = self.config.get('top_k_candidates', 3)
        top_candidates = candidate_scores[:top_k]

        if top_candidates:
            best_candidate = top_candidates[0]
            selected_dev_id = best_candidate['developer_id']

            self.developer_workload[selected_dev_id] = (
                self.developer_workload.get(selected_dev_id, 0) + 1
            )

            allocation_result = format_allocation_result(
                selected_dev_id,
                task_row['task_id'],
                best_candidate['score'],
                {
                    'task_title': task_row['title'],
                    'developer_name': best_candidate['developer_name'],
                    'top_candidates': top_candidates,
                    'priority': task_row['priority'],
                    'complexity': task_row['complexity']
                }
            )

            self.allocations.append(allocation_result)

            self.logger.info(
                f"Task {task_row['task_id']} allocated to {selected_dev_id} "
                f"({best_candidate['developer_name']}) with confidence {best_candidate['score']:.3f}"
            )

            return allocation_result

        self.logger.warning(f"No suitable candidate found for task {task_row['task_id']}")
        return {}

    def allocate_all_tasks(self, tasks_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Process all tasks using GNN-learned embeddings."""
        self.logger.info(f"Beginning GNN-based allocation for {len(tasks_df)} tasks")

        tasks_sorted = tasks_df.copy()
        priority_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
        tasks_sorted['priority_rank'] = tasks_sorted['priority'].map(priority_order)
        tasks_sorted = tasks_sorted.sort_values('priority_rank')

        for _, task_row in tasks_sorted.iterrows():
            self.allocate_task(task_row)

        self.logger.info(f"GNN-based allocation complete: {len(self.allocations)} tasks assigned")
        return self.allocations

    def generate_allocation_report(self) -> pd.DataFrame:
        """Create comprehensive report of all task allocations."""
        if not self.allocations:
            self.logger.warning("No allocations to report")
            return pd.DataFrame()

        report_data = []

        for allocation in self.allocations:
            report_data.append({
                'Task ID': allocation['task_id'],
                'Developer ID': allocation['developer_id'],
                'Developer Name': allocation['metadata']['developer_name'],
                'Task Title': allocation['metadata']['task_title'],
                'Confidence Score': allocation['confidence_score'],
                'Priority': allocation['metadata']['priority'],
                'Complexity': allocation['metadata']['complexity'],
                'Timestamp': allocation['timestamp']
            })

        report_df = pd.DataFrame(report_data)
        self.logger.info(f"Generated GNN-based allocation report with {len(report_df)} entries")
        return report_df
