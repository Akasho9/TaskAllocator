"""
GNN-based Task Allocation System.

Authors: Group-27
    - Nikita Dhiman (M25AI1001)(G24AIT001)
    - Hensi Solanki (M25AI1090)(G24AIT115)
    - Akash Singh (M25AI1043)(G24AIT054)
"""

import logging
import pandas as pd
import numpy as np
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


class DeveloperExpertiseModeler:
    """Model developer expertise using heterogeneous graph representations."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.graph_builder = GraphBuilder(logger)
        self.feature_encoder = FeatureEncoder(logger)
        self.developer_embeddings = {}
        self.technology_mapping = {}

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

    def compute_developer_embeddings(self) -> None:
        """Generate embedding representations for all developers."""
        self.logger.info("Computing developer expertise embeddings")
        developer_nodes = self.graph_builder.nodes['developer']

        for dev_node_id in developer_nodes:
            base_features = self.graph_builder.node_features[dev_node_id]['features']

            tech_edges = self.graph_builder.edges[('developer', 'masters', 'technology')]
            tech_contributions = [
                edge[2]['weight'] for edge in tech_edges if edge[0] == dev_node_id
            ]
            tech_score = sum(tech_contributions) / max(len(tech_contributions), 1)

            contrib_edges = self.graph_builder.edges[('developer', 'contributes_to', 'file')]
            contrib_counts = len([e for e in contrib_edges if e[0] == dev_node_id])
            contrib_score = min(float(contrib_counts) / 20.0, 1.0)

            proj_edges = self.graph_builder.edges[('developer', 'works_on', 'project')]
            proj_contributions = [
                edge[2]['weight'] for edge in proj_edges if edge[0] == dev_node_id
            ]
            proj_score = sum(proj_contributions) / max(len(proj_contributions), 1)

            aggregated_features = base_features + [tech_score, contrib_score, proj_score]
            self.developer_embeddings[dev_node_id] = np.array(aggregated_features)

        self.logger.info(f"Generated embeddings for {len(self.developer_embeddings)} developers")

    def prepare_technology_mapping(self, technologies_df: pd.DataFrame) -> None:
        """Create technology index mapping for feature encoding."""
        self.technology_mapping = self.feature_encoder.create_technology_mapping(
            technologies_df
        )


class TaskAllocator:
    """Allocate tasks to developers using GNN-derived expertise embeddings."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger,
                 expertise_modeler: DeveloperExpertiseModeler):
        self.config = config
        self.logger = logger
        self.expertise_modeler = expertise_modeler
        self.developer_workload = {}
        self.allocations = []

    def compute_expertise_match_score(self, developer_embedding: np.ndarray,
                                      task_features: np.ndarray) -> float:
        """Calculate how well a developer's expertise matches task requirements."""
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
        """Calculate comprehensive allocation score combining multiple factors."""
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
        """Find the best developer for a specific task."""
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
        """Process all tasks and generate allocation recommendations."""
        self.logger.info(f"Beginning allocation for {len(tasks_df)} tasks")

        tasks_sorted = tasks_df.copy()
        priority_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
        tasks_sorted['priority_rank'] = tasks_sorted['priority'].map(priority_order)
        tasks_sorted = tasks_sorted.sort_values('priority_rank')

        for _, task_row in tasks_sorted.iterrows():
            self.allocate_task(task_row)

        self.logger.info(f"Allocation complete: {len(self.allocations)} tasks assigned")
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
        self.logger.info(f"Generated allocation report with {len(report_df)} entries")
        return report_df
