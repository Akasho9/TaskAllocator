"""
Core library functions for graph construction and data processing.

Authors: Group-27
    - Nikita Dhiman (M25AI1001)(G24AIT001)
    - Hensi Solanki (M25AI1090)(G24AIT115)
    - Akash Singh (M25AI1043)(G24AIT054)
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict

from .helpers import (
    generate_node_id,
    normalize_feature_vector,
    parse_technology_list
)


class GraphBuilder:
    """Construct heterogeneous graphs representing developer expertise networks."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.nodes = defaultdict(list)
        self.edges = defaultdict(list)
        self.node_features = {}

    def add_developer_nodes(self, developers_df: pd.DataFrame) -> None:
        """Create developer nodes with experience-based features."""
        self.logger.info(f"Adding {len(developers_df)} developer nodes")

        for _, row in developers_df.iterrows():
            node_id = generate_node_id("dev", row['developer_id'])
            features = [
                float(row['years_experience']) / 15.0,
                hash(row['primary_language']) % 100 / 100.0,
                hash(row['specialization']) % 100 / 100.0
            ]
            normalized_features = normalize_feature_vector(features)
            self.nodes['developer'].append(node_id)
            self.node_features[node_id] = {
                'features': normalized_features,
                'original_id': row['developer_id'],
                'name': row['name']
            }

    def add_file_nodes(self, contributions_df: pd.DataFrame) -> None:
        """Create file nodes representing code artifacts."""
        unique_files = contributions_df['file_path'].unique()
        self.logger.info(f"Adding {len(unique_files)} file nodes")

        for file_path in unique_files:
            node_id = generate_node_id("file", file_path)
            file_contributions = contributions_df[contributions_df['file_path'] == file_path]
            total_changes = (file_contributions['lines_added'].sum() + 
                           file_contributions['lines_deleted'].sum())
            commit_frequency = len(file_contributions)

            features = [
                np.log1p(total_changes) / 10.0,
                float(commit_frequency) / 100.0,
                len(file_path.split('/')) / 10.0
            ]
            normalized_features = normalize_feature_vector(features)
            self.nodes['file'].append(node_id)
            self.node_features[node_id] = {
                'features': normalized_features,
                'original_path': file_path
            }

    def add_technology_nodes(self, technologies_df: pd.DataFrame) -> None:
        """Create technology nodes representing skills and tools."""
        unique_techs = technologies_df['technology'].unique()
        self.logger.info(f"Adding {len(unique_techs)} technology nodes")

        for tech in unique_techs:
            node_id = generate_node_id("tech", tech)
            tech_records = technologies_df[technologies_df['technology'] == tech]
            avg_years = tech_records['years_used'].mean()
            user_count = len(tech_records)

            features = [
                float(avg_years) / 10.0,
                float(user_count) / len(technologies_df['developer_id'].unique()),
                hash(tech) % 100 / 100.0
            ]
            normalized_features = normalize_feature_vector(features)
            self.nodes['technology'].append(node_id)
            self.node_features[node_id] = {
                'features': normalized_features,
                'original_name': tech
            }

    def add_project_nodes(self, projects_df: pd.DataFrame) -> None:
        """Create project nodes representing collaborative work units."""
        unique_projects = projects_df['project_id'].unique()
        self.logger.info(f"Adding {len(unique_projects)} project nodes")

        for project_id in unique_projects:
            node_id = generate_node_id("proj", project_id)
            project_records = projects_df[projects_df['project_id'] == project_id]
            team_size = len(project_records)
            avg_contribution = project_records['contribution_score'].mean()

            features = [
                float(team_size) / 10.0,
                float(avg_contribution),
                hash(project_id) % 100 / 100.0
            ]
            normalized_features = normalize_feature_vector(features)
            self.nodes['project'].append(node_id)
            self.node_features[node_id] = {
                'features': normalized_features,
                'original_id': project_id,
                'name': project_records.iloc[0]['project_name']
            }

    def create_contribution_edges(self, contributions_df: pd.DataFrame) -> None:
        """Establish edges between developers and files."""
        self.logger.info("Creating developer-file contribution edges")

        for _, row in contributions_df.iterrows():
            dev_node_id = generate_node_id("dev", row['developer_id'])
            file_node_id = generate_node_id("file", row['file_path'])
            contribution_strength = (
                float(row['commits']) / 100.0 +
                np.log1p(row['lines_added'] + row['lines_deleted']) / 10.0
            ) / 2.0

            edge_data = {
                'weight': min(contribution_strength, 1.0),
                'commits': row['commits']
            }
            self.edges[('developer', 'contributes_to', 'file')].append(
                (dev_node_id, file_node_id, edge_data)
            )

    def create_technology_edges(self, technologies_df: pd.DataFrame) -> None:
        """Establish edges between developers and technologies."""
        self.logger.info("Creating developer-technology mastery edges")
        proficiency_map = {'Expert': 1.0, 'Advanced': 0.75, 
                          'Intermediate': 0.5, 'Beginner': 0.25}

        for _, row in technologies_df.iterrows():
            dev_node_id = generate_node_id("dev", row['developer_id'])
            tech_node_id = generate_node_id("tech", row['technology'])
            proficiency_weight = proficiency_map.get(row['proficiency_level'], 0.5)
            experience_factor = min(float(row['years_used']) / 10.0, 1.0)

            edge_data = {
                'weight': (proficiency_weight + experience_factor) / 2.0,
                'proficiency': row['proficiency_level']
            }
            self.edges[('developer', 'masters', 'technology')].append(
                (dev_node_id, tech_node_id, edge_data)
            )

    def create_project_edges(self, projects_df: pd.DataFrame) -> None:
        """Establish edges between developers and projects."""
        self.logger.info("Creating developer-project work edges")

        for _, row in projects_df.iterrows():
            dev_node_id = generate_node_id("dev", row['developer_id'])
            proj_node_id = generate_node_id("proj", row['project_id'])
            edge_data = {
                'weight': float(row['contribution_score']),
                'role': row['role']
            }
            self.edges[('developer', 'works_on', 'project')].append(
                (dev_node_id, proj_node_id, edge_data)
            )

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics for the constructed graph."""
        stats = {
            'node_counts': {node_type: len(nodes) 
                           for node_type, nodes in self.nodes.items()},
            'edge_counts': {str(edge_type): len(edges) 
                           for edge_type, edges in self.edges.items()},
            'total_nodes': sum(len(nodes) for nodes in self.nodes.values()),
            'total_edges': sum(len(edges) for edges in self.edges.values())
        }
        return stats


class FeatureEncoder:
    """Transform raw data into numerical feature representations."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.encodings = {}

    def encode_task_features(self, task_row: pd.Series, 
                           technology_mapping: Dict[str, int]) -> np.ndarray:
        """Convert task specifications into feature vectors."""
        required_techs = parse_technology_list(task_row['required_technologies'])
        tech_vector = np.zeros(len(technology_mapping))
        for tech in required_techs:
            if tech in technology_mapping:
                tech_vector[technology_mapping[tech]] = 1.0

        priority_map = {'Critical': 1.0, 'High': 0.8, 'Medium': 0.6, 'Low': 0.4}
        complexity_map = {'Expert': 1.0, 'Advanced': 0.7, 'Intermediate': 0.4}

        priority_score = priority_map.get(task_row['priority'], 0.5)
        complexity_score = complexity_map.get(task_row['complexity'], 0.5)
        effort_normalized = float(task_row['estimated_hours']) / 200.0
        meta_features = np.array([priority_score, complexity_score, effort_normalized])

        return np.concatenate([tech_vector, meta_features])

    def create_technology_mapping(self, technologies_df: pd.DataFrame) -> Dict[str, int]:
        """Build index mapping for all technologies."""
        unique_technologies = sorted(technologies_df['technology'].unique())
        mapping = {tech: idx for idx, tech in enumerate(unique_technologies)}
        self.logger.info(f"Created mapping for {len(mapping)} technologies")
        return mapping
