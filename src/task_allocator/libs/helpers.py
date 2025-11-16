"""
Helper utility functions for the Task Allocation System.

Authors: Group-27
    - Nikita Dhiman (M25AI1001)(G24AIT001)
    - Hensi Solanki (M25AI1090)(G24AIT115)
    - Akash Singh (M25AI1043)(G24AIT054)
"""

import logging
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime


def generate_node_id(prefix: str, identifier: str) -> str:
    """Generate a unique node identifier."""
    hash_suffix = hashlib.md5(identifier.encode()).hexdigest()[:8]
    return f"{prefix}_{hash_suffix}"


def normalize_feature_vector(features: List[float]) -> List[float]:
    """Normalize feature vector to unit length using L2 normalization."""
    magnitude = sum(x ** 2 for x in features) ** 0.5
    if magnitude == 0:
        return features
    return [x / magnitude for x in features]


def calculate_similarity_score(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two feature vectors."""
    if len(vec1) != len(vec2):
        logging.warning("Vector length mismatch in similarity calculation")
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(x ** 2 for x in vec1) ** 0.5
    magnitude2 = sum(x ** 2 for x in vec2) ** 0.5

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def parse_technology_list(tech_string: str) -> List[str]:
    """Parse semicolon-separated technology string."""
    if not tech_string:
        return []
    technologies = [tech.strip() for tech in tech_string.split(';')]
    return [tech for tech in technologies if tech]


def compute_weighted_score(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """Calculate weighted sum of multiple scoring components."""
    total_weight = sum(weights.values())
    if total_weight == 0:
        logging.warning("Zero total weight in score calculation")
        return 0.0
    weighted_sum = sum(scores.get(key, 0) * weight for key, weight in weights.items())
    return weighted_sum / total_weight


def format_allocation_result(developer_id: str, task_id: str, 
                            confidence: float, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Structure allocation results into standardized format."""
    return {
        "developer_id": developer_id,
        "task_id": task_id,
        "confidence_score": round(confidence, 4),
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata
    }


def validate_graph_structure(node_counts: Dict[str, int], 
                            edge_counts: Dict[str, int]) -> bool:
    """Verify graph construction meets minimum requirements."""
    min_nodes_per_type = 2
    min_edges_per_type = 1

    for node_type, count in node_counts.items():
        if count < min_nodes_per_type:
            logging.error(f"Insufficient {node_type} nodes: {count}")
            return False

    for edge_type, count in edge_counts.items():
        if count < min_edges_per_type:
            logging.error(f"Insufficient {edge_type} edges: {count}")
            return False

    return True


def log_performance_metrics(metrics: Dict[str, float], phase: str) -> None:
    """Log model performance metrics in structured format."""
    logger = logging.getLogger(__name__)
    logger.info(f"Performance Metrics - {phase}")
    for metric_name, metric_value in sorted(metrics.items()):
        logger.info(f"  {metric_name}: {metric_value:.4f}")
