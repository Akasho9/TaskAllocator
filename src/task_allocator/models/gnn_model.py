"""
Graph Neural Network Models for Developer Expertise Learning.

Authors: Group-27
    - Nikita Dhiman (M25AI1001)(G24AIT001)
    - Hensi Solanki (M25AI1090)(G24AIT115)
    - Akash Singh (M25AI1043)(G24AIT054)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Any
import logging
import numpy as np


class HeterogeneousGNN(nn.Module):
    
    def __init__(self, 
                 metadata: Tuple[List[str], List[Tuple[str, str, str]]],
                 hidden_channels: int = 128,
                 out_channels: int = 64,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 logger: logging.Logger = None):
        super(HeterogeneousGNN, self).__init__()
        
        self.logger = logger if logger else logging.getLogger(__name__)
        self.num_layers = num_layers
        self.dropout = dropout
        self.out_channels = out_channels
        
        node_types, edge_types = metadata
        
        self.logger.info(f"Initializing HeterogeneousGNN with {num_layers} layers")
        self.logger.info(f"Hidden channels: {hidden_channels}, Output channels: {out_channels}")
        
        self.input_projections = nn.ModuleDict()
        for node_type in node_types:
            self.input_projections[node_type] = Linear(-1, hidden_channels)
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleDict()
        
        for layer_idx in range(num_layers):
            conv_dict = {}
            
            in_channels = hidden_channels
            out_channels_layer = out_channels if layer_idx == num_layers - 1 else hidden_channels
            
            for edge_type in edge_types:
                conv_dict[edge_type] = SAGEConv(
                    in_channels=(in_channels, in_channels),
                    out_channels=out_channels_layer,
                    aggr='mean'
                )
            
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
            
            if layer_idx < num_layers - 1:
                for node_type in node_types:
                    bn_key = f"{node_type}_{layer_idx}"
                    self.batch_norms[bn_key] = nn.BatchNorm1d(hidden_channels)
        
        self.logger.info(f"GNN model initialized with {len(self.convs)} convolutional layers")
    
    def forward(self, x_dict: Dict[str, torch.Tensor], 
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> Dict[str, torch.Tensor]:
        for node_type in x_dict.keys():
            x_dict[node_type] = self.input_projections[node_type](x_dict[node_type])
        
        for layer_idx, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            
            if layer_idx < self.num_layers - 1:
                for node_type in x_dict.keys():
                    bn_key = f"{node_type}_{layer_idx}"
                    if bn_key in self.batch_norms:
                        x_dict[node_type] = self.batch_norms[bn_key](x_dict[node_type])
                    x_dict[node_type] = F.relu(x_dict[node_type])
                    x_dict[node_type] = F.dropout(x_dict[node_type], p=self.dropout, training=self.training)
        
        return x_dict


class GNNTrainer:
    
    def __init__(self, model: HeterogeneousGNN, learning_rate: float = 0.001,
                 weight_decay: float = 5e-4, logger: logging.Logger = None):
        self.model = model
        self.logger = logger if logger else logging.getLogger(__name__)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.logger.info(f"Initialized GNN trainer with lr={learning_rate}, wd={weight_decay}")
    
    def compute_contrastive_loss(self, developer_embeddings: torch.Tensor, task_embeddings: torch.Tensor,
                                 positive_pairs: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        developer_embeddings = F.normalize(developer_embeddings, p=2, dim=1)
        task_embeddings = F.normalize(task_embeddings, p=2, dim=1)
        similarities = torch.matmul(developer_embeddings, task_embeddings.t()) / temperature
        losses = []
        for dev_idx, task_idx in positive_pairs:
            positive_sim = similarities[dev_idx, task_idx]
            exp_sims = torch.exp(similarities[dev_idx])
            loss = -positive_sim + torch.log(exp_sims.sum())
            losses.append(loss)
        return torch.stack(losses).mean()
    
    def train_epoch(self, hetero_data: HeteroData, positive_pairs: List[Tuple[int, int]], epoch: int) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        x_dict = {node_type: hetero_data[node_type].x for node_type in hetero_data.node_types}
        edge_index_dict = {edge_type: hetero_data[edge_type].edge_index for edge_type in hetero_data.edge_types}
        embeddings_dict = self.model(x_dict, edge_index_dict)
        developer_embeddings = embeddings_dict['developer']
        num_tasks = len(positive_pairs)
        task_embeddings = torch.randn(num_tasks, developer_embeddings.size(1)).to(developer_embeddings.device)
        positive_pairs_tensor = torch.tensor(positive_pairs, dtype=torch.long)
        loss = self.compute_contrastive_loss(developer_embeddings, task_embeddings, positive_pairs_tensor)
        loss.backward()
        self.optimizer.step()
        loss_value = loss.item()
        if epoch % 10 == 0:
            self.logger.info(f"Epoch {epoch}: Loss = {loss_value:.4f}")
        return loss_value
    
    def train(self, hetero_data: HeteroData, positive_pairs: List[Tuple[int, int]],
              num_epochs: int = 100, early_stopping_patience: int = 15) -> List[float]:
        self.logger.info(f"Starting GNN training for {num_epochs} epochs")
        self.logger.info(f"Training with {len(positive_pairs)} positive pairs")
        losses = []
        best_loss = float('inf')
        patience_counter = 0
        for epoch in range(1, num_epochs + 1):
            loss = self.train_epoch(hetero_data, positive_pairs, epoch)
            losses.append(loss)
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
                if epoch % 10 == 0:
                    self.logger.info(f"New best loss: {best_loss:.4f}")
            else:
                patience_counter += 1
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        self.logger.info(f"Training completed. Final loss: {losses[-1]:.4f}")
        return losses
    
    def get_embeddings(self, hetero_data: HeteroData) -> Dict[str, np.ndarray]:
        self.model.eval()
        with torch.no_grad():
            x_dict = {node_type: hetero_data[node_type].x for node_type in hetero_data.node_types}
            edge_index_dict = {edge_type: hetero_data[edge_type].edge_index for edge_type in hetero_data.edge_types}
            embeddings_dict = self.model(x_dict, edge_index_dict)
        numpy_embeddings = {}
        for node_type, embeddings in embeddings_dict.items():
            numpy_embeddings[node_type] = embeddings.cpu().numpy()
            self.logger.info(f"Extracted {len(embeddings)} embeddings for {node_type} nodes")
        return numpy_embeddings
