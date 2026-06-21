import asyncio
from client_server.game_agent_server import GameServerAgent
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import torch
import torch.nn as nn
 


# ==========================================
# GNN FEATURE EXTRACTOR
# ==========================================
class SpatialGNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, n_planets: int = 30):        
        self.n_planets, self.n_globals = n_planets, 4 
        self.features_per_node = (observation_space.shape[0] - self.n_globals) // self.n_planets
        self.node_feature_dim, self.hidden_dim, self.hidden_dim_3, self.gnn_output_dim = self.features_per_node, 256, 128, 96
        actual_features_dim = self.gnn_output_dim + self.features_per_node + (self.n_planets * self.gnn_output_dim) + self.n_globals   
        super().__init__(observation_space, actual_features_dim)

        self.spatial_decay = nn.Parameter(torch.tensor([2.0]))
        self.embed = nn.Linear(self.node_feature_dim, self.hidden_dim); self.ln_embed = nn.LayerNorm(self.hidden_dim) 
        self.update1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim); self.ln1 = nn.LayerNorm(self.hidden_dim)      
        self.update2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim); self.ln2 = nn.LayerNorm(self.hidden_dim)
        self.update3 = nn.Linear(self.hidden_dim * 2, self.hidden_dim_3); self.ln3 = nn.LayerNorm(self.hidden_dim_3) 
        self.update4 = nn.Linear(self.hidden_dim_3 * 2, self.gnn_output_dim); self.ln4 = nn.LayerNorm(self.gnn_output_dim) 

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        B = observations.shape[0]
        global_features = observations[:, -self.n_globals:] 
        x = observations[:, :-self.n_globals].view(B, self.n_planets, self.features_per_node)
        
        coords, is_acting_mask = x[:, :, -2:], x[:, :, -3].unsqueeze(-1) 
        valid_nodes_mask = x[:, :, 0:3].sum(dim=-1, keepdim=True)
        adj_mask = valid_nodes_mask * valid_nodes_mask.transpose(1, 2)
        
        h = self.ln_embed(torch.relu(self.embed(x))) * valid_nodes_mask
        dist_matrix = torch.cdist(coords, coords) 
        adj_weights = torch.exp(-dist_matrix * torch.clamp(self.spatial_decay, min=0.1, max=10.0)) * adj_mask
        adj_weights = adj_weights / (adj_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        messages1 = torch.bmm(adj_weights, h); h1 = self.ln1(torch.relu(self.update1(torch.cat([h, messages1], dim=-1))) + h) * valid_nodes_mask
        messages2 = torch.bmm(adj_weights, h1); h2 = self.ln2(torch.relu(self.update2(torch.cat([h1, messages2], dim=-1))) + h1) * valid_nodes_mask
        messages3 = torch.bmm(adj_weights, h2); h3 = self.ln3(torch.relu(self.update3(torch.cat([h2, messages3], dim=-1)))) * valid_nodes_mask
        messages4 = torch.bmm(adj_weights, h3); h_updated = self.ln4(torch.relu(self.update4(torch.cat([h3, messages4], dim=-1)))) * valid_nodes_mask
        
        acting_gnn = torch.sum(h_updated * is_acting_mask, dim=1)
        acting_raw = torch.sum(x * is_acting_mask, dim=1) 
        flat_gnn_rep = torch.flatten(h_updated, start_dim=1)
        return torch.cat([acting_gnn, acting_raw, flat_gnn_rep, global_features], dim=-1)

