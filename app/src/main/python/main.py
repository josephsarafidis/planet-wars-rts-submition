import asyncio
from client_server.game_agent_server import GameServerAgent
from agents.gnn_agent import EventDrivenAllPlanetsGNNAgent
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import torch
import torch.nn as nn
 




class SpatialGNNExtractor(BaseFeaturesExtractor):
    """
    Produces: [acting_gnn (64)] + [acting_raw (23)] + [all_planets_gnn (N×64)] + [globals (4)]
    Total = 64 + 23 + (20×64) + 4 = 1367  (for N=20)
    """
    def __init__(self, observation_space: gym.spaces.Box, n_planets: int = 30):
        self.n_planets         = n_planets
        self.n_globals         = 4
        total_obs_size         = observation_space.shape[0]
        self.features_per_node = (total_obs_size - self.n_globals) // n_planets

        self.node_feature_dim  = self.features_per_node
        self.hidden_dim        = 256
        self.gnn_output_dim    = 64

        # [acting_gnn] + [acting_raw] + [all_planets_gnn] + [globals]
        actual_features_dim = (
            self.gnn_output_dim
            + self.features_per_node
            + (self.n_planets * self.gnn_output_dim)
            + self.n_globals
        )
        super().__init__(observation_space, actual_features_dim)

        self.spatial_decay = nn.Parameter(torch.tensor([2.0]))

        self.embed    = nn.Linear(self.node_feature_dim, self.hidden_dim)
        self.ln_embed = nn.LayerNorm(self.hidden_dim)

        self.update1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.ln1     = nn.LayerNorm(self.hidden_dim)

        self.update2 = nn.Linear(self.hidden_dim * 2, self.gnn_output_dim)
        self.ln2     = nn.LayerNorm(self.gnn_output_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        B = observations.shape[0]

        global_features = observations[:, -self.n_globals:]
        planet_obs_flat = observations[:, :-self.n_globals]

        x = planet_obs_flat.view(B, self.n_planets, self.features_per_node)

        coords        = x[:, :, -2:]                      # (B, N, 2)
        is_acting_mask = x[:, :, -3].unsqueeze(-1)        # (B, N, 1)

        h = torch.relu(self.embed(x))
        h = self.ln_embed(h)

        dist_matrix = torch.cdist(coords, coords)
        decay       = torch.clamp(self.spatial_decay, min=0.1, max=10.0)
        adj_weights = torch.exp(-dist_matrix * decay)
        adj_weights = adj_weights / (adj_weights.sum(dim=-1, keepdim=True) + 1e-6)

        messages1 = torch.bmm(adj_weights, h)
        h1        = torch.relu(self.update1(torch.cat([h, messages1], dim=-1)))
        h1        = self.ln1(h1)

        messages2 = torch.bmm(adj_weights, h1)
        h_updated = torch.relu(self.update2(torch.cat([h1, messages2], dim=-1)))
        h_updated = self.ln2(h_updated)

        # Acting planet: GNN embedding + raw features
        acting_gnn = torch.sum(h_updated * is_acting_mask, dim=1)   # (B, 64)
        acting_raw = torch.sum(x        * is_acting_mask, dim=1)    # (B, 23)

        # Full map representation
        flat_gnn_rep = torch.flatten(h_updated, start_dim=1)        # (B, N*64)

        return torch.cat([acting_gnn, acting_raw, flat_gnn_rep, global_features], dim=-1)





if __name__ == "__main__":
    print("Running Agent Server")
    agent =EventDrivenAllPlanetsGNNAgent()
    asyncio.run(GameServerAgent(host="0.0.0.0", port=8080, agent=agent).start())

