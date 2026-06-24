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
        self.n_planets = n_planets
        self.n_globals = 4 
        
        # Calculate features per node by removing globals first
        total_obs_size = observation_space.shape[0]
        self.features_per_node = (total_obs_size - self.n_globals) // self.n_planets
        assert self.features_per_node == 25
        # --- HIDDEN DIMENSIONS (The Funnel Strategy) ---
        self.node_feature_dim = self.features_per_node 
        self.hidden_dim = 256
        self.hidden_dim_3 = 128
        self.gnn_output_dim = 96 # Αυξημένο για να κρατήσει την πολύπλοκη χωρική πληροφορία

        # ΝΕΟ ΜΕΓΕΘΟΣ: 
        # [Actor GNN (96)] + [Actor Raw (22)] + [All N Planets (N * 96)] + [Globals (4)]
        actual_features_dim = self.gnn_output_dim + self.features_per_node + (self.n_planets * self.gnn_output_dim) + self.n_globals   
        super().__init__(observation_space, actual_features_dim)

        self.spatial_decay = nn.Parameter(torch.tensor([2.0]))
        
        # --- EMBEDDING ---
        self.embed = nn.Linear(self.node_feature_dim, self.hidden_dim)
        self.ln_embed = nn.LayerNorm(self.hidden_dim) 
        
        # --- LAYER 1 (256 -> 256) ---
        self.update1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.ln1 = nn.LayerNorm(self.hidden_dim)      
        
        # --- LAYER 2 (256 -> 256) ---
        self.update2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)
        
        # --- LAYER 3 (256 -> 128) Funnel Down ---
        self.update3 = nn.Linear(self.hidden_dim * 2, self.hidden_dim_3)
        self.ln3 = nn.LayerNorm(self.hidden_dim_3) 

        # --- LAYER 4 (128 -> 96) Final Output ---
        self.update4 = nn.Linear(self.hidden_dim_3 * 2, self.gnn_output_dim)
        self.ln4 = nn.LayerNorm(self.gnn_output_dim) 

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        B = observations.shape[0]
        
        # 1. Isolate Globals and Planet Features
        global_features = observations[:, -self.n_globals:] 
        planet_obs_flat = observations[:, :-self.n_globals] 
        
        x = planet_obs_flat.view(B, self.n_planets, self.features_per_node)
        
        # 2. Safe Slicing
        coords = x[:, :, -2:]            
        is_acting_mask = x[:, :, -3].unsqueeze(-1) 
        
        # Το άθροισμα των πρώτων 3 features (Owner One-Hot) είναι 1.0 για αληθινούς, 0.0 για άδειους
        valid_nodes_mask = x[:, :, 0:3].sum(dim=-1, keepdim=True) # [B, N, 1]
        
        # Μάσκα γειτνίασης: Αν έστω και ένας από τους 2 πλανήτες είναι φάντασμα, το link γίνεται 0.0
        adj_mask = valid_nodes_mask * valid_nodes_mask.transpose(1, 2) # [B, N, N]
        
        # --- Initial Embedding ---
        h = torch.relu(self.embed(x))
        h = self.ln_embed(h) 
        h = h * valid_nodes_mask # Σκοτώνουμε τα biases των φαντασμάτων
        
        # 3. Spatial Attention / Adjacency Matrix
        decay = torch.clamp(self.spatial_decay, min=0.1, max=10.0)

        dx = coords[:, :, 0:1] - coords[:, :, 0:1].transpose(1, 2)
        dy = coords[:, :, 1:2] - coords[:, :, 1:2].transpose(1, 2)

        dist_sq = dx.square() + dy.square()

        adj_weights = torch.exp(-dist_sq * decay)
        
        # Εφαρμογή της μάσκας ΠΡΙΝ το normalization! 
        adj_weights = adj_weights * adj_mask
        
        # Normalization (το 1e-6 σώζει από διαίρεση με το μηδέν αν το row έχει μόνο φαντάσματα)
        adj_weights = adj_weights / (adj_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        # --- 4. Deep Message Passing with Residuals ---
        
        # LAYER 1
        messages1 = torch.bmm(adj_weights, h)
        h1_new = torch.relu(self.update1(torch.cat([h, messages1], dim=-1)))
        h1 = self.ln1(h1_new + h) 
        h1 = h1 * valid_nodes_mask # SANITIZE
        
        # LAYER 2
        messages2 = torch.bmm(adj_weights, h1)
        h2_new = torch.relu(self.update2(torch.cat([h1, messages2], dim=-1))) 
        h2 = self.ln2(h2_new + h1) 
        h2 = h2 * valid_nodes_mask # SANITIZE

        # LAYER 3
        messages3 = torch.bmm(adj_weights, h2)
        h3 = torch.relu(self.update3(torch.cat([h2, messages3], dim=-1))) 
        h3 = self.ln3(h3)
        h3 = h3 * valid_nodes_mask # SANITIZE
        
        # LAYER 4
        messages4 = torch.bmm(adj_weights, h3)
        h_updated = torch.relu(self.update4(torch.cat([h3, messages4], dim=-1))) 
        h_updated = self.ln4(h_updated)
        h_updated = h_updated * valid_nodes_mask # SANITIZE
        
        # --- 5. Output Construction ---        
        acting_gnn = torch.sum(h_updated * is_acting_mask, dim=1)  # [Batch, 96]
        acting_raw = torch.sum(x * is_acting_mask, dim=1)          # [Batch, 23]
        
        # Τα φαντάσματα είναι τώρα ΟΛΟΚΛΗΡΩΤΙΚΑ ΜΗΔΕΝ. 
        # Όταν γίνουν flatten, το policy head θα πάρει τέλεια, καθαρά μηδενικά.
        flat_gnn_rep = torch.flatten(h_updated, start_dim=1)       # [Batch, N * 96]
        
        output = torch.cat([acting_gnn, acting_raw, flat_gnn_rep, global_features], dim=-1)
        
        return output

