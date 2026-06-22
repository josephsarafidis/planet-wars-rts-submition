import os
import numpy as np
import gymnasium as gym
from typing import Optional
from sb3_contrib import MaskablePPO
from core.game_state import GameParams, Player, Action, GameState
from agents.planet_wars_agent import PlanetWarsPlayer
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from gymnasium import spaces 
import torch
import torch.nn as nn
import importlib
from pathlib import Path
import traceback
import random

from model_architecture import SpatialGNNExtractor 

N_PLANETS = 30



import zipfile
import pickle
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO

# Φτιάχνουμε μια κλάση-μαϊμού που συμπεριφέρεται σαν περιβάλλον
class DummyTrainingEnv(gym.Env):
    def __init__(self, obs_space, act_space):
        self.observation_space = obs_space
        self.action_space = act_space
    def reset(self, seed=None, options=None): return np.zeros(self.observation_space.shape, dtype=np.float32), {}
    def step(self, action): return np.zeros(self.observation_space.shape), 0, False, False, {}



import zipfile
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO

def manual_load_model(model_path):
    print("Initializing fresh model architecture...", flush=True)
    
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(844,), dtype=np.float32)
    act_space = spaces.Discrete(165)
    dummy_env = DummyVecEnv([lambda: DummyTrainingEnv(obs_space, act_space)])
    

    
    policy_kwargs = {
        "features_extractor_class": SpatialGNNExtractor,
        "features_extractor_kwargs": {"n_planets": 30},
        "net_arch": {"pi": [1024, 512, 256], "vf": [1024, 512, 256]}
    }
    
    new_model = MaskablePPO(
        "MlpPolicy", 
        dummy_env,
        policy_kwargs=policy_kwargs,
        device="cpu"
    )
    # 2. Φορτώνεις τα weights κατευθείαν (χωρίς pickle, χωρίς classes, χωρίς imports)
    state_dict = torch.load(model_path, map_location='cpu')
    new_model.policy.load_state_dict(state_dict)
    
    return new_model
    
    
    

class GNNAgent(PlanetWarsPlayer):
    def __init__(self, model_path: str ="clean_weights.pth", max_planets: int = 30, det: bool = True):
        super().__init__()
        self.model_name = os.path.basename(model_path)
        try:
            print("Loading model...")


            BASE_DIR = Path(__file__).resolve().parent.parent
            MODEL_PATH = BASE_DIR / "models" / "clean_weights.pth"
            
            self.model = manual_load_model(str(MODEL_PATH))
                    
            print(f"Loaded Advanced GNN Model successfully from {model_path}")
            self.agent_type = self.model_name 

        except Exception:
            traceback.print_exc()
            raise

        self.max_planets = max_planets
        self.det = det
        
        # --- ARCHITECTURE CONFIG ---
        self.features_per_node = 28 # 24 + 4 zeroed out
        self.n_globals = 4
        self.fleet_buckets = [0.1, 0.25, 0.5, 0.75, 1.0] # The 5 buckets
        self.old_flag_count = 5 # Kept to maintain the action bins size
        
        self.base_action_bins = (self.max_planets + 1) * len(self.fleet_buckets)
        self.meta_action_bins = self.old_flag_count * 2
        self.total_action_bins = self.base_action_bins + self.meta_action_bins
        
        self.planet_cooldown = 50
        
        # --- INTERNAL MEMORY ---
        self.planet_ready_tick = None

    def prepare_to_play_as(self, player: Player, params: GameParams, opponent: Optional[str] = None):
        super().prepare_to_play_as(player, params, opponent)
        self.params = params
        self.player = player
        self.planet_ready_tick = None

    # ==========================================
    # ACTION MASKING LOGIC 
    # ==========================================
    def is_action_allowed(self, target_idx: int, ratio_idx: int, active_planet, game_state: GameState, obs_mapping: dict) -> bool:
        if target_idx == self.max_planets: 
            return True # NOOP is always allowed
        
        real_target_id = obs_mapping.get(target_idx, None)
        if real_target_id is None: 
            return False
        
        target_planet = next((p for p in game_state.planets if p.id == real_target_id), None)
        if not target_planet or target_planet.id == active_planet.id: 
            return False
        
        ratio_to_send = self.fleet_buckets[ratio_idx]
        if int(active_planet.n_ships * ratio_to_send) < 1: 
            return False
            
        return True

    def get_action_mask(self, active_planet, game_state: GameState, obs_mapping: dict) -> np.ndarray:
        masks = np.zeros(self.total_action_bins, dtype=np.int8)
        
        # Base Dispatch Actions
        for target_idx in range(self.max_planets + 1):
            for ratio_idx in range(len(self.fleet_buckets)):
                if self.is_action_allowed(target_idx, ratio_idx, active_planet, game_state, obs_mapping):
                    flat_action_idx = (target_idx * len(self.fleet_buckets)) + ratio_idx
                    masks[flat_action_idx] = 1

        # Fallback for safety
        if not masks.any():
            for r in range(len(self.fleet_buckets)):
                masks[(self.max_planets * len(self.fleet_buckets)) + r] = 1
        return masks

    # ==========================================
    # OBSERVATION EXTRACTION
    # ==========================================
    def _get_obs(self, game_state: GameState, active_planet) -> tuple[np.ndarray, dict]:
        obs = np.zeros((self.max_planets * self.features_per_node) + self.n_globals, dtype=np.float32)
        mapping = {}
        
        inc_f = {p.id: 0.0 for p in game_state.planets}
        inc_e = {p.id: 0.0 for p in game_state.planets}
        incoming_friendly_fleets = {p.id: [] for p in game_state.planets}
        incoming_enemy_fleets    = {p.id: [] for p in game_state.planets}
        
        global_f_ships = global_e_ships = global_f_prod = global_e_prod = 0.0

        for p in game_state.planets:
            if p.owner == self.player:
                global_f_ships += p.n_ships
                global_f_prod += p.growth_rate
            elif p.owner == self.player.opponent():
                global_e_ships += p.n_ships
                global_e_prod += p.growth_rate

            if getattr(p, 'transporter', None) is not None:
                t = p.transporter
                dest = game_state.planets[t.destination_index]
                eta = p.position.distance(dest.position) / self.params.transporter_speed
                
                if t.owner == self.player:
                    inc_f[dest.id] += t.n_ships
                    incoming_friendly_fleets[dest.id].append((t.n_ships, eta))
                    global_f_ships += t.n_ships 
                else:
                    inc_e[dest.id] += t.n_ships
                    incoming_enemy_fleets[dest.id].append((t.n_ships, eta))
                    global_e_ships += t.n_ships 

        for p in game_state.planets:
            incoming_friendly_fleets[p.id].sort(key=lambda x: x[1])
            incoming_enemy_fleets[p.id].sort(key=lambda x: x[1])

        ordered_planets = sorted(game_state.planets, key=lambda p: p.id)
        max_coord = max(self.params.width, self.params.height)
        max_time = float(self.params.max_ticks)
        n_planets_current = len(ordered_planets)

        for i, planet in enumerate(ordered_planets):
            if i >= self.max_planets: break
            mapping[i] = planet.id
            idx = i * self.features_per_node
            
            obs[idx]   = 1.0 if planet.owner == self.player else 0.0
            obs[idx+1] = 1.0 if planet.owner == self.player.opponent() else 0.0
            obs[idx+2] = 1.0 if planet.owner == Player.Neutral else 0.0
            
            safe_ships = max(0.0, float(planet.n_ships))
            obs[idx + 3] = np.log1p(safe_ships)
            
            estimated_ships = float(planet.n_ships)
            if planet.owner == self.player:
                estimated_ships += inc_f[planet.id] - inc_e[planet.id] 
            elif planet.owner == self.player.opponent():
                estimated_ships -= inc_f[planet.id] - inc_e[planet.id] 
            else:
                estimated_ships -= inc_e[planet.id] + inc_f[planet.id] 

            obs[idx + 4] = np.sign(estimated_ships) * np.log1p(np.abs(estimated_ships))
            obs[idx + 5] = np.log1p(max(0.0, inc_f[planet.id]))
            obs[idx + 6] = np.log1p(max(0.0, inc_e[planet.id]))

            f_fleets = incoming_friendly_fleets[planet.id]
            for f_idx in range(3):
                if f_idx < len(f_fleets):
                    ships, eta = f_fleets[f_idx]
                    obs[idx + 7 + (f_idx * 2)] = np.log1p(max(0.0, ships))
                    obs[idx + 8 + (f_idx * 2)] = 1.0 - (eta / max_time) 
                else:
                    obs[idx + 7 + (f_idx * 2)] = 0.0
                    obs[idx + 8 + (f_idx * 2)] = 0.0

            e_fleets = incoming_enemy_fleets[planet.id]
            for e_idx in range(3):
                if e_idx < len(e_fleets):
                    ships, eta = e_fleets[e_idx]
                    obs[idx + 13 + (e_idx * 2)] = np.log1p(max(0.0, ships))
                    obs[idx + 14 + (e_idx * 2)] = 1.0 - (eta / max_time)
                else:
                    obs[idx + 13 + (e_idx * 2)] = 0.0
                    obs[idx + 14 + (e_idx * 2)] = 0.0

            obs[idx + 19] = float(planet.growth_rate) / self.params.max_growth_rate
            
            # --- Feature Repurposing matching the Training Arch ---
            cd_left = max(0, self.planet_ready_tick.get(planet.id, 0) - game_state.game_tick)
            obs[idx + 20] = min(1.0, cd_left / 50.0)
            
            for f_idx in range(self.old_flag_count - 1):
                obs[idx + 21 + f_idx] = 0.0
                
            obs[idx + 25] = 1.0 if planet.id == active_planet.id else 0.0
            obs[idx + 26] = planet.position.x / max_coord
            obs[idx + 27] = planet.position.y / max_coord
            
        for i in range(n_planets_current, self.max_planets):
            idx = i * self.features_per_node
            obs[idx : idx + self.features_per_node] = 0.0

        global_idx = self.max_planets * self.features_per_node
        obs[global_idx] = np.log1p(max(0.0, global_f_ships))
        obs[global_idx + 1] = np.log1p(max(0.0, global_e_ships))
        obs[global_idx + 2] = global_f_prod / (self.params.max_growth_rate * self.max_planets)
        obs[global_idx + 3] = global_e_prod / (self.params.max_growth_rate * self.max_planets)
            
        return obs, mapping

    # ==========================================
    # PURE AI DECISION LOOP
    # ==========================================
    def get_action(self, game_state: GameState) -> Action:
        if self.model is None:
            return Action.do_nothing()
        
        current_tick = game_state.game_tick

        if self.planet_ready_tick is None:
            self.planet_ready_tick = {p.id: 0 for p in game_state.planets}

        friendly_planets = [p for p in game_state.planets if p.owner == self.player]

        # 1. Map incoming enemy fleets to destinations and track total fleets
        incoming_enemy_ships = {p.id: 0 for p in game_state.planets}
        incoming_friendly_ships = {p.id: 0 for p in game_state.planets}

        current_enemy_fleets = 0
        
        for p in game_state.planets:
            if getattr(p, 'transporter', None) is not None and p.transporter.owner == self.player.opponent():
                current_enemy_fleets += 1
                incoming_enemy_ships[p.transporter.destination_index] += p.transporter.n_ships
            elif getattr(p, 'transporter', None) is not None and p.transporter.owner == self.player:
                incoming_friendly_ships[p.transporter.destination_index] += p.transporter.n_ships


        
        # ==========================================
        # 2. HEURISTIC OVERRIDE LOGIC: OPPORTUNISTIC SNIPER
        # ==========================================
        valid_sources = [
            p for p in friendly_planets 
            if p.n_ships > 5 
            and getattr(p, 'transporter', None) is None 
            and current_tick >= self.planet_ready_tick.get(p.id, 0)
        ]
        
        non_friendly_planets = [p for p in game_state.planets if p.owner != self.player]
        
        best_attack = None
        best_roi_score = -1.0  # Threshold for "is this attack worth overriding the NN?"

        for source in valid_sources:
            # Localized Danger Check: Ensure THIS source planet is safe, ignore the rest of the board.
            # We keep enough ships to survive incoming enemies, plus a buffer of 3 just in case.
            minimum_safe_garrison = incoming_enemy_ships[source.id] + 3

            for target in non_friendly_planets:
                # If we are already sending ships there, trust the existing fleet and move on
                if incoming_friendly_ships[target.id] > 0:
                    continue
                
                eta = source.position.distance(target.position) / self.params.transporter_speed
                
                growth_defense = target.growth_rate * eta if target.owner == self.player.opponent() else 0
                target_defense = target.n_ships + incoming_enemy_ships[target.id] + growth_defense
                
                required_attack_ships = int(target_defense) + 2
                
                # Check if we have enough to attack AND defend our home simultaneously
                if source.n_ships >= (required_attack_ships + minimum_safe_garrison):
                    
                    # --- Calculate ROI (Return on Investment) ---
                    # High growth, cheap required ships, and short distance score the highest.
                    roi_score = (target.growth_rate * 5000.0) / (required_attack_ships * eta)
                    #print(required_attack_ships, eta)
                    # Neutral planets don't spawn enemy fleets, making them safer investments early game
                    if target.owner == Player.Neutral:
                        roi_score *= 1.5 
                    #print(roi_score)
                    # Find the absolute best move on the board
                    if roi_score > best_roi_score:
                        best_roi_score = roi_score
                        best_attack = (source, target, required_attack_ships)

        # Execute ONLY if we found a highly profitable move (adjust 2.0 threshold based on testing)
        if best_attack and best_roi_score > 1:
            source, target, required_attack_ships = best_attack
            
            self.planet_ready_tick[source.id] = current_tick + 5
            
                
            #print(f"[Heuristic] Sniping planet {target.id} (Growth: {target.growth_rate}) with {required_attack_ships} ships from {source.id}. ROI: {best_roi_score:.2f}")
            return Action(
                player_id=self.player,
                source_planet_id=source.id,
                destination_planet_id=target.id,
                num_ships=required_attack_ships
            )

        # ==========================================
        # 3. AI FALLBACK LOGIC
        # ==========================================
        
        # 1. Βρίσκουμε όλους τους φίλιους πλανήτες που ΜΠΟΡΟΥΝ να δράσουν αυτό το tick.
        valid_sources = [
            p for p in game_state.planets 
            if p.owner == self.player 
            and p.n_ships >= 1 
            and getattr(p, 'transporter', None) is None 
            and current_tick >= self.planet_ready_tick.get(p.id, 0)
        ]

        # 2. Ανακατεύουμε για να μην ευνοείται πάντα ο πλανήτης με ID 0
        random.shuffle(valid_sources)

        # 3. Ρωτάμε το GNN για τον καθένα ξεχωριστά
        for active_planet in valid_sources:
            obs, obs_mapping = self._get_obs(game_state, active_planet)
            action_mask = self.get_action_mask(active_planet, game_state, obs_mapping)
            
            action_flat, _ = self.model.predict(obs, action_masks=action_mask, deterministic=self.det)
            action_int = int(action_flat)
            
            if action_int < self.base_action_bins:
                sorted_target_idx = action_int // len(self.fleet_buckets)
                ratio_idx = action_int % len(self.fleet_buckets)
                
                # Αν διάλεξε πλανήτη (Άρα Επίθεση/Μεταφορά)
                if sorted_target_idx < self.max_planets:
                    real_target_id = obs_mapping.get(sorted_target_idx, None)
                    
                    if real_target_id is not None and real_target_id != active_planet.id:
                        ratio_to_send = self.fleet_buckets[ratio_idx]
                        ships_to_send = int(active_planet.n_ships * ratio_to_send)
                        
                        if ships_to_send >= 1:
                            # Επιτυχημένο action: Βάζουμε το 5-tick cooldown και επιστρέφουμε!
                            self.planet_ready_tick[active_planet.id] = current_tick + 5
                            #print(real_target_id, ships_to_send, game_state.game_tick)
                            return Action(
                                player_id=self.player,
                                source_planet_id=active_planet.id,
                                destination_planet_id=real_target_id,
                                num_ships=ships_to_send
                            )
                            
                self.planet_ready_tick[active_planet.id] = current_tick + 5
            else:
                # Masked Meta Actions (Safeguard)
                self.planet_ready_tick[active_planet.id] = current_tick + 5

        # Αν φτάσαμε εδώ, όλοι οι έτοιμοι πλανήτες επέλεξαν NOOP.
        return Action.do_nothing()

    def get_agent_type(self) -> str:
        return self.agent_type

    
    
    
