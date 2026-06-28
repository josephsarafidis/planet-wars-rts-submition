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
    
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(754,), dtype=np.float32)
    act_space = spaces.Discrete(151)
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
    def __init__(self, model_path: str = "clean_weights.pth", max_planets: int = 30, det: bool = True, heuristic_fallback: bool = True):
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
        self.features_per_node = 25
        self.n_globals = 4
        self.fleet_buckets = [0.1, 0.25, 0.5, 0.75, 1.0] 
        
        self.heuristic_enabled = heuristic_fallback
        self.attack_action_bins = self.max_planets * len(self.fleet_buckets)
        self.noop_action_idx = self.attack_action_bins
        self.total_action_bins = self.attack_action_bins + 1
        
        self.planet_cooldown = 50
        
        # --- INTERNAL MEMORY ---
        self.planet_ready_tick = None
        self.max_eta = None

    def prepare_to_play_as(self, player: Player, params: GameParams, opponent: Optional[str] = None):
        super().prepare_to_play_as(player, params, opponent)
        self.params = params
        self.player = player
        self.planet_ready_tick = None
        self.max_eta = np.sqrt(self.params.width**2 + self.params.height**2) / self.params.transporter_speed
    # ==========================================
    # ACTION MASKING LOGIC 
    # ==========================================
    def is_action_allowed(self, target_idx: int, ratio_idx: int, active_planet, game_state: GameState, obs_mapping: dict) -> bool:
        if target_idx == 0: 
            return False # Cannot attack self (slot 0 is always active_planet)
        
        real_target_id = obs_mapping.get(target_idx, None)
        if real_target_id is None: 
            return False
        
        target_planet = next((p for p in game_state.planets if p.id == real_target_id), None)
        if not target_planet or target_planet.id == active_planet.id: 
            return False
        
        ratio_to_send = self.fleet_buckets[ratio_idx]
        if int(active_planet.n_ships * ratio_to_send) <= 1: 
            return False
            
        return True

    def get_action_mask(self, active_planet, game_state: GameState, obs_mapping: dict) -> np.ndarray:
        masks = np.zeros(self.total_action_bins, dtype=np.int8)
        
        # NOOP is always allowed
        masks[self.noop_action_idx] = 1
        
        for target_idx in range(self.max_planets):
            for ratio_idx in range(len(self.fleet_buckets)):
                if self.is_action_allowed(target_idx, ratio_idx, active_planet, game_state, obs_mapping):
                    flat_action_idx = (target_idx * len(self.fleet_buckets)) + ratio_idx
                    masks[flat_action_idx] = 1
        return masks

    # ==========================================
    # OBSERVATION EXTRACTION
    # ==========================================
    def _get_sorted_planets_by_distance(self, game_state: GameState, active_planet):
        if active_planet is None:
            return sorted(game_state.planets, key=lambda p: p.id)

        other_planets = [p for p in game_state.planets if p.id != active_planet.id]
        other_planets.sort(
            key=lambda p: (active_planet.position.distance(p.position), p.id)
        )
        return [active_planet] + other_planets

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

        # --- EGOCENTRIC SORTING ---
        ordered_planets = self._get_sorted_planets_by_distance(game_state, active_planet)
        
        max_coord = max(self.params.width, self.params.height)
        n_planets_current = len(ordered_planets)

        for i, planet in enumerate(ordered_planets):
            if i >= self.max_planets: break
            mapping[i] = planet.id
            idx = i * self.features_per_node
            
            obs[idx]   = 1.0 if planet.owner == self.player else 0.0
            obs[idx+1] = 1.0 if planet.owner == self.player.opponent() else 0.0
            obs[idx+2] = 1.0 if planet.owner == Player.Neutral else 0.0
            obs[idx+3] = np.log1p(float(planet.n_ships))
            
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
                    obs[idx + 8 + (f_idx * 2)] = (eta / self.max_eta)
                else:
                    obs[idx + 7 + (f_idx * 2)] = 0.0
                    obs[idx + 8 + (f_idx * 2)] = 0.0

            e_fleets = incoming_enemy_fleets[planet.id]
            for e_idx in range(3):
                if e_idx < len(e_fleets):
                    ships, eta = e_fleets[e_idx]
                    obs[idx + 13 + (e_idx * 2)] = np.log1p(max(0.0, ships))
                    obs[idx + 14 + (e_idx * 2)] = (eta / self.max_eta)
                else:
                    obs[idx + 13 + (e_idx * 2)] = 0.0
                    obs[idx + 14 + (e_idx * 2)] = 0.0

            obs[idx + 19] = float(planet.growth_rate) / self.params.max_growth_rate
            
            total_inc_e = sum(ships for ships, eta in incoming_enemy_fleets.get(planet.id, []))
            obs[idx + 20] = 1.0 if (total_inc_e > 0 and planet.owner == self.player) else 0.0
            
            distance_to_active = active_planet.position.distance(planet.position)
            obs[idx + 21] = distance_to_active / max_coord
            obs[idx + 22] = 1.0 if planet.id == active_planet.id else 0.0

            obs[idx + 23] = planet.position.x / max_coord
            obs[idx + 24] = planet.position.y / max_coord
            
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

        # 1. Map incoming enemy fleets
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
            if p.n_ships >= 1
            and getattr(p, 'transporter', None) is None 
            and current_tick >= self.planet_ready_tick.get(p.id, 0)
        ]
        
        non_friendly_planets = [p for p in game_state.planets if p.owner != self.player]
        if self.heuristic_enabled:
            best_attack = None
            best_roi_score = -1.0 

            for source in valid_sources:
                minimum_safe_garrison = incoming_enemy_ships[source.id] + 3

                for target in non_friendly_planets:
                    if incoming_friendly_ships[target.id] > 0:
                        continue
                    
                    eta = source.position.distance(target.position) / self.params.transporter_speed
                    growth_defense = target.growth_rate * eta if target.owner == self.player.opponent() else 0
                    target_defense = target.n_ships + incoming_enemy_ships[target.id] + growth_defense
                    required_attack_ships = int(target_defense) + 2
                    
                    if source.n_ships >= (required_attack_ships + minimum_safe_garrison):
                        roi_score = (target.growth_rate * 5000.0) / (required_attack_ships * eta)
                        if target.owner == Player.Neutral:
                            roi_score *= 1.5 
                            
                        if roi_score > best_roi_score:
                            best_roi_score = roi_score
                            best_attack = (source, target, required_attack_ships)

            if best_attack and best_roi_score > 1.0:
                source, target, required_attack_ships = best_attack
                self.planet_ready_tick[source.id] = current_tick + 5
                return Action(
                    player_id=self.player,
                    source_planet_id=source.id,
                    destination_planet_id=target.id,
                    num_ships=required_attack_ships
                )

        # ==========================================
        # 3. AI FALLBACK LOGIC (EGOCENTRIC EVALUATION)
        # ==========================================
        # Evaluate active planet
        if valid_sources:
            active_planet = min(valid_sources, key=lambda p: self.planet_ready_tick[p.id])
    
            obs, obs_mapping = self._get_obs(game_state, active_planet)
            action_mask = self.get_action_mask(active_planet, game_state, obs_mapping)
            
            action_flat, _ = self.model.predict(obs, action_masks=action_mask, deterministic=self.det)
            action_int = int(action_flat)
            
            if action_int == self.noop_action_idx:
                self.planet_ready_tick[active_planet.id] = current_tick + self.planet_cooldown
                return Action.do_nothing()
            
            sorted_target_idx = action_int // len(self.fleet_buckets)
            ratio_idx = action_int % len(self.fleet_buckets)
            
            real_target_id = obs_mapping.get(sorted_target_idx, None)
            
            if real_target_id is not None and real_target_id != active_planet.id:
                ratio_to_send = self.fleet_buckets[ratio_idx]
                ships_to_send = int(active_planet.n_ships * ratio_to_send)
                
                if ships_to_send >= 1:
                    self.planet_ready_tick[active_planet.id] = current_tick + self.planet_cooldown
                    return Action(
                        player_id=self.player,
                        source_planet_id=active_planet.id,
                        destination_planet_id=real_target_id,
                        num_ships=ships_to_send
                    )
        else:
            return Action.do_nothing()
        print("NOOP")
        return Action.do_nothing()

    def get_agent_type(self) -> str:
        return self.agent_type

