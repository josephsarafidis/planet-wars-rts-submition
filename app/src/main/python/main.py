import asyncio
from client_server.game_agent_server import GameServerAgent
from agents.gnn_agent import EventDrivenAllPlanetsGNNAgent
from agents.random_agents import PureRandomAgent
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import torch
import torch.nn as nn
 


if __name__ == "__main__":
    print("Running Agent Server")
    agent =EventDrivenAllPlanetsGNNAgent()
    asyncio.run(GameServerAgent(host="0.0.0.0", port=8080, agent=agent).start())

