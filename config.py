"""
Configuration and hyperparameters for Pong RL AI
"""

import torch

# Device configuration - Use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Game settings
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 60
BALL_SIZE = 10
PADDLE_SPEED = 8
BALL_SPEED = 7

# Neural Network architecture
STATE_SIZE = 6  # (paddle_y, opponent_y, ball_x, ball_y, ball_vx, ball_vy)
ACTION_SIZE = 3  # (stay, up, down)
HIDDEN_SIZE = 256

# Training hyperparameters
LEARNING_RATE = 0.001
GAMMA = 0.99  # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
MEMORY_SIZE = 100000
TARGET_UPDATE = 10  # Update target network every N episodes

# Visualization settings
FPS = 60
RENDER_EVERY = 1  # Render every N frames during training
PLOT_UPDATE_FREQ = 10  # Update plots every N episodes
