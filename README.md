# 🏓 Pong RL AI

A Deep Q-Network (DQN) agent that learns to play Pong using reinforcement learning. Built with PyTorch and Pygame, featuring real-time visualization of training progress.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎮 Features

- **Deep Q-Network (DQN)** with experience replay and target network
- **GPU Acceleration** - Automatically uses CUDA if available
- **Real-time Visualization** - Watch the AI learn during training
- **Live Training Metrics** - Reward plots updated in real-time
- **Pre-trained Models** - Ready-to-play trained models included

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/itsalhamzah/Pong-RL-AI.git
cd Pong-RL-AI

# Install dependencies
pip install -r requirements.txt
```

### Watch the AI Play

```bash
python play.py
```

### Train from Scratch

```bash
python train.py
```

## 📁 Project Structure

```
Pong-RL-AI/
├── config.py        # Hyperparameters and settings
├── dqn_agent.py     # DQN agent implementation
├── pong_env.py      # Pong game environment
├── train.py         # Training script with visualization
├── play.py          # Play with trained model
├── models/          # Saved model weights
│   ├── pong_best.pth
│   └── pong_final.pth
└── requirements.txt
```

## ⚙️ Configuration

Key hyperparameters in `config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `LEARNING_RATE` | 0.001 | Adam optimizer learning rate |
| `GAMMA` | 0.99 | Discount factor for future rewards |
| `EPSILON_DECAY` | 0.995 | Exploration rate decay |
| `BATCH_SIZE` | 64 | Experience replay batch size |
| `MEMORY_SIZE` | 100,000 | Replay buffer capacity |
| `HIDDEN_SIZE` | 256 | Neural network hidden layer size |

## 🧠 How It Works

The agent observes 6 state variables:
- Paddle Y position
- Opponent paddle Y position  
- Ball X and Y position
- Ball velocity (X and Y)

It learns to choose between 3 actions:
- **Stay** - Don't move
- **Up** - Move paddle up
- **Down** - Move paddle down

The DQN uses experience replay and a target network for stable learning.

## 📊 Training Progress

The training script displays:
- Real-time game visualization
- Episode rewards plot
- Epsilon (exploration rate) decay
- Best reward achieved

## 🎯 Results

After training, the AI learns to:
- Track the ball effectively
- Anticipate ball trajectory
- Position optimally for returns

## 📝 License

MIT License - feel free to use and modify!

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.
