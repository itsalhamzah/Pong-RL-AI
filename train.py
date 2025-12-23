"""
Pong RL Training Script with Real-Time Visualization
"""

import pygame
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import torch
import os
from datetime import datetime

from pong_env import PongEnv
from dqn_agent import DQNAgent
from config import DEVICE, FPS, TARGET_UPDATE, PLOT_UPDATE_FREQ


def create_training_plot(episode_rewards, avg_rewards, epsilons, losses):
    """Create training metrics plot as pygame surface"""
    fig, axes = plt.subplots(2, 2, figsize=(4.5, 4), facecolor='#0f0f19', dpi=100)
    
    for ax in axes.flat:
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='white', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#333355')
    
    # Episode rewards
    if episode_rewards:
        axes[0, 0].plot(episode_rewards, color='#00c8ff', alpha=0.5, linewidth=0.5)
        if avg_rewards:
            axes[0, 0].plot(avg_rewards, color='#ff3296', linewidth=2)
        axes[0, 0].set_title('Rewards', color='white', fontsize=10)
        axes[0, 0].set_xlabel('Episode', color='gray', fontsize=8)
    
    # Epsilon decay
    if epsilons:
        axes[0, 1].plot(epsilons, color='#32ff64', linewidth=2)
        axes[0, 1].set_title('Epsilon (Exploration)', color='white', fontsize=10)
        axes[0, 1].set_xlabel('Episode', color='gray', fontsize=8)
        axes[0, 1].set_ylim(0, 1)
    
    # Loss
    if losses:
        axes[1, 0].plot(losses[-500:], color='#ffaa00', alpha=0.7, linewidth=0.5)
        axes[1, 0].set_title('Training Loss', color='white', fontsize=10)
        axes[1, 0].set_xlabel('Step', color='gray', fontsize=8)
    
    # Stats
    axes[1, 1].axis('off')
    if episode_rewards:
        stats_text = f"""
Training Statistics
─────────────────
Episodes: {len(episode_rewards)}
Best Reward: {max(episode_rewards):.1f}
Avg (Last 50): {np.mean(episode_rewards[-50:]):.2f}
Win Rate: {sum(1 for r in episode_rewards[-50:] if r > 0) / min(50, len(episode_rewards)) * 100:.1f}%
GPU: {DEVICE.type.upper()}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, color='white', fontsize=9,
                       verticalalignment='center', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Convert to pygame surface
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    
    # Get raw data from canvas buffer (works with newer matplotlib)
    buf = canvas.buffer_rgba()
    size = canvas.get_width_height()
    
    plt.close(fig)
    
    # Create surface from RGBA buffer
    surf = pygame.image.frombuffer(buf, size, "RGBA")
    return surf


def train():
    """Main training loop with visualization"""
    print("=" * 60)
    print("[PONG] REINFORCEMENT LEARNING AI")
    print("=" * 60)
    print(f"[DEVICE] {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    
    # Initialize environment and agent
    env = PongEnv(render_mode=True)
    agent = DQNAgent()
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Try to load existing checkpoint to resume training
    start_episode = 0
    episode_rewards = []
    avg_rewards = []
    epsilons = []
    losses = []
    best_reward = float('-inf')
    
    checkpoint_path = "models/pong_checkpoint.pth"
    if os.path.exists(checkpoint_path):
        print(f"\n[RESUME] Found existing checkpoint, loading...")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        agent.policy_net.load_state_dict(checkpoint['policy_net'])
        agent.target_net.load_state_dict(checkpoint['target_net'])
        agent.optimizer.load_state_dict(checkpoint['optimizer'])
        agent.epsilon = checkpoint['epsilon']
        agent.training_step = checkpoint['training_step']
        start_episode = checkpoint.get('episode', 0)
        episode_rewards = checkpoint.get('episode_rewards', [])
        avg_rewards = checkpoint.get('avg_rewards', [])
        epsilons = checkpoint.get('epsilons', [])
        best_reward = checkpoint.get('best_reward', float('-inf'))
        print(f"[RESUME] Continuing from episode {start_episode + 1}")
        print(f"[RESUME] Current epsilon: {agent.epsilon:.3f}")
        print(f"[RESUME] Best reward so far: {best_reward:.1f}")
    else:
        print("\n[NEW] Starting fresh training session...")
    
    # Create combined display window
    pygame.init()
    display_width = 600 + 450  # Game (600) + Plots (450)
    display_height = 400
    display = pygame.display.set_mode((display_width, display_height))
    pygame.display.set_caption("Pong RL AI - Training in Progress")
    clock = pygame.time.Clock()
    
    num_episodes = 1000
    
    # Helper function to save full checkpoint
    def save_checkpoint(episode_num, is_final=False):
        checkpoint = {
            'policy_net': agent.policy_net.state_dict(),
            'target_net': agent.target_net.state_dict(),
            'optimizer': agent.optimizer.state_dict(),
            'epsilon': agent.epsilon,
            'training_step': agent.training_step,
            'episode': episode_num,
            'episode_rewards': episode_rewards,
            'avg_rewards': avg_rewards,
            'epsilons': epsilons,
            'best_reward': best_reward
        }
        torch.save(checkpoint, "models/pong_checkpoint.pth")
        if is_final:
            torch.save(checkpoint, "models/pong_final.pth")
        print(f"[SAVE] Checkpoint saved at episode {episode_num + 1}")
    
    # Initialize cached plot surface
    plot_surf = None
    
    print(f"\n[TRAINING] Started! Watch the AI learn to play Pong...")
    print(f"[INFO] Training will run from episode {start_episode + 1} to {num_episodes}\n")
    
    for episode in range(start_episode, num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        while not done:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    save_checkpoint(episode)
                    env.close()
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        save_checkpoint(episode)
                        env.close()
                        pygame.quit()
                        return
            
            # Select and execute action
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Store transition and learn
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
            
            state = next_state
            total_reward += reward
            step += 1
            
            # Render game to its surface
            env.render(episode=episode + 1, epsilon=agent.epsilon, total_reward=total_reward)
            
            # Get game surface and blit to display
            if env.screen is not None:
                display.blit(env.screen, (0, 0))
            
            # Keep showing cached plot surface (update only at episode end)
            if plot_surf is not None:
                display.blit(plot_surf, (600, 0))
            
            pygame.display.flip()
            clock.tick(FPS)
        
        # Update plot only at end of episode (eliminates flickering)
        plot_surf = create_training_plot(
            episode_rewards, avg_rewards, epsilons, losses
        )
        
        # End of episode
        episode_rewards.append(total_reward)
        epsilons.append(agent.epsilon)
        
        # Calculate moving average
        if len(episode_rewards) >= 50:
            avg_rewards.append(np.mean(episode_rewards[-50:]))
        else:
            avg_rewards.append(np.mean(episode_rewards))
        
        # Update epsilon
        agent.update_epsilon()
        
        # Update target network
        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()
        
        # Save best model and periodic checkpoint
        if total_reward > best_reward:
            best_reward = total_reward
            save_checkpoint(episode)
            agent.save("models/pong_best.pth")
        
        # Save checkpoint every 50 episodes
        if (episode + 1) % 50 == 0:
            save_checkpoint(episode)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode + 1:4d} | "
                  f"Reward: {total_reward:6.1f} | "
                  f"Avg(50): {avg_reward:6.2f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Loss: {agent.get_recent_loss():.4f}")
    
    # Training complete
    print("\n" + "=" * 60)
    print("[COMPLETE] TRAINING COMPLETE!")
    print(f"[STATS] Best Reward: {best_reward:.1f}")
    print(f"[STATS] Final Avg Reward (50): {np.mean(episode_rewards[-50:]):.2f}")
    print("=" * 60)
    
    save_checkpoint(num_episodes - 1, is_final=True)
    env.close()
    pygame.quit()


if __name__ == "__main__":
    train()
