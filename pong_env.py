"""
Pong Game Environment for Reinforcement Learning
"""

import pygame
import numpy as np
import random
from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, PADDLE_WIDTH, PADDLE_HEIGHT,
    BALL_SIZE, PADDLE_SPEED, BALL_SPEED
)


class PongEnv:
    """Custom Pong environment with Gym-like API"""
    
    def __init__(self, render_mode=True):
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        # Colors
        self.BLACK = (15, 15, 25)
        self.WHITE = (255, 255, 255)
        self.NEON_BLUE = (0, 200, 255)
        self.NEON_PINK = (255, 50, 150)
        self.NEON_GREEN = (50, 255, 100)
        
        if self.render_mode:
            pygame.init()
            pygame.display.set_caption("🎮 Pong RL AI - Training")
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
        
        self.reset()
    
    def reset(self):
        """Reset the game state"""
        # Player paddle (AI) - right side
        self.player_y = SCREEN_HEIGHT // 2 - PADDLE_HEIGHT // 2
        
        # Opponent paddle - left side (simple AI)
        self.opponent_y = SCREEN_HEIGHT // 2 - PADDLE_HEIGHT // 2
        
        # Ball
        self.ball_x = SCREEN_WIDTH // 2
        self.ball_y = SCREEN_HEIGHT // 2
        self.ball_vx = BALL_SPEED * random.choice([-1, 1])
        self.ball_vy = BALL_SPEED * random.uniform(-0.5, 0.5)
        
        # Scores
        self.player_score = 0
        self.opponent_score = 0
        
        return self._get_state()
    
    def _get_state(self):
        """Get normalized state representation"""
        state = np.array([
            self.player_y / SCREEN_HEIGHT,
            self.opponent_y / SCREEN_HEIGHT,
            self.ball_x / SCREEN_WIDTH,
            self.ball_y / SCREEN_HEIGHT,
            self.ball_vx / BALL_SPEED,
            self.ball_vy / BALL_SPEED
        ], dtype=np.float32)
        return state
    
    def step(self, action):
        """
        Execute one step in the environment
        action: 0 = stay, 1 = up, 2 = down
        Returns: (state, reward, done, info)
        """
        # Move player paddle based on action
        if action == 1:  # Up
            self.player_y = max(0, self.player_y - PADDLE_SPEED)
        elif action == 2:  # Down
            self.player_y = min(SCREEN_HEIGHT - PADDLE_HEIGHT, self.player_y + PADDLE_SPEED)
        
        # Simple opponent AI - follows ball
        opponent_center = self.opponent_y + PADDLE_HEIGHT // 2
        if opponent_center < self.ball_y - 10:
            self.opponent_y = min(SCREEN_HEIGHT - PADDLE_HEIGHT, self.opponent_y + PADDLE_SPEED * 0.7)
        elif opponent_center > self.ball_y + 10:
            self.opponent_y = max(0, self.opponent_y - PADDLE_SPEED * 0.7)
        
        # Move ball
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy
        
        reward = 0
        done = False
        
        # Ball collision with top/bottom walls
        if self.ball_y <= 0 or self.ball_y >= SCREEN_HEIGHT - BALL_SIZE:
            self.ball_vy = -self.ball_vy
            self.ball_y = max(0, min(SCREEN_HEIGHT - BALL_SIZE, self.ball_y))
        
        # Ball collision with player paddle (right)
        if (self.ball_x + BALL_SIZE >= SCREEN_WIDTH - PADDLE_WIDTH - 10 and
            self.player_y <= self.ball_y <= self.player_y + PADDLE_HEIGHT):
            self.ball_vx = -abs(self.ball_vx)
            # Add some spin based on where ball hits paddle
            hit_pos = (self.ball_y - self.player_y) / PADDLE_HEIGHT
            self.ball_vy = BALL_SPEED * (hit_pos - 0.5) * 1.5
            reward = 0.5  # Small reward for hitting the ball
        
        # Ball collision with opponent paddle (left)
        if (self.ball_x <= PADDLE_WIDTH + 10 and
            self.opponent_y <= self.ball_y <= self.opponent_y + PADDLE_HEIGHT):
            self.ball_vx = abs(self.ball_vx)
            hit_pos = (self.ball_y - self.opponent_y) / PADDLE_HEIGHT
            self.ball_vy = BALL_SPEED * (hit_pos - 0.5) * 1.5
        
        # Scoring
        if self.ball_x <= 0:  # Player scores
            self.player_score += 1
            reward = 1.0
            self._reset_ball(direction=1)
        elif self.ball_x >= SCREEN_WIDTH:  # Opponent scores
            self.opponent_score += 1
            reward = -1.0
            self._reset_ball(direction=-1)
        
        # Check if game is done (first to 5)
        if self.player_score >= 5 or self.opponent_score >= 5:
            done = True
        
        info = {
            'player_score': self.player_score,
            'opponent_score': self.opponent_score
        }
        
        return self._get_state(), reward, done, info
    
    def _reset_ball(self, direction=1):
        """Reset ball to center with given direction"""
        self.ball_x = SCREEN_WIDTH // 2
        self.ball_y = SCREEN_HEIGHT // 2
        self.ball_vx = BALL_SPEED * direction
        self.ball_vy = BALL_SPEED * random.uniform(-0.5, 0.5)
    
    def render(self, episode=0, epsilon=0, total_reward=0):
        """Render the game state"""
        if not self.render_mode or self.screen is None:
            return
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        
        # Clear screen with dark background
        self.screen.fill(self.BLACK)
        
        # Draw center line
        for y in range(0, SCREEN_HEIGHT, 20):
            pygame.draw.rect(self.screen, (50, 50, 70), 
                           (SCREEN_WIDTH // 2 - 2, y, 4, 10))
        
        # Draw paddles with glow effect
        # Player paddle (right) - neon blue
        pygame.draw.rect(self.screen, self.NEON_BLUE,
                        (SCREEN_WIDTH - PADDLE_WIDTH - 10, self.player_y, 
                         PADDLE_WIDTH, PADDLE_HEIGHT), border_radius=5)
        
        # Opponent paddle (left) - neon pink
        pygame.draw.rect(self.screen, self.NEON_PINK,
                        (10, self.opponent_y, PADDLE_WIDTH, PADDLE_HEIGHT), 
                        border_radius=5)
        
        # Draw ball with glow
        pygame.draw.circle(self.screen, self.NEON_GREEN,
                          (int(self.ball_x), int(self.ball_y)), BALL_SIZE)
        
        # Draw scores
        player_text = self.font.render(str(self.player_score), True, self.NEON_BLUE)
        opponent_text = self.font.render(str(self.opponent_score), True, self.NEON_PINK)
        self.screen.blit(player_text, (SCREEN_WIDTH * 3 // 4, 20))
        self.screen.blit(opponent_text, (SCREEN_WIDTH // 4, 20))
        
        # Draw training info
        info_font = pygame.font.Font(None, 24)
        episode_text = info_font.render(f"Episode: {episode}", True, self.WHITE)
        epsilon_text = info_font.render(f"ε: {epsilon:.3f}", True, self.WHITE)
        reward_text = info_font.render(f"Reward: {total_reward:.1f}", True, self.WHITE)
        
        self.screen.blit(episode_text, (10, SCREEN_HEIGHT - 70))
        self.screen.blit(epsilon_text, (10, SCREEN_HEIGHT - 45))
        self.screen.blit(reward_text, (10, SCREEN_HEIGHT - 20))
        
        pygame.display.flip()
        return True
    
    def close(self):
        """Clean up resources"""
        if self.render_mode and pygame.get_init():
            pygame.quit()
