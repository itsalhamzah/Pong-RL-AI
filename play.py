"""
Play Pong Against the Trained AI
Use W/S or Arrow keys to control your paddle
"""

import pygame
import numpy as np
import torch
import os

from dqn_agent import DQNAgent
from config import (
    DEVICE, SCREEN_WIDTH, SCREEN_HEIGHT, PADDLE_WIDTH, PADDLE_HEIGHT,
    BALL_SIZE, PADDLE_SPEED, BALL_SPEED, FPS
)
import random


class PongGame:
    """Pong game for human vs AI play"""
    
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("🎮 Pong - You vs AI")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 48)
        self.small_font = pygame.font.Font(None, 28)
        
        # Colors
        self.BLACK = (15, 15, 25)
        self.WHITE = (255, 255, 255)
        self.NEON_BLUE = (0, 200, 255)
        self.NEON_PINK = (255, 50, 150)
        self.NEON_GREEN = (50, 255, 100)
        self.GOLD = (255, 215, 0)
        
        # Start button dimensions
        self.button_width = 200
        self.button_height = 60
        self.button_x = SCREEN_WIDTH // 2 - self.button_width // 2
        self.button_y = SCREEN_HEIGHT // 2 + 40
        
        self.game_started = False
        self.reset_game()
    
    def reset_game(self):
        """Reset game state"""
        # Human paddle - left side
        self.human_y = SCREEN_HEIGHT // 2 - PADDLE_HEIGHT // 2
        
        # AI paddle - right side
        self.ai_y = SCREEN_HEIGHT // 2 - PADDLE_HEIGHT // 2
        
        # Ball
        self.ball_x = SCREEN_WIDTH // 2
        self.ball_y = SCREEN_HEIGHT // 2
        self.ball_vx = BALL_SPEED * random.choice([-1, 1])
        self.ball_vy = BALL_SPEED * random.uniform(-0.5, 0.5)
        
        # Scores
        self.human_score = 0
        self.ai_score = 0
        
        self.game_over = False
        self.winner = None
    
    def get_ai_state(self):
        """Get normalized state for AI (from AI's perspective)"""
        # AI sees itself as the player (right side)
        state = np.array([
            self.ai_y / SCREEN_HEIGHT,
            self.human_y / SCREEN_HEIGHT,
            self.ball_x / SCREEN_WIDTH,
            self.ball_y / SCREEN_HEIGHT,
            self.ball_vx / BALL_SPEED,
            self.ball_vy / BALL_SPEED
        ], dtype=np.float32)
        return state
    
    def step(self, human_action, ai_action):
        """Execute one game step"""
        # Move human paddle
        if human_action == 1:  # Up
            self.human_y = max(0, self.human_y - PADDLE_SPEED)
        elif human_action == 2:  # Down
            self.human_y = min(SCREEN_HEIGHT - PADDLE_HEIGHT, self.human_y + PADDLE_SPEED)
        
        # Move AI paddle
        if ai_action == 1:  # Up
            self.ai_y = max(0, self.ai_y - PADDLE_SPEED)
        elif ai_action == 2:  # Down
            self.ai_y = min(SCREEN_HEIGHT - PADDLE_HEIGHT, self.ai_y + PADDLE_SPEED)
        
        # Move ball
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy
        
        # Ball collision with top/bottom walls
        if self.ball_y <= 0 or self.ball_y >= SCREEN_HEIGHT - BALL_SIZE:
            self.ball_vy = -self.ball_vy
            self.ball_y = max(0, min(SCREEN_HEIGHT - BALL_SIZE, self.ball_y))
        
        # Ball collision with AI paddle (right)
        if (self.ball_x + BALL_SIZE >= SCREEN_WIDTH - PADDLE_WIDTH - 10 and
            self.ai_y <= self.ball_y <= self.ai_y + PADDLE_HEIGHT):
            self.ball_vx = -abs(self.ball_vx)
            hit_pos = (self.ball_y - self.ai_y) / PADDLE_HEIGHT
            self.ball_vy = BALL_SPEED * (hit_pos - 0.5) * 1.5
        
        # Ball collision with human paddle (left)
        if (self.ball_x <= PADDLE_WIDTH + 10 and
            self.human_y <= self.ball_y <= self.human_y + PADDLE_HEIGHT):
            self.ball_vx = abs(self.ball_vx)
            hit_pos = (self.ball_y - self.human_y) / PADDLE_HEIGHT
            self.ball_vy = BALL_SPEED * (hit_pos - 0.5) * 1.5
        
        # Scoring
        if self.ball_x <= 0:  # AI scores
            self.ai_score += 1
            self._reset_ball(direction=1)
        elif self.ball_x >= SCREEN_WIDTH:  # Human scores
            self.human_score += 1
            self._reset_ball(direction=-1)
        
        # Check game over (first to 5)
        if self.human_score >= 5:
            self.game_over = True
            self.winner = "YOU"
        elif self.ai_score >= 5:
            self.game_over = True
            self.winner = "AI"
    
    def _reset_ball(self, direction=1):
        """Reset ball to center"""
        self.ball_x = SCREEN_WIDTH // 2
        self.ball_y = SCREEN_HEIGHT // 2
        self.ball_vx = BALL_SPEED * direction
        self.ball_vy = BALL_SPEED * random.uniform(-0.5, 0.5)
    
    def render_start_screen(self):
        """Render the start screen with a start button"""
        self.screen.fill(self.BLACK)
        
        # Draw center line
        for y in range(0, SCREEN_HEIGHT, 20):
            pygame.draw.rect(self.screen, (50, 50, 70), 
                           (SCREEN_WIDTH // 2 - 2, y, 4, 10))
        
        # Title
        title_font = pygame.font.Font(None, 72)
        title = title_font.render("PONG", True, self.NEON_GREEN)
        self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 80))
        
        # Subtitle
        subtitle = self.font.render("You vs AI", True, self.WHITE)
        self.screen.blit(subtitle, (SCREEN_WIDTH // 2 - subtitle.get_width() // 2, 150))
        
        # Draw start button with hover effect
        mouse_pos = pygame.mouse.get_pos()
        is_hovered = (self.button_x <= mouse_pos[0] <= self.button_x + self.button_width and
                     self.button_y <= mouse_pos[1] <= self.button_y + self.button_height)
        
        button_color = self.NEON_BLUE if is_hovered else (0, 150, 200)
        
        # Button glow effect when hovered
        if is_hovered:
            glow_rect = pygame.Rect(self.button_x - 4, self.button_y - 4,
                                   self.button_width + 8, self.button_height + 8)
            pygame.draw.rect(self.screen, (0, 100, 150), glow_rect, border_radius=15)
        
        # Button background
        button_rect = pygame.Rect(self.button_x, self.button_y, 
                                 self.button_width, self.button_height)
        pygame.draw.rect(self.screen, button_color, button_rect, border_radius=10)
        pygame.draw.rect(self.screen, self.WHITE, button_rect, 2, border_radius=10)
        
        # Button text
        button_text = self.font.render("START", True, self.WHITE)
        text_x = self.button_x + (self.button_width - button_text.get_width()) // 2
        text_y = self.button_y + (self.button_height - button_text.get_height()) // 2
        self.screen.blit(button_text, (text_x, text_y))
        
        # Controls hint
        controls = self.small_font.render("W/S or ↑/↓ to move | First to 5 wins!", True, (100, 100, 120))
        self.screen.blit(controls, (SCREEN_WIDTH // 2 - controls.get_width() // 2, SCREEN_HEIGHT - 60))
        
        click_hint = self.small_font.render("Click START or press SPACE to begin", True, (150, 150, 170))
        self.screen.blit(click_hint, (SCREEN_WIDTH // 2 - click_hint.get_width() // 2, SCREEN_HEIGHT - 30))
        
        pygame.display.flip()
    
    def check_start_button_click(self, pos):
        """Check if the start button was clicked"""
        return (self.button_x <= pos[0] <= self.button_x + self.button_width and
                self.button_y <= pos[1] <= self.button_y + self.button_height)

    def render(self):
        """Render the game"""
        self.screen.fill(self.BLACK)
        
        # Draw center line
        for y in range(0, SCREEN_HEIGHT, 20):
            pygame.draw.rect(self.screen, (50, 50, 70), 
                           (SCREEN_WIDTH // 2 - 2, y, 4, 10))
        
        # Draw human paddle (left) - pink
        pygame.draw.rect(self.screen, self.NEON_PINK,
                        (10, self.human_y, PADDLE_WIDTH, PADDLE_HEIGHT), 
                        border_radius=5)
        
        # Draw AI paddle (right) - blue
        pygame.draw.rect(self.screen, self.NEON_BLUE,
                        (SCREEN_WIDTH - PADDLE_WIDTH - 10, self.ai_y, 
                         PADDLE_WIDTH, PADDLE_HEIGHT), border_radius=5)
        
        # Draw ball
        pygame.draw.circle(self.screen, self.NEON_GREEN,
                          (int(self.ball_x), int(self.ball_y)), BALL_SIZE)
        
        # Draw scores
        human_text = self.font.render(str(self.human_score), True, self.NEON_PINK)
        ai_text = self.font.render(str(self.ai_score), True, self.NEON_BLUE)
        self.screen.blit(human_text, (SCREEN_WIDTH // 4, 20))
        self.screen.blit(ai_text, (SCREEN_WIDTH * 3 // 4, 20))
        
        # Draw labels
        you_label = self.small_font.render("YOU", True, self.NEON_PINK)
        ai_label = self.small_font.render("AI", True, self.NEON_BLUE)
        self.screen.blit(you_label, (SCREEN_WIDTH // 4 - 10, 60))
        self.screen.blit(ai_label, (SCREEN_WIDTH * 3 // 4, 60))
        
        # Draw controls hint
        controls = self.small_font.render("W/S or ↑/↓ to move | ESC to quit", True, (100, 100, 120))
        self.screen.blit(controls, (SCREEN_WIDTH // 2 - controls.get_width() // 2, SCREEN_HEIGHT - 30))
        
        # Game over overlay
        if self.game_over:
            # Semi-transparent overlay
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            # Winner text
            if self.winner == "YOU":
                win_text = self.font.render("🎉 YOU WIN! 🎉", True, self.GOLD)
            else:
                win_text = self.font.render("AI WINS!", True, self.NEON_BLUE)
            
            self.screen.blit(win_text, 
                           (SCREEN_WIDTH // 2 - win_text.get_width() // 2, 
                            SCREEN_HEIGHT // 2 - 40))
            
            restart_text = self.small_font.render("Press SPACE to play again | ESC to quit", True, self.WHITE)
            self.screen.blit(restart_text,
                           (SCREEN_WIDTH // 2 - restart_text.get_width() // 2,
                            SCREEN_HEIGHT // 2 + 20))
        
        pygame.display.flip()


def main():
    """Main game loop"""
    print("=" * 60)
    print("🎮 PONG - YOU vs TRAINED AI")
    print("=" * 60)
    print(f"[DEVICE] {DEVICE}")
    
    # Load trained AI agent
    agent = DQNAgent()
    
    model_path = None
    if os.path.exists("models/pong_best.pth"):
        model_path = "models/pong_best.pth"
    elif os.path.exists("models/pong_final.pth"):
        model_path = "models/pong_final.pth"
    
    if model_path:
        agent.load(model_path)
        print(f"[AI] Loaded trained model from {model_path}")
    else:
        print("[WARNING] No trained model found! AI will play randomly.")
        print("          Run train.py first to train the AI.")
    
    # Set AI to evaluation mode (no exploration)
    agent.epsilon = 0
    
    print("=" * 60)
    print("[CONTROLS] W/S or Arrow Up/Down to move your paddle")
    print("[GOAL] First to 5 points wins!")
    print("=" * 60)
    
    game = PongGame()
    running = True
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_SPACE:
                    if not game.game_started:
                        game.game_started = True
                    elif game.game_over:
                        game.reset_game()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if not game.game_started and game.check_start_button_click(event.pos):
                    game.game_started = True
        
        if not game.game_started:
            # Show start screen
            game.render_start_screen()
        elif not game.game_over:
            # Get human input
            keys = pygame.key.get_pressed()
            human_action = 0  # Stay
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                human_action = 1  # Up
            elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
                human_action = 2  # Down
            
            # Get AI action
            ai_state = game.get_ai_state()
            ai_action = agent.select_action(ai_state, training=False)
            
            # Update game
            game.step(human_action, ai_action)
            
            # Render game
            game.render()
        else:
            # Game over - just render
            game.render()
        
        game.clock.tick(FPS)
    
    pygame.quit()
    print("\n[GAME] Thanks for playing!")


if __name__ == "__main__":
    main()
