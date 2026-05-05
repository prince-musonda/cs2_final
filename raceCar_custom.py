import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
from collections import deque

# --- CONFIGURATION ---
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
FPS = 60

# RL Hyperparameters
GAMMA = 0.95
LEARNING_RATE = 0.001
MEMORY_SIZE = 50000
BATCH_SIZE = 64
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995


# --- DQN MODEL (PyTorch) ---
class DQN(nn.Module):
    def __init__(self, state_size=3, action_size=4):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.output = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.output(x)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPS_START

        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.update_target()

        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return None

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)

        current_q = self.model(states).gather(1, actions)
        next_q = self.target_model(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + (1 - dones) * GAMMA * next_q

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > EPS_END:
            self.epsilon *= EPS_DECAY

        return loss.item()


# --- CAR PHYSICS ---
class Car:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0
        self.speed = 0
        self.max_speed = 5
        self.acceleration = 0.2
        self.friction = 0.05
        self.width = 20
        self.height = 30
        self.collided = False

    def update(self, action):
        # 0: Left, 1: Right, 2: Forward, 3: Nothing
        if action == 2:
            self.speed += self.acceleration
        elif action == 3:
            pass  # Coasting

        if self.speed != 0:
            if action == 0:
                self.angle -= 0.08
            if action == 1:
                self.angle += 0.08

        # Physics
        self.speed -= self.friction
        if self.speed < 0:
            self.speed = 0
        if self.speed > self.max_speed:
            self.speed = self.max_speed

        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed

    def get_mask_points(self):
        # Simple polygon for collision
        points = []
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        w, h = self.width / 2, self.height / 2

        offsets = [(-w, -h), (w, -h), (w, h), (-w, h)]
        for dx, dy in offsets:
            rx = self.x + dx * cos_a - dy * sin_a
            ry = self.y + dx * sin_a + dy * cos_a
            points.append((rx, ry))
        return points


# --- ENVIRONMENT & TRACK ---
class RaceEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("JARVIS v2.0 - DQNPytorch Race System")
        self.clock = pygame.time.Clock()

        # Font Fallback
        try:
            self.font = pygame.font.SysFont("Courier New", 14, bold=True)
        except:
            self.font = pygame.font.SysFont(None, 14)

        # Track Boundaries (Longer, More Winding Path)
        self.outer_track = [
            (0, 0),
            (600, -100),
            (1000, 200),
            (1400, 100),
            (1800, 300),
            (2200, 100),
            (2600, 400),
            (2400, 800),
            (2000, 700),
            (1600, 1000),
            (1000, 900),
            (600, 1100),
            (0, 900),
            (-400, 1000),
            (-800, 700),
            (-600, 300),
            (-200, 200),
        ]
        self.inner_track = [
            (100, 150),
            (550, 100),
            (900, 350),
            (1350, 250),
            (1750, 450),
            (2050, 300),
            (2300, 500),
            (2200, 700),
            (1900, 600),
            (1550, 850),
            (1050, 750),
            (650, 950),
            (100, 750),
            (-300, 850),
            (-600, 650),
            (-450, 400),
            (-100, 350),
        ]
        self.obstacles = []
        self.reset()

    def reset(self):
        # Start the car at the beginning of the new track
        self.car = Car(50, 75)
        self.steps = 0
        self.total_reward = 0
        self.generate_obstacles()
        return self.get_state()

    def generate_obstacles(self):
        self.obstacles = []
        # Generate random circular obstacles inside the new larger track bounds
        # Reduced count and radius to prevent blocking
        for _ in range(random.randint(8, 12)):
            for attempt in range(150):
                x = random.randint(-800, 2600)
                y = random.randint(-100, 1100)
                radius = random.randint(10, 25)  # Slightly smaller max radius

                # Basic track containment check
                if not self.is_inside((x, y), self.inner_track) and self.is_inside(
                    (x, y), self.outer_track
                ):
                    # 1. Keep obstacles away from start
                    if math.sqrt((x - 50) ** 2 + (y - 75) ** 2) < 250:
                        continue

                    # 2. Check overlap with existing obstacles
                    overlap = False
                    for existing in self.obstacles:
                        dist = math.sqrt(
                            (x - existing["pos"][0]) ** 2
                            + (y - existing["pos"][1]) ** 2
                        )
                        if dist < (
                            radius + existing["radius"] + 60
                        ):  # Ensure gap between them
                            overlap = True
                            break
                    if overlap:
                        continue

                    # 3. Add to system
                    self.obstacles.append({"pos": (x, y), "radius": radius})
                    break

    def get_state(self):
        readings = []
        angles = [-0.6, 0, 0.6]  # 3 rays
        ray_len = 150

        for a in angles:
            ray_angle = self.car.angle + a
            dist = 1.0  # 1.0 = clear, < 1.0 = obstructed

            for i in range(1, ray_len, 5):
                rx = self.car.x + math.cos(ray_angle) * i
                ry = self.car.y + math.sin(ray_angle) * i

                # Check collision with borders
                if self.check_point_collision(rx, ry):
                    dist = i / ray_len
                    break
            readings.append(dist)
        return readings

    def check_point_collision(self, x, y):
        # Borders
        p = (x, y)
        if self.is_inside(p, self.inner_track):
            return True
        if not self.is_inside(p, self.outer_track):
            return True

        # Obstacles
        for obs in self.obstacles:
            dx = x - obs["pos"][0]
            dy = y - obs["pos"][1]
            if math.sqrt(dx * dx + dy * dy) < obs["radius"]:
                return True
        return False

    def is_inside(self, point, polygon):
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if point[1] > min(p1y, p2y):
                if point[1] <= max(p1y, p2y):
                    if point[0] <= max(p1x, p2x):
                        if p1y != p2y:
                            xints = (point[1] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or point[0] <= xints:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def step(self, action):
        self.car.update(action)
        self.steps += 1

        state = self.get_state()
        collided = self.check_point_collision(self.car.x, self.car.y)

        reward = 0.1  # Survival
        reward += (self.car.speed / self.car.max_speed) * 1.5

        # Reward for staying centered (sensors equal)
        reward += (1.0 - abs(state[0] - state[2])) * 0.5

        if collided:
            reward = -100  # Heavy penalty as requested

        self.total_reward += reward
        return state, reward, collided

    def render(self, agent, loss_val):
        # Background: Dark blue space with a dynamic tracking grid
        self.screen.fill((15, 23, 42))

        cam_x = SCREEN_WIDTH // 2 - self.car.x
        cam_y = SCREEN_HEIGHT // 2 - self.car.y

        # Draw Dynamic Tech Grid
        grid_size = 100
        start_x = int((self.car.x - SCREEN_WIDTH // 2) // grid_size) * grid_size
        start_y = int((self.car.y - SCREEN_HEIGHT // 2) // grid_size) * grid_size
        for x in range(
            start_x - grid_size, start_x + SCREEN_WIDTH + grid_size * 2, grid_size
        ):
            pygame.draw.line(
                self.screen,
                (30, 41, 59),
                (x + cam_x, -5000 + cam_y),
                (x + cam_x, 5000 + cam_y),
                1,
            )
        for y in range(
            start_y - grid_size, start_y + SCREEN_HEIGHT + grid_size * 2, grid_size
        ):
            pygame.draw.line(
                self.screen,
                (30, 41, 59),
                (-5000 + cam_x, y + cam_y),
                (5000 + cam_x, y + cam_y),
                1,
            )

        # Draw Asphalt Track (Fill between boundaries)
        outer_pts = [(p[0] + cam_x, p[1] + cam_y) for p in self.outer_track]
        inner_pts = [(p[0] + cam_x, p[1] + cam_y) for p in self.inner_track]

        pygame.draw.polygon(self.screen, (39, 39, 42), outer_pts)  # Asphalt color
        pygame.draw.polygon(self.screen, (15, 23, 42), inner_pts)  # Grass/Inner color

        # Helper to draw racing curbs (Red/White segments)
        def draw_curbs(poly):
            for i in range(len(poly)):
                p1 = poly[i]
                p2 = poly[(i + 1) % len(poly)]
                dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                chunks = max(1, int(dist // 25))
                for c in range(chunks):
                    t1 = c / chunks
                    t2 = (c + 1) / chunks
                    cp1 = (
                        p1[0] + (p2[0] - p1[0]) * t1 + cam_x,
                        p1[1] + (p2[1] - p1[1]) * t1 + cam_y,
                    )
                    cp2 = (
                        p1[0] + (p2[0] - p1[0]) * t2 + cam_x,
                        p1[1] + (p2[1] - p1[1]) * t2 + cam_y,
                    )
                    color = (
                        (244, 244, 245) if c % 2 == 0 else (244, 63, 94)
                    )  # White / Rose
                    pygame.draw.line(self.screen, color, cp1, cp2, 6)

        draw_curbs(self.outer_track)
        draw_curbs(self.inner_track)

        # Draw Obstacles (Cones/Barrels)
        for obs in self.obstacles:
            ox, oy = int(obs["pos"][0] + cam_x), int(obs["pos"][1] + cam_y)
            r = obs["radius"]
            # Glow/Shadow
            pygame.draw.circle(self.screen, (120, 30, 30), (ox, oy), r + 4)
            # Body
            pygame.draw.circle(self.screen, (244, 63, 94), (ox, oy), r)
            # Highlight
            pygame.draw.circle(
                self.screen, (255, 100, 100), (ox - r // 3, oy - r // 3), r // 2
            )
            pygame.draw.circle(self.screen, (255, 255, 255), (ox, oy), r, 2)

        # Draw Car (Detailed Model)
        pts = self.car.get_mask_points()
        draw_pts = [(p[0] + cam_x, p[1] + cam_y) for p in pts]
        pygame.draw.polygon(self.screen, (14, 165, 233), draw_pts)  # Sky Blue Body
        pygame.draw.polygon(self.screen, (56, 189, 248), draw_pts, 3)  # Neon edge

        # Headlights
        head_dist = 16
        lx = self.car.x + math.cos(self.car.angle + 0.5) * head_dist + cam_x
        ly = self.car.y + math.sin(self.car.angle + 0.5) * head_dist + cam_y
        rx = self.car.x + math.cos(self.car.angle - 0.5) * head_dist + cam_x
        ry = self.car.y + math.sin(self.car.angle - 0.5) * head_dist + cam_y
        pygame.draw.circle(self.screen, (255, 255, 255), (int(lx), int(ly)), 4)
        pygame.draw.circle(self.screen, (255, 255, 255), (int(rx), int(ry)), 4)

        # --- DASHBOARD HUD ---
        # Main Info Panel
        overlay = pygame.Surface((280, 200), pygame.SRCALPHA)
        pygame.draw.rect(overlay, (15, 23, 42, 220), (0, 0, 280, 200), border_radius=12)
        pygame.draw.rect(
            overlay, (56, 189, 248, 100), (0, 0, 280, 200), width=1, border_radius=12
        )
        self.screen.blit(overlay, (20, 20))

        # Header Label
        hdr = self.font.render("JARVIS INTEL - RL RACE CORE", True, (56, 189, 248))
        self.screen.blit(hdr, (40, 35))

        stats = [
            f"REWARD TOTAL:  {int(self.total_reward)}",
            f"LEARN RATE:    {LEARNING_RATE}",
            f"EXPLORATION:   {agent.epsilon * 100:.1f}%",
            f"ACTIVE STEPS:  {self.steps}",
            f"SYSTEM LOSS:   {loss_val:.5f}" if loss_val else "LOSS: CALC...",
            f"OBSTACLE CT:   {len(self.obstacles)}",
        ]
        for i, s in enumerate(stats):
            txt = self.font.render(s, True, (244, 244, 245))
            self.screen.blit(txt, (40, 65 + i * 20))

        # Sensor Visualizer (Bottom Data Stream)
        state_vals = self.get_state()
        for i, val in enumerate(state_vals):
            # val is normalized distance 0.0 (crash) to 1.0 (clear)
            is_danger = val < 0.3
            bar_color = (225, 29, 72) if is_danger else (14, 165, 233)

            # Label
            lbl_color = (255, 255, 255) if is_danger else (161, 161, 170)
            lbl = self.font.render(f"ULTRA_SONIC_{['L', 'C', 'R'][i]}", True, lbl_color)
            self.screen.blit(lbl, (20, 530 + i * 45))

            # Bar Background
            pygame.draw.rect(
                self.screen, (30, 41, 59), (20, 550 + i * 45, 250, 12), border_radius=4
            )
            # Active Bar
            pygame.draw.rect(
                self.screen,
                bar_color,
                (20, 550 + i * 45, int(250 * val), 12),
                border_radius=4,
            )

            # Distance Text
            dist_text = self.font.render(f"{val * 150:.1f}cm", True, bar_color)
            self.screen.blit(dist_text, (280, 548 + i * 45))

        pygame.display.flip()
        self.clock.tick(FPS)


# --- MAIN LOOP ---
if __name__ == "__main__":
    env = RaceEnv()
    agent = DQNAgent(3, 4)
    episodes = 500

    current_loss = None

    for e in range(episodes):
        state = env.reset()
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            action = agent.act(state)
            next_state, reward, done = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            loss = agent.train()
            if loss:
                current_loss = loss

            env.render(agent, current_loss)

        agent.update_target()
        print(
            f"Episode: {e + 1}/{episodes}, Reward: {env.total_reward:.1f}, Epsilon: {agent.epsilon:.2f}"
        )
