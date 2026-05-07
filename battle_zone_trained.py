from collections import deque

import ale_py
import gymnasium as gym
import cv2
import numpy as np
import torch
import torch.nn as nn


# =========================================================
# CONFIG
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STACK_SIZE = 4
IMG_SIZE = 84
FEATURE_DIM = 512

MODEL_PATH = "battle_zone_dqn (3).pth"

NUM_EPISODES = 10


# =========================================================
# ENV
# =========================================================
gym.register_envs(ale_py)

env = gym.make("ALE/BattleZone-v5", render_mode="human")
NUM_ACTIONS = env.action_space.n


# =========================================================
# PREPROCESSING
# =========================================================
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


class FrameStack:
    def __init__(self, stack_size=4):
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)

    def reset(self, frame):
        processed = preprocess_frame(frame)
        self.frames.clear()

        for _ in range(self.stack_size):
            self.frames.append(processed)

        return np.stack(self.frames, axis=0)

    def step(self, frame):
        processed = preprocess_frame(frame)
        self.frames.append(processed)

        return np.stack(self.frames, axis=0)


# =========================================================
# MODEL
# =========================================================
class CNNEncoder(nn.Module):
    def __init__(self, in_channels=4, feature_dim=512):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.fc = nn.Sequential(nn.Linear(64 * 7 * 7, feature_dim), nn.ReLU())

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


class QHead(nn.Module):
    def __init__(self, feature_dim=512, num_actions=5):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(), nn.Linear(256, num_actions)
        )

    def forward(self, features):
        return self.net(features)


class DQNAgent(nn.Module):
    def __init__(self, encoder, q_head):
        super().__init__()
        self.encoder = encoder
        self.q_head = q_head

    def forward(self, x):
        features = self.encoder(x)
        q_values = self.q_head(features)
        return q_values


# =========================================================
# LOAD MODEL
# =========================================================
policy_net = DQNAgent(
    encoder=CNNEncoder(in_channels=STACK_SIZE, feature_dim=FEATURE_DIM),
    q_head=QHead(feature_dim=FEATURE_DIM, num_actions=NUM_ACTIONS),
).to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

policy_net.load_state_dict(checkpoint["model_state_dict"])
policy_net.eval()

print("Model loaded successfully.")


# =========================================================
# ACTION SELECTION
# =========================================================
def select_action(model, state):
    state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        q_values = model(state_tensor)
        action = torch.argmax(q_values, dim=1).item()

    return action


# =========================================================
# PLAY LOOP
# =========================================================
frame_stack = FrameStack(STACK_SIZE)

for episode in range(NUM_EPISODES):
    state_raw, info = env.reset()
    state = frame_stack.reset(state_raw)

    done = False
    total_reward = 0

    while not done:
        action = select_action(policy_net, state)

        next_state_raw, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        state = frame_stack.step(next_state_raw)

        total_reward += reward

    print(f"Episode {episode + 1}: total_reward = {total_reward}")

env.close()
