import gymnasium as gym
import json
import random
from collections import deque, namedtuple
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# =========================================================
# CONFIG
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPISODES = 2000
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 64
BUFFER_SIZE = 100000
MIN_REPLAY_SIZE = 5000
TARGET_UPDATE_FREQ = 1000  # update target net every N env steps
TRAIN_FREQ = 4  # train every N env steps

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 200000

STACK_SIZE = 4
IMG_SIZE = 84
FEATURE_DIM = 512

TARGET_REWARD = 500
JSON_PATH = "natural_cnn_data.json"


# =========================================================
# ENV
# =========================================================
env = gym.make("CarRacing-v3", continuous=False, render_mode="human")
NUM_ACTIONS = env.action_space.n


# =========================================================
# REPLAY BUFFER
# =========================================================
Transition = namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done"]
)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states = np.stack([t.state for t in batch], axis=0)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch], axis=0)
        dones = np.array([t.done for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# =========================================================
# PREPROCESSING
# =========================================================
def preprocess_frame(frame):
    """
    frame: (96, 96, 3)
    return: (84, 84) grayscale float32 in [0, 1]
    """
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
        return np.stack(self.frames, axis=0)  # (4, 84, 84)

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
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),  # -> 20x20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # -> 9x9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # -> 7x7
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
# HELPERS
# =========================================================
def epsilon_by_step(step):
    if step >= EPS_DECAY_STEPS:
        return EPS_END
    frac = step / EPS_DECAY_STEPS
    return EPS_START + frac * (EPS_END - EPS_START)


def select_action(model, state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()

    state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state_tensor)
        action = torch.argmax(q_values, dim=1).item()
    return action


def optimize_model(policy_net, target_net, replay_buffer, optimizer, loss_fn):
    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

    states = torch.tensor(states, dtype=torch.float32, device=DEVICE)
    actions = torch.tensor(actions, dtype=torch.long, device=DEVICE).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
    next_states = torch.tensor(next_states, dtype=torch.float32, device=DEVICE)
    dones = torch.tensor(dones, dtype=torch.float32, device=DEVICE)

    current_q_values = policy_net(states).gather(1, actions).squeeze(1)

    with torch.no_grad():
        max_next_q_values = target_net(next_states).max(dim=1)[0]
        target_q_values = rewards + GAMMA * max_next_q_values * (1.0 - dones)

    loss = loss_fn(current_q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# =========================================================
# INIT NETWORKS
# =========================================================
policy_net = DQNAgent(
    encoder=CNNEncoder(in_channels=STACK_SIZE, feature_dim=FEATURE_DIM),
    q_head=QHead(feature_dim=FEATURE_DIM, num_actions=NUM_ACTIONS),
).to(DEVICE)

target_net = DQNAgent(
    encoder=CNNEncoder(in_channels=STACK_SIZE, feature_dim=FEATURE_DIM),
    q_head=QHead(feature_dim=FEATURE_DIM, num_actions=NUM_ACTIONS),
).to(DEVICE)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
loss_fn = nn.SmoothL1Loss()
replay_buffer = ReplayBuffer(BUFFER_SIZE)
frame_stack = FrameStack(STACK_SIZE)


# torch.save({
#         "model_state_dict": policy_net.state_dict(),
#         "optimizer_state_dict": optimizer.state_dict(),
#         }, "cnn_dqn.pth")


# =========================================================
# WARMUP REPLAY BUFFER
# =========================================================
print("Filling replay buffer...")

state_raw, info = env.reset()
state = frame_stack.reset(state_raw)

while len(replay_buffer) < MIN_REPLAY_SIZE:
    action = env.action_space.sample()
    next_state_raw, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    next_state = frame_stack.step(next_state_raw)
    replay_buffer.add(state, action, reward, next_state, done)

    state = next_state

    if done:
        state_raw, info = env.reset()
        state = frame_stack.reset(state_raw)

print("Replay buffer ready.")


# =========================================================
# TRAINING LOOP
# =========================================================
rewards = []
losses = []
target_reward = TARGET_REWARD
first_reached = None
ended_by_terminated = 0
ended_by_truncated = 0

global_step = 0

start_time = time.time()
for episode in range(NUM_EPISODES):
    state_raw, info = env.reset()
    state = frame_stack.reset(state_raw)

    total_reward = 0
    done = False
    episode_losses = []

    while not done:
        epsilon = epsilon_by_step(global_step)

        action = select_action(policy_net, state, epsilon)

        next_state_raw, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        next_state = frame_stack.step(next_state_raw)

        replay_buffer.add(state, action, reward, next_state, done)

        total_reward += reward
        state = next_state
        global_step += 1

        if len(replay_buffer) >= MIN_REPLAY_SIZE and global_step % TRAIN_FREQ == 0:
            loss = optimize_model(
                policy_net, target_net, replay_buffer, optimizer, loss_fn
            )
            episode_losses.append(loss)

        if global_step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            if terminated:
                ended_by_terminated += 1
            if truncated:
                ended_by_truncated += 1

    rewards.append(total_reward)

    avg_loss = float(np.mean(episode_losses)) if episode_losses else None
    losses.append(avg_loss)

    if first_reached is None and total_reward >= target_reward:
        first_reached = episode + 1

    print(
        f"Episode {episode + 1}, "
        f"total_reward: {total_reward:.2f}, "
        f"epsilon: {epsilon:.4f}, "
        f"avg_loss: {avg_loss}"
    )

    data = {
        "rewards": rewards,
        "losses": losses,
        "target_reward": target_reward,
        "first_reached": first_reached,
        "ended_by_terminated": ended_by_terminated,
        "ended_by_truncated": ended_by_truncated,
    }

    with open(JSON_PATH, "w") as f:
        json.dump(data, f)

    # save weights
    torch.save(
        {
            "model_state_dict": policy_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "cnn_dqn.pth",
    )

# save time taken
end_time = time.time()
time_taken = end_time - start_time
with open(JSON_PATH, "r") as f:
    data = json.load(f)
    data["time_taken"] = time_taken
with open(JSON_PATH, "w") as f:
    json.dump(data, f)

env.close()
