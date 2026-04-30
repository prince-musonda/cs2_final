# Race Track Reinforcement Learning Environment

This is a Pygame-based race track environment designed for reinforcement learning agents, similar to OpenAI Gym environments. The track is a complex, winding path with multiple turns, making it more challenging than simple oval tracks.

## Features

- **Agent**: A simulated self-driving car with ultrasonic sensors
- **Track**: Complex bitmap-based winding race track with sharp turns and varying width
- **State Space**: 6-dimensional vector consisting of 5 ultrasonic sensor readings and current speed
- **Action Space**: 5 discrete actions (turn left, turn right, speed up, slow down, do nothing)
- **Reward System**:
  - Positive reward for forward progress along the track (based on x-position)
  - Negative reward for moving backward or staying in the same position
  - Large negative reward for colliding with track boundaries

## Installation

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   **Note**: If pygame installation fails due to compilation issues on Windows with Python 3.12+, try:
   - Using conda: `conda install pygame`
   - Downloading a pre-built wheel from https://www.pygame.org/download.shtml
   - Or use `pip install pygame --pre` for pre-release versions

## Usage

```python
from race_track_env import RaceTrackEnv

env = RaceTrackEnv()
state = env.reset()

done = False
while not done:
    action = 4  # Choose action (0-4)
    next_state, reward, done, info = env.step(action)
    env.render()  # Optional: render the environment

env.close()
```

## Environment Details

### State

- 5 ultrasonic sensor distances (front-left, front, front-right, left, right)
- Current speed of the car

### Actions

0. Turn left
1. Turn right
2. Speed up
3. Slow down
4. Do nothing

### Reward Function

- Progress reward: +1 per unit of forward distance
- No progress penalty: -0.1 for staying or moving backward
- Collision penalty: -10

### Termination

- Episode ends on collision with boundaries
- Maximum 1000 steps per episode

## Customization

You can modify the environment parameters in the `__init__` method:

- Track shape and size
- Car physics (speed limits, turn angles)
- Sensor configuration
- Reward values
