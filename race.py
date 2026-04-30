import math
import pygame
import numpy as np
import torch
import json
import random

# =========================
# CONFIG
# =========================
WIDTH, HEIGHT = 900, 700
WORLD_WIDTH, WORLD_HEIGHT = 4000, 4000

CAR_SIZE_X, CAR_SIZE_Y = 24, 14
MAX_RADAR_LENGTH = 400

BACKGROUND = (40, 120, 40)
GREY = (120, 120, 120)
WHITE = (255, 255, 255)
BLUE_RAY = (0, 180, 255)
GREEN = (100, 255, 100)
RED = (255, 0, 0)
MAGENTA = (100, 170, 215)


class CarRacingEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("RL Car Racing - 3 Ultrasonic Sensors")
        self.clock = pygame.time.Clock()

        self.map_surface = pygame.Surface((WORLD_WIDTH, WORLD_HEIGHT))
        self.path_data = []
        self.radars = []

        self.max_speed = 8.0
        self.min_speed = 2.0
        self.turn_amount = 5.0

        self.progress_scale = 4.0
        self.center_bonus_scale = 0.8
        self.step_penalty = -0.12
        self.no_progress_penalty = -1.5
        self.backward_penalty_scale = 3.0
        self.crash_penalty = -100.0

        self.no_progress_steps = 0
        self.no_progress_limit = 18
        self.progress_epsilon = 0.35

        self._draw_track()
        self._precompute_centerline()
        self.reset()

    # =========================
    # TRACK
    # =========================
    def _draw_track(self):
        self.map_surface.fill(BACKGROUND)
        center_x, center_y = WORLD_WIDTH // 2, WORLD_HEIGHT // 2

        for i in range(0, 720):
            angle = i * 0.5
            rad = math.radians(angle)

            radius_offset = math.sin(rad * 5) * 350 + math.cos(rad * 3) * 200
            x = center_x + math.cos(rad) * (1300 + radius_offset)
            y = center_y + math.sin(rad) * (1300 + radius_offset)

            track_width = 90 + math.sin(rad * 8) * 20
            self.path_data.append((x, y, track_width))

        # outer road
        for p in self.path_data:
            pygame.draw.circle(
                self.map_surface, GREY, (int(p[0]), int(p[1])), int(p[2])
            )

        # inner drivable lane
        for p in self.path_data:
            inner_radius = max(5, int(p[2] - 15))
            pygame.draw.circle(
                self.map_surface, WHITE, (int(p[0]), int(p[1])), inner_radius
            )

    def _precompute_centerline(self):
        self.centerline = np.array(
            [(p[0], p[1]) for p in self.path_data], dtype=np.float32
        )
        self.track_widths = np.array([p[2] for p in self.path_data], dtype=np.float32)
        self.num_nodes = len(self.centerline)

    # =========================
    # RESET
    # =========================
    def reset(self):
        start_idx = 0
        next_idx = 10

        self.car_x, self.car_y = self.centerline[start_idx]

        dx = self.centerline[next_idx][0] - self.car_x
        dy = self.centerline[next_idx][1] - self.car_y
        self.angle = 360 - math.degrees(math.atan2(dy, dx))

        self.speed = 4.0
        self.alive = True
        self.score = 0

        self.closest_idx = start_idx
        self.total_progress = 0.0
        self.no_progress_steps = 0

        self._update_radars()
        return self.get_state()

    # =========================
    # STEP
    # =========================
    def step(self, action):
        """
        Action mapping:
        0 = do nothing
        1 = turn left
        2 = turn right
        3 = accelerate
        4 = brake
        """

        old_idx = self.closest_idx

        # apply action
        if action == 1:
            self.angle += self.turn_amount
        elif action == 2:
            self.angle -= self.turn_amount
        elif action == 3:
            self.speed += 0.5
        elif action == 4:
            self.speed -= 0.7

        self.speed = max(self.min_speed, min(self.speed, self.max_speed))

        # move
        self.car_x += math.cos(math.radians(360 - self.angle)) * self.speed
        self.car_y += math.sin(math.radians(360 - self.angle)) * self.speed

        self.score += 1

        # collision
        self._check_collision()

        if self.alive:
            self.closest_idx = self._find_closest_centerline_idx(local_window=40)

        reward = self._compute_reward(old_idx)

        self._update_radars()
        done = not self.alive

        return self.get_state(), reward, done, self.score

    # =========================
    # REWARD
    # =========================
    def _compute_reward(self, old_idx):
        if not self.alive:
            return self.crash_penalty

        new_idx = self.closest_idx
        progress_delta = self._forward_index_delta(old_idx, new_idx)
        self.total_progress += progress_delta

        # distance from centerline
        cx, cy = self.centerline[new_idx]
        dist_from_center = math.hypot(self.car_x - cx, self.car_y - cy)

        road_radius = self.track_widths[new_idx] - 15.0
        normalized_offset = min(dist_from_center / max(road_radius, 1.0), 1.5)

        # core rewards
        progress_reward = self.progress_scale * max(progress_delta, 0.0)
        center_reward = self.center_bonus_scale * (1.0 - normalized_offset)

        reward = progress_reward + center_reward + self.step_penalty

        # punish going backward strongly
        if progress_delta < 0:
            reward -= self.backward_penalty_scale * abs(progress_delta)

        # punish no progress / spinning / stalling
        if progress_delta < self.progress_epsilon:
            self.no_progress_steps += 1
            reward -= 0.6
        else:
            self.no_progress_steps = 0

        if self.no_progress_steps >= self.no_progress_limit:
            reward += self.no_progress_penalty

        return float(reward)

    # =========================
    # STATE
    # =========================
    def get_state(self):
        """
        state = [left_sensor, front_sensor, right_sensor, speed]
        all normalized to roughly 0..1
        """
        distances = [radar[1] / MAX_RADAR_LENGTH for radar in self.radars]
        return np.array(distances + [self.speed / self.max_speed], dtype=np.float32)

    # =========================
    # RENDER
    # =========================
    def render(self):
        cam_x = max(0, min(int(self.car_x) - WIDTH // 2, WORLD_WIDTH - WIDTH))
        cam_y = max(0, min(int(self.car_y) - HEIGHT // 2, WORLD_HEIGHT - HEIGHT))

        self.screen.blit(self.map_surface, (0, 0), (cam_x, cam_y, WIDTH, HEIGHT))

        screen_car_x = int(self.car_x) - cam_x
        screen_car_y = int(self.car_y) - cam_y

        # draw car
        car_color = GREEN if self.alive else RED
        car_surface = pygame.Surface((CAR_SIZE_X, CAR_SIZE_Y), pygame.SRCALPHA)
        car_surface.fill(car_color)

        rotated_car = pygame.transform.rotate(car_surface, self.angle)
        rect = rotated_car.get_rect(center=(screen_car_x, screen_car_y))
        self.screen.blit(rotated_car, rect.topleft)

        # draw sensors
        for radar in self.radars:
            hit_pos = radar[0]
            screen_hit_x = hit_pos[0] - cam_x
            screen_hit_y = hit_pos[1] - cam_y

            pygame.draw.line(
                self.screen,
                BLUE_RAY,
                (screen_car_x, screen_car_y),
                (screen_hit_x, screen_hit_y),
                2,
            )
            pygame.draw.circle(self.screen, BLUE_RAY, (screen_hit_x, screen_hit_y), 4)

        # optional current centerline point
        cx, cy = self.centerline[self.closest_idx]
        pygame.draw.circle(self.screen, MAGENTA, (int(cx) - cam_x, int(cy) - cam_y), 4)

        pygame.display.flip()
        self.clock.tick(60)

    # =========================
    # COLLISION
    # =========================
    def _check_collision(self):
        try:
            color = self.map_surface.get_at((int(self.car_x), int(self.car_y)))
            if color == GREY or color == BACKGROUND:
                self.alive = False
        except IndexError:
            self.alive = False

    # =========================
    # SENSORS
    # =========================
    def _update_radars(self):
        self.radars = []

        # Arduino-style 3 ultrasonic directions:
        # left, front, right
        angles = [-45, 0, 45]

        for degree in angles:
            length = 0
            x, y = int(self.car_x), int(self.car_y)

            while length < MAX_RADAR_LENGTH:
                test_x = int(
                    self.car_x
                    + math.cos(math.radians(360 - (self.angle + degree))) * length
                )
                test_y = int(
                    self.car_y
                    + math.sin(math.radians(360 - (self.angle + degree))) * length
                )

                if not (0 < test_x < WORLD_WIDTH and 0 < test_y < WORLD_HEIGHT):
                    break

                color = self.map_surface.get_at((test_x, test_y))
                if color == GREY or color == BACKGROUND:
                    break

                x, y = test_x, test_y
                length += 4

            dist = int(math.sqrt((x - self.car_x) ** 2 + (y - self.car_y) ** 2))

            # Add +/- 5% random noise to simulate real-world sensor jitter
            noise_factor = random.uniform(0.95, 1.05)
            noisy_dist = int(dist * noise_factor)

            # Clamp it so it doesn't exceed the max length
            noisy_dist = min(noisy_dist, MAX_RADAR_LENGTH)

            self.radars.append([(x, y), noisy_dist])

    # =========================
    # PROGRESS HELPERS
    # =========================
    def _find_closest_centerline_idx(self, local_window=40):
        best_idx = self.closest_idx
        best_dist = float("inf")

        for k in range(-local_window, local_window + 1):
            idx = (self.closest_idx + k) % self.num_nodes
            px, py = self.centerline[idx]
            d = (self.car_x - px) ** 2 + (self.car_y - py) ** 2

            if d < best_dist:
                best_dist = d
                best_idx = idx

        return best_idx

    def _forward_index_delta(self, old_idx, new_idx):
        raw = new_idx - old_idx

        if raw > self.num_nodes // 2:
            raw -= self.num_nodes
        elif raw < -self.num_nodes // 2:
            raw += self.num_nodes

        return float(raw)


if __name__ == "__main__":
    env = CarRacingEnv()
    state = env.reset()
    running = True

    model = torch.nn.Sequential(
        torch.nn.Linear(4, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 300),
        torch.nn.ReLU(),
        torch.nn.Linear(300, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 5),
    )

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    state = env.reset()
    episode_length = []
    i = 0
    while True:
        i += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # train model
        y_pred = model(torch.Tensor(state))
        print(state)
        av_softmax = torch.softmax(y_pred / 5, dim=0)
        action = torch.multinomial(av_softmax, num_samples=1, replacement=True).item()
        next_state, reward, done, score = env.step(action)
        av_reward = y_pred.detach().clone()
        av_reward[action] = reward
        loss = loss_fn(y_pred, av_reward)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # save model progress
        if i % 200 == 0:
            print("saving")
            torch.save(model.state_dict(), "./car_model_weights (20).pth")
        env.render()

        if done:
            episode_length.append(score)
            with open("./learning_results.json", "w") as f:
                json.dump(episode_length, f)
            state = env.reset()
        else:
            state = next_state

    pygame.quit()
