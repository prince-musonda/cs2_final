import ale_py
import gymnasium as gym
import pygame

gym.register_envs(ale_py)

env = gym.make("ALE/BattleZone-v5", render_mode="human")
obs, info = env.reset()

pygame.init()
pygame.display.set_mode((300, 200))
clock = pygame.time.Clock()

# Show action meanings
print(env.unwrapped.get_action_meanings())
# Usually something like:
# ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', ...]


def get_action():
    keys = pygame.key.get_pressed()

    if keys[pygame.K_SPACE]:
        return 1  # FIRE
    elif keys[pygame.K_UP]:
        return 2  # UP
    elif keys[pygame.K_RIGHT]:
        return 3  # RIGHT
    elif keys[pygame.K_LEFT]:
        return 4  # LEFT
    elif keys[pygame.K_DOWN]:
        return 5  #
    elif keys[pygame.K_a]:
        return 7
    elif keys[pygame.K_d]:
        return 6
    else:
        return 0  # NOOP


running = True
total_reward = 0
episode = 1

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    action = get_action()

    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward

    if reward != 0:
        print(
            f"Episode {episode} | Action: {action} | Reward: {reward} | Total: {total_reward}"
        )

    if terminated or truncated:
        print(f"Episode {episode} finished. Total reward: {total_reward}")
        episode += 1
        total_reward = 0
        obs, info = env.reset()

    clock.tick(60)

env.close()
pygame.quit()
