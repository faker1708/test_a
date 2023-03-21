import gym
import sys
import time

env = gym.make('CartPole-v1', render_mode="human")
for episode in range(10):
    env.reset()
    # print("Episode finished after {} timesteps".format(episode))
    for ik in range(100):
        # env.render()

        # ac = env.action_space.sample()
        # print(ac)
        # ac = 0
        # print(type(ac))
        ac = 0
        state, reward, terminated,truncated, info = env.step(0)
        print(ac)




        # print(reward)
        if terminated:
            break
        time.sleep(0.02)
env.close()
# sys.pause(10)