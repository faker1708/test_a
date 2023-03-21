import gym
from gym import envs
env_list = envs.registry.keys()
env_ids = [env_item for env_item in env_list]
print('There are {0} envs in gym'.format(len(env_ids)))
print(env_ids)


env = gym.make('CartPole-v1')
print('观测空间 = {}'.format(env.observation_space))
print('动作空间 = {}'.format(env.action_space))
print('动作数 = {}'.format(env.action_space.n))
print('初始状态 = {}'.format(env.state))

init_state = env.reset()
print('初始状态 = {}'.format(init_state))
print('初始状态 = {}'.format(env.state))

for k in range(5):
    action = env.action_space.sample()
    state, reward, done, info,xxx = env.step(action)
    # aaa = env.step(action)
    # print(aaa)
    print('动作 = {0}: 当前状态 = {1}, 奖励 = {2}, 结束标志 = {3}, 日志信息 = {4}'.format(action, state, reward, done, info))
    print(xxx)