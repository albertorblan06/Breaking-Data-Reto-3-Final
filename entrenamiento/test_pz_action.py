from pettingzoo.atari import boxing_v2
env = boxing_v2.env()
env.reset()
print(env.action_space("first_0"))
print(env.action_space("second_0"))
