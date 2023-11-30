import openai
from iphyre.games import PARAS, GAMES
from iphyre.simulator import IPHYRE
import json


def reward_from_action(action, env):
    env.reset()
    total_step = len(action)
    step, time_count, total_reward = 0, 0, 0
    while time_count < 15:
        if step < total_step:
            p, t = action[step][0:2], action[step][2]
            if time_count >= t:
                _, reward, done = env.step(p)
                total_reward += reward
                step += 1
                time_count += 1 / 10
                continue
        p = [0., 0.]
        _, reward, done = env.step(p)
        total_reward += reward
        time_count += 1 / 10
        if done:
            break
    return total_reward

prefix_IPHYRE = """You are given a simulated square-shaped 2D world of size 600*600 consisting of some objects. Your goal is to drop the ball into the abyss by eliminateing eliminable blocks. You have 15.0 seconds to finish a task.

The object configuration array is as follows (blocks and balls are all objects):

For blocks:
    [x1, y1, x2, y2, radius, eli, dynamic, joint, spring]
    [x1, y1] indicates the left end point and [x2, y2] indicates the right end point. The height (twice the 'radius') of blocks is 20.
For balls:
    [x, y, x, y, radius, eli, dynamic, joint, spring]
    [x, y] indicates the center of the ball. 'radius' is the radius of the ball.
ell: 0/1 indicates whether the corresponding object can be eliminated. 1 is eliminable and 0 is not eliminable.
dynamic: 0/1 indicates whether the corresponding object can move by external force. 1 is dynamic and 0 is static.
joint: 0/1 indicates whether the corresponding object is connected to a stick. 1 is connected and 0 is not connected.
spring: 0/1 indicates whether the corresponding object is connected to a spring. 1 is connected and 0 is not connected.

Objects in this world follow the following rules:
- Coordinates of objects are given by (x, y), where x is horizontal (0 to 600 represents left to right) and y is vertical (0 to 600 represents bottom to top).
- When the simulation starts, all objects move due to gravity except some static objects. They cannot affected by external force. The parameter of 'dynamic' in the object configuration dict indicates whether the object is static.
- During the simulation, objects move and interact (collide) with each other over time.
- If you don't want to eliminate a block, then you can do nothing. Doing nothing is also an action.

Given a specific object configuration array, generate an action sequence to tell when to eliminate which block. Output an array of dicts containing time and index of blocks.
The index is from 0 to 5. Here is an example of the output:
[
{"time": 0, "index": 0},
{"time": 0.2, "index": 1}
]

Given the spcific object configuration dict below:
"""

suffix_IPHYRE = """

The action sequence is:
"""


prefix = prefix_IPHYRE
tasks = GAMES
suffix = suffix_IPHYRE

openai.api_key = "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" 

for task_id, task in enumerate(tasks):
    print(f'\n[--------------------task {task_id} {task}--------------------]\n')
    ENV = IPHYRE(task, fps=10)
    paras = ENV.reset()
    print(paras)
    text = prefix + str(paras) + suffix
    print(text)
    responses = openai.ChatCompletion.create(
            model='gpt-4', # 'gpt-4' 'gpt-3.5-turbo'
            messages=[{'role': 'system', 'content': "You are a helpful assistant."},
                      {'role': 'user', 'content': text}],
            temperature=0.5,
            max_tokens=512,
            n=2
        )
    print(f'[-----response-----]')
    response = responses['choices'][1]['message']['content']
    # response = """[
    # {"time": 0, "index": 0},
    # {"time": 1.3, "index": 1}
    # ]
    # """
    print(response)
    actions = response.replace('\n', '').replace(' ', '').replace(']', '').replace('[', '').replace('},', '}|').split('|')
    actions = [eval(action) for action in actions]
    print(actions)
    action_space = ENV.get_action_space()[1:]
    for action in actions:
        if action['index'] >= len(action_space):
            action['index'] = len(action_space) - 1
    # convert actions in to an array of [x, y, t] using the action space; the actions are [{'time': 0, 'index': 0}, {'time': 1, 'index': 1}]
    actions = [[action_space[action['index']][0], action_space[action['index']][1], action['time']] for action in actions]
    print(actions)
    reward = reward_from_action(actions, ENV)
    print('reward:', reward)
    # save game, action, and reward in a json file
    json_data = {'task': task, 'actions': actions, 'reward': reward}
    with open(f'./gpt_IPHYRE.json', 'a') as f:
        json.dump(json_data, f)
        f.write('\n')
        