import pickle
import numpy as np
from toto_benchmark.sim.dm_pour import DMWaterPouringEnv

# -------- load your model -------

# --------------------------------

init_positions = pickle.load(open('init_positions.pickle', 'rb'))
init_tank_pos = init_positions['init_tank_pos']
init_joint_pos = init_positions['init_joint_pos']

env = DMWaterPouringEnv(has_viewer=False)
rewards = []
for i in range(len(init_tank_pos)):
    print(f'Evaluating Traj {i} ...')
    tank_pos = init_tank_pos[i]
    joint_pos = init_joint_pos[i]
    env.seed(0)
    obs = env._reset_with_states(tank_pos=tank_pos, robot_qpos=joint_pos, robot_qvel=np.zeros(len(joint_pos)))
    while not env.done:
        # ---- TODO: add your model's predict code ----
        #  a = model(obs)
        a = [0.18, 0.2, 0.3, -1.8, 0, 1.5, 0.7-np.pi/4] # dummy action
        # -------------------------------------------
        obs, reward, done, _ = env.step(a)
    rewards.append(reward) # TODO: take last reward or max reward along the trajectory?
    print(f'Traj {i} Final reward: {reward}')
print(f'Finished evaluating {len(init_tank_pos)} trajectories.')
mean_reward = np.mean(rewards)
max_reward = np.max(rewards)
success_rate = sum(np.array(rewards) > 0) / len(rewards)
# internal TODO: any other metrics we want to include?
# internal TODO: add print statements correspondingly
