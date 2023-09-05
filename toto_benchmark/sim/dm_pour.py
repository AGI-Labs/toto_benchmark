import hydra
import mujoco
import os
import numpy as np
from mujoco import _enums, _functions
from toto_benchmark.sim.rand import np_random
import toto_benchmark


class DMWaterPouringEnv:
    def __init__(self, has_viewer=False, use_real_robot=False, ignore_done=False):
        self.seed()
        self.has_viewer = has_viewer
        self.use_real_robot = use_real_robot
        if self.use_real_robot:
            self.has_viewer = False
        self.num_particles = 12
        self.tank_size = np.array((0.15, 0.15, 0.06))
        self.tank_pos = np.array((0.42, -0.1, 0.55))  # will be randomized during reset
        self.robot_min_positions = [-0.18, -0.2, -0.3, -2.1, -0.5, 2, 0.7-np.pi/4]
        self.robot_max_positions = [0.18, 0.2, 0.3, -1.8, 0, 1.5, 0.7-np.pi/4]
        self.tank_range_low = [0.3, -0.25, 0.55]
        self.tank_range_high = [0.5, 0.25, 0.55]
        self.camera_name = 'newerview'
        self.ignore_done = ignore_done
        self.horizon = 2000 # max demo length is 905
        self.control_freq = 100
        self.control_timestep = 1. / self.control_freq

        self._initialize_sim()
        self.model_timestep = self.model.opt.timestep

        if self.has_viewer:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(self.model, self.data)
            viewer.cam.fixedcamid = _functions.mj_name2id(self.model, _enums.mjtObj.mjOBJ_CAMERA, self.camera_name)
            viewer.cam.type = _enums.mjtCamera.mjCAMERA_FIXED
            self.viewer = viewer
        else:
            self.viewer = None
        
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0])
        
        self.reset()
    
    def _initialize_sim(self, xml_string=None): # xml_string not used
        base_path = os.path.dirname(toto_benchmark.__file__)
        self.model = mujoco.MjModel.from_xml_path(os.path.join(base_path, "sim/franka_panda_pouring.xml"))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)
        self.renderer = mujoco.Renderer(self.model)
        self.renderer.update_scene(self.data, camera=self.camera_name)
    
    def seed(self, seed=None):
        self.np_random, seed = np_random(seed)
        return [seed]
    
    def set_seed(self, seed=None):
        return self.seed(seed)
    
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        # Randomize the robot start position
        init_joints = self.np_random.uniform(low=self.robot_min_positions, high=self.robot_max_positions)
        for joint in range(1, 8):
            self.data.joint("panda0_joint" + str(joint)).qpos = init_joints[joint-1]
        
        self.data.joint("panda0_finger_joint1").qpos = 0.04
        self.data.joint("panda0_finger_joint2").qpos = 0.04

        # Randomize tank position
        tank_init_pos = self.np_random.uniform(low=self.tank_range_low, high=self.tank_range_high)
        self.model.body_pos[self.data.body('water_tank').id] = tank_init_pos
        self.tank_pos = tank_init_pos
            
        # Put the water particles inside the mug
        mujoco.mj_forward(self.model, self.data)
        for i in range(1, self.num_particles + 1):
            mug_pos = self.data.body('mug_0').xpos
            for i in range(1, self.num_particles + 1):
                self.model.body_pos[-i] = mug_pos
        mujoco.mj_forward(self.model, self.data)

        self.timestep = 0
        self.done = False

        if self.has_viewer:
            self.viewer.sync()

        return self._get_observations()
    
    def _reset_with_states(self, tank_pos, robot_qpos, robot_qvel): # used for traj replay
        mujoco.mj_resetData(self.model, self.data)
        for joint in range(1, 8):
            self.data.joint("panda0_joint" + str(joint)).qpos = robot_qpos[joint-1]
            self.data.joint("panda0_joint" + str(joint)).qvel = robot_qvel[joint-1]

        self.data.joint("panda0_finger_joint1").qpos = 0.04
        self.data.joint("panda0_finger_joint2").qpos = 0.04
        
        self.model.body_pos[self.data.body('water_tank').id] = tank_pos
        self.tank_pos = tank_pos
            
        # Put the water particles inside the mug
        mujoco.mj_forward(self.model, self.data)
        for i in range(1, self.num_particles + 1):
            mug_pos = self.data.body('mug_0').xpos
            for i in range(1, self.num_particles + 1):
                self.model.body_pos[-i] = mug_pos
        mujoco.mj_forward(self.model, self.data)

        self.timestep = 0
        self.done = False

        if self.has_viewer:
            self.viewer.sync()

        return self._get_observations()


    def _get_observations(self):
        pixels = self.renderer.render()
        proprio = self._get_proprio()
        return {'image':pixels, 'proprioception':proprio}
    
    def _get_proprio(self):
        return self.data.qpos[:7]
    
    def reward(self, action):
        particle_pos = self.data.xpos[-self.num_particles:, :].copy()
        upper_limit = (self.tank_size / 2 + self.tank_pos)
        lower_limit = (-self.tank_size / 2 + self.tank_pos)
        x_within = np.logical_and(particle_pos[:, 0] < upper_limit[0], particle_pos[:, 0] > lower_limit[0])
        y_within = np.logical_and(particle_pos[:, 1] < upper_limit[1], particle_pos[:, 1] > lower_limit[1])
        z_within = np.logical_and(particle_pos[:, 2] < upper_limit[2], particle_pos[:, 2] > lower_limit[2])
        xy_within = np.logical_and(x_within, y_within)
        num_success_particles = np.logical_and(z_within, xy_within)
        return np.sum(num_success_particles) * 100 / self.num_particles

    def step(self, action):
        if self.done:
            raise ValueError("executing action in terminated episode")

        self.timestep += 1

        for i in range(int(self.control_timestep / self.model_timestep)):
            self._pre_action(action)
            mujoco.mj_step(self.model, self.data)

        self.renderer.update_scene(self.data, camera=self.camera_name)

        obs = self._get_observations()
        reward, done, info = self._post_action(action)
        if self.has_viewer:
            self.viewer.sync()
        
        return obs, reward, done, info

    def _pre_action(self, action):
        action = np.clip(action, -1.0, 1.0)
        action = self.act_mid + action * self.act_rng  # mean center and scale
        self.data.ctrl[:] = action

    def _post_action(self, action):
        reward = self.reward(action)
        self.done = (self.timestep >= self.horizon) or (reward == 100.0) and not self.ignore_done
        info = {'proprioception': self._get_proprio()}
        return reward, self.done, info

    @property
    def action_spec(self):
        high = np.ones_like(self.model.actuator_ctrlrange[:, 1])
        low = -1.0 * np.ones_like(self.model.actuator_ctrlrange[:, 0])
        return low, high

    @property
    def spec(self):
        this_spec = Spec(self._get_observations().shape[0], self.action_spec[0].shape[0])
        return this_spec


class Spec:
    def __init__(self, observation_dim, action_dim):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
