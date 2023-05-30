# TOTO Dataset

### Format
The data is parsed into a python list of dictionaries, where each dictionary is a trajectory. The keys of each dictionary are as follows:
- `'traj_id'`: string. The name of the folder containing all images of the trajectory
- `'observations'`: numpy array of shape (*horizon*, 7). Contains the robot's jointstates (as absolute joint angles) at each timestep
- `'cam0c'`: python array of length *horizon*. Contains the filename of RGB images
- `'cam0d'`: python array of length *horizon*. Contains the filename of depth images
- `'actions'`: numpy array of shape (*horizon*, 7). Contains the robot's actions (as absolute joint angles) at each timestep
- `'rewards'`: numpy array of length (*horizon*, ). The reward is sparse, which means the reward at all but the last step is 0. We haven’t done any normalization for the reward in the current version
- `'terminated'`: numpy array of length (*horizon*, ). This denotes whether the trajectory is terminated at each timestep, so all the entries is 0 except the last entry being 1

- End Effector Pose keys
  - In addition to keys listed above, we have added support to observe and predict end-effector poses.  The new folder `eef_poses` contains pickle files as well as  instructions on how to use the new keys.
      - `'eef_pose_observations'`: numpy array of shape (*horizon*, 7). Contains the robot's absolute end effector poses (position + quaternion) at each timestep
      -  `'eef_pose_actions'`: numpy array of shape (*horizon*, 7). Contains the robot's actions (as absolute end effector poses) at each timestep

Sample trajectories: data_samples.zip. This contains 20 trajectories.

### Tasks

#### Scooping
- Zip file(s): cloud-datset-scooping.zip
- 3 containers
- 3 scooping materials
- 6 positions
- 1900 trajectories
#### Pouring
- Zip file(s): cloud-datset-pouring.zip
- 4 containers
- 2 pouring materials
- 6 positions
- 1003 trajectories
