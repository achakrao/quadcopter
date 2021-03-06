"""Takeoff task."""

import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask

class Landing(BaseTask):
    """Simple task where the goal is to lift off the ground and reach a target height."""

    def __init__(self):
        # State space: <position_x, .._y, .._z, orientation_x, .._y, .._z, .._w>
        cube_size = 300.0  # env is cube_size x cube_size x cube_size
        self.observation_space = spaces.Box(
            np.array([- cube_size / 2, - cube_size / 2,       0.0, -1.0, -1.0, -1.0, -1.0]),
            np.array([  cube_size / 2,   cube_size / 2, cube_size,  1.0,  1.0,  1.0,  1.0]))

        # Action space: <force_x, .._y, .._z, torque_x, .._y, .._z>
        max_force = 20.0
        max_torque = 20.0
        self.action_space = spaces.Box(
            np.array([0, 0, -max_force, 0, 0, -max_torque]),
            np.array([0, 0,  max_force, 0, 0,  max_torque]))
        #print("Takeoff(): action_space = {}".format(self.action_space))  # [debug]

        # Task-specific parameters
        self.max_duration = 10.0  # secs
        self.target_z = 0.0001  # target height (z position) to reach for successful landing
        # reward specific parameters
        self.target_position = np.array([0.0, 0.0, 0.0])

    def set_agent(self, agent):
        self.agent = agent

    def reset(self):
        self.last_timestamp = None
        self.last_position = None
        return Pose(
                position=Point(0.0, 0.0, 10.0 + np.random.normal(0.5, 0.1)),
                orientation=Quaternion(0.0, 0.0, 0.0, 0.0),
            ), Twist(
                linear=Vector3(0.0, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, 0.0)
            )

    def calculate_reward(self, timestamp, pose, velocity, linear_acceleration):
        # Compute reward / penalty and check if this episode is complete
        done = False

        reward = 0
        if pose.position.z <= self.target_z:  # agent has landed
            reward += 1.0  # bonus reward
            done = True
        elif timestamp > self.max_duration:  # agent has run out of time
            reward -= 0.9  # extra penalty
            done = True

        return reward, done

    def compute_velocity(self, position, timestamp):
        if self.last_timestamp is None:
            velocity = np.array([0.0, 0.0, 0.0])
        else:
            velocity = (position -self.last_position)/ max(timestamp - self.last_timestamp,
                    0.001)
        return velocity

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        #compute position and velocity
        position = np.array([pose.position.x, pose.position.y, pose.position.z])
        orientation = np.array([pose.orientation.x, pose.orientation.y,
            pose.orientation.z, pose.orientation.w])
        # compute velocity
        velocity = self.compute_velocity(position, timestamp)
        # Prepare state vector (pose only; ignore angular_velocity, linear_acceleration)
        state = np.concatenate([position, orientation, velocity])
        self.last_timestamp = timestamp
        self.last_position = position
        # Compute reward / penalty and check if this episode is complete
        reward, done = self.calculate_reward(timestamp, pose, velocity, linear_acceleration)
        # Take one RL step, passing in current state and reward, and obtain action
        # Note: The reward passed in here is the result of past action(s)
        action = self.agent.step(state, reward, done)  # note: action = <force; torque> vector

        # Convert to proper force command (a Wrench object) and return it
        if action is not None:
            action = np.clip(action.flatten(), self.action_space.low, self.action_space.high)  # flatten, clamp to action space limits
            return Wrench(
                    force=Vector3(action[0], action[1], action[2]),
                    torque=Vector3(action[3], action[4], action[5])
                ), done
        else:
            return Wrench(), done
