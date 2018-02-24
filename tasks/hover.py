"""Takeoff task."""

import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask

class Hover(BaseTask):
    """Simple task where the goal is to lift off the ground and reach a target height."""

    def __init__(self):
        # State space: <position_x, .._y, .._z, orientation_x, .._y, .._z, .._w>
        cube_size = 300.0  # env is cube_size x cube_size x cube_size
        self.observation_space = spaces.Box(
            np.array([- cube_size / 2, - cube_size / 2,       0.0, -1.0, -1.0, -1.0, -1.0]),
            np.array([  cube_size / 2,   cube_size / 2, cube_size,  1.0,  1.0,  1.0,  1.0]))
        #print("Takeoff(): observation_space = {}".format(self.observation_space))  # [debug]

        # Action space: <force_x, .._y, .._z, torque_x, .._y, .._z>
        max_force = 25.0
        max_torque = 20.0
        self.action_space = spaces.Box(
            np.array([-max_force, -max_force, -max_force, -max_torque, -max_torque, -max_torque]),
            np.array([ max_force,  max_force,  max_force,  max_torque,  max_torque,  max_torque]))
        #print("Takeoff(): action_space = {}".format(self.action_space))  # [debug]

        # Task-specific parameters
        self.max_duration = 20.0  # secs
        self.max_error_delta = 1.0
        self.max_error_velocity = 1.0
        self.max_error_orientation = 0.5
        self.target_position = np.array([0.0, 0.0, 10.0])
        self.weight_position = 0.6
        self.target_orientation = np.array([0.0, 0.0, 0.0, 1.0])
        self.weight_orientation = 0.1
        self.target_velocity = np.array([0.0, 0.0, 0.0])
        self.weight_velocity = 0.3

    def set_agent(self, agent):
        self.agent = agent

    def reset(self):
        self.last_timestamp = None
        self.last_position = None
        self.combined_reward = None

        self.p =  (0.0, 0.0, 10.0 + np.random.normal(0.5, 0.1))
        # Nothing to reset; just return initial condition
        return Pose(
                position=Point(*self.p),  # drop off from a slight random height
                orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
            ), Twist(
                linear=Vector3(0.0, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, 0.0)
            )

    def calculate_reward(self, timestamp, position, orientation, velocity):
        error_position = np.linalg.norm(self.p - position) 
        error_orientation = np.linalg.norm(self.target_orientation - orientation)
        error_velocity = np.linalg.norm(self.target_velocity - velocity)

        reward = -(self.weight_position * error_position +
                self.weight_orientation * error_orientation +
                self.weight_velocity * error_velocity)
        if self.combined_reward:
            reward += self.combined_reward

        done = False
        # check agent achieved hovering altitude
        if position[2] > self.target_position[2] and\
                position[2] - self.target_position[2] <= self.max_error_delta: 
            reward += 1.0  # bonus reward
            done = True
        elif timestamp > self.max_duration:  # agent has run out of time
            reward -= 1.0  # extra penalty
            done = True
        elif error_velocity > self.max_error_velocity:
            reward -= 0.3
            done = True
        elif np.abs(error_position) <= self.max_error_delta:
            reward -= 0.5
            done = True
        return reward, done

    def set_combined_reward(self, reward):
        self.combined_reward = reward

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        #compute position and velocity
        position = np.array([pose.position.x, pose.position.y, pose.position.z])
        orientation = np.array([pose.orientation.x, pose.orientation.y,
            pose.orientation.z, pose.orientation.w])

        if self.last_timestamp is None:
            velocity = np.array([0.0, 0.0, 0.0])
        else:
            velocity = (position -self.last_position)/ max(timestamp - self.last_timestamp,
                    0.001)
        state = np.concatenate([position, orientation, velocity])
        self.last_timestamp = timestamp
        self.last_position = position
        # Compute reward / penalty and check if this episode is complete
        reward, done = self.calculate_reward(timestamp, position, orientation, velocity)

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
