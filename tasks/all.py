"""Combined task."""

import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask
from quad_controller_rl.tasks.takeoff import Takeoff
from quad_controller_rl.tasks.hover import Hover
from quad_controller_rl.tasks.landing import Landing

class All(BaseTask):
    """Simple task where the goal is to lift off the ground and reach a target height."""

    def __init__(self):
        self.flag = "takeoff"
        self.takeoff = Takeoff()
        self.hover = Hover()
        self.landing = Landing()
        # State space: <position_x, .._y, .._z, orientation_x, .._y, .._z, .._w>
        cube_size = 300.0  # env is cube_size x cube_size x cube_size
        self.observation_space = spaces.Box(
            np.array([- cube_size / 2, - cube_size / 2,       0.0, -1.0, -1.0, -1.0, -1.0]),
            np.array([  cube_size / 2,   cube_size / 2, cube_size,  1.0,  1.0,  1.0,  1.0]))

        self.target_z = 10.0
        self.hover_height_delta = 1.0
        self.action_space = self.takeoff.action_space
        self.max_duration = 20

    def set_action_space(self, task):
        if isinstance(task, Takeoff):
            self.action_space = self.takeoff.action_space
        elif isinstance(task, Hover):
            self.action_space = self.hover.action_space
        elif isinstance(task, Landing):
            self.action_space = self.landing.action_space

    def reset(self):
        agt = super().get_agent()
        self.landing.set_agent(agt)
        self.takeoff.set_agent(agt)
        self.hover.set_agent(agt)

        if self.flag == "takeoff":
            self.set_action_space(self.takeoff)
            return self.takeoff.reset()
        elif self.flag == "hover":
            self.set_action_space(self.hover)
            return self.hover.reset()
        elif self.flag == "landing":
            self.set_action_space(self.landing)
            return self.landing.reset()

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        if self.flag == "takeoff":
            wrench = self.takeoff.update(timestamp, pose, angular_velocity, linear_acceleration)
            if pose.position.z >= self.target_z:
                self.flag = "hover"
            return wrench
        elif self.flag == "hover":
            wrench = self.hover.update(timestamp, pose, angular_velocity, linear_acceleration)
            if abs(pose.position.z - self.target_z) <= self.hover_height_delta:
                self.flag = "landing"
            else:
                self.hover.set_combined_reward(-0.9)
            if timestamp > self.max_duration:
                self.hover.set_combined_reward(-0.8)
            return wrench
        elif self.flag == "landing":
            wrench = self.landing.update(timestamp, pose, angular_velocity, linear_acceleration)
            if pose.position.z <= 0.1:
                print('Simulation completed')
                return None
            return wrench
