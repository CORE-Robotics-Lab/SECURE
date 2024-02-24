from typing import Any, Dict

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.utils import distance


class SafePush(Task):
    def __init__(
        self,
        sim,
        reward_type="sparse",
        distance_threshold=0.03,  # ori: 0.01
        goal_xy_range=0.3,
        obj_xy_range=0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.04
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, 0])
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        box_shape = np.array([2, 2, 10])
        small_box_shape = np.array([1, 1, 4])
        self.box_z_off = box_shape[2] * self.object_size / 2
        self.small_box_z_off = (box_shape[2] + (small_box_shape[2] / 2)) * self.object_size
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.2, width=1.2, height=0.4, x_offset=0)
        self.sim.create_box(
            body_name="object",
            half_extents=box_shape * self.object_size / 2,
            mass=1.0,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )
        self.sim.create_box(
            body_name="target",
            half_extents=box_shape * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )

        # self.sim.create_box(
        #     body_name="small_box",
        #     half_extents=small_box_shape * self.object_size / 2,
        #     mass=0.0,
        #     position=np.array([0.4, 0.0, self.small_box_z_off]),
        #     rgba_color=np.array([0.2, 0.2, 0.8, 1.0]),
        # )

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        object_position = np.array(self.sim.get_base_position("object"))
        object_rotation = np.array(self.sim.get_base_rotation("object"))
        object_velocity = np.array(self.sim.get_base_velocity("object"))
        object_angular_velocity = np.array(self.sim.get_base_angular_velocity("object"))
        observation = np.concatenate(
            [
                object_position,
                object_rotation,
                object_velocity,
                object_angular_velocity,
            ]
        )
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        object_position = np.array(self.sim.get_base_position("object"))
        return object_position

    def reset(self) -> None:
        # todo: change the height of the block

        # yy: make the height changeable -> random height
        # self.object_size = np.random.choice(np.arange(0.04, 0.08, 0.001), 1)
        # yy: make the height fixed to 0.2 * 2
        self.object_size = 0.04
        # print("Obj Size: ", self.object_size)
        # self.sim.close()
        with self.sim.no_rendering():
            self.sim.remove_geometry("object")
            self.sim.remove_geometry("target")
            # self.sim.remove_geometry("small_box")
            # self.sim.resetSimulation()
            self._create_scene()
            # self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)


        self.goal = self._sample_goal()
        object_position = self._sample_object()
        # self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("target", np.array([0.55, 0, self.box_z_off]), np.array([0.0, 0.0, 0.0, 1.0]))

        # self.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object", np.array([0.4, 0, self.box_z_off]), np.array([0.0, 0.0, 0.0, 1.0]))

        # self.sim.set_base_pose("small_box", np.array([0.4, 0, 0.4]), np.array([0.0, 0.0, 0.0, 1.0]))
        # print(self.small_box_z_off)

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = np.array([0.0, 0.0, self.object_size / 2])  # z offset for the cube center
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        goal += noise
        return goal

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        object_position = np.array([0.0, 0.0, self.box_z_off])
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        return object_position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        # d = distance(achieved_goal, desired_goal)
        d = distance(self.sim.get_base_position("target"), self.sim.get_base_position("object"))
        # print(d, self.distance_threshold)

        # if bool(np.array(d < self.distance_threshold, dtype=np.bool8)):
        #     print(">> Success! Distance: {}, Threshold: {}".format(d, self.distance_threshold))

        # print(self.sim.get_base_position("target"), self.sim.get_base_position("object"), d, self.distance_threshold)
        # return np.array(d < self.distance_threshold, dtype=np.bool8)
        return d < self.distance_threshold


    def is_terminate(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(self.sim.get_base_position("target"), self.sim.get_base_position("object"))
        return d < 0.01

    def is_fall(self):
        # print("fall_angle: ", np.array(self.sim.get_base_rotation("object"))[1])
        # print("fall_angle_small_box: ", abs(np.array(self.sim.get_base_rotation("small_box"))[1]))
        # return abs(np.array(self.sim.get_base_rotation("object"))[1]), abs(np.array(self.sim.get_base_rotation("small_box"))[1])
        return abs(np.array(self.sim.get_base_rotation("object"))[1]), 0
        # return np.array(self.sim.get_base_rotation("object"))[2] >= 0.15
        # return self.sim.get_base_position("object")[2] <= ((self.object_size / 2) / 8)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32)
        else:
            return -d.astype(np.float32)
