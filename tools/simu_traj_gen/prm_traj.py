import pickle
from typing import Sequence

import numpy as np
from scipy.spatial import KDTree

from pyrdw import DiscreteScene
from tools.geometry import chk_line_bound_cross, norm_vec
from tools.algebra import l2_norm


class PRMTrajectoryGenerator:
    class Node:
        """
        Node class for dijkstra search
        """

        def __init__(self, x, y, cost, parent_index):
            self.x = x
            self.y = y
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + \
                str(self.cost) + "," + str(self.parent_index)

    def __init__(self, scene):
        self.scene: DiscreteScene = scene
        self.sample_num = 1000
        self.sample_points = []
        self.sample_KDTree: KDTree = None
        self.sample_roadmap: list[list] = None
        self.sample_rm_weights = None

    def __sample_scene(self):
        max_x, max_y = self.scene.max_size

        self.sample_points = []
        for _ in range(self.sample_num):
            find = False
            while not find:
                x, y = np.random.rand(2)
                p = [x * max_x, y * max_y]
                cur_tiling, _ = self.scene.calc_user_tiling_conv(p)
                if cur_tiling.type:
                    self.sample_points.append(p)
                    find = True

        self.sample_KDTree = KDTree(self.sample_points)

    def __prepare_roadmap(self, N_KNN=10):
        """
        use KNN algorithm to prepare the PRM.

        Args:
            N_KNN: number of the K-nearest neighbors

        Returns:

        """
        self.sample_roadmap = []
        for point in self.sample_points:
            # returns all nearest sample points sorted by the distance to current point
            dists, indexes = self.sample_KDTree.query(point, k=self.sample_num)
            neighbors_id = []
            for i in range(1, len(indexes)):
                other_id = indexes[i]
                other_p = self.sample_points[other_id]

                if not chk_line_bound_cross(point, other_p, self.scene.bounds):
                    neighbors_id.append(other_id)

                if len(neighbors_id) >= N_KNN:
                    break
            self.sample_roadmap.append(neighbors_id)

        self.sample_rm_weights = [[1] * len(rm) for rm in self.sample_roadmap]

    def generate_prm(self):
        self.__sample_scene()
        self.__prepare_roadmap()

    def generate_single_traj(self, traj_dis=100 * 1e3):
        """

        Args:
            traj_dis: the target walking distance of a trajectory, 100 meters by default

        Returns:

        """
        traj_points = []
        all_id = list(range(self.sample_num))
        current_id = np.random.choice(all_id)
        walked_dis = 0
        select_weights = pickle.loads(pickle.dumps(self.sample_rm_weights))

        while True:
            current_point = self.sample_points[current_id]
            traj_points.append(current_point)
            nei_ids = self.sample_roadmap[current_id]

            nei_weight = np.array(select_weights[current_id])
            nei_weight = nei_weight / np.sum(nei_weight)

            nei_ids_id = np.random.choice(list(range(len(nei_ids))), p=nei_weight)

            next_id = nei_ids[nei_ids_id]
            next_point = self.sample_points[next_id]
            v = np.array(next_point) - np.array(current_point)
            d = l2_norm(v)
            wd = d + walked_dis
            if wd < traj_dis:
                select_weights[current_id][nei_ids_id] -= 0.1
                if select_weights[current_id][nei_ids_id] <= 0:
                    select_weights[current_id] = (np.array(select_weights[current_id]) + 1).tolist()
                current_id = next_id
                walked_dis = wd
            else:
                over_d = wd - traj_dis
                final_d = d - over_d
                final_point = (np.array(current_point) + norm_vec(v) * final_d).tolist()
                traj_points.append(final_point)
                break
        return traj_points

    def dijkstra_planning(self, start: Sequence[float], target: Sequence[float]):
        self.sample_points.append(start)
        self.sample_points.append(target)

        self.__prepare_roadmap()

        start_node = PRMTrajectoryGenerator.Node(*start, 0.0, -1)
        goal_node = PRMTrajectoryGenerator.Node(*target, 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[len(self.sample_roadmap) - 2] = start_node

        path_found = True

        while True:
            if not open_set:
                print("Cannot find path")
                path_found = False
                break

            c_id = min(open_set, key=lambda o: open_set[o].cost)
            current = open_set[c_id]

            if c_id == (len(self.sample_roadmap) - 1):
                print("goal is found!")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]
            # Add it to the closed set
            closed_set[c_id] = current

            # expand search grid based on motion model
            for i in range(len(self.sample_roadmap[c_id])):
                n_id = self.sample_roadmap[c_id][i]
                dx = self.sample_points[n_id][0] - current.x
                dy = self.sample_points[n_id][1] - current.y
                d = l2_norm(np.array([dx, dy]))
                node = PRMTrajectoryGenerator.Node(*self.sample_points[n_id], current.cost + d, c_id)

                if n_id in closed_set:
                    continue
                # Otherwise if it is already in the open set
                if n_id in open_set:
                    if open_set[n_id].cost > node.cost:
                        open_set[n_id].cost = node.cost
                        open_set[n_id].parent_index = c_id
                else:
                    open_set[n_id] = node

        self.sample_points.pop()
        self.sample_points.pop()

        if path_found is False:
            return None
        else:
            # generate final course
            traj = []
            parent_index = goal_node.parent_index
            while parent_index != -1:
                n = closed_set[parent_index]
                traj.append([n.x, n.y])
                parent_index = n.parent_index
            traj.reverse()

            return traj


def obtain_prm_trajectories(scene, traj_num=100, traj_len=100 * 1e3):
    all_trajs = []
    prm_gen = PRMTrajectoryGenerator(scene)
    prm_gen.generate_prm()

    for progress in range(traj_num):
        t = prm_gen.generate_single_traj(traj_len)
        all_trajs.append(t)
        print('\rgenerating road with {:.2f}%'.format(progress / traj_num * 100), end="")
    print()
