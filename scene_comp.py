import os

import numpy as np

from common import data_path
from tools.gen import load_scene

if __name__ == '__main__':
    v_path = data_path + '\\vir'
    v_files = [os.path.join(v_path, file) for file in os.listdir(v_path) if 'json' in file]
    scenes_name = []
    scenes_size = []
    scenes_area = []
    for f_path in v_files:
        tar_scene = load_scene(f_path)
        scenes_name.append(tar_scene.name)
        scenes_size.append(tar_scene.max_size)
        scenes_area.append(tar_scene.max_size[0] * tar_scene.max_size[1])
    scenes_size = np.array(scenes_size)
    scenes_area = np.array(scenes_area)
    max_i = np.argmax(scenes_area)
    min_i = np.argmin(scenes_area)
    print(f"{scenes_name[min_i]}, size {scenes_size[min_i]}")
    print(f"{scenes_name[max_i]}, size {scenes_size[max_i]}")



    # for i, name in enumerate(scenes_name):
    #     print(f"{name}, size {scenes_size[i]}")
