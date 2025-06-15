import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

from tools.gen import TrajectoryModPopupUI, load_json, load_scene, load_trajectories, TrajectoryType

if __name__ == '__main__':
    cur_path = os.path.abspath(os.path.dirname(__file__))
    root_path = cur_path[:cur_path.find('pyrdwdata') + len('pyrdwdata')]
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    file_path = filedialog.askopenfilename(initialdir=root_path)  # 打开文件对话框

    pui = TrajectoryModPopupUI(load_json("./tools/trajmod_popup_ui.json"))
    scene = load_scene(file_path)
    trajs = load_trajectories(str(Path(file_path).parent), Path(file_path).name.split('.')[0])
    abs_road = []
    prox_road = []
    abs_rand = []
    for traj_i, traj in enumerate(trajs):
        traj.range_distance(20000)
        tars = [[*td, t_i] for t_i, td in enumerate(traj.walkable().tolist())]
        if traj.type == 'absolute roadmap':
            abs_road.append(tars)
        elif traj.type == 'approximate roadmap':
            prox_road.append(tars)
        elif traj.type == 'absolute random' or traj.type == 'grid_rand':
            abs_rand.append(tars)
        print(f"\rtraj read:{traj_i / len(trajs) * 100}%", end="")
    print()
    scene.simu_trajs_abs_road_targets = abs_road
    scene.simu_trajs_prox_road_targets = prox_road
    scene.simu_trajs_abs_rand_targets = abs_rand

    pui.process_v_scene(scene)
    pui.proc_callback()
    pui.mainloop()
