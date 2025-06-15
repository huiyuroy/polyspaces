from tools.gen import load_scene
from tools.simu_traj_gen import obtain_prm_trajectories

if __name__ == '__main__':
    v_path = 'E:\\polyspaces\\vir'
    p_path = 'E:\\polyspaces\\phy'

    vname = 'abnormal_s3'
    pname = 'test8'

    tar_path = v_path + '\\' + vname + '.json'

    tar_scene = load_scene(tar_path)
    tar_scene.update_roadmap()
    tar_scene.update_grids_runtime_attr()

    all_trajectories = obtain_prm_trajectories(scene=tar_scene)
