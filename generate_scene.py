import os

from tools.gen import SceneGenWindowUI,load_json

if __name__ == '__main__':
    cur_path = os.path.abspath(os.path.dirname(__file__))
    root_path = cur_path[:cur_path.find('pyrdwdata') + len('pyrdwdata')]

    pui = SceneGenWindowUI(load_json(root_path+'\\tools\\gen_scene.json'))
    pui.proc_callback()
    pui.mainloop()
