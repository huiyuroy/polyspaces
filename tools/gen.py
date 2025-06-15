import sys

import ujson
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from tools import *
import tools.geometry as geo
import tools.algebra as alg

from tools.space import PI
from tools.space.boundary import Boundary
from tools.space.scene import Scene
from tools.space.trajectory import Trajectory


class DrawType(Enum):
    Triangle = 0
    Rectangle = 1
    Pentagon = 2
    Hexagon = 3
    Circle = 4
    Custom = 5


class TrajectoryType(Enum):
    AbsRoad = 0
    ProxRoad = 1
    AbsRand = 2
    TilingRand = 3


def get_files(directory, extension):
    files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.' + extension):
                filepath = os.path.join(root, filename)
                files.append(filepath)
    return files


def load_bound(bound_attr):
    bound = Boundary()
    bound.set_contour(bound_attr["is_out_bound"], bound_attr["points"])
    bound.center = np.array(bound_attr["center"])
    bound.barycenter = bound_attr["barycenter"]
    bound.cir_rect = bound_attr["cir_rect"]
    return bound


def load_contours(scene, data):
    scene.name = data['name']
    scene.bounds = []
    for bound_attr in data['bounds']:
        scene.bounds.append(load_bound(bound_attr))
    scene.max_size = data["max_size"]
    scene.scene_center = np.array(data["scene_center"])


def load_scene(s_path):
    scene_attri = load_json(s_path)
    s = Scene()
    load_contours(s, scene_attri)
    return s


def load_json(j_path):
    if j_path is not None:
        with open(j_path, mode='r') as f:
            j_result = ujson.load(f)
            return j_result


def load_trajectories(tar_path, scene_name):
    all_traj_files = get_files(tar_path + '\\simu_trajs\\{}'.format(scene_name), 'json')
    trajs = []

    for t_type, t_tars in list(map(load_trajectory, all_traj_files)):
        t = Trajectory()
        t.type = t_type
        t.tar_data = tuple(t_tars)
        t.tar_num = len(t_tars)
        t.end_idx = t.tar_num - 1
        trajs.append(t)
    return tuple(trajs)


def load_trajectory(tar_path):
    traj_data = load_json(tar_path)
    traj_type = traj_data['type']
    traj_tars = traj_data['targets']
    return traj_type, traj_tars


def save_bound(bound):
    return {"is_out_bound": bound.is_out_bound,
            "points": np.array(np.around(bound.points, decimals=4), dtype='float').tolist(),
            "center": np.array(np.around(bound.center, decimals=4), dtype='float').tolist(),
            "barycenter": np.array(np.around(bound.barycenter, decimals=4), dtype='float').tolist(),
            "cir_rect": np.array(np.around(bound.cir_rect, decimals=4), dtype='float').tolist()}


def save_convex_poly(convex):
    return {"vertices": np.array(np.around(convex.vertices, decimals=4), dtype='float').tolist(),
            "center": np.array(np.around(convex.center, decimals=4), dtype='float').tolist(),
            "barycenter": np.array(np.around(convex.barycenter, decimals=4), dtype='float').tolist(),
            "cir_circle": [np.array(np.around(convex.cir_circle[0], decimals=4), dtype='float').tolist(),
                           float(np.around(convex.cir_circle[1], decimals=4))],
            "in_circle": [np.array(np.around(convex.in_circle[0], decimals=4), dtype='float').tolist(),
                          float(np.around(convex.in_circle[1], decimals=4))],
            "cir_rect": np.array(convex.cir_rect).copy().tolist(),
            "out_edges": np.array(convex.out_edges).copy().tolist(),
            "in_edges": np.array(convex.in_edges).copy().tolist()}


def save_contours(scene):
    return {'name': scene.name,
            'bounds': list(map(save_bound, scene.bounds)),
            'max_size': np.array(np.around(scene.max_size, decimals=4), dtype='float').tolist(),
            "out_bound_conv": save_convex_poly(scene.out_bound_conv),
            "out_conv_hull": save_convex_poly(scene.out_conv_hull),
            "scene_center": np.array(np.around(scene.scene_center, decimals=4), dtype='float').tolist()}


def save_scene(scene, tar_path=None):
    contour_data = save_contours(scene)
    # if not os.path.exists(tar_path):
    #     os.makedirs(tar_path)
    done = save_json(contour_data, tar_path + '.json'.format(scene.name))
    return done


def save_json(data, file_path=None):
    with open(file_path, mode='w') as f:
        ujson.dump(data, f, ensure_ascii=True, indent=2)
        return True


def save_trajectories(tar_path, scene_name, trajs, traj_type='abs_road'):
    """

    Args:
        tar_path:
        scene_name:
        trajs:
        traj_type:
            - abs_road
            - approx_road
            - abs_rand

    Returns:

    """
    tar_path += '\\simu_trajs\\{}'.format(scene_name)
    for tid, t in enumerate(trajs):
        p = tar_path + '\\' + traj_type + '{}.json'.format(tid)


class BaseWindowUI(tk.Tk):
    def __init__(self, ui_spec):
        super().__init__()
        self.ui_spec = ui_spec
        self.title(self.ui_spec['title'])
        w, h, x, y = self.ui_spec['width'], self.ui_spec['height'], self.ui_spec['x'], self.ui_spec['y']
        self.geometry('{}x{}+{}+{}'.format(w, h, x, y))
        self.config(background=self.ui_spec['bg'])
        self.resizable(0, 0)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.__create_components()

    def __create_components(self):
        com_spec = self.ui_spec['components']
        for key in com_spec:
            c_spec = com_spec[key]
            c_loc = c_spec['size']
            setattr(self, key, None)
            if 'label' in key:
                setattr(self, key,
                        tk.Label(self, text=c_spec['text'], font=c_spec['font'], anchor=c_spec['anchor']))
                getattr(self, key).config(bg=c_spec['bg'])
            elif 'scale' in key:
                setattr(self, key, tk.Scale(self, from_=c_spec['range'][0], to=c_spec['range'][1],
                                            orient=tk.HORIZONTAL, length=c_spec['length'], showvalue=False,
                                            relief=tk.RIDGE, font=c_spec['font']))
                getattr(self, key).config(bg=c_spec['bg'])
            elif 'entry' in key:
                setattr(self, key, tk.Entry(self, relief=tk.RIDGE, font=c_spec['font']))
                getattr(self, key).config(bg=c_spec['bg'])
            elif 'button' in key:
                setattr(self, key, tk.Button(self, text=c_spec['text'], font=c_spec['font']))
                c_state = tk.NORMAL if c_spec['state'] == 'Enable' else tk.DISABLED
                getattr(self, key).config(bg=c_spec['bg'], state=c_state)
            elif 'canvas' in key:
                setattr(self, key, tk.Canvas(self))
                getattr(self, key).pack()
                getattr(self, key).config(bg=c_spec['bg'])
            elif 'listbox' in key:
                setattr(self, key, tk.Listbox(self))
                getattr(self, key).pack()
            elif 'progress' in key:
                setattr(self, key, ttk.Progressbar(self))
                getattr(self, key).pack(side=tk.TOP)
                getattr(self, key)['maximum'] = 100
                getattr(self, key)['value'] = 0
            c_obj = getattr(self, key)
            if c_obj is not None:
                c_obj.place(x=c_loc[0], y=c_loc[1], width=c_loc[2], height=c_loc[3])

    @abstractmethod
    def proc_callback(self):
        raise NotImplementedError

    def on_closing(self):
        self.destroy()
        sys.exit()


class SceneGenWindowUI(BaseWindowUI):

    def __init__(self, ui_spec):
        super().__init__(ui_spec=ui_spec)
        cur_path = os.path.abspath(os.path.dirname(__file__))
        self.root_path = cur_path[:cur_path.find('pyrdwdata') + len('pyrdwdata')]
        self.phy_draw_type = None
        self.phy_canvas_size = []
        self.phy_canvas_center = []
        self.cur_focus_center = []
        self.phy_max_len = 20  # 最长20m
        self.phy_scale = 25
        self.phy_pre_scale = 25
        self.phy_scene_size = []
        self.phy_bounds = []
        self.phy_cur_contour_bounds = []
        self.phy_bounds_idx = -1
        self.phy_rect_area = [None] * 2
        self.phy_rect_idx = 0
        self.phy_custom_contour_done = False
        self.canvas_vel = 0.5  # 0.5m/time
        self.show_tri = False
        self.show_mer_bound = False

    def proc_callback(self):
        com_spec = self.ui_spec['components']
        for key in com_spec:
            c_obj = getattr(self, key)
            c_spec = com_spec[key]
            c_loc = c_spec['size']
            if 'canvas' in key:
                if key == "phy_canvas":
                    self.phy_canvas_size = c_loc[2:4]
                    self.phy_canvas_center = [int(i * 0.5) for i in self.phy_canvas_size]
                    self.cur_focus_center = [10, 10]  # 10m,10m
                    self.phy_scene_size = [0, 0]
                    c_obj.bind("<B1-Motion>", self.callback_motion)
                    c_obj.bind("<ButtonPress-1>", self.callback_left_click)
                    c_obj.bind("<ButtonPress-3>", self.callback_right_click)
                    c_obj.bind('<MouseWheel>', self.callback_pc_wheel)
                    c_obj.bind('<ButtonRelease-1>', self.callback_left_release)
            elif 'button' in key:
                callback_func = None
                if 'quest' in key:
                    callback_func = self.callback_quest_btn
                elif 'add_contour' in key:
                    callback_func = self.callback_add_contour_btn
                elif 'tri_draw' in key:
                    callback_func = self.callback_add_tri_btn
                elif 'rect_draw' in key:
                    callback_func = self.callback_add_rect_btn
                elif 'pent_draw' in key:
                    callback_func = self.callback_add_pent_btn
                elif 'hex_draw' in key:
                    callback_func = self.callback_add_hex_btn
                elif 'custom_draw' in key:
                    callback_func = self.callback_add_custom_btn
                elif 'end_contour' in key:
                    callback_func = self.callback_end_contour_btn
                elif 'rmv_last' in key:
                    callback_func = self.callback_remove_last_btn
                elif 'rmv_all' in key:
                    callback_func = self.callback_remove_all_btn
                elif 'tri_contour' in key:
                    callback_func = self.callback_tri_btn
                elif 'open' in key:
                    callback_func = self.callback_open_btn
                elif 'save' in key:
                    callback_func = self.callback_save_btn

                c_obj.bind('<Button-1>', callback_func)
        self.bind('<KeyPress>', self.callback_keyboard)
        # self.phy_bounds = self.env_mg.gen_phy_contour()
        self.refresh_prender()

    def callback_left_click(self, event):
        click_x, click_y = event.x, self.phy_canvas_size[1] - event.y
        tar_x = round((click_x - self.phy_canvas_center[0]) / self.phy_scale + self.cur_focus_center[0], 4)
        tar_y = round((click_y - self.phy_canvas_center[1]) / self.phy_scale + self.cur_focus_center[1], 4)
        if self.phy_draw_type != DrawType.Custom.value:
            self.phy_rect_area[0] = [tar_x, tar_y]
            self.phy_rect_area[1] = None
        else:
            self.phy_cur_contour_bounds.append([tar_x, tar_y])
        self.refresh_prender()

    def callback_right_click(self, event):
        click_x, click_y = event.x, self.phy_canvas_size[1] - event.y
        tar_x = round((click_x - self.phy_canvas_center[0]) / self.phy_scale + self.cur_focus_center[0], 4)
        tar_y = round((click_y - self.phy_canvas_center[1]) / self.phy_scale + self.cur_focus_center[1], 4)
        if self.phy_draw_type == DrawType.Custom.value:
            cur_num = len(self.phy_cur_contour_bounds)
            if cur_num >= 3:
                for i in range(-1, cur_num - 1):
                    ls, le = self.phy_cur_contour_bounds[i], self.phy_cur_contour_bounds[i + 1]
                    dis, _, t = geo.calc_point_pro2line([tar_x, tar_y], ls, le)
                    if 1 > t > 0 and dis <= 0.05:
                        self.phy_cur_contour_bounds.pop(i + 1)
                        break
            elif cur_num == 2:
                ls, le = self.phy_cur_contour_bounds[0], self.phy_cur_contour_bounds[1]
                dis, _, t = geo.calc_point_pro2line([tar_x, tar_y], ls, le)
                if 1 > t > 0 and dis <= 0.05:
                    self.phy_cur_contour_bounds.clear()
        self.refresh_prender()

    def callback_left_release(self, event):
        click_x, click_y = event.x, self.phy_canvas_size[1] - event.y
        tar_x = round((click_x - self.phy_canvas_center[0]) / self.phy_scale + self.cur_focus_center[0], 4)
        tar_y = round((click_y - self.phy_canvas_center[1]) / self.phy_scale + self.cur_focus_center[1], 4)
        if self.phy_draw_type != DrawType.Custom.value:
            if len(self.phy_cur_contour_bounds) > 0:
                self.phy_end_contour_button.config(state=tk.NORMAL)
        else:
            if len(self.phy_cur_contour_bounds) >= 3:
                head_x, head_y = self.phy_cur_contour_bounds[0]
                if alg.l2_norm(np.array([tar_x - head_x, tar_y - head_y])) <= 0.05:
                    self.phy_cur_contour_bounds[-1] = [head_x, head_y]  # 精确到小数点后4位即可
                    self.phy_end_contour_button.config(state=tk.NORMAL)
                    self.phy_custom_contour_done = True
            # print('cur', self.phy_cur_contour_bounds)
        self.refresh_prender()

    def callback_motion(self, event):
        # 精确到小数点后4位即可
        click_x, click_y = event.x, self.phy_canvas_size[1] - event.y
        tar_x = round((click_x - self.phy_canvas_center[0]) / self.phy_scale + self.cur_focus_center[0], 4)
        tar_y = round((click_y - self.phy_canvas_center[1]) / self.phy_scale + self.cur_focus_center[1], 4)
        if self.phy_draw_type != DrawType.Custom.value:
            self.phy_rect_area[1] = [tar_x, tar_y]
            self.calc_cur_regpoly_contour()
        else:
            self.phy_cur_contour_bounds[-1] = [tar_x, tar_y]
        self.refresh_prender()

    def callback_pc_wheel(self, event):
        if event.delta > 0:
            self.phy_scale += 1
        else:
            self.phy_scale -= 1
            if self.phy_scale < 25:
                self.phy_scale = 25
        self.refresh_prender()

    def callback_keyboard(self, event):
        if event.keysym == 'd':
            self.cur_focus_center[0] += self.canvas_vel
            if self.cur_focus_center[0] + self.phy_canvas_size[0] / self.phy_scale * 0.5 > self.phy_max_len:
                self.cur_focus_center[0] = self.phy_max_len - self.phy_canvas_size[0] / self.phy_scale * 0.5
        elif event.keysym == 'a':
            self.cur_focus_center[0] -= self.canvas_vel
            if self.cur_focus_center[0] - self.phy_canvas_size[0] / self.phy_scale * 0.5 < 0:
                self.cur_focus_center[0] = self.phy_canvas_size[0] / self.phy_scale * 0.5
        elif event.keysym == 'w':
            self.cur_focus_center[1] += self.canvas_vel
            if self.cur_focus_center[1] + self.phy_canvas_size[1] / self.phy_scale * 0.5 > self.phy_max_len:
                self.cur_focus_center[1] = self.phy_max_len - self.phy_canvas_size[1] / self.phy_scale * 0.5
        elif event.keysym == 's':
            self.cur_focus_center[1] -= self.canvas_vel
            if self.cur_focus_center[1] - self.phy_canvas_size[1] / self.phy_scale * 0.5 < 0:
                self.cur_focus_center[1] = self.phy_canvas_size[1] / self.phy_scale * 0.5
        self.refresh_prender()

    def callback_add_contour_btn(self, event):
        self.phy_draw_type = None
        self.phy_custom_contour_done = False
        self.phy_bounds_idx += 1
        self.phy_bounds.append([])
        self.phy_add_contour_button.config(state=tk.DISABLED)
        self.phy_tri_draw_button.config(state=tk.NORMAL)
        self.phy_rect_draw_button.config(state=tk.NORMAL)
        self.phy_pent_draw_button.config(state=tk.NORMAL)
        self.phy_hex_draw_button.config(state=tk.NORMAL)
        self.phy_custom_draw_button.config(state=tk.NORMAL)
        self.phy_tri_contour_button.config(state=tk.DISABLED)
        self.phy_save_button.config(state=tk.DISABLED)
        self.show_tri = False
        self.refresh_prender()

        # print('info:', self.phy_bounds_idx, self.phy_bounds)

    def callback_add_tri_btn(self, event):
        self.phy_draw_type = DrawType.Triangle.value
        self.phy_cur_contour_bounds.clear()
        self.refresh_prender()

    def callback_add_rect_btn(self, event):
        self.phy_draw_type = DrawType.Rectangle.value
        self.phy_cur_contour_bounds.clear()
        self.refresh_prender()

    def callback_add_pent_btn(self, event):
        self.phy_draw_type = DrawType.Pentagon.value
        self.phy_cur_contour_bounds.clear()
        self.refresh_prender()

    def callback_add_hex_btn(self, event):
        self.phy_draw_type = DrawType.Hexagon.value
        self.phy_cur_contour_bounds.clear()
        self.refresh_prender()

    def callback_add_custom_btn(self, event):
        self.phy_draw_type = DrawType.Custom.value
        self.phy_custom_contour_done = False
        self.phy_cur_contour_bounds.clear()
        self.refresh_prender()

    def callback_end_contour_btn(self, event):
        if self.phy_draw_type == DrawType.Custom.value:
            self.phy_cur_contour_bounds.pop()
        self.phy_bounds[-1] = pickle.loads(pickle.dumps(self.phy_cur_contour_bounds))
        self.phy_cur_contour_bounds.clear()
        self.phy_end_contour_button.config(state=tk.DISABLED)
        self.phy_tri_draw_button.config(state=tk.DISABLED)
        self.phy_rect_draw_button.config(state=tk.DISABLED)
        self.phy_pent_draw_button.config(state=tk.DISABLED)
        self.phy_hex_draw_button.config(state=tk.DISABLED)
        self.phy_custom_draw_button.config(state=tk.DISABLED)
        self.phy_add_contour_button.config(state=tk.NORMAL)
        self.phy_rmv_last_button.config(state=tk.NORMAL)
        self.phy_rmv_all_button.config(state=tk.NORMAL)
        self.phy_tri_contour_button.config(state=tk.NORMAL)
        self.phy_save_button.config(state=tk.NORMAL)
        self.phy_draw_type = None

    def callback_remove_last_btn(self, event):
        if len(self.phy_bounds) > 0:
            if len(self.phy_bounds[-1]):
                self.phy_bounds.pop()
        if len(self.phy_bounds) == 0:
            self.phy_rmv_last_button.config(state=tk.DISABLED)
        self.refresh_prender()

    def callback_remove_all_btn(self, event):
        if len(self.phy_bounds) > 0:
            self.phy_bounds.clear()
            self.phy_rmv_all_button.config(state=tk.DISABLED)
        self.refresh_prender()

    def callback_tri_btn(self, event):
        self.show_tri = not self.show_tri
        self.refresh_prender()

    def callback_open_btn(self, event):
        target_scene_path = filedialog.askopenfilename(title='Open scene', initialdir=self.root_path,
                                                       filetypes=[('json', '*.json'), ('All Files', '*')])

        s = load_scene(target_scene_path)
        self.phy_bounds.clear()
        self.phy_cur_contour_bounds.clear()
        self.phy_rect_area = [0, 0]
        small_offset = 0.2
        bounds_num = len(s.bounds)
        out_bound = []
        inner_bounds = []
        for i in range(bounds_num):
            b_ps = s.bounds[i].points
            b_ps_num = s.bounds[i].points_num
            bound = []
            for j in range(b_ps_num):
                x = b_ps[j][0] / 100 + small_offset
                y = b_ps[j][1] / 100 + small_offset
                bound.append([x, y])
            if s.bounds[i].is_out_bound:
                out_bound.append(bound)
            else:
                inner_bounds.append(bound)
        self.phy_bounds = out_bound + inner_bounds
        self.refresh_prender()

    def callback_save_btn(self, event):
        target_scene_path = filedialog.asksaveasfilename(title='Save scene', initialdir=self.root_path,
                                                         filetypes=[('json', '*.json'), ('All Files', '*')])
        name = target_scene_path.split('/')[-1]
        if self.phy_bounds is not None and len(self.phy_bounds) > 0:
            target_scene = self.gen_scene(name)
            if save_scene(target_scene, target_scene_path):
                self.destroy()
            else:
                return
        else:
            mb.askokcancel(title='Warning', message='No physical scene is selected.')

    def callback_quest_btn(self, event):
        info1 = '1. choose \"add contour\" \n'
        info2 = '2. select \"draw rect\", \"draw pent\" ... to draw regular contour \n'
        info3 = '3. if select \"custom\", draw any thing you like, and use mouse right to delete unwanted lines\n'
        info4 = '4. click \"end contour\" to save current contour'
        mb.askokcancel(title='How to use', message=info1 + info2 + info3 + info4)

    def refresh_label(self):
        c_w = round(self.phy_canvas_size[0] / self.phy_scale, 2)
        c_h = round(self.phy_canvas_size[1] / self.phy_scale, 2)
        s_w, s_h = round(self.phy_scene_size[0], 1), round(self.phy_scene_size[1], 1)
        if len(self.phy_cur_contour_bounds) > 2:
            _, s_w, s_h, _ = geo.calc_poly_min_cir_rect(np.array(self.phy_cur_contour_bounds))
        yb = self.cur_focus_center[1] - self.phy_canvas_size[1] / self.phy_scale * 0.5
        yc = self.cur_focus_center[1]
        yt = self.cur_focus_center[1] + self.phy_canvas_size[1] / self.phy_scale * 0.5
        xl = self.cur_focus_center[0] - self.phy_canvas_size[0] / self.phy_scale * 0.5
        xc = self.cur_focus_center[0]
        xr = self.cur_focus_center[0] + self.phy_canvas_size[0] / self.phy_scale * 0.5
        self.phy_can_y0_label.configure(text=str(round(yb, 2)))
        self.phy_can_y1_label.configure(text=str(round(yc, 2)))
        self.phy_can_y2_label.configure(text=str(round(yt, 2)))
        self.phy_can_x0_label.configure(text=str(round(xl, 2)))
        self.phy_can_x1_label.configure(text=str(round(xc, 2)))
        self.phy_can_x2_label.configure(text=str(round(xr, 2)))
        self.phy_canvas_label.configure(text='window size: {} m * {} m'.format(c_w, c_h))
        self.phy_scene_label.configure(text='cur contour size: {} m * {} m'.format(round(s_w, 2), round(s_h, 2)))

    # --------------------------------------------render----------------------------------------------------------------

    def refresh_prender(self):
        self.phy_canvas.delete('all')
        self.phy_pre_canvas.delete('all')

        cur_center = [self.cur_focus_center[0] * self.phy_scale,
                      (self.phy_max_len - self.cur_focus_center[1]) * self.phy_scale]
        trans = [self.phy_canvas_center[0] - cur_center[0], self.phy_canvas_center[1] - cur_center[1]]
        # self.draw_out_bound(trans)
        self.draw_existing_bounds(trans)
        self.draw_cur_contour(trans)
        self.draw_bound_min_rect()
        self.draw_canvas_area()
        self.refresh_label()

    def draw_existing_bounds(self, trans):
        if self.phy_bounds is None or len(self.phy_bounds) < 1:
            return
        if self.show_tri:
            if self.phy_bounds[-1] is None or len(self.phy_bounds[-1]) < 1:
                return
            tris = geo.calc_poly_triangulation(self.phy_bounds)  # 求解当前三角剖分结果
            if len(tris) > 1:
                inter_polys = []
                for i in range(len(tris) - 1):
                    for j in range(i + 1, len(tris)):
                        inter_poly = geo.calc_con_polys_intersect(tris[i].vertices, tris[i].in_circle,
                                                                  tris[j].vertices,
                                                                  tris[j].in_circle)
                        if inter_poly is not None:
                            p_pre_contour = []
                            for v in inter_poly:
                                vp_x = v[0] * self.phy_pre_scale
                                vp_y = (self.phy_max_len - v[1]) * self.phy_pre_scale
                                p_pre_contour += [vp_x, vp_y]
                            self.phy_pre_canvas.create_polygon(p_pre_contour, outline='black', fill='green')
                            print('triangulation error occur! discover intersection area!', inter_poly)
                            inter_polys.append(inter_poly)
                if len(inter_polys) > 0:
                    print('triangulation error occur!fix problem')
                else:
                    print('triangulation success!')

            for tri in tris:
                tri_bound = []
                for v in tri.vertices:
                    vp_x = v[0] * self.phy_pre_scale
                    vp_y = (self.phy_max_len - v[1]) * self.phy_pre_scale
                    tri_bound += [vp_x, vp_y]
                self.phy_pre_canvas.create_polygon(tri_bound, fill='lightgray')
                for oe in tri.out_edges:
                    x1, y1 = tri.vertices[oe[0]]
                    x2, y2 = tri.vertices[oe[1]]
                    pp1x = x1 * self.phy_scale + trans[0]
                    pp1y = (self.phy_max_len - y1) * self.phy_scale + trans[1]
                    pp2x = x2 * self.phy_scale + trans[0]
                    pp2y = (self.phy_max_len - y2) * self.phy_scale + trans[1]
                    self.phy_canvas.create_line(pp1x, pp1y, pp2x, pp2y, fill='black', width=1)
                    pe1x = x1 * self.phy_pre_scale
                    pe1y = (self.phy_max_len - y1) * self.phy_pre_scale
                    pe2x = x2 * self.phy_pre_scale
                    pe2y = (self.phy_max_len - y2) * self.phy_pre_scale
                    self.phy_pre_canvas.create_line(pe1x, pe1y, pe2x, pe2y, fill='blue', width=1)
                for ie in tri.in_edges:
                    x1, y1 = tri.vertices[ie[0]]
                    x2, y2 = tri.vertices[ie[1]]
                    pp1x = x1 * self.phy_scale + trans[0]
                    pp1y = (self.phy_max_len - y1) * self.phy_scale + trans[1]
                    pp2x = x2 * self.phy_scale + trans[0]
                    pp2y = (self.phy_max_len - y2) * self.phy_scale + trans[1]
                    self.phy_canvas.create_line(pp1x, pp1y, pp2x, pp2y, fill='red', width=1)
                    pe1x = x1 * self.phy_pre_scale
                    pe1y = (self.phy_max_len - y1) * self.phy_pre_scale
                    pe2x = x2 * self.phy_pre_scale
                    pe2y = (self.phy_max_len - y2) * self.phy_pre_scale
                    self.phy_pre_canvas.create_line(pe1x, pe1y, pe2x, pe2y, fill='red', width=1)
        else:
            for pb in self.phy_bounds:
                if len(pb) > 0:
                    p_contour = []
                    p_pre_contour = []
                    for v in pb:
                        vx = v[0] * self.phy_scale + trans[0]
                        vy = (self.phy_max_len - v[1]) * self.phy_scale + trans[1]
                        p_contour += [vx, vy]
                        vp_x = v[0] * self.phy_pre_scale
                        vp_y = (self.phy_max_len - v[1]) * self.phy_pre_scale
                        p_pre_contour += [vp_x, vp_y]
                    self.phy_canvas.create_polygon(p_contour, outline='black', fill='white')
                    self.phy_pre_canvas.create_polygon(p_pre_contour, outline='black', fill='white')
                    '''barycenter = calc_poly_barycenter(pb)
                    cx = barycenter[0] * self.phy_pre_scale
                    cy = (self.phy_max_len - barycenter[1]) * self.phy_pre_scale
                    self.phy_pre_canvas.create_oval(cx - 2, cy - 2, cx + 2, cy + 2, fill='red')'''

    def draw_cur_contour(self, trans):
        p_num = len(self.phy_cur_contour_bounds)
        if p_num >= 2:
            for i in range(0, p_num - 1):
                p1x, p1y = self.phy_cur_contour_bounds[i]
                p2x, p2y = self.phy_cur_contour_bounds[i + 1]
                self.draw_contour_line(p1x, p1y, p2x, p2y, trans)
            if self.phy_draw_type == DrawType.Custom.value:
                if self.phy_custom_contour_done:
                    p1x, p1y = self.phy_cur_contour_bounds[-1]
                    p2x, p2y = self.phy_cur_contour_bounds[0]
                    self.draw_contour_line(p1x, p1y, p2x, p2y, trans)
            else:
                p1x, p1y = self.phy_cur_contour_bounds[-1]
                p2x, p2y = self.phy_cur_contour_bounds[0]
                self.draw_contour_line(p1x, p1y, p2x, p2y, trans)
        elif p_num == 1:
            p1x, p1y = self.phy_cur_contour_bounds[0]
            pp1x = p1x * self.phy_scale + trans[0]
            pp1y = (self.phy_max_len - p1y) * self.phy_scale + trans[1]
            pe1x = p1x * self.phy_pre_scale
            pe1y = (self.phy_max_len - p1y) * self.phy_pre_scale
            self.phy_canvas.create_oval(pp1x - 1, pp1y - 1, pp1x + 1, pp1y + 1, fill='black')
            self.phy_pre_canvas.create_oval(pe1x - 1, pe1y - 1, pe1x + 1, pe1y + 1, fill='black')

    def draw_bound_min_rect(self):
        if len(self.phy_cur_contour_bounds) > 2:
            rect, s_w, s_h, _ = geo.calc_poly_min_cir_rect(np.array(self.phy_cur_contour_bounds))
            num = len(rect)
            for i in range(-1, num - 1):
                p1x, p1y = rect[i]
                p2x, p2y = rect[i + 1]
                pe1x = p1x * self.phy_pre_scale
                pe1y = (self.phy_max_len - p1y) * self.phy_pre_scale
                pe2x = p2x * self.phy_pre_scale
                pe2y = (self.phy_max_len - p2y) * self.phy_pre_scale
                self.phy_pre_canvas.create_line(pe1x, pe1y, pe2x, pe2y, fill='red')

    def draw_contour_line(self, p1x, p1y, p2x, p2y, trans):
        pp1x = p1x * self.phy_scale + trans[0]
        pp1y = (self.phy_max_len - p1y) * self.phy_scale + trans[1]
        pp2x = p2x * self.phy_scale + trans[0]
        pp2y = (self.phy_max_len - p2y) * self.phy_scale + trans[1]
        self.phy_canvas.create_line(pp1x, pp1y, pp2x, pp2y)
        self.phy_canvas.create_oval(pp1x - 1, pp1y - 1, pp1x + 1, pp1y + 1, fill='black')
        pe1x = p1x * self.phy_pre_scale
        pe1y = (self.phy_max_len - p1y) * self.phy_pre_scale
        pe2x = p2x * self.phy_pre_scale
        pe2y = (self.phy_max_len - p2y) * self.phy_pre_scale
        self.phy_pre_canvas.create_line(pe1x, pe1y, pe2x, pe2y)
        self.phy_pre_canvas.create_oval(pe1x - 1, pe1y - 1, pe1x + 1, pe1y + 1, fill='black')

    def draw_canvas_area(self):
        c_x_l = (self.cur_focus_center[0] - self.phy_canvas_size[0] / self.phy_scale * 0.5) * self.phy_pre_scale
        c_y_b = self.phy_canvas_size[1] - (
                self.cur_focus_center[1] - self.phy_canvas_size[1] / self.phy_scale * 0.5) * self.phy_pre_scale
        c_x_r = (self.cur_focus_center[0] + self.phy_canvas_size[0] / self.phy_scale * 0.5) * self.phy_pre_scale
        c_y_t = self.phy_canvas_size[1] - (
                self.cur_focus_center[1] + self.phy_canvas_size[1] / self.phy_scale * 0.5) * self.phy_pre_scale
        self.phy_pre_canvas.create_rectangle(c_x_l, c_y_b, c_x_r, c_y_t, outline='blue', dash=1, width=2)

    def calc_cur_regpoly_contour(self):
        self.phy_cur_contour_bounds.clear()
        if self.phy_draw_type == DrawType.Triangle.value:
            x0, y0 = self.phy_rect_area[0][0], self.phy_rect_area[0][1]
            x1, y1 = self.phy_rect_area[1][0], self.phy_rect_area[1][1]
            self.phy_cur_contour_bounds.append([x0, y0])
            self.phy_cur_contour_bounds.append([x1, y0])
            self.phy_cur_contour_bounds.append([(x1 + x0) / 2, y1])
        elif self.phy_draw_type == DrawType.Rectangle.value:
            x0, y0 = self.phy_rect_area[0][0], self.phy_rect_area[0][1]
            x1, y1 = self.phy_rect_area[1][0], self.phy_rect_area[1][1]
            self.phy_cur_contour_bounds.append([x0, y0])
            self.phy_cur_contour_bounds.append([x1, y0])
            self.phy_cur_contour_bounds.append([x1, y1])
            self.phy_cur_contour_bounds.append([x0, y1])
        elif self.phy_draw_type == DrawType.Pentagon.value:
            x0, y0 = self.phy_rect_area[0][0], self.phy_rect_area[0][1]
            x1, y1 = self.phy_rect_area[1][0], self.phy_rect_area[1][1]
            self.phy_cur_contour_bounds.append([x1 / 3 + 2 * x0 / 3, y0])
            self.phy_cur_contour_bounds.append([2 * x1 / 3 + x0 / 3, y0])
            self.phy_cur_contour_bounds.append([x1, y1 / 2 + y0 / 2])
            self.phy_cur_contour_bounds.append([(x1 + x0) / 2, y1])
            self.phy_cur_contour_bounds.append([x0, y1 / 2 + y0 / 2])
        elif self.phy_draw_type == DrawType.Hexagon.value:
            x0, y0 = self.phy_rect_area[0][0], self.phy_rect_area[0][1]
            x1, y1 = self.phy_rect_area[1][0], self.phy_rect_area[1][1]
            self.phy_cur_contour_bounds.append([(x1 + x0) / 2, y0])
            self.phy_cur_contour_bounds.append([x1, y1 / 3 + 2 * y0 / 3])
            self.phy_cur_contour_bounds.append([x1, 2 * y1 / 3 + y0 / 3])
            self.phy_cur_contour_bounds.append([(x1 + x0) / 2, y1])
            self.phy_cur_contour_bounds.append([x0, 2 * y1 / 3 + y0 / 3])
            self.phy_cur_contour_bounds.append([x0, y1 / 3 + 2 * y0 / 3])

    def gen_scene(self, name):
        temp_bounds = pickle.loads(pickle.dumps(self.phy_bounds))
        if temp_bounds is not None and len(temp_bounds) > 0:
            out_bound = temp_bounds[0]
            x_min, y_min = float('inf'), float('inf')
            for v in out_bound:
                x, y = v
                if x < x_min:
                    x_min = x
                if y < y_min:
                    y_min = y
            for pb in temp_bounds:
                for v in pb:
                    v[0] -= x_min
                    v[1] -= y_min
                    v[0] = round(v[0], 4) * 100
                    v[1] = round(v[1], 4) * 100
        scene = Scene()
        scene.update_contours(name, temp_bounds)
        return scene


class TrajectoryModPopupUI(BaseWindowUI):

    def __init__(self, ui_spec):
        super().__init__(ui_spec)
        self.vir_scene = None
        self.abs_patch_repeat = 10
        self.prox_patch_repeat = 10
        self.abs_rand_repeat = 10
        self.tiling_rand_repeat = 10
        self.vir_max_w = 0
        self.vir_max_h = 0
        self.recorded_paths_data_x = []
        self.recorded_paths_data_y = []
        self.recorded_paths_data_z = []
        self.wall_x = []
        self.wall_y = []
        self.cur_trajs = []
        self.cur_traj_type = TrajectoryType.AbsRoad.value
        self.traj_id = 0
        self.traj_type_opt = []
        self.drawing_point = 0
        self.figure = Figure(figsize=(10, 10), dpi=100)
        self.axes = self.figure.add_subplot(111, projection='3d')

        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        im = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.jet)
        self.figure.colorbar(im, ax=self.axes, fraction=0.1, pad=0.15, shrink=0.9, anchor=(0.0, 0.3),
                             orientation='vertical',
                             ticks=np.linspace(0, 100, 11), label='walking depth')

    def process_v_scene(self, vir_scene):
        self.vir_scene = vir_scene
        self.update_ui()
        if vir_scene is not None:
            self.process_scene_data()
            self.update_scene_drawing()
            self.traj_id = 0
            if vir_scene.simu_trajs_abs_road_targets is not None and len(vir_scene.simu_trajs_abs_road_targets) > 0:
                self.process_traj_data()
            self.update_drawing(self.traj_id)

    def proc_callback(self):
        self.title('traj gen')
        com_spec = self.ui_spec['components']
        for key in com_spec:
            c_obj = getattr(self, key)
            c_spec = com_spec[key]
            c_loc = c_spec['size']
            if "button" in key:
                callback_func = None
                if 'abs_road_repeat' in key:
                    callback_func = self.callback_road_repeat_ensure_btn
                elif 'prox_road_repeat' in key:
                    callback_func = self.callback_prox_road_repeat_ensure_btn
                elif 'abs_rand_repeat' in key:
                    callback_func = self.callback_abs_rand_repeat_ensure_btn
                elif 'tiling_rand_repeat' in key:
                    callback_func = self.callback_tiling_rand_repeat_ensure_btn
                elif 'traj_gen' in key:
                    callback_func = self.callback_gen_traj_btn
                    c_obj['text'] = 'generate'
                elif 'traj_save' in key:
                    callback_func = self.callback_save_traj_btn
                elif 'pre_traj' in key:
                    callback_func = self.callback_previous_btn
                elif 'next_traj' in key:
                    callback_func = self.callback_next_btn
                c_obj.bind("<Button-1>", callback_func)
            elif 'entry' in key:
                if 'abs_road_repeat' in key:
                    c_obj.delete(0, "end")
                    c_obj.insert(tk.END, str(self.abs_patch_repeat))
                elif 'prox_road_repeat' in key:
                    c_obj.delete(0, "end")
                    c_obj.insert(tk.END, str(self.prox_patch_repeat))
                elif 'abs_rand_repeat' in key:
                    c_obj.delete(0, "end")
                    c_obj.insert(tk.END, str(self.abs_rand_repeat))
                elif 'tiling_rand_repeat' in key:
                    c_obj.delete(0, "end")
                    c_obj.insert(tk.END, str(self.tiling_rand_repeat))
            elif 'canvas' in key:
                setattr(self, key, FigureCanvasTkAgg(self.figure, self))
                c_obj = getattr(self, key)
                c_obj.get_tk_widget().place(x=c_loc[0], y=c_loc[1], width=c_loc[2], height=c_loc[3])
                c_obj.get_tk_widget().config(bg='black')
                c_obj.draw()
            elif 'optmenu' in key:
                variable = tk.StringVar()
                variable.set(c_spec["option"][0])
                self.traj_type_opt = c_spec["option"]
                callback_func = None
                if 'traj_type' in key:
                    callback_func = self.callback_draw_canvas_type_select

                setattr(self, key, tk.OptionMenu(self, variable, *c_spec["option"], command=callback_func))
                getattr(self, key).place(x=c_loc[0], y=c_loc[1], width=c_loc[2], height=c_loc[3])

    def update_ui(self):
        v_scene = self.vir_scene
        if v_scene is not None:
            self.label_name.configure(text='name: ' + v_scene.name)
            self.label_patch.configure(text='patches: {}'.format(len(v_scene.patches)))
            self.label_node.configure(text='nodes: {}'.format(len(v_scene.nodes)))
            self.label_conv.configure(text='convs: {}'.format(len(v_scene.conv_polys)))
            road_traj_num = len(v_scene.simu_trajs_abs_road_targets) + len(v_scene.simu_trajs_prox_road_targets)
            self.label_follow_patch.configure(text='trajectories road-targets: {}'.format(road_traj_num))
            rand_traj_num = len(v_scene.simu_trajs_abs_rand_targets) + len(v_scene.simu_trajs_tiling_rand_targets)
            self.label_follow_random.configure(text='trajectories random-targets: {}'.format(rand_traj_num))
            self.abs_road_repeat_entry.delete(0, "end")
            self.abs_road_repeat_entry.insert(tk.END, str(self.abs_patch_repeat))
            self.label_patch_count.configure(text='{}'.format(len(v_scene.patches)))
            total_road_traj = self.abs_patch_repeat * len(v_scene.patches)
            self.label_abs_road_total.configure(text='generation number: {}'.format(total_road_traj))
            self.prox_road_repeat_entry.delete(0, "end")
            self.prox_road_repeat_entry.insert(tk.END, str(self.prox_patch_repeat))
            self.label_patch_count1.configure(text='{}'.format(len(v_scene.patches)))
            total_road_traj = self.prox_patch_repeat * len(v_scene.patches)
            self.label_prox_road_total.configure(text='generation number: {}'.format(total_road_traj))
            self.abs_rand_repeat_entry.delete(0, "end")
            self.abs_rand_repeat_entry.insert(tk.END, str(self.abs_rand_repeat))
            self.label_conv_count.configure(text='{}'.format(len(v_scene.conv_polys)))
            total_rand_traj = self.abs_rand_repeat * len(v_scene.conv_polys)
            self.label_rand_total.configure(text='generation number: {}'.format(total_rand_traj))
            self.tiling_rand_repeat_entry.delete(0, "end")
            self.tiling_rand_repeat_entry.insert(tk.END, str(self.tiling_rand_repeat))
            self.label_conv_count1.configure(text='{}'.format(len(v_scene.conv_polys)))
            total_rand_traj = self.tiling_rand_repeat * len(v_scene.conv_polys)
            self.label_tiling_rand_total.configure(text='generation number: {}'.format(total_rand_traj))
        else:
            self.label_abs_road_total.configure(text='generation repeat: {}'.format(self.abs_patch_repeat))
            self.label_prox_road_total.configure(text='generation repeat: {}'.format(self.prox_patch_repeat))
            self.label_rand_total.configure(text='generation repeat: {}'.format(self.abs_rand_repeat))
            self.label_tiling_rand_total.configure(text='generation repeat: {}'.format(self.tiling_rand_repeat))

    def process_scene_data(self):
        self.wall_x = []
        self.wall_y = []
        for bound in self.vir_scene.bounds:
            wall_points_x = [0.0 for _ in range(len(bound.points))]
            wall_points_y = [0.0 for _ in range(len(bound.points))]
            for j in range(-1, len(bound.points) - 1):
                wall_points_x[j] = bound.points[j][0]
                wall_points_y[j] = bound.points[j][1]
            self.wall_x.append(wall_points_x)
            self.wall_y.append(wall_points_y)
        self.vir_max_w, self.vir_max_h = self.vir_scene.max_size

    def process_traj_data(self):
        vir_scene = self.vir_scene
        if self.cur_traj_type == TrajectoryType.AbsRoad.value:
            self.cur_trajs = vir_scene.simu_trajs_abs_road_targets
        elif self.cur_traj_type == TrajectoryType.ProxRoad.value:
            self.cur_trajs = vir_scene.simu_trajs_prox_road_targets
        elif self.cur_traj_type == TrajectoryType.AbsRand.value:
            self.cur_trajs = vir_scene.simu_trajs_abs_rand_targets
        elif self.cur_traj_type == TrajectoryType.TilingRand.value:
            self.cur_trajs = vir_scene.simu_trajs_tiling_rand_targets
        self.recorded_paths_data_x, self.recorded_paths_data_y, self.recorded_paths_data_z = [], [], []
        for traj in self.cur_trajs:
            x_seq, y_seq, z_seq = [], [], []
            for i in range(len(traj)):
                x, y, z = traj[i]
                x_seq.append(x)
                y_seq.append(y)
                z_seq.append(z)
            self.recorded_paths_data_x.append(x_seq)
            self.recorded_paths_data_y.append(y_seq)
            self.recorded_paths_data_z.append(z_seq)
        self.label_cur_traj_num.configure(text='Trajectories Number: {}'.format(len(self.cur_trajs)))

    def update_traj_display(self):
        if len(self.cur_trajs) > 0:
            self.update_drawing(self.traj_id)
            self.label_cur_traj_id.configure(text='Cur Trajectory id: {}'.format(self.traj_id))
            self.label_cur_traj_depth.configure(
                text='Cur Trajectory Steps: {}'.format(len(self.cur_trajs[self.traj_id])))

    def generate_v_trajs(self):
        vir_scene = self.vir_scene
        self.traj_proc_label.configure(text='generating')
        vir_scene.gen_simu_road_trajs(self.abs_patch_repeat, prox=False, dis_ang=20)
        self.traj_proc_progress['value'] = 25
        self.window.update()
        vir_scene.gen_simu_road_trajs(self.prox_patch_repeat, prox=True, dis_ang=20)
        self.traj_proc_progress['value'] = 50
        self.window.update()
        vir_scene.gen_simu_abs_rand_trajs(self.abs_rand_repeat, walk_range=[0.2, 0.4], rot_range=[0, PI])
        self.traj_proc_progress['value'] = 75
        self.window.update()
        vir_scene.gen_simu_tiling_rand_trajs(self.tiling_rand_repeat)
        self.traj_proc_progress['value'] = 100
        self.window.update()
        self.cur_traj_type = TrajectoryType.AbsRoad.value
        self.process_traj_data()
        self.traj_id = 0
        self.update_traj_display()
        self.traj_proc_label.configure(text='done')

    def save_v_trajs(self):
        vir_scene = self.vir_scene
        abs_road_trajs = vir_scene.simu_trajs_abs_road_targets
        prox_road_trajs = vir_scene.simu_trajs_prox_road_targets
        abs_rand_trajs = vir_scene.simu_trajs_abs_rand_targets
        tiling_rand_trajs = vir_scene.simu_trajs_tiling_rand_targets
        self.app_config.create_trajectory_save_path()
        self.traj_proc_label.configure(text='saving')
        self.app_config.save_recorded_vir_trajectory(abs_road_trajs, 'abs_road')
        self.traj_proc_progress['value'] = 25
        self.window.update()
        self.app_config.save_recorded_vir_trajectory(prox_road_trajs, 'prox_road')
        self.traj_proc_progress['value'] = 50
        self.window.update()
        self.app_config.save_recorded_vir_trajectory(abs_rand_trajs, 'abs_rand')
        self.traj_proc_progress['value'] = 75
        self.window.update()
        self.app_config.save_recorded_vir_trajectory(tiling_rand_trajs, 'tiling_rand')
        self.traj_proc_progress['value'] = 100
        self.window.update()
        self.traj_proc_label.configure(text='done')

    def callback_road_repeat_ensure_btn(self, event):
        self.abs_patch_repeat = int(self.abs_road_repeat_entry.get())

        total_road_traj = self.abs_patch_repeat * len(self.env_mg.env.vir_scene.patches)
        self.label_abs_road_total.configure(text='generation number: {}'.format(total_road_traj))

    def callback_prox_road_repeat_ensure_btn(self, event):
        self.prox_patch_repeat = int(self.prox_road_repeat_entry.get())

        total_road_traj = self.prox_patch_repeat * len(self.env_mg.env.vir_scene.patches)
        self.label_prox_road_total.configure(text='generation number: {}'.format(total_road_traj))

    def callback_abs_rand_repeat_ensure_btn(self, event):
        self.abs_rand_repeat = int(self.abs_rand_repeat_entry.get())

        total_rand_traj = self.abs_rand_repeat * len(self.env_mg.env.vir_scene.conv_polys)
        self.label_rand_total.configure(text='generation number: {}'.format(total_rand_traj))

    def callback_tiling_rand_repeat_ensure_btn(self, event):
        self.tiling_rand_repeat = int(self.tiling_rand_repeat_entry.get())

        total_rand_traj = self.tiling_rand_repeat * len(self.env_mg.env.vir_scene.conv_polys)
        self.label_tiling_rand_total.configure(text='generation number: {}'.format(total_rand_traj))

    def callback_gen_traj_btn(self, event):
        self.generate_v_trajs()

    def callback_save_traj_btn(self, event):
        self.save_v_trajs()

    def callback_batch_proc_btn(self, event):
        v_scene_list = self.app_config.vir_all_scene_names
        for v_file_name in v_scene_list:
            v_name = v_file_name.split(".")[0]
            s_type = 'vir'
            xml_tree, scene_name = self.app_config.load_scene_spec(v_name, s_type)
            done, scene = self.app_config.load_scene_attr(scene_name, s_type)
            if not done:
                self.env_mg.gen_scene(scene_name, xml_tree, s_type)
                self.env_mg.gen_roadmap(xml_tree, s_type)
                self.env_mg.gen_patches(s_type)
                self.env_mg.gen_scene_segmentation(s_type)
                self.env_mg.gen_scene_tilings(s_type)
                self.env_mg.gen_scene_tilings_weights(s_type)
            else:
                self.env_mg.set_scene(scene, s_type)
                self.env_mg.gen_roadmap(xml_tree, s_type)
                self.env_mg.gen_patches(s_type)
                self.env_mg.gen_scene_tilings_weights(s_type)
            self.update_ui()
            self.update_scene_drawing()
            self.generate_v_trajs()
            self.save_v_trajs()

    def callback_draw_canvas_type_select(self, value):
        if value == self.traj_type_opt[0]:
            self.cur_traj_type = TrajectoryType.AbsRoad.value
        elif value == self.traj_type_opt[1]:
            self.cur_traj_type = TrajectoryType.ProxRoad.value
        elif value == self.traj_type_opt[2]:
            self.cur_traj_type = TrajectoryType.AbsRand.value
        elif value == self.traj_type_opt[3]:
            self.cur_traj_type = TrajectoryType.TilingRand.value
        self.process_traj_data()
        self.traj_id = 0
        self.update_traj_display()

    def callback_next_btn(self, event):
        self.traj_id += 1
        self.traj_id %= len(self.cur_trajs)
        self.update_traj_display()

    def callback_previous_btn(self, event):
        self.traj_id -= 1
        if self.traj_id < 0:
            self.traj_id = 0
        self.update_traj_display()

    def update_drawing(self, traj_id):
        if len(self.cur_trajs) > 0:
            x = self.recorded_paths_data_x[traj_id]
            y = self.recorded_paths_data_y[traj_id]
            z = self.recorded_paths_data_z[traj_id]
            nx = np.array(x)
            ny = np.array(y)
            nz = np.array(z)  # np.expand_dims(z, axis=0)

            self.axes.clear()
            w = max(int(self.vir_max_w) + 1, int(self.vir_max_h) + 1)
            self.axes.set_xlim(0, w)
            self.axes.set_ylim(0, w)
            self.axes.set_zlim(0, math.ceil(z[-1]))
            self.axes.set_xlabel('x')
            self.axes.set_ylabel('y')
            self.axes.set_zlabel('time')
            for i in range(len(self.wall_x)):
                b_x = np.array(self.wall_x[i] + [self.wall_x[i][0]])
                b_y = np.array(self.wall_y[i] + [self.wall_y[i][0]])
                b_z = np.zeros(len(b_x))
                self.axes.plot(b_x, b_y, b_z, color='black')
            # 设定每个图的color map和color bar所表示范围是一样的，即归一化
            norm = matplotlib.colors.Normalize(vmin=nz.min(), vmax=nz.max())
            z_colors = []
            for i in range(len(nz)):
                z_colors.append(matplotlib.cm.jet(int(norm(255 * nz[i]))))
            for i in range(len(nz) - 1):
                self.axes.plot(nx[i:i + 2], ny[i:i + 2], nz[i:i + 2], color=z_colors[i])
            self.figure.canvas.draw()

    def update_scene_drawing(self):
        self.axes.clear()
        w = max(int(self.vir_max_w) + 1, int(self.vir_max_h) + 1)
        self.axes.set_xlim(0, w)
        self.axes.set_ylim(0, w)
        self.axes.set_zlim(0, 100)
        self.axes.set_xlabel('x-value')
        self.axes.set_ylabel('y-value')
        self.axes.set_zlabel('time-horizon')
        for i in range(len(self.wall_x)):
            b_x = np.array(self.wall_x[i] + [self.wall_x[i][0]])
            b_y = np.array(self.wall_y[i] + [self.wall_y[i][0]])
            b_z = np.zeros(len(b_x))
            self.axes.plot(b_x, b_y, b_z, color='black')
        self.figure.canvas.draw()
