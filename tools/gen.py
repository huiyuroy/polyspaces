import ujson

from tools import *
import tools.base_math as bmath


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


class Scene:
    def __init__(self):
        self.name = None
        self.scene_type = 'vir'
        self.bounds = []
        self.max_size = [0, 0]  # w,h
        self.out_bound_conv = None
        self.out_conv_hull = None
        self.scene_center = np.array([0, 0])


class Boundary:
    def __init__(self):
        self.is_out_bound = False
        self.points = []
        self.points_num = 0
        self.center = np.array([0, 0])
        self.barycenter = []
        self.cir_rect = []
        self.orth_cir_rect = []

    def set_contour(self, out_boundary, points):
        self.is_out_bound = out_boundary
        self.points = points
        self.points_num = len(points)
        self.__calc_surround_rect()

    def clean_repeat(self):
        need_clean = []
        for i in range(len(self.points) - 1):
            for j in range(i + 1, len(self.points)):
                if bmath.chk_p_same(self.points[i], self.points[j]):
                    need_clean.append(j)
        for idx in need_clean:
            self.points.pop(idx)
        self.points_num = len(self.points)

    def __calc_surround_rect(self):
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = 0, 0
        for px, py in self.points:
            if px <= min_x:
                min_x = px
            elif px >= max_x:
                max_x = px
            if py <= min_y:
                min_y = py
            elif py >= max_y:
                max_y = py
        self.orth_cir_rect = [min_x, min_y, max_x, max_y]

    def clone(self):
        c_bound = Boundary()
        c_bound.is_out_bound = self.is_out_bound
        c_bound.points = np.array(self.points).copy().tolist()
        c_bound.points_num = self.points_num
        c_bound.center = self.center.copy()
        c_bound.barycenter = np.array(self.barycenter).copy().tolist()
        c_bound.cir_rect = np.array(self.cir_rect).copy().tolist()
        return c_bound

    def print_info(self):
        info = ''
        for b in self.points:
            info += str(b)
        print("is out bound", self.is_out_bound, "point list:", info)
        print('center:', self.center, ' barycenter:', self.barycenter)


class DrawType(Enum):
    Triangle = 0
    Rectangle = 1
    Pentagon = 2
    Hexagon = 3
    Circle = 4
    Custom = 5


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


class SceneGenWindowUI(BaseWindowUI):

    def __init__(self, ui_spec):
        super().__init__(ui_spec=ui_spec)
        cur_path = os.path.abspath(os.path.dirname(__file__))
        self.roo_path = cur_path[:cur_path.find('polyspaces') + len('polyspaces')]
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
                    dis, _, t = bmath.calc_point_pro2line([tar_x, tar_y], ls, le)
                    if 1 > t > 0 and dis <= 0.05:
                        self.phy_cur_contour_bounds.pop(i + 1)
                        break
            elif cur_num == 2:
                ls, le = self.phy_cur_contour_bounds[0], self.phy_cur_contour_bounds[1]
                dis, _, t = bmath.calc_point_pro2line([tar_x, tar_y], ls, le)
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
                if bmath.l2_norm([tar_x - head_x, tar_y - head_y]) <= 0.05:
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
        target_scene_path = filedialog.askopenfilename(title='Open scene', initialdir=self.roo_path,
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
        if self.phy_bounds is not None and len(self.phy_bounds) > 0:
            target_bounds = self.calc_phy_bound_regular()
            if self.app_config.save_phy_scene(target_bounds):
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
            _, s_w, s_h, _ = bmath.calc_poly_min_cir_rect(np.array(self.phy_cur_contour_bounds))
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
            tris = bmath.calc_poly_triangulation(self.phy_bounds)  # 求解当前三角剖分结果
            if len(tris) > 1:
                inter_polys = []
                for i in range(len(tris) - 1):
                    for j in range(i + 1, len(tris)):
                        inter_poly = bmath.calc_con_polys_intersect(tris[i].vertices, tris[i].in_circle,
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
            rect, s_w, s_h, _ = bmath.calc_poly_min_cir_rect(np.array(self.phy_cur_contour_bounds))
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

    def calc_phy_bound_regular(self):
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
                    v[0] = round(v[0], 4)
                    v[1] = round(v[1], 4)
        return temp_bounds
