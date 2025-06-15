import numpy as np
import shapely

import shapely.geometry as geometry
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.text import Text, Annotation
from matplotlib.patches import Polygon, Rectangle, Circle, Arrow, ConnectionPatch, Ellipse, FancyBboxPatch
from matplotlib.widgets import Button, Slider, Widget
from matplotlib.lines import Line2D

import tools.geometry as geo
from common import data_path
from tools.gen import load_scene
from tools.space.scene import DiscreteScene


class SceneDrawer:

    def __init__(self):
        self.mPress = False
        self.start_x = 0
        self.start_y = 0
        self.fig = None
        self.ax = None
        self.scene: DiscreteScene = None

    def call_move(self, event):  # event mouse press/release
        if event.name == 'button_press_event':
            axtemp = event.inaxes
            # Whether mouse in a coordinate system or not, yes is the figure in the mouse location, no is None
            if axtemp and event.button == 3:
                # print(event)
                self.mPress = True
                self.start_x = event.xdata
                self.start_y = event.ydata
                # self.update(np.array([self.start_x, self.start_y]))
        elif event.name == 'button_release_event':
            axtemp = event.inaxes
            if axtemp and event.button == 3:
                self.mPress = False
        elif event.name == 'motion_notify_event':
            axtemp = event.inaxes
            if axtemp and event.button == 3 and self.mPress:  # the mouse continuing press
                x_min, x_max = axtemp.get_xlim()
                y_min, y_max = axtemp.get_ylim()
                w = x_max - x_min
                h = y_max - y_min
                # mouse movement
                mx = event.xdata - self.start_x
                my = event.ydata - self.start_y
                axtemp.set(xlim=(x_min - mx, x_min - mx + w))
                axtemp.set(ylim=(y_min - my, y_min - my + h))
                self.fig.canvas.draw_idle()  # Delay drawing
        return

    def call_scroll(self, event):
        axtemp = event.inaxes
        # caculate the xlim and ylim after zooming
        if axtemp:
            x_min, x_max = axtemp.get_xlim()
            y_min, y_max = axtemp.get_ylim()
            w = x_max - x_min
            h = y_max - y_min
            curx = event.xdata
            cury = event.ydata
            curXposition = (curx - x_min) / w
            curYposition = (cury - y_min) / h
            # Zoom the figure for 1.1 times
            if event.button == 'down':
                w = w * 1.1
                h = h * 1.1
            elif event.button == 'up':
                w = w / 1.1
                h = h / 1.1
            newx = curx - w * curXposition
            newy = cury - h * curYposition
            axtemp.set(xlim=(newx, newx + w))
            axtemp.set(ylim=(newy, newy + h))
            self.fig.canvas.draw_idle()  # drawing

    # def update_annot(self, ind, l1, annot, x_str, y_str):
    #     posx = np.array(l1.get_data())[0][ind["ind"][0]]  # get the x in the line
    #     posy = np.array(l1.get_data())[1][ind["ind"][0]]  # get the y in the line
    #     annot.xy = ([posx, posy])
    #     text = "{}, {}".format(" ".join([x_str[n] for n in ind["ind"]]),
    #                            " ".join([y_str[n] for n in ind["ind"]]))
    #
    #     annot.set_text(text)
    #     cmap = plt.cm.RdYlGn
    #     norm = plt.Normalize(1, 4)
    #     c = np.random.randint(1, 5, size=10)  # the upper colour
    #     annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
    #     annot.get_bbox_patch().set_alpha(0.4)
    #
    # def hover(self, event, l1, annot, ax, x_str, y_str):
    #     vis = annot.get_visible()
    #     if event.inaxes == ax:
    #         cont, ind = l1.contains(event)
    #         if cont:  # the mouse in the point
    #             self.update_annot(ind, l1, annot, x_str, y_str)
    #             annot.set_visible(True)
    #         else:
    #             if vis:
    #                 annot.set_visible(False)
    #                 self.fig.canvas.draw_idle()

    def draw_scene_bounds(self):
        minx, miny, maxx, maxy = self.scene.poly_contour.bounds
        b = Polygon(np.array([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]),
                    facecolor='gray',
                    edgecolor='gray')
        self.ax.add_patch(b)
        for bound in self.scene.bounds:
            ecolor = 'grey'
            bps = np.array(shapely.Polygon(bound.points).exterior.coords)
            if bound.is_out_bound:
                polygon = Polygon(bps, facecolor='white', edgecolor=ecolor)
                polygon1 = Polygon(np.array([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]),
                                   facecolor='white', fill=False, edgecolor=ecolor, linewidth=9)
                self.ax.add_patch(polygon1)
            else:
                polygon = Polygon(bps, facecolor='gray', edgecolor=ecolor)
            self.ax.add_patch(polygon)
        # polygon = Polygon(np.array(self.scene.poly_contour_safe.exterior.coords), facecolor='none', edgecolor='blue')
        # self.ax.add_patch(polygon)
        # for geo in self.scene.poly_contour_safe.interiors:
        #     polygon = Polygon(np.array(geo.coords), facecolor='none', edgecolor='blue')
        #     self.ax.add_patch(polygon)

        # for conv in self.scene.conv_polys:
        #     ecolor = 'gray'
        #     polygon = Polygon(conv.vertices, facecolor='gray', edgecolor=ecolor, alpha=0.3)
        #     self.ax.add_patch(polygon)
        # # if 'vir' in s_type:
        # #     radius = 2
        # #     tc1 = [4.5928, 6.46457]
        # #     print(check_point_in_bound(tc1, self.rdw_env.vir_scene.bounds))
        # #     cx, cy = vertex_transfer(tc1, max_len, scale, trans, scene_center)
        # #     canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, fill='blue')
        # #     tc2 = [4.78852, 6.42338]
        # #     print(check_point_in_bound(tc2, self.rdw_env.vir_scene.bounds))
        # #     cx, cy = vertex_transfer(tc2, max_len, scale, trans, scene_center)
        # #     canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, fill='blue')
        # #     print(check_line_bound_cross(tc1,tc2,self.rdw_env.vir_scene.bounds))
        # #     print(len_vec(np.array(tc2)-np.array(tc1)))
        # cx, cy, bx, by = 0, 0, 0, 0
        # radius = 2
        # for b in scene.bounds:
        #     contour = []
        #     for v in b.b_points:
        #         contour += vertex_transfer(v, max_len, scale, trans, scene_center)
        #     canvas.create_polygon(contour, fill='', outline='black', width=1)
        #     if b.is_out_bound:
        #         cx, cy = vertex_transfer(b.center, max_len, scale, trans, scene_center)
        #         bx, by = vertex_transfer(b.barycenter, max_len, scale, trans, scene_center)
        # canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, fill='black')  # 画中心
        # # canvas.create_oval(bx - radius, by - radius, bx + radius, by + radius, fill='blue')  # 画重心
        # if click is not None and len(click) > 0:
        #     tc = [click[0] - max_len * 0.5 + scene_center[0], click[1] - max_len * 0.5 + scene_center[1]]
        #     cx, cy = vertex_transfer(tc, max_len, scale, trans, scene_center)
        #     canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, fill='red')

    def draw_scene_convs(self):
        for conv in self.scene.conv_polys:
            polygon = Polygon(conv.vertices, facecolor='white', edgecolor='blue', alpha=0.5)
            self.ax.add_patch(polygon)
        # for bound in self.scene.bounds:
        #     if not bound.is_out_bound:
        #         ecolor = 'red'
        #         polygon = Polygon(bound.points, facecolor='white', edgecolor=ecolor)
        #         self.ax.add_patch(polygon)

    def draw_scene_grids(self, click_pos=None):
        click_tiling = None
        for t in self.scene.tilings:
            if t.type:
                polygon = Polygon(t.rect, facecolor='white', alpha=0.5)
            else:
                polygon = Polygon(t.rect, facecolor='black', alpha=0.5)
            self.ax.add_patch(polygon)
            if click_pos is not None:
                if geo.chk_p_in_tiling_simple(click_pos, t):
                    click_tiling = t
        if click_tiling is not None:
            if len(click_tiling.corr_conv_ids) > 0:
                for conv_id in click_tiling.corr_conv_ids:
                    cpoly = Polygon(self.scene.conv_polys[conv_id].vertices, facecolor='gray', edgecolor='green')
                    self.ax.add_patch(cpoly)
            polygon = Polygon(click_tiling.rect, facecolor='blue', edgecolor='red')
            self.ax.add_patch(polygon)
            print(click_tiling.type, click_tiling.center)

    def draw_tiling(self, click_pos):
        tiling, tconv = self.scene.calc_located_tiling_conv(click_pos)
        # tiling.corr_conv_ids = []
        # tiling.corr_conv_inters = []
        # tiling.corr_conv_cin = -1
        # if (self.scene.poly_contour.contains(tiling.poly_contour) and
        #         not self.scene.poly_contour.boundary.intersects(tiling.poly_contour.boundary)):
        #     tiling.type = 1
        # else:
        #     tiling.intersection_scene(self)
        # if tiling.type or tiling.cross_bound.shape[0] > 0:  # tiling is within or partially within the scene
        #     for i in range(len(self.scene.conv_polys)):
        #         conv = self.scene.conv_polys[i]
        #         inter = conv.poly_contour.intersection(tiling.poly_contour)
        #         if not inter.is_empty and isinstance(inter, shapely.Polygon):
        #             tiling.corr_conv_ids.append(i)
        #             tiling.corr_conv_inters.append(np.array(inter.exterior.coords))
        #         if conv.poly_contour.contains(shapely.Point(tiling.center)):
        #             tiling.corr_conv_cin = i
        # print(tiling.corr_conv_ids)

        polygon = Polygon(tiling.rect, facecolor='green', edgecolor='yellow')
        self.ax.add_patch(polygon)

        if len(tiling.cross_bound) > 0:
            for l in tiling.cross_bound:
                cross_line = Line2D([l[0][0], l[1][0]], [l[0][1], l[1][1]], linewidth=2)
                self.ax.add_line(cross_line)

    def update(self, click_pos=None):
        # self.ax.cla()
        self.draw_scene_bounds()
        # self.draw_scene_grids()
        # self.draw_scene_convs()
        if click_pos is not None:
            # print(self.scene.poly_contour_safe.covers(geometry.Point(click_pos)),
            #       geo.calc_point_mindis2bound(click_pos, self.scene.bounds))
            self.draw_tiling(click_pos)
            plt.draw()

    def draw(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        self.update()
        # plt.grid(alpha=0.2)
        plt.gca().set_aspect(1)

        self.fig.canvas.mpl_connect('scroll_event', lambda event: self.call_scroll(event))  # Event mouse wheel
        self.fig.canvas.mpl_connect('button_press_event', lambda event: self.call_move(event))
        self.fig.canvas.mpl_connect('button_release_event', lambda event: self.call_move(event))
        self.fig.canvas.mpl_connect('motion_notify_event', lambda event: self.call_move(event))  # Event mouse move
        self.ax.set_xlim(0, 2000)  # xlabel start limition
        self.ax.set_ylim(0, 2000)  # ylabel start limition

        plt.show()


if __name__ == '__main__':
    v_path = data_path + '\\vir'
    p_path = data_path + '\\rl\\phy'

    vname = 'orthogonal_s4'
    pname = 'complex'

    tar_path = v_path + '\\' + vname + '.json'

    tar_scene = load_scene(tar_path)
    tar_scene.update_grids_runtime_attr()

    drawer = SceneDrawer()
    drawer.scene = tar_scene
    drawer.draw()
    print('scene size= {} * {}'.format(*tar_scene.max_size))
