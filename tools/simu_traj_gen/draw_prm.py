import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle, Circle, Arrow, ConnectionPatch, Ellipse, FancyBboxPatch

from prm_traj import PRMTrajectoryGenerator
import pyrdw.generator as generator


class SceneDrawer:

    def __init__(self):
        self.mPress = False
        self.start_x = 0
        self.start_y = 0
        self.fig = None
        self.ax = None
        self.scene = None
        self.prm_traj = None
        self.gen_traj = False
        # self.start_pos = None
        # self.end_pos = None

    def call_move(self, event):  # event mouse press/release
        if event.name == 'button_press_event':
            axtemp = event.inaxes
            # Whether mouse in a coordinate system or not, yes is the figure in the mouse location, no is None
            if axtemp:
                if event.button == 1:
                    # print(event)
                    self.mPress = True
                    self.start_x = event.xdata
                    self.start_y = event.ydata

                elif event.button == 3:
                    # x, y = event.xdata, event.ydata
                    # print(x,y)
                    # if self.start_pos is None:
                    #     self.start_pos = [x, y]
                    # elif self.end_pos is None:
                    #     self.end_pos = [x, y]
                    # else:
                    #     self.start_pos = None
                    #     self.end_pos = None
                    self.gen_traj = True
                    self.update()

        elif event.name == 'button_release_event':
            axtemp = event.inaxes
            if axtemp and event.button == 1:
                self.mPress = False
        elif event.name == 'motion_notify_event':
            axtemp = event.inaxes
            if axtemp and event.button == 1 and self.mPress:  # the mouse continuing press
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

    def draw_scene_bounds(self):
        back_poly = ((-10000, -10000), (10000, -10000), (10000, 10000), (-10000, 10000))
        back_poly = Polygon(back_poly, facecolor='white', edgecolor='white')
        self.ax.add_patch(back_poly)
        for bound in self.scene.bounds:
            ecolor = 'black'
            if bound.is_out_bound:
                ecolor = 'red'
            polygon = Polygon(bound.points, facecolor='white', edgecolor=ecolor)
            self.ax.add_patch(polygon)

    def draw_prm_samples(self):
        samples = np.array(self.prm_traj.sample_points)
        samples_x = samples[:, 0]
        samples_y = samples[:, 1]
        # self.ax.scatter(samples_x, samples_y, s=5, color='r')

    def draw_prm_traj(self, traj):
        # se = np.array([self.start_pos, self.end_pos])
        #
        # self.ax.scatter(se[:, 0], se[:, 1], s=5, color='g')

        traj = np.array(traj)

        self.ax.plot(traj[:, 0], traj[:, 1], color='r')

    def update(self):
        # self.ax.cla()
        self.draw_scene_bounds()
        self.draw_prm_samples()
        if self.gen_traj:
            self.gen_traj = False
            trajs = self.prm_traj.generate_single_traj()
            self.draw_prm_traj(trajs)
        # plt.draw()

    def draw(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        self.update()
        plt.grid(alpha=0.2)
        plt.gca().set_aspect(1)

        self.fig.canvas.mpl_connect('scroll_event', lambda event: self.call_scroll(event))  # Event mouse wheel
        self.fig.canvas.mpl_connect('button_press_event', lambda event: self.call_move(event))
        self.fig.canvas.mpl_connect('button_release_event', lambda event: self.call_move(event))
        self.fig.canvas.mpl_connect('motion_notify_event', lambda event: self.call_move(event))  # Event mouse move
        self.ax.set_xlim(0, 2000)  # xlabel start limition
        self.ax.set_ylim(0, 2000)  # ylabel start limition

        plt.show()


if __name__ == '__main__':
    v_name = 'abnormal_s17'
    v_path = 'E:\\polyspaces\\vir'
    vscene = generator.load_scene(v_path + '\\' + v_name + '.json')
    vscene.update_roadmap()
    vscene.update_grids_runtime_attr()

    prm_t = PRMTrajectoryGenerator(vscene)
    prm_t.generate_prm()

    drawer = SceneDrawer()
    drawer.scene = vscene
    drawer.prm_traj = prm_t
    drawer.draw()
