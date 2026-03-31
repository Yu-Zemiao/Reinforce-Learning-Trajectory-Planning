#引入模块----------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
#------------------------------------------
#自定义模块--------------------------------
from environment import Environment

#------------------------------------------

# 主体-------------------------------------

class Visiualization:

    def __init__(self, environment:Environment):
        
        self.environment = environment
        self.robot = None
        self.sliders = []




    # ============================
    # 坐标轴等比例
    # ============================

    def set_axes_equal(self, ax):

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()

        x_center = np.mean(xlim)
        y_center = np.mean(ylim)
        z_center = np.mean(zlim)

        span = max(
            xlim[1] - xlim[0],
            ylim[1] - ylim[0],
            zlim[1] - zlim[0]
        )

        ax.set_xlim(x_center - span/2, x_center + span/2)
        ax.set_ylim(y_center - span/2, y_center + span/2)
        ax.set_zlim(z_center - span/2, z_center + span/2)

        ax.set_box_aspect([1,1,1])


    # ============================
    # 绘制坐标系
    # ============================

    def draw_frame(self, ax, T, length=120):

        o = T[:3,3]

        x = T[:3,0]
        y = T[:3,1]
        z = T[:3,2]

        ax.quiver(*o, *(x*length), color='r')
        ax.quiver(*o, *(y*length), color='g')
        ax.quiver(*o, *(z*length), color='b')


    # ============================
    # 圆柱连杆
    # ============================

    def draw_cylinder(self, ax, p1, p2, radius):

        v = p2 - p1
        L = np.linalg.norm(v)

        if L < 1e-6:
            return

        v = v / L

        theta = np.linspace(0, 2*np.pi, 20)
        z = np.linspace(0, L, 2)

        theta, z = np.meshgrid(theta, z)

        x = radius*np.cos(theta)
        y = radius*np.sin(theta)

        z_axis = np.array([0,0,1])

        dot = np.clip(np.dot(z_axis, v), -1, 1)
        angle = np.arccos(dot)

        axis = np.cross(z_axis, v)

        if np.linalg.norm(axis) > 1e-6:

            axis = axis / np.linalg.norm(axis)

            K = np.array([
                [0,-axis[2],axis[1]],
                [axis[2],0,-axis[0]],
                [-axis[1],axis[0],0]
            ])

            R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)

            xyz = np.vstack((x.flatten(),y.flatten(),z.flatten()))
            xyz = R @ xyz

            x = xyz[0].reshape(x.shape)
            y = xyz[1].reshape(y.shape)
            z = xyz[2].reshape(z.shape)

        x += p1[0]
        y += p1[1]
        z += p1[2]

        ax.plot_surface(x,y,z,alpha=0.6)


    # ============================
    # 绘制机器人
    # ============================

    def draw_robot(self, ax, frames):

        points = self.robot.points
        radii = self.robot.axial_body_width

        # 连杆
        for i in range(6):

            self.draw_cylinder(
                ax,
                points[i],
                points[i+1],
                radii[i]
            )

        # 关节点
        ax.plot(
            points[:,0],
            points[:,1],
            points[:,2],
            'ko',
            markersize=6
        )

        # 编号
        for i,p in enumerate(points):

            ax.text(
                p[0]+30,
                p[1]+30,
                p[2]+30,
                f"J{i}"
            )

        # 坐标系
        for T in frames:

            self.draw_frame(ax, T)


    # ============================
    # 更新函数
    # ============================

    def update(self, val):

        ax = self.ax

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()

        elev = ax.elev
        azim = ax.azim

        ax.cla()

        # 更新角度
        for i,s in enumerate(self.sliders):

            self.robot.theta[i] = s.val

        # 正运动学
        frames = self.robot.forward_kinematics()

        # 绘图
        self.draw_robot(ax, frames)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)

        ax.view_init(elev=elev, azim=azim)

        self.set_axes_equal(ax)

        self.fig.canvas.draw_idle()


    # ============================
    # 主入口
    # ============================

    def visiualize(self, robot):

        self.robot = robot

        robot.forward_kinematics()

        self.fig = plt.figure(figsize=(9,9))
        self.ax = self.fig.add_subplot(111, projection='3d')

        plt.subplots_adjust(left=0.25)

        self.sliders = []

        # 创建滑块
        for i in range(6):

            ax_slider = plt.axes((0.05, 0.85-i*0.1, 0.15, 0.03))

            s = Slider(
                ax_slider,
                f"J{i + 1}",
                robot.theta_limit[i][0],
                robot.theta_limit[i][1],
                valinit=robot.theta[i]
            )

            s.on_changed(self.update)

            self.sliders.append(s)

        workspace = 1000

        self.ax.set_xlim(-workspace,workspace)
        self.ax.set_ylim(-workspace,workspace)
        self.ax.set_zlim(-workspace,workspace)

        self.ax.set_box_aspect([1,1,1])

        self.update(None)

        plt.show()
