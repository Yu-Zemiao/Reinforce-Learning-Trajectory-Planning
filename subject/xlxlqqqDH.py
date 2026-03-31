import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


# test:

# ============================================================
# DH 参数 [alpha(rad), a(mm), d(mm), home_offset(rad)]
# ============================================================
# DH_PARAMS = [
#     [-0.000441,   0.000000,   141.3209,  -0.017764],
#     [ 1.571090,   0.000000,     0.0000,  -0.012528],
#     [ 0.003779, 596.023500,     0.0000,  -0.003989],
#     [-0.000261, 571.965900,  -131.9707,  -0.008874],
#     [ 1.570481,   0.000000,   115.2260,  -0.006943],
#     [-1.571710,   0.000000,   104.3891,   0.009412],
# ]

DH_PARAMS = [
    [-0.0,   0.000000,   141.3209,  -0.0],
    [ 1.571090,   0.000000,     0.0000,  -0.0],
    [ 0.0, 596.023500,     0.0000,  -0.0],
    [-0.0, 571.965900,  -131.9707,  -0.0],
    [ 1.570481,   0.000000,   115.2260,  -0.0],
    [-1.571710,   0.000000,   104.3891,   0.0],
]

THETA_LIMIT = [
    (-360, 360),
    (-360, 360),
    (-360, 360),
    (-360, 360),
    (-360, 360),
    (-360, 360),
]

# ============================================================
# Modified DH 变换矩阵
# ============================================================
def dh_transform(alpha, a, d, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct,    -st,     0,     a   ],
        [st*ca, ct*ca,  -sa,  -sa*d ],
        [st*sa, ct*sa,   ca,   ca*d ],
        [0,      0,       0,    1   ]
    ])

def forward_kinematics(theta_deg):
    """返回各坐标系变换矩阵列表（含 Base）"""
    theta_rad = np.deg2rad(theta_deg)
    T_list = [np.eye(4)]
    T = np.eye(4)
    for i, (alpha, a, d, ho) in enumerate(DH_PARAMS):
        T = T @ dh_transform(alpha, a, d, theta_rad[i] + ho)
        T_list.append(T.copy())
    return T_list

def get_positions(T_list):
    return np.array([T[:3, 3] for T in T_list])

# ============================================================
# 可视化
# ============================================================
class Visualization:

    def __init__(self):
        self.sliders = []
        self.theta = [0.0] * 6

    # 坐标轴等比例
    def set_axes_equal(self, ax):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()
        x_center = np.mean(xlim)
        y_center = np.mean(ylim)
        z_center = np.mean(zlim)
        span = max(xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0])
        ax.set_xlim(x_center - span/2, x_center + span/2)
        ax.set_ylim(y_center - span/2, y_center + span/2)
        ax.set_zlim(z_center - span/2, z_center + span/2)
        ax.set_box_aspect([1, 1, 1])

    # 绘制坐标系
    def draw_frame(self, ax, T, length=60):
        o = T[:3, 3]
        ax.quiver(*o, *(T[:3,0]*length), color='r')
        ax.quiver(*o, *(T[:3,1]*length), color='g')
        ax.quiver(*o, *(T[:3,2]*length), color='b')

    # 圆柱连杆
    def draw_cylinder(self, ax, p1, p2, radius=18):
        v = p2 - p1
        L = np.linalg.norm(v)
        if L < 1e-6:
            return
        v = v / L

        theta = np.linspace(0, 2*np.pi, 20)
        z     = np.linspace(0, L, 2)
        theta, z = np.meshgrid(theta, z)

        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        z_axis = np.array([0, 0, 1])
        dot    = np.clip(np.dot(z_axis, v), -1, 1)
        angle  = np.arccos(dot)
        axis   = np.cross(z_axis, v)

        if np.linalg.norm(axis) > 1e-6:
            axis = axis / np.linalg.norm(axis)
            K = np.array([
                [0,       -axis[2],  axis[1]],
                [axis[2],  0,       -axis[0]],
                [-axis[1], axis[0],  0      ]
            ])
            R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K @ K)
            xyz = R @ np.vstack((x.flatten(), y.flatten(), z.flatten()))
            x = xyz[0].reshape(theta.shape)
            y = xyz[1].reshape(theta.shape)
            z = xyz[2].reshape(theta.shape)

        ax.plot_surface(x + p1[0], y + p1[1], z + p1[2], alpha=0.6)

    # 绘制机器人
    def draw_robot(self, ax, T_list):
        positions = get_positions(T_list)
        radii = [30, 25, 22, 18, 15, 12]

        # 连杆
        for i in range(6):
            self.draw_cylinder(ax, positions[i], positions[i+1], radii[i])

        # 关节点
        ax.plot(positions[:,0], positions[:,1], positions[:,2],
                'ko', markersize=6)

        # 编号
        for i, p in enumerate(positions):
            label = 'Base' if i == 0 else f'J{i}'
            ax.text(p[0]+30, p[1]+30, p[2]+30, label, fontsize=8)

        # 末端标记
        tcp = positions[-1]
        ax.scatter(*tcp, color='red', s=60, zorder=5)
        ax.text(tcp[0]+30, tcp[1]+30, tcp[2]+30, 'TCP', fontsize=8, color='red')

        # 坐标系
        for T in T_list:
            self.draw_frame(ax, T)

    # 更新函数
    def update(self, val):
        ax = self.ax

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()
        elev = ax.elev
        azim = ax.azim

        ax.cla()

        for i, s in enumerate(self.sliders):
            self.theta[i] = s.val

        T_list = forward_kinematics(self.theta)

        self.draw_robot(ax, T_list)

        # 末端位姿信息
        end_T = T_list[-1]
        xyz   = end_T[:3, 3]
        r31, r11, r21 = end_T[2,0], end_T[0,0], end_T[1,0]
        r32, r33      = end_T[2,1], end_T[2,2]
        ry = np.arctan2(-r31, np.sqrt(r11**2 + r21**2))
        rz = np.arctan2(r21, r11)   if abs(np.cos(ry)) > 1e-6 else 0.0
        rx = np.arctan2(r32, r33)   if abs(np.cos(ry)) > 1e-6 else 0.0

        info = (
            f"TCP\n"
            f"X={xyz[0]:.1f} mm\n"
            f"Y={xyz[1]:.1f} mm\n"
            f"Z={xyz[2]:.1f} mm\n"
            f"Rx={np.rad2deg(rx):.1f}°\n"
            f"Ry={np.rad2deg(ry):.1f}°\n"
            f"Rz={np.rad2deg(rz):.1f}°"
        )
        ax.text2D(0.02, 0.98, info, transform=ax.transAxes,
                  fontsize=8, va='top', fontfamily='monospace')

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.view_init(elev=elev, azim=azim)
        self.set_axes_equal(ax)

        self.fig.canvas.draw_idle()

    # 主入口
    def visualize(self):
        self.fig = plt.figure(figsize=(9, 9))
        self.ax  = self.fig.add_subplot(111, projection='3d')

        plt.subplots_adjust(left=0.25)

        # 创建滑块
        for i in range(6):
            ax_slider = plt.axes([0.05, 0.85 - i*0.1, 0.15, 0.03])
            s = Slider(ax_slider, f"J{i+1}",
                       THETA_LIMIT[i][0], THETA_LIMIT[i][1],
                       valinit=self.theta[i])
            s.on_changed(self.update)
            self.sliders.append(s)

        workspace = 1000
        self.ax.set_xlim(-workspace, workspace)
        self.ax.set_ylim(-workspace, workspace)
        self.ax.set_zlim(-workspace, workspace)
        self.ax.set_box_aspect([1, 1, 1])

        self.update(None)
        plt.show()


# ============================================================
if __name__ == '__main__':

    # 直接计算，打印结果
    # 对应末端点：266.331，-886.461，936.642，-90.917°，-22.951°，179.988°
    # angles = [-80.339, 64.029, -49.259, 164.373, -280.029, -22.804]

    # 对应末端点：-349.451，-930.015，934.170，-88.757°，-5.908°，-179.532°
    angles = [-121.464, 53.761, -33.861, 161.560, -237.948, -5.135]

    # 初始姿态
    # angles = [0.0, 90.0, 0.0, 0.0, 0.0, 0.0]
    
    T_list = forward_kinematics(angles)
    end_T  = T_list[-1]
    xyz    = end_T[:3, 3]

    r31, r11, r21 = end_T[2,0], end_T[0,0], end_T[1,0]
    r32, r33      = end_T[2,1], end_T[2,2]
    ry = np.arctan2(-r31, np.sqrt(r11**2 + r21**2))
    rz = np.arctan2(r21, r11)  if abs(np.cos(ry)) > 1e-6 else 0.0
    rx = np.arctan2(r32, r33)  if abs(np.cos(ry)) > 1e-6 else 0.0

    print(f"关节角: {angles}")
    print(f"X  = {xyz[0]:.3f} mm")
    print(f"Y  = {xyz[1]:.3f} mm")
    print(f"Z  = {xyz[2]:.3f} mm")
    print(f"Rx = {np.rad2deg(rx):.3f}°")
    print(f"Ry = {np.rad2deg(ry):.3f}°")
    print(f"Rz = {np.rad2deg(rz):.3f}°")

    # 同时打开可视化
    vis = Visualization()
    vis.theta = angles
    vis.visualize()