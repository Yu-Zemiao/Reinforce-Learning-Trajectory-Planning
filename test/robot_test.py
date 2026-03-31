import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ================================
# 数学工具
# ================================

def dh_transform(theta, d, a, alpha):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ])

# ================================
# 机器人类
# ================================

class Robot:
    def __init__(self):
        # DH参数 (theta初始值, d, a, alpha)
        self.dh = np.array([
            [0, 141.32,  0,       0   ],
            [0, 0,       181.65,  90  ],
            [0, 0,       571.97,  0   ],
            [0, -131.97, 0,       90  ],
            [0, 115.23,  0,       -90 ],
            [0, 104.39,  0,       0   ]
        ], dtype=float)

        # self.dh = np.array([# 这个参数是根据程序读取的参数输出的，相对更正确一些，但是可能也有问题
        #     [0,  141.32,   0,       0   ],
        #     [0,  0,        0,       90  ],
        #     [0,  0,        596.02,  0   ],
        #     [0,  -131.97,  571.97,  0   ],
        #     [0,  115.23,   0,       90  ],
        #     [0,  104.39,   0,       -90 ]
        # ], dtype=float)

        # 关节角
        self.q = np.zeros(6)

        # 角度限制
        self.limit = np.array([
            [-360, 360],
            [-85, 265],
            [-175, 175],
            [-85, 265],
            [-360, 360],
            [-360, 360]
        ])

    # ================================
    # 前向运动学
    # ================================
    def forward(self):
        T = np.eye(4)
        points = [np.zeros(3)]
        frames = []

        for i in range(6):
            theta = self.q[i] + self.dh[i][0]
            d = self.dh[i][1]
            a = self.dh[i][2]
            alpha = self.dh[i][3]

            Ti = dh_transform(theta, d, a, alpha)
            T = T @ Ti

            points.append(T[:3, 3])
            frames.append(T.copy())

        return np.array(points), frames

# ================================
# 真实比例调整函数 (核心修改点)
# ================================

def set_axes_equal_dynamic(ax):
    """动态强制 3D 坐标轴显示真实的物理比例 (1:1:1)"""
    # 1. 获取当前 X, Y, Z 轴的数据范围
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()

    # 2. 计算每个轴的中心点
    x_center = np.mean(xlim)
    y_center = np.mean(ylim)
    z_center = np.mean(zlim)

    # 3. 计算所有轴中的最大跨度
    max_span = max(xlim[1] - xlim[0], 
                   ylim[1] - ylim[0], 
                   zlim[1] - zlim[0])

    # 4. 以各自的中心点为基础，将所有轴的跨度都设置为相同的最大跨度
    ax.set_xlim(x_center - max_span / 2, x_center + max_span / 2)
    ax.set_ylim(y_center - max_span / 2, y_center + max_span / 2)
    ax.set_zlim(z_center - max_span / 2, z_center + max_span / 2)
    
    # 5. 强制绘图框的各边比例一致
    ax.set_box_aspect([1, 1, 1])

# ================================
# 绘制坐标系
# ================================

def draw_frame(ax, T, length=120):
    o = T[:3, 3]
    x = T[:3, 0]
    y = T[:3, 1]
    z = T[:3, 2]

    ax.quiver(*o, *(x*length), color='r') # x为红色
    ax.quiver(*o, *(y*length), color='g') # y为绿色
    ax.quiver(*o, *(z*length), color='b') # z为蓝色

# ================================
# 绘制圆柱连杆
# ================================

def draw_cylinder(ax, p1, p2, radius=40):
    v = p2 - p1
    L = np.linalg.norm(v)

    if L == 0:
        return

    v = v / L

    theta = np.linspace(0, 2*np.pi, 20)
    z = np.linspace(0, L, 2)
    theta, z = np.meshgrid(theta, z)

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    z_axis = np.array([0, 0, 1])
    
    # 防止浮点数精度导致的 arccos NaN 警告
    dot_product = np.clip(np.dot(z_axis, v), -1.0, 1.0)
    angle = np.arccos(dot_product)
    axis = np.cross(z_axis, v)

    # 处理普通旋转
    if np.linalg.norm(axis) > 1e-6:
        axis = axis / np.linalg.norm(axis)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle)*K + (1 - np.cos(angle))*(K @ K)
        
        xyz = np.vstack((x.flatten(), y.flatten(), z.flatten()))
        xyz = R @ xyz
        x = xyz[0].reshape(x.shape)
        y = xyz[1].reshape(y.shape)
        z = xyz[2].reshape(z.shape)
        
    # 处理两向量反向共线的边缘情况
    elif dot_product < -0.99:
        R = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        xyz = np.vstack((x.flatten(), y.flatten(), z.flatten()))
        xyz = R @ xyz
        x = xyz[0].reshape(x.shape)
        y = xyz[1].reshape(y.shape)
        z = xyz[2].reshape(z.shape)

    x += p1[0]
    y += p1[1]
    z += p1[2]

    ax.plot_surface(x, y, z, alpha=0.6)

# ================================
# 绘制机器人
# ================================

def draw_robot(ax, points, frames):
    # 连杆
    for i in range(len(points)-1):
        draw_cylinder(ax, points[i], points[i+1], 30)

    # 关节
    ax.plot(points[:, 0], points[:, 1], points[:, 2], 'ko', markersize=6)

    # 编号
    for i, p in enumerate(points):
        offset = np.array([30, 30, 30])
        ax.text(
            p[0]+offset[0], p[1]+offset[1], p[2]+offset[2],
            f"J{i}", fontsize=11, color="black", weight="bold"
        )

    # --- 新增：画第0个点（基座）的坐标系 ---
    # 构建一个 4x4 的单位矩阵，并把平移向量替换为 points[0] 的坐标
    T_base = np.eye(4)
    T_base[:3, 3] = points[0]
    draw_frame(ax, T_base, 120)
    # --------------------------------------

    # 坐标系
    for T in frames:
        draw_frame(ax, T, 120)

# ================================
# 主程序
# ================================

robot = Robot()

fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.25)

sliders = []

# 创建滑块
for i in range(6):
    ax_slider = plt.axes([0.05, 0.85 - i*0.1, 0.15, 0.03])
    s = Slider(
        ax_slider, f"J{i}",
        robot.limit[i][0], robot.limit[i][1], valinit=0
    )
    sliders.append(s)

# 初始化时给定一个足够大的工作空间作为起始画布
workspace = 1800
ax.set_xlim(-workspace, workspace)
ax.set_ylim(-workspace, workspace)
ax.set_zlim(-workspace, workspace)
ax.set_box_aspect([1, 1, 1])

# ================================
# 更新函数
# ================================

def update(val):
    # 1. 在清除画布前，记录当前视角和坐标轴缩放状态
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    elev = ax.elev
    azim = ax.azim

    # 清空画布
    ax.cla()

    # 更新关节角度
    for i in range(6):
        robot.q[i] = np.deg2rad(sliders[i].val)

    # 正向运动学计算及渲染
    points, frames = robot.forward()
    draw_robot(ax, points, frames)

    # 重新添加标签
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    # 2. 将视角和缩放范围塞回去
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.view_init(elev=elev, azim=azim)
    
    # 3. 动态修正为绝对真实的 1:1:1 比例
    set_axes_equal_dynamic(ax)
    
    # 刷新显示
    fig.canvas.draw_idle()

for s in sliders:
    s.on_changed(update)

# 初始化视图
update(None)

plt.show()