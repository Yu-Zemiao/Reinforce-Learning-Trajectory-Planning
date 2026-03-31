import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

def plot_capsule(ax, p0, p1, R, color='blue', alpha=0.3):
    """
    在 3D 坐标系中绘制一个胶囊体（圆柱 + 两端球体）
    p0: 起点 (x, y, z)
    p1: 终点 (x, y, z)
    R: 胶囊体半径
    """
    p0 = np.array(p0)
    p1 = np.array(p1)
    v = p1 - p0
    length = np.linalg.norm(v)
    
    # 1. 如果长度极小，退化为一个球
    if length < 1e-5:
        u, v_sph = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = p0[0] + R * np.cos(u) * np.sin(v_sph)
        y = p0[1] + R * np.sin(u) * np.sin(v_sph)
        z = p0[2] + R * np.cos(v_sph)
        # 【关键修复】：加入 shade=False 绕过计算光照时的矩阵广播 Bug
        ax.plot_surface(x, y, z, color=color, alpha=alpha, shade=False, edgecolors='none')
        return

    # 2. 计算圆柱体的法向量，用于生成表面
    v = v / length
    not_v = np.array([1, 0, 0])
    if abs(np.dot(v, not_v)) > 0.99:
        not_v = np.array([0, 1, 0])
    n1 = np.cross(v, not_v)
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(v, n1)
    
    # 3. 绘制圆柱部分 (Cylinder)
    t = np.linspace(0, length, 10)
    theta = np.linspace(0, 2 * np.pi, 20)
    t, theta = np.meshgrid(t, theta)
    X, Y, Z = [p0[i] + v[i] * t + R * np.sin(theta) * n1[i] + R * np.cos(theta) * n2[i] for i in [0, 1, 2]]
    # 【关键修复】：加入 shade=False
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, shade=False, edgecolors='none')
    
    # 4. 绘制两端球体 (Spheres at ends)
    u, v_sph = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    for p in [p0, p1]:
        x = p[0] + R * np.cos(u) * np.sin(v_sph)
        y = p[1] + R * np.sin(u) * np.sin(v_sph)
        z = p[2] + R * np.cos(v_sph)
        # 【关键修复】：加入 shade=False
        ax.plot_surface(x, y, z, color=color, alpha=alpha, shade=False, edgecolors='none')

def main():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("DH Model (Red Line) vs Physical Model (Blue Capsule)", fontsize=14)

    # --- 依据你提供的图纸定义的关键节点 (单位: mm) ---
    # 为了视觉上与图纸对应，我们在 Y 轴上应用水平偏移
    offsets = 181.65 
    
    # 节点坐标 (DH模型中的关节中心)
    p_base = [0, 0, 0]
    p_shoulder = [0, 0, 142.65]
    p_upper_arm_start = [0, -offsets, 142.65]
    p_elbow = [0, -offsets, 142.65 + 595]
    p_wrist_start = [0, -offsets, 142.65 + 595 + 571.5]
    p_wrist_end = [0, -offsets + 115, 142.65 + 595 + 571.5] # 腕部向前伸出

    # 包装成列表，方便循环
    joints = [p_base, p_shoulder, p_upper_arm_start, p_elbow, p_wrist_start, p_wrist_end]
    
    # 对应的胶囊体半径 (含安全膨胀)
    # [底座, 肩部偏移(不画胶囊), 大臂, 小臂, 腕部]
    radii = [80, 0, 65, 50, 45]

    # --- 1. 绘制物理模型 (蓝色的半透明胶囊体) ---
    for i in range(len(joints)-1):
        if radii[i] > 0: # 过滤掉纯偏移的虚拟连杆
            plot_capsule(ax, joints[i], joints[i+1], radii[i], color='cyan', alpha=0.3)

    # --- 2. 绘制 DH 运动学骨架 (红色的线段和黑色的关节点) ---
    xs = [p[0] for p in joints]
    ys = [p[1] for p in joints]
    zs = [p[2] for p in joints]
    
    # 画线段 (DH连杆)
    ax.plot(xs, ys, zs, color='red', linewidth=3, label="DH Skeleton (Kinematics)")
    # 画散点 (DH关节)
    ax.scatter(xs, ys, zs, color='black', s=50, zorder=5, label="Joints")

    # --- 设置显示参数 ---
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.legend()
    
    # 强制让 X, Y, Z 轴比例一致 (避免模型被拉伸或压扁)
    # 提取所有坐标的范围来计算一个中心和最大极差
    max_range = np.array([max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs)]).max() / 2.0
    mid_x = (max(xs)+min(xs)) * 0.5
    mid_y = (max(ys)+min(ys)) * 0.5
    mid_z = (max(zs)+min(zs)) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()

if __name__ == "__main__":
    main()