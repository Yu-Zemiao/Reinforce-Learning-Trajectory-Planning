import numpy as np
import matplotlib.pyplot as plt

def plot_robot_cylinders(points, radii):
    # 检查输入长度是否匹配
    if len(points) - 1 != len(radii):
        raise ValueError("radii 的长度必须是 points 长度减 1")

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 生成圆柱体的基础网格数据 (沿 Z 轴)
    # 稍微增加分辨率让圆柱更平滑
    resolution = 20
    theta = np.linspace(0, 2 * np.pi, resolution)
    z_base = np.linspace(0, 1, resolution)
    Theta, Z_base = np.meshgrid(theta, z_base)

    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i + 1]
        r = radii[i]

        # 计算两点之间的向量和距离
        vec = p1 - p0
        length = np.linalg.norm(vec)
        
        # 如果两点重合，跳过
        if length < 1e-6:
            continue
            
        # 目标方向单位向量
        dir_vec = vec / length

        # 基础圆柱的 X 和 Y (未旋转平移前)
        X_base = r * np.cos(Theta)
        Y_base = r * np.sin(Theta)
        # Z轴拉伸至实际长度
        Z_scaled = Z_base * length

        # 将 X, Y, Z 展平，方便矩阵乘法
        base_coords = np.vstack((X_base.flatten(), Y_base.flatten(), Z_scaled.flatten()))

        # 计算从 [0, 0, 1] 到 dir_vec 的旋转矩阵
        # 基准向量
        k = np.array([0, 0, 1])
        
        # 如果方向正好是Z轴或者反Z轴，直接处理以避免叉乘为0的奇异点
        if np.allclose(dir_vec, k):
            R = np.eye(3)
        elif np.allclose(dir_vec, -k):
            R = np.diag([1, -1, -1])
        else:
            # 叉乘求旋转轴向量
            v = np.cross(k, dir_vec)
            # 旋转角度的余弦和正弦
            c = np.dot(k, dir_vec)
            s = np.linalg.norm(v)
            
            # v的反对称矩阵 (Skew-symmetric matrix)
            v_x = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
            
            # Rodrigues 旋转公式
            R = np.eye(3) + v_x + np.dot(v_x, v_x) * ((1 - c) / (s ** 2))

        # 应用旋转
        rotated_coords = np.dot(R, base_coords)

        # 应用平移并还原形状
        X_final = rotated_coords[0, :].reshape(X_base.shape) + p0[0]
        Y_final = rotated_coords[1, :].reshape(Y_base.shape) + p0[1]
        Z_final = rotated_coords[2, :].reshape(Z_scaled.shape) + p0[2]

        # 绘制圆柱体表面
        ax.plot_surface(X_final, Y_final, Z_final, color='cyan', alpha=0.8, edgecolors='k', linewidth=0.1)

    # ================= 坐标系格式化 =================
    
    # 获取所有点的最大绝对值边界，以确保原点 (0,0,0) 在正中心
    # 加上最大的半径，防止圆柱体边缘超出画幅
    max_range = np.max(np.abs(points)) + np.max(radii)
    
    # 1. 原点固定在中心: 强制 X, Y, Z 轴的显示范围是以 0 为中心的对称区间
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    # 2. 坐标轴单位长度固定 (真实比例)
    ax.set_box_aspect([1, 1, 1])

    # 绘制原点标记以供参考
    ax.scatter(0, 0, 0, color='red', s=50, label='Origin (0,0,0)', zorder=5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title("3D Robot Model Visualization")
    plt.show()

# ================= 测试代码 =================
if __name__ == "__main__":
    # 虚拟机器人的 5 个关节节点 (N=5)
    test_points = np.array([
        [0.0, 0.0, 0.0],     # 原点 Base
        [0.0, 0.0, 2.0],     # 第一段垂直
        [1.5, 0.0, 3.5],     # 第二段倾斜
        [3.0, 1.0, 4.0],     # 第三段转向
        [4.0, 2.0, 3.0]      # 末端执行器
    ])
    
    # 4 段轴体的半径 (N-1 = 4)
    # 通常机器人越往末端走越细
    test_radii = np.array([0.4, 0.3, 0.2, 0.1])
    
    plot_robot_cylinders(test_points, test_radii)