# 引入模块----------------------------------
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import imageio
# ------------------------------------------
# 自定义模块--------------------------------
from environment.environment import Environment
from utils.logger import logger
# ------------------------------------------

current_dir = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(current_dir, "log", "image")

class Visualization:

    def __init__(self, environment: Environment):
        self.env = environment
        self.robot = environment.robot
        self.save_picture_path = os.path.join(data_path, "posture.png")
        self.save_trajectory_path = os.path.join(data_path, "trajectory.gif")

    # ============================
    # 坐标轴等比例
    # ============================
    def set_axes_equal(self, ax):
        limits = np.array([
            ax.get_xlim(),
            ax.get_ylim(),
            ax.get_zlim()
        ])

        center = np.mean(limits, axis=1)
        span = np.max(limits[:, 1] - limits[:, 0])

        ax.set_xlim(center[0] - span/2, center[0] + span/2)
        ax.set_ylim(center[1] - span/2, center[1] + span/2)
        ax.set_zlim(center[2] - span/2, center[2] + span/2)
        ax.set_box_aspect([1, 1, 1])

    # ============================
    # 两点之间画圆柱
    # ============================
    def draw_cylinder_between_points(self, ax, p_start, p_end, radius, color='lightblue', alpha=0.8):
        """在两点之间画一个圆柱体，具有真实三维宽度"""
        p_start = np.array(p_start)
        p_end = np.array(p_end)
        
        v = p_end - p_start
        length = np.linalg.norm(v)
        
        if length < 1e-9:
            return  # 两点重合，跳过
        
        # 圆柱局部坐标：默认沿 z 轴，半径为 radius
        n_theta = 20
        theta = np.linspace(0, 2 * np.pi, n_theta)
        z_local = np.linspace(0, length, 10)
        theta, z_local = np.meshgrid(theta, z_local)
        
        x_local = radius * np.cos(theta)
        y_local = radius * np.sin(theta)
        
        # 圆柱方向向量（从 start 到 end）
        v_norm = v / length
        
        # 构建旋转矩阵：将 z 轴旋转到 v_norm 方向
        z_axis = np.array([0, 0, 1])
        if np.allclose(v_norm, z_axis):
            rot_matrix = np.eye(3)
        elif np.allclose(v_norm, -z_axis):
            rot_matrix = -np.eye(3)
        else:
            axis = np.cross(z_axis, v_norm)
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(np.dot(z_axis, v_norm))
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            rot_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        
        # 将圆柱局部坐标旋转并平移
        n_points = x_local.shape[0] * x_local.shape[1]
        local_coords = np.column_stack([x_local.ravel(), y_local.ravel(), z_local.ravel()])
        global_coords = np.dot(local_coords, rot_matrix.T) + p_start
        
        X = global_coords[:, 0].reshape(x_local.shape)
        Y = global_coords[:, 1].reshape(y_local.shape)
        Z = global_coords[:, 2].reshape(z_local.shape)
        
        ax.plot_surface(X, Y, Z, color=color, alpha=alpha)

    # ============================
    # 画球体（关节）
    # ============================
    def draw_sphere(self, ax, center, radius, color='steelblue', alpha=0.9):
        """在指定位置画一个球体"""
        u = np.linspace(0, 2 * np.pi, 15)
        v = np.linspace(0, np.pi, 15)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(x, y, z, color=color, alpha=alpha)

    # ============================
    # 画机械臂（真实宽度版）
    # ============================
    def draw_robot(self, ax, posture=None, initial_posture=None, target_posture=None):

        # ======================
        # 当前机械臂
        # ======================
        if posture is None:
            posture, _ = self.robot.forward_kinematics()
        point = np.array(posture[:, :3])

        # ======================
        # 初始点
        # ======================
        if initial_posture is None:
            initial_posture, _ = self.robot.forward_kinematics(self.env.initial_angles)
        initial_point = np.array(initial_posture[-1, :3])  # 末端

        # ======================
        # 目标点
        # ======================
        if target_posture is None:
            target_posture, _ = self.robot.forward_kinematics(self.env.target_angles)
        target_point = np.array(target_posture[-1, :3])  # 末端

        # ======================
        # 安全检查
        # ======================
        if point.ndim != 2 or point.shape[1] != 3:
            raise ValueError(f"FK输出错误，应该是(N,3)，但得到 {point.shape}")

        pts = point
        arm_r = self.robot.arm_radius

        # ======================
        # 画连杆（圆柱）
        # ======================
        for i in range(len(pts) - 1):
            self.draw_cylinder_between_points(
                ax, pts[i], pts[i+1], arm_r[i],
                color='lightblue', alpha=0.8
            )

        # ======================
        # 画关节（球）
        # ======================
        for i, p in enumerate(pts):
            joint_radius = arm_r[i] if i < len(arm_r) else arm_r[-1]
            self.draw_sphere(ax, p, joint_radius * 1.2, color='steelblue', alpha=0.9)

        # ======================
        # 标关节
        # ======================
        for i, p in enumerate(pts):
            ax.text(p[0], p[1], p[2], f"J{i}", fontsize=9)

        # ======================
        # 画初始点（绿色）
        # ======================
        self.draw_sphere(ax, initial_point, arm_r[-1]*1.5, color='green', alpha=0.9)
        ax.text(initial_point[0], initial_point[1], initial_point[2],
                "Initial", color='green', fontsize=10)

        # ======================
        # 画目标点（红色）
        # ======================
        self.draw_sphere(ax, target_point, arm_r[-1]*1.5, color='red', alpha=0.9)
        ax.text(target_point[0], target_point[1], target_point[2],
                "Target", color='red', fontsize=10)

        # ======================
        # 可选：画连线（目标方向）
        # ======================
        ax.plot(
            [pts[-1, 0], target_point[0]],
            [pts[-1, 1], target_point[1]],
            [pts[-1, 2], target_point[2]],
            linestyle='--',
            color='red',
            linewidth=1
        )

    # ============================
    # 画立方体
    # ============================
    def draw_cube(self, ax, obs, label):
        x, y, z, l, w, h, _ = obs

        xx = [x, x + l]
        yy = [y, y + w]
        zz = [z, z + h]

        for X, Y in [(xx, yy)]:
            X, Y = np.meshgrid(X, Y)
            ax.plot_surface(X, Y, np.full_like(X, z), alpha=0.3, color='red')
            ax.plot_surface(X, Y, np.full_like(X, z + h), alpha=0.3, color='red')

        for Y, Z in [(yy, zz)]:
            Y, Z = np.meshgrid(Y, Z)
            ax.plot_surface(np.full_like(Y, x), Y, Z, alpha=0.3, color='red')
            ax.plot_surface(np.full_like(Y, x + l), Y, Z, alpha=0.3, color='red')

        for X, Z in [(xx, zz)]:
            X, Z = np.meshgrid(X, Z)
            ax.plot_surface(X, np.full_like(X, y), Z, alpha=0.3, color='red')
            ax.plot_surface(X, np.full_like(X, y + w), Z, alpha=0.3, color='red')

        cx, cy, cz = x + l/2, y + w/2, z + h/2
        ax.text(cx, cy, cz, label, color='black', fontsize=10, ha='center', va='center')

    # ============================
    # 画圆柱
    # ============================
    def draw_cylinder(self, ax, obs, label):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        x, y, z, r, h, direction, _ = obs

        theta = np.linspace(0, 2 * np.pi, 30)
        t = np.linspace(0, h, 10)
        theta, t = np.meshgrid(theta, t)

        if direction == 0:
            X = x + t
            Y = y + r * np.cos(theta)
            Z = z + r * np.sin(theta)
            cx, cy, cz = x + h/2, y, z
            
            theta_circle = np.linspace(0, 2 * np.pi, 30, endpoint=False)
            bottom_verts = [[x, y + r * np.cos(th), z + r * np.sin(th)] for th in theta_circle]
            top_verts = [[x + h, y + r * np.cos(th), z + r * np.sin(th)] for th in theta_circle]
        elif direction == 1:
            X = x + r * np.cos(theta)
            Y = y + t
            Z = z + r * np.sin(theta)
            cx, cy, cz = x, y + h/2, z
            
            theta_circle = np.linspace(0, 2 * np.pi, 30, endpoint=False)
            bottom_verts = [[x + r * np.cos(th), y, z + r * np.sin(th)] for th in theta_circle]
            top_verts = [[x + r * np.cos(th), y + h, z + r * np.sin(th)] for th in theta_circle]
        else:
            X = x + r * np.cos(theta)
            Y = y + r * np.sin(theta)
            Z = z + t
            cx, cy, cz = x, y, z + h/2
            
            theta_circle = np.linspace(0, 2 * np.pi, 30, endpoint=False)
            bottom_verts = [[x + r * np.cos(th), y + r * np.sin(th), z] for th in theta_circle]
            top_verts = [[x + r * np.cos(th), y + r * np.sin(th), z + h] for th in theta_circle]

        ax.plot_surface(X, Y, Z, alpha=0.3, color='blue')
        
        if bottom_verts:
            bottom_poly = Poly3DCollection([bottom_verts], alpha=0.3, color='blue')
            ax.add_collection3d(bottom_poly)
        
        if top_verts:
            top_poly = Poly3DCollection([top_verts], alpha=0.3, color='blue')
            ax.add_collection3d(top_poly)
        
        ax.text(cx, cy, cz, label, color='darkblue', fontsize=10, ha='center', va='center')

    def draw_obstacles(self, ax):
        cube_idx = cylinder_idx = 1
        for obs in self.env.ce.obstacles:
            if int(obs[6]) == 0:
                self.draw_cube(ax, obs, f"cube{cube_idx}")
                cube_idx += 1
            else:
                self.draw_cylinder(ax, obs, f"cylinder{cylinder_idx}")
                cylinder_idx += 1

    # ============================
    # 主入口
    # ============================
    def save_picture(self, posture = None, save_path=None):

        if save_path is None:
            save_path = self.save_picture_path
        
        if posture is None:
            posture = self.robot.forward_kinematics()

        self.fig = plt.figure(figsize=(9, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.ax.set_xlim(-1000, 1000)
        self.ax.set_ylim(-1000, 1000)
        self.ax.set_zlim(-1000, 1000)

        # 关键：确保比例正确
        self.set_axes_equal(self.ax)

        self.draw_robot(self.ax, posture)
        self.draw_obstacles(self.ax)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(self.fig)
        

    def save_trajectory(self, trajectory=None, save_path=None, sample_step=10, fps=40):

            if save_path is None:
                save_path = self.save_trajectory_path

            if trajectory is None:
                trajectory = self.env.trajectory

            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            frames = []

            # --- 核心修改部分 ---
            # 预先计算好需要采样的索引列表
            indices = list(range(0, len(trajectory), sample_step))
            
            # 如果轨迹不为空，并且最后一个索引不是轨迹的最后一步，则强制把最后一步加进去
            if len(trajectory) > 0 and indices[-1] != len(trajectory) - 1:
                indices.append(len(trajectory) - 1)
            # --------------------

            # 遍历修改后的索引列表
            for i in indices:

                theta = trajectory[i]

                # 正确解包 FK
                posture, _ = self.robot.forward_kinematics(theta)

                # 创建图像
                fig = plt.figure(figsize=(6, 6))
                ax = fig.add_subplot(111, projection='3d')

                ax.set_xlim(-1000, 1000)
                ax.set_ylim(-1000, 1000)
                ax.set_zlim(-1000, 1000)

                self.set_axes_equal(ax)

                # 画机械臂 + 障碍物
                self.draw_robot(ax, posture)
                self.draw_obstacles(ax)

                # 标题
                ax.set_title(f"Step: {i}", fontsize=10)

                # 获取图像（兼容新 matplotlib）
                fig.canvas.draw()
                img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]

                frames.append(img)

                plt.close(fig)

            # 保存 GIF
            imageio.mimsave(save_path, frames, fps=fps)

            logger.info(f"Trajectory GIF saved to: {save_path}")
