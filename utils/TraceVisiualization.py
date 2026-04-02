from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from robot.robot import Robot


def _set_axes_equal(ax):
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
    span = max(span, 1.0)

    ax.set_xlim(x_center - span / 2, x_center + span / 2)
    ax.set_ylim(y_center - span / 2, y_center + span / 2)
    ax.set_zlim(z_center - span / 2, z_center + span / 2)
    ax.set_box_aspect([1, 1, 1])


def _draw_cylinder(ax, p1, p2, radius):
    v = p2 - p1
    length = np.linalg.norm(v)
    if length < 1e-9:
        return

    direction = v / length
    theta = np.linspace(0, 2 * np.pi, 20)
    z = np.linspace(0, length, 2)
    theta, z = np.meshgrid(theta, z)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    z_axis = np.array([0.0, 0.0, 1.0])
    dot = np.clip(np.dot(z_axis, direction), -1.0, 1.0)
    angle = np.arccos(dot)
    axis = np.cross(z_axis, direction)

    if np.linalg.norm(axis) > 1e-9:
        axis = axis / np.linalg.norm(axis)
        k = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0]
        ])
        r = np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)
        xyz = np.vstack((x.ravel(), y.ravel(), z.ravel()))
        xyz = r @ xyz
        x = xyz[0].reshape(x.shape)
        y = xyz[1].reshape(y.shape)
        z = xyz[2].reshape(z.shape)

    x += p1[0]
    y += p1[1]
    z += p1[2]
    ax.plot_surface(x, y, z, color="tab:blue", alpha=0.35, linewidth=0, antialiased=False)


def _set_scatter_points(scatter, points):
    scatter._offsets3d = (points[:, 0], points[:, 1], points[:, 2])


def trace_trajectory_to_gif(trajectory, step, save_path):
    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.ndim != 2 or trajectory.shape[1] != 6:
        raise ValueError("trajectory 必须是 n*6 的矩阵")
    if trajectory.shape[0] == 0:
        raise ValueError("trajectory 至少需要 1 行")

    if int(step) != step or step <= 0:
        raise ValueError("step 必须是正整数")
    step = int(step)

    sampled_index = np.arange(0, trajectory.shape[0], step, dtype=int)
    if sampled_index[-1] != trajectory.shape[0] - 1:
        sampled_index = np.append(sampled_index, trajectory.shape[0] - 1)

    robot = Robot()
    dh_robot = robot.dr
    sampled_points = []
    for idx in sampled_index:
        posture = dh_robot.forward_kinematics(trajectory[idx])
        sampled_points.append(posture[:, :3].copy())

    all_points = np.concatenate(sampled_points, axis=0)
    points_min = all_points.min(axis=0)
    points_max = all_points.max(axis=0)
    center = (points_min + points_max) / 2.0
    span = np.max(points_max - points_min)
    span = max(span, 1.0)
    limit_radius = span * 0.6

    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(center[0] - limit_radius, center[0] + limit_radius)
    ax.set_ylim(center[1] - limit_radius, center[1] + limit_radius)
    ax.set_zlim(center[2] - limit_radius, center[2] + limit_radius)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    _set_axes_equal(ax)

    init_points = sampled_points[0]
    line_robot, = ax.plot(init_points[:, 0], init_points[:, 1], init_points[:, 2], color="k", linewidth=1.6)
    joints = ax.scatter(init_points[:, 0], init_points[:, 1], init_points[:, 2], color="k", s=16)
    tcp = ax.scatter(init_points[-1:, 0], init_points[-1:, 1], init_points[-1:, 2], color="r", s=35)
    title = ax.set_title("")

    def update(frame_id):
        points = sampled_points[frame_id]
        line_robot.set_data(points[:, 0], points[:, 1])
        line_robot.set_3d_properties(points[:, 2]) # type: ignore
        _set_scatter_points(joints, points)
        _set_scatter_points(tcp, points[-1:])
        title.set_text(f"Frame {frame_id + 1}/{len(sampled_points)}  StepIndex={sampled_index[frame_id]}")
        return line_robot, joints, tcp, title

    animation = FuncAnimation(
        fig,
        update,
        frames=len(sampled_points),
        interval=100,
        repeat=True,
        blit=False
    )

    save_file = Path(save_path)
    if save_file.suffix.lower() != ".gif":
        save_file = save_file.with_suffix(".gif")
    save_file.parent.mkdir(parents=True, exist_ok=True)

    writer = PillowWriter(fps=10)
    animation.save(str(save_file), writer=writer, dpi=72)
    plt.close(fig)

    return str(save_file)
