import os

import numpy as np
import torch

from environment import Environment
from read_and_write_file import ReadAndWritefile
from train import Train

from utils.TraceVisiualization import trace_trajectory_to_gif
from utils.logger import logger


def generate_trajectory_from_model(max_steps=4000):
    env = Environment()
    env.max_steps = int(max_steps)

    theta_limits = env.robot.theta_limits.astype(float)
    end_angle = np.array([-80.339, 64.029, -49.259, 164.373, -280.029, -22.804])
    begin_angle = np.array([0, 0, 0, 0, 0, 0])

    env.initial_set(begin_angle, end_angle)
    train = Train(env)
    fileio = ReadAndWritefile()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    training_path = os.path.join(current_dir, "log", "train")
    best_training_parameters_path = os.path.join(training_path, "best_training_parameters.pt")
    model_path = best_training_parameters_path

    fileio.read_training_parameters_file(train.agent, model_path, inference=True)

    env.use_random_reset = False
    policy_device = next(train.agent.policy.parameters()).device
    state = torch.FloatTensor(env.train_reset()).to(policy_device)
    trajectory = [env.theta.copy()]

    for _ in range(env.max_steps):
        with torch.no_grad():
            # 使用确定性策略：直接使用actor输出的均值
            mean_action = torch.tanh(train.agent.policy.actor(state)) * train.agent.policy.action_bound
            action = mean_action.detach().cpu().numpy()

        next_state, reward, done, success = env.step(action)
        trajectory.append(env.theta.copy())
        state = torch.FloatTensor(next_state).to(policy_device)

        if done:
            break

    trajectory = np.asarray(trajectory, dtype=float)

    logger.info("Begin angle: %s", np.round(begin_angle, 3))
    logger.info("End angle: %s", np.round(end_angle, 3))
    logger.info("Trajectory shape: %s", trajectory.shape)
    logger.info("angle error: %s", np.round(trajectory[-1] - end_angle, 3))
    logger.info("%s", trajectory)

    return trajectory


def test_batch(batch_size=100):
    logger.info("training start")
    env = Environment()
    theta_limits = env.robot.theta_limits.astype(float)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    training_path = os.path.join(current_dir, "log", "train")
    best_training_parameters_path = os.path.join(training_path, "best_training_parameters.pt")
    model_path = best_training_parameters_path

    train = Train(env)
    fileio = ReadAndWritefile()
    fileio.read_training_parameters_file(train.agent, model_path, inference=True)
    policy_device = next(train.agent.policy.parameters()).device

    for i in range(batch_size):
        begin_angle = np.array([
            np.random.uniform(theta_limits[j, 0], theta_limits[j, 1]) for j in range(6)
        ], dtype=float)
        end_angle = np.array([
            np.random.uniform(theta_limits[j, 0], theta_limits[j, 1]) for j in range(6)
        ], dtype=float)
        env.initial_set(begin_angle, end_angle)
        env.use_random_reset = False
        state = torch.FloatTensor(env.train_reset()).to(policy_device)
        trajectory = [env.theta.copy()]

        for _ in range(env.max_steps):
            with torch.no_grad():
                # 使用确定性策略：直接使用actor输出的均值
                mean_action = torch.tanh(train.agent.policy.actor(state)) * train.agent.policy.action_bound
                action = mean_action.detach().cpu().numpy()

            next_state, reward, done, success = env.step(action)
            trajectory.append(env.theta.copy())
            state = torch.FloatTensor(next_state).to(policy_device)

            if done:
                break
        
        trajectory = np.asarray(trajectory, dtype=float)

        angle_error = np.round(trajectory[-1] - end_angle, 3)
        # trajectory_path = os.path.join(output_dir, f"trajectory_{i + 1:04d}.txt")
        # np.savetxt(trajectory_path, trajectory, fmt="%f\n")
        logger.info(f"[{i + 1}/{batch_size}]trajectory angle_error: {angle_error}, begin_angle: {np.round(begin_angle, 3)}, end_angle: {np.round(end_angle, 3)}")
        

def test_trajectory():
    logger.info("Begin test trace.")

    trajectory = generate_trajectory_from_model(max_steps=4000)

    GIF_path = "test_trajectory.gif"
    trace_trajectory_to_gif(trajectory, step=10, save_path=GIF_path)
    logger.info(f"GIF generated, {GIF_path}")


if __name__ == "__main__":
    
    # test_trajectory()
    test_batch(batch_size=100)


