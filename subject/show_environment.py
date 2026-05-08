#引入模块----------------------------------
import numpy as np
import os
#------------------------------------------

#自定义模块--------------------------------
from visualization import Visualization
from environment.environment import Environment
#------------------------------------------

if __name__ == "__main__":

    env = Environment()
    visualization = Visualization(env)
    
    env.ce.create_cube([500, 0, 500], [200, 200, 200])
    # env.ce.create_cylinder([-500, 0, 0], [50, 200, 1])

    initial_angles = np.array([0, 0, 0, 0, 0, 0])
    target_angles = np.array([-80.339, 64.029, -49.259, 164.373, -280.029, -22.804])

    initial_posture, _ = env.robot.forward_kinematics(initial_angles)
    target_posture, _ = env.robot.forward_kinematics(target_angles)

    angles = []
    for i in range(101):
        temp = initial_angles + (target_angles - initial_angles) * i / 100
        angles.append(temp)
    angles = np.array(angles)

    visualization.save_picture(posture = target_posture)
    # visualization.save_trajectory(trajectory = angles, sample_step = 10, fps = 1)
