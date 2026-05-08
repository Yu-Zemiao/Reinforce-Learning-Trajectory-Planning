# # 用于测试碰撞函数所需时间
# import time
# from environment.collision_environment import CollisionEnvironment
# import numpy as np
# from robot.robot import Robot

# if __name__ == "__main__":
#     env = CollisionEnvironment()
#     env.create_cube([5000, 0, 1], [1, 1, 1])
#     # env.create_cylinder([0, 0, 2], [1, 1, 1])
#     robot = Robot()
#     env.robot = robot
#     angles = np.array([0, 0, 0, 0, 0, 0])
#     start_time = time.time()
#     env.detect_collision(angles)
#     end_time = time.time()
#     print(f"碰撞检测所需时间: {end_time - start_time} 秒")

import time
import torch
from environment.collision_environment import CollisionEnvironment
import numpy as np
from robot.robot import Robot


if __name__ == "__main__":
    env = CollisionEnvironment()
    # env.create_cube([5000, 0, 1], [1, 1, 1])
    env.create_cylinder([5000, 0, 2], [1, 1, 1])

    robot = Robot()
    env.robot = robot

    angles = np.array([0, 0, 0, 0, 0, 0])

    # -------------------------
    # 🔥 1. GPU 预热（非常重要）
    # -------------------------
    for _ in range(5):
        env.detect_collision(angles)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # -------------------------
    # 🔥 2. 正式计时（多次取平均）
    # -------------------------
    repeat = 100

    start_time = time.time()

    for _ in range(repeat):
        env.detect_collision(angles)


    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()

    print(f"平均碰撞检测时间: {(end_time - start_time)/repeat:.6f} 秒")