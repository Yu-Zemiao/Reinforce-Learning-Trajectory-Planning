#引入模块----------------------------------
import numpy as np
from sympy.geometry import point
from utils.logger import logger
import torch
#------------------------------------------
#自定义模块--------------------------------
from robot.robot import Robot
#------------------------------------------
#注意事项----------------------------------
# 1. 圆柱体和立方体的方向只能是x轴、y轴或z轴
# 2. 圆柱体的direction参数只能是0、1或2，分别代表x轴、y轴、z轴
# 3. 对于立方体，求解是否直接碰撞采用精确解，求解是否间接碰撞采用采样解
# 4. 对于圆柱体，求解是否直接碰撞和是否间接碰撞均采用采样解
# 5. 所谓采样解，就是采样一段线段的点，判断是否有点在圆柱体内
#------------------------------------------
# 主体-------------------------------------

class CollisionEnvironment:
    def __init__(self):
        self.robot = Robot()

        # 障碍物
        self.obstacles = np.empty((0, 7)) 

    # ----------------------------------------------------------------------------------------
    # 创建障碍物
    # 0 代表立方体
    # 1 代表圆柱体

    # 创建立方体障碍物
    # 最后一个参数为0，代表立方体
    # 不考虑旋转
    def create_cube(self, initial_point, cube_size):
        x, y, z = initial_point
        l, w, h = cube_size
        obstacle = np.array([[x, y, z, l, w, h, 0]])
        self.obstacles = np.append(self.obstacles, obstacle, axis=0)

    # 创建圆柱障碍物
    # 最后一个参数为1，代表圆柱体
    # 不考虑旋转
    # direction: 0 代表x轴，1 代表y轴，2 代表z轴
    def create_cylinder(self, initial_point, cylinder_parameters):
        x, y, z = initial_point
        r, h, direction = cylinder_parameters
        obstacle = np.array([[x, y, z, r, h, direction, 1]])
        self.obstacles = np.append(self.obstacles, obstacle, axis=0)

    # ----------------------------------------------------------------------------------------
    # 碰撞检测
    def collision_detect(self, angles):

        min_dist = float('inf')

        if len(self.obstacles) == 0:
            return False, 0

        for obstacle in self.obstacles:
            if obstacle[6] == 0:
                is_collision, dist = self.cube_collision_detect(angles, obstacle)
                if is_collision:
                    return True, 0
                min_dist = min(min_dist, dist)
            elif obstacle[6] == 1:
                is_collision, dist = self.cylinder_collision_detect(angles, obstacle)
                if is_collision:
                    return True, 0
                min_dist = min(min_dist, dist)

        return False, min_dist

    # -----------------------------------------------------------------
    # 立方体碰撞检测
    # 存在两种会被视为碰撞的情况
    # 情况1：线段在立方体内
    # 情况2：线段在立方体外，但是距离小于阈值（即机械臂半径）
    def cube_collision_detect(self, angles, obstacle):

        dr_joint_posture, cr_joint_posture = self.robot.forward_kinematics(angles)
        joint_point = dr_joint_posture[:, :3]

        x, y, z, l, w, h, _ = obstacle
        cube = np.array([
            [x, x + l],
            [y, y + w],
            [z, z + h]
        ])

        arm_radius = self.robot.arm_radius
        min_dist = float('inf')

        for i in range(len(joint_point) - 1):
            
            joint_point1 = np.array(joint_point[i])
            joint_point2 = np.array(joint_point[i+1])
            
            wether_collision, dist = self.cube_point_collision_detect(joint_point1, joint_point2, cube, arm_radius[i])
            min_dist = min(min_dist, dist)
            if wether_collision:
                return True, 0      

        return False, min_dist

    def cube_point_collision_detect(self, joint_point1, joint_point2, cube, arm_radius, device="cuda"):

        joint_point1 = torch.tensor(joint_point1, dtype=torch.float32, device=device)
        joint_point2 = torch.tensor(joint_point2, dtype=torch.float32, device=device)
        cube = torch.tensor(cube, dtype=torch.float32, device=device)

        d = joint_point2 - joint_point1
        length = torch.norm(d)

        # 每1mm采样
        steps = max(int(length.item() / 1.0), 1)

        # 所有采样点
        t = torch.linspace(0, 1, steps + 1, device=device).unsqueeze(1)
        points = joint_point1 + t * d

        x_min, y_min, z_min = cube[:, 0]
        x_max, y_max, z_max = cube[:, 1]

        px = points[:, 0]
        py = points[:, 1]
        pz = points[:, 2]

        # -------------------------
        # 点到AABB距离
        # -------------------------
        dx = torch.maximum(x_min - px, torch.zeros_like(px))
        dx = torch.maximum(dx, px - x_max)

        dy = torch.maximum(y_min - py, torch.zeros_like(py))
        dy = torch.maximum(dy, py - y_max)

        dz = torch.maximum(z_min - pz, torch.zeros_like(pz))
        dz = torch.maximum(dz, pz - z_max)

        dist = torch.sqrt(dx**2 + dy**2 + dz**2)

        # 全局最小距离
        min_dist = torch.min(dist)

        # 是否碰撞
        wether_collision = min_dist <= arm_radius

        return wether_collision.item(), min_dist.item()

    # -----------------------------------------------------------------
    # 圆柱碰撞检测
    # 存在两种会被视为碰撞的情况
    # 情况1：线段在圆柱体内
    # 情况2：线段在圆柱体外，但是距离小于阈值（即机械臂半径）
    # 均采用采样解
    def cylinder_collision_detect(self, angles, obstacle):
        
        dr_joint_posture, cr_joint_posture = self.robot.forward_kinematics(angles)
        joint_point = dr_joint_posture[:, :3]
        arm_radius = self.robot.arm_radius

        cylinder = obstacle
        min_dist = float('inf')
        
        for i in range(len(joint_point) - 1):
            joint_point1 = np.array(joint_point[i])
            joint_point2 = np.array(joint_point[i+1])

            wether_collision, dist = self.cylinder_point_collision_detect(joint_point1, joint_point2, cylinder, arm_radius[i])
            min_dist = min(min_dist, dist)
            if wether_collision:
                return True, 0

        return False, min_dist

    def cylinder_point_collision_detect(self, joint_point1, joint_point2, cylinder, arm_radius, device="cuda"):

        joint_point1 = torch.tensor(joint_point1, dtype=torch.float32, device=device)
        joint_point2 = torch.tensor(joint_point2, dtype=torch.float32, device=device)

        d = joint_point2 - joint_point1
        length = torch.norm(d)

        x, y, z, r, h, direction, _ = cylinder

        # -------------------------
        # 采样点
        # -------------------------
        steps = max(int(length.item() / 1.0), 1)

        t = torch.linspace(0, 1, steps + 1, device=device).unsqueeze(1)
        points = joint_point1 + t * d

        px, py, pz = points[:, 0], points[:, 1], points[:, 2]

        # -------------------------
        # 统一坐标系
        # -------------------------
        if direction == 0:

            axis = px
            radial1 = py - y
            radial2 = pz - z

            axis_min = x
            axis_max = x + h

        elif direction == 1:

            axis = py
            radial1 = px - x
            radial2 = pz - z

            axis_min = y
            axis_max = y + h

        else:

            axis = pz
            radial1 = px - x
            radial2 = py - y

            axis_min = z
            axis_max = z + h

        # =========================
        # 径向距离
        # =========================
        radial_dist = torch.sqrt(radial1**2 + radial2**2)

        # =========================
        # 轴向超出距离
        # =========================
        axis_outside = torch.zeros_like(axis)

        axis_outside = torch.where(
            axis < axis_min,
            axis_min - axis,
            axis_outside
        )

        axis_outside = torch.where(
            axis > axis_max,
            axis - axis_max,
            axis_outside
        )

        # =========================
        # 径向超出距离
        # =========================
        radial_outside = torch.clamp(radial_dist - r, min=0.0)

        # =========================
        # 点到有限圆柱体距离
        # =========================
        dist = torch.sqrt(radial_outside**2 + axis_outside**2)

        # =========================
        # 最小距离
        # =========================
        min_dist = torch.min(dist)

        # =========================
        # 是否碰撞
        # =========================
        wether_collision = (min_dist <= arm_radius).item()

        return wether_collision, min_dist.item()
