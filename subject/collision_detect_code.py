class collision_environment:
    def cube_point_collision_detect(self, joint_point1, joint_point2, cube, arm_radius):

        d = joint_point2 - joint_point1  # 方向向量

        x_min, y_min, z_min = cube[:, 0]
        x_max, y_max, z_max = cube[:, 1]

        # 采样点，每1mm取1个点
        steps = int(np.linalg.norm(d) / 1.0)
        steps = max(steps, 1)
        for i in range(steps + 1):
            t = i / steps
            point = joint_point1 + t * d
            point_x, point_y, point_z = point

            # 情况1：点在立方体内
            if (x_min <= point_x <= x_max and
                y_min <= point_y <= y_max and
                z_min <= point_z <= z_max):
                return True

            # 情况2：点在立方体外，但是距离小于阈值（即机械臂半径）
            dx = max(x_min - point_x, 0, point_x - x_max)
            dy = max(y_min - point_y, 0, point_y - y_max)
            dz = max(z_min - point_z, 0, point_z - z_max)

            dist = np.sqrt(dx**2 + dy**2 + dz**2)

            if dist <= arm_radius:
                return True

        return False

    # 检测线段是否与立方体直接碰撞
    def segment_cube_direct_collision(self, point1: np.ndarray, point2: np.ndarray, cube):

        t_min = 0.0
        t_max = 1.0

        d = point2 - point1  # 方向向量

        for i in range(3):  # x,y,z
            if abs(d[i]) < 1e-8:
                # 平行
                if point1[i] < cube[i][0] or point1[i] > cube[i][1]:
                    return False
            else:
                t1 = (cube[i][0] - point1[i]) / d[i]
                t2 = (cube[i][1] - point1[i]) / d[i]

                t_enter = min(t1, t2)
                t_exit = max(t1, t2)

                t_min = max(t_min, t_enter)
                t_max = min(t_max, t_exit)

                if t_min > t_max:
                    return False # 无交集

        return True # 有交集，说明直接碰撞

    # 线段没有直接与立方体碰撞，但是距离小于阈值（即机械臂半径）
    def segment_cube_outside_collision(self, point1: np.ndarray, point2: np.ndarray, cube: np.ndarray, arm_radius):
        
        x_min, y_min, z_min = cube[:, 0]
        x_max, y_max, z_max = cube[:, 1]

        # 点到AABB距离
        def point_to_aabb_distance(point):
            dx = max(x_min - point[0], 0, point[0] - x_max)
            dy = max(y_min - point[1], 0, point[1] - y_max)
            dz = max(z_min - point[2], 0, point[2] - z_max)
            return np.sqrt(dx*dx + dy*dy + dz*dz)

        # 采样点
        steps = 100   # 可以调（20~100都行）
        for i in range(steps + 1):
            t = i / steps
            point = point1 + t * (point2 - point1)

            if point_to_aabb_distance(point) < arm_radius:
                return True

        return False

    def cylinder_point_collision_detect(self, joint_point1, joint_point2, cylinder, arm_radius):

        d = joint_point2 - joint_point1
        x, y, z, r, h, direction, _ = cylinder

        steps = int(np.linalg.norm(d) / 1.0)
        steps = max(steps, 1)

        for i in range(steps + 1):
            t = i / steps
            px, py, pz = joint_point1 + t * d

            # =========================
            # 根据方向统一坐标系
            # =========================
            if direction == 0:      # x轴
                axis = px
                radial1, radial2 = py - y, pz - z
                axis_min, axis_max = x, x + h

            elif direction == 1:    # y轴
                axis = py
                radial1, radial2 = px - x, pz - z
                axis_min, axis_max = y, y + h

            elif direction == 2:    # z轴
                axis = pz
                radial1, radial2 = px - x, py - y
                axis_min, axis_max = z, z + h

            # =========================
            # 计算径向距离
            # =========================
            radial_dist = np.hypot(radial1, radial2)

            # =========================
            # 情况1：点在圆柱内部
            # =========================
            if (axis_min <= axis <= axis_max) and (radial_dist <= r):
                return True

            # =========================
            # 情况2：侧面距离
            # =========================
            if axis_min <= axis <= axis_max:
                if radial_dist <= r + arm_radius:
                    return True

            # =========================
            # 情况3：端面距离
            # =========================
            axis_dist = min(abs(axis - axis_min), abs(axis - axis_max))

            if radial_dist <= r:
                if axis_dist <= arm_radius:
                    return True

            # =========================
            # 情况4：角落（边缘）
            # =========================
            dist = np.sqrt((radial_dist - r)**2 + axis_dist**2)

            if dist <= arm_radius:
                return True
        
        return False