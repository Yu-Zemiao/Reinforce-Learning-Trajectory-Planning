import numpy as np

def dh_matrix(theta, d, a, alpha):
    """
    根据给定的四个参数，计算并返回标准 DH 齐次变换矩阵
    """
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

# 简化的 UR5 机械臂标准 DH 参数表 (单位：米, 弧度)
# 常数项：d (连杆偏置), a (连杆长度), alpha (连杆转角)
# 变量项：theta 将在下面单独给出
dh_table = [
    {"d": 0.089159, "a": 0.0,      "alpha": np.pi/2},   # 关节 1
    {"d": 0.0,      "a": -0.42500, "alpha": 0.0},       # 关节 2
    {"d": 0.0,      "a": -0.39225, "alpha": 0.0},       # 关节 3
    {"d": 0.10915,  "a": 0.0,      "alpha": np.pi/2},   # 关节 4
    {"d": 0.09465,  "a": 0.0,      "alpha": -np.pi/2},  # 关节 5
    {"d": 0.0823,   "a": 0.0,      "alpha": 0.0}        # 关节 6
]

# 设定我们要测试的当前 6 个关节的转角 (这里全设为 0 弧度，即机械臂的初始伸直状态)
joint_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# 1. 初始化一个 4x4 的单位矩阵，代表基坐标系 (Base Frame)
T_end = np.eye(4)

# 2. 从关节 1 到关节 6，依次计算并相乘变换矩阵
for i in range(6):
    # 计算当前关节的坐标变换矩阵: i-1_T_i
    T_i = dh_matrix(joint_angles[i], dh_table[i]["d"], dh_table[i]["a"], dh_table[i]["alpha"])
    
    # 矩阵右乘 (基坐标系 -> 关节1 -> 关节2 ... -> 末端)
    T_end = np.dot(T_end, T_i)

# 3. 提取最终的姿态 (Rotation) 和 位置 (Position)
position = T_end[:3, 3]    # 提取第4列的前3个元素 (X, Y, Z)
rotation = T_end[:3, :3]   # 提取左上角的 3x3 旋转矩阵

np.set_printoptions(suppress=True, precision=4) # 格式化输出
print("===== 最终的齐次变换矩阵 (0_T_6) =====")
print(T_end)

print("\n===== 末端执行器位置 (米) =====")
print(f"X: {position[0]:.4f}, Y: {position[1]:.4f}, Z: {position[2]:.4f}")

print("\n===== 末端执行器姿态 (3x3旋转矩阵) =====")
print(rotation)