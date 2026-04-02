import numpy as np
from utils.TraceVisiualization import trace_trajectory_to_gif

traj = np.zeros((20, 6))

for i in range(20):
    traj[i, 0] = 1 / 20 * i

gif_path = trace_trajectory_to_gif(traj, step=1, save_path=r"./trace.gif")
print(gif_path)