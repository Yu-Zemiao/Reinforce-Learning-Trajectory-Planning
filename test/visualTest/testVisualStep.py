from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.TraceVisiualization import trace_trajectory_to_gif

traj = np.zeros((20, 6))

for i in range(20):
    traj[i, 0] = 1 / 20 * i

gif_path = trace_trajectory_to_gif(
    traj,
    step=1,
    save_path=Path(__file__).resolve().parent / "trace.gif"
)
print(gif_path)
