#引入模块----------------------------------
import numpy as np
import os
import argparse
import torch
#------------------------------------------

#自定义模块--------------------------------
from visiualization import Visiualization
from environment import Environment
from train import Train
from read_and_write_file import ReadAndWritefile
from utils.logger import logger
#------------------------------------------

#注意事项----------------------------------
# 1.所有的输入和输出为角度制
# 2.所有数据保留三维小数
#------------------------------------------

#                    _ooOoo_
#                   o8888888o
#                   88" . "88
#                   (| -_- |)
#                   O\  =  /O
#                ____/`---'\____
#              .'  \\|     |//  `.
#             /  \\|||  :  |||//  \
#            /  _||||| -:- |||||-  \
#            |   | \\\  -  /// |   |
#            | \_|  ''\---/''  |   |
#            \  .-\__  `-`  ___/-. /
#          ___`. .'  /--.--\  `. . __
#       ."" '<  `.___\_<|>_/___.'  >'"".
#      | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#      \  \ `-.   \_ __\ /__ _/   .-` /  /
# ======`-.____`-.___\_____/___.-`____.-'======
#                    `=---='
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#             佛祖保佑       永无BUG


# 主体-------------------------------------

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='机器人轨迹规划训练')
    parser.add_argument('--device', type=str, default='0', 
                        help='指定训练使用的设备: CPU 或 GPU编号(0, 1, 2...) (default: 0)')
    parser.add_argument('--algorithm', type=str, default='PPO', choices=['PPO', 'SAC'],
                        help='选择训练算法: PPO 或 SAC (default: PPO)')
    args = parser.parse_args()
    
    # 设置设备
    if args.device.upper() == 'CPU':
        device = torch.device("cpu")
        print(f"使用设备: CPU")
    else:
        # 尝试解析为GPU编号
        try:
            gpu_id = int(args.device)
            if torch.cuda.is_available():
                if gpu_id < torch.cuda.device_count():
                    device = torch.device(f"cuda:{gpu_id}")
                    print(f"使用设备: cuda:{gpu_id} ({torch.cuda.get_device_name(gpu_id)})")
                else:
                    print(f"错误: GPU {gpu_id} 不存在，可用GPU数量: {torch.cuda.device_count()}")
                    print("回退使用 CPU")
                    device = torch.device("cpu")
            else:
                print("CUDA不可用，使用CPU")
                device = torch.device("cpu")
        except ValueError:
            print(f"错误: 无效的设备参数 '{args.device}'")
            print("使用方式: --device CPU 或 --device 0 或 --device 1 等")
            print("回退使用 CPU")
            device = torch.device("cpu")
    
    print(f"选择算法: {args.algorithm}")
    print("="*80)

    env = Environment()
    
    target_angles = np.array([-80.339, 64.029, -49.259, 164.373, -280.029, -22.804])
    # target_angles = np.array([-121.464, 53.761, -33.861, 161.560, -237.948, -5.135])
    initial_angles = np.array([0, 0, 0, 0, 0, 0])

    env.target_angles = target_angles
    env.initial_angles = initial_angles

    train = Train(env, device=device, algorithm=args.algorithm)
    env.use_random_reset = True
    fileio = ReadAndWritefile()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    training_path = os.path.join(current_dir, "log", "pt_version","0.1.3", args.algorithm)
    best_training_parameters_path = os.path.join(training_path, "best_training_parameters.pt")
    os.makedirs(os.path.dirname(best_training_parameters_path), exist_ok=True)
    last_training_parameters_path = os.path.join(training_path, "last_training_parameters.pt")
    os.makedirs(os.path.dirname(last_training_parameters_path), exist_ok=True)
    

    read_training_parameters_file_path = best_training_parameters_path

    # 这个决定是否重新训练，还是依据上一次的结果再次训练
    if os.path.exists(read_training_parameters_file_path):
        logger.info(f"读取训练参数文件: {read_training_parameters_file_path}")
        fileio.read_training_parameters_file(train.agent, read_training_parameters_file_path)
    else:
        logger.info(f"训练参数文件不存在: {read_training_parameters_file_path}")
    
    train.train()

    
