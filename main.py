#引入模块----------------------------------
import numpy as np
import os
#------------------------------------------

#自定义模块--------------------------------
from visiualization import Visiualization
from environment import Environment
from train import Train
from read_and_write_file import ReadAndWritefile
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

    env = Environment()
    
    target_angles = np.array([-80.339, 64.029, -49.259, 164.373, -280.029, -22.804])
    # target_angles = np.array([-121.464, 53.761, -33.861, 161.560, -237.948, -5.135])
    initial_angles = np.array([0, 0, 0, 0, 0, 0])

    env.target_angles = target_angles
    env.initial_angles = initial_angles

    train = Train(env)
    env.use_random_reset = True
    fileio = ReadAndWritefile()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    training_path = os.path.join(current_dir, "log", "pt_version","0.1.3")
    best_training_parameters_path = os.path.join(training_path, "best_training_parameters.pt")
    os.makedirs(os.path.dirname(best_training_parameters_path), exist_ok=True)
    last_training_parameters_path = os.path.join(training_path, "last_training_parameters.pt")
    os.makedirs(os.path.dirname(last_training_parameters_path), exist_ok=True)
    

    read_training_parameters_file_path = best_training_parameters_path

    # 这个决定是否重新训练，还是依据上一次的结果再次训练
    # fileio.read_training_parameters_file(train.agent, read_training_parameters_file_path)

    train.train()

    
