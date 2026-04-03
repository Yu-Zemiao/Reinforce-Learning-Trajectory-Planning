#引入模块----------------------------------
import numpy as np
import torch
#------------------------------------------

#自定义模块--------------------------------
from agent.PPO_agent import PPOAgent
from utils.logger import logger
#------------------------------------------

#注意事项----------------------------------
# 1.所有的输入和输出为角度制
# 2.所有数据保留三维小数
#------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReadAndWritefile:

    def __init__(self):

        # 轨迹文件
        self.read_trajectory_file_path = None
        self.write_trajectory_file_path = None
        # 训练参数文件
        self.read_training_parameters_file_path = None
        self.write_training_parameters_file_path = None

        self.write_reward_file_path = None



    # 检查路径是否存在
    def file_path_exist_detect(self, file_path):
        
        if not file_path:
            raise ValueError("路径为空！")

        return 1
    
    # 检查容器数据是否存在
    def container_exist_detect(self, container):
        if container is None:
            raise ValueError("容器为空！")
        
        return 1
    
    # 读取轨迹文件
    def read_trajectory_file(self, read_trajectory_file_path = None):

        path = read_trajectory_file_path if read_trajectory_file_path is not None else self.read_trajectory_file_path
        
        self.file_path_exist_detect(path) # 先检查路径是否存在

        container = np.loadtxt(path) # type: ignore

        logger.info(f"读取轨迹成功！共读取{container.shape[0]}个数据")

        return container

    # 输出轨迹文件
    def write_trajectory_file(self, write_trajectory_file_path = None, trajectory_container = None):

        container = trajectory_container
        path = write_trajectory_file_path if write_trajectory_file_path is not None else self.write_trajectory_file_path

        self.container_exist_detect(container)
        self.file_path_exist_detect(path) # 检查路径是否存在

        np.savetxt(path, container, fmt='%.3f')  # type: ignore

        logger.info(f"输出轨迹成功！共输出{container.shape[0] if container is not None else 0}个数据，训练数据保存至{path}")

    # 读取训练数据
    def read_training_parameters_file(self, agent:PPOAgent, read_training_parameters_file_path = None, inference = False):
        
        path = read_training_parameters_file_path if read_training_parameters_file_path is not None else self.read_training_parameters_file_path
        self.file_path_exist_detect(path) # 检查路径是否存在
        
        state_dict = torch.load(path, map_location = device) # type: ignore

        # 加载模型参数
        agent.policy.load_state_dict(state_dict)

        # 推理模式暂时用不了
        if inference:
            agent.policy.eval() # 推理模式
        else:
            agent.policy.train() # 训练模式

        logger.info("读取训练数据成功！")

    # 输出训练数据
    def write_training_parameters_file(self, agent, write_training_parameters_file_path = None):

        path = write_training_parameters_file_path if write_training_parameters_file_path is not None else self.write_training_parameters_file_path
        self.file_path_exist_detect(path) # 检查路径是否存在

        try:
            torch.save(agent.policy.state_dict(), path) # type: ignore
        except RuntimeError as e:
            # 如果文件被占用或无权限，这里会给出明确的中文提示，而不是抛出一长串底层的 Traceback
            raise RuntimeError(f"保存失败！请检查文件是否被占用或是否有写入权限。\n目标路径: {path}\n底层报错: {e}")

    def write_reward_file(self, reward_container = None, write_reward_file_path = None):
        
        path = write_reward_file_path if write_reward_file_path is not None else self.write_reward_file_path
        self.file_path_exist_detect(path) # 检查路径是否存在
        self.container_exist_detect(reward_container)
        np.savetxt(path, reward_container, fmt='%.3f')  # type: ignore

        logger.info(f"输出奖励成功！共输出{len(reward_container)}个数据，奖励数据保存至{path}") # type: ignore
              
