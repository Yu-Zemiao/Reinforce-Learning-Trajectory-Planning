#引入模块----------------------------------
import argparse
import torch
#------------------------------------------

def get_device(device_str='auto'):
    """根据字符串获取 torch.device 对象"""
    if device_str == 'cpu':
        return torch.device('cpu')
    elif device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        # 假设是 GPU 索引
        return torch.device(f'cuda:{device_str}')

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='强化学习轨迹规划')
    parser.add_argument('--device', type=str, default='auto',
                        help='使用的设备: cpu, auto(默认), 或 GPU 编号 (如 0, 1)')
    args = parser.parse_args()

    # 设置全局 device
    global device
    device = get_device(args.device)

    return args, device

# 默认 device，会被 parse_args() 覆盖
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
