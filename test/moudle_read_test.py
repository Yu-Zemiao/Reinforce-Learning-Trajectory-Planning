from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_FACE, TopAbs_EDGE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BndLib import bndlib_AddSolid
from OCC.Core.Bnd import Bnd_Box
from OCC.Display.SimpleGui import init_display

def read_step_file(file_path):
    """
    读取STEP文件，返回解析后的模型和几何信息
    :param file_path: STEP文件路径（.step/.stp）
    :return: shape - 整体模型, solid_list - 所有实体列表
    """
    # 1. 初始化STEP读取器
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(file_path)
    
    # 检查文件读取状态
    if status != 1:  # 1=读取成功，0=失败，2=警告
        raise Exception(f"读取STEP文件失败！状态码：{status}，文件路径：{file_path}")
    
    # 2. 转换STEP数据为OCCT的几何形状（Shape）
    step_reader.TransferRoots()  # 转换所有根节点
    shape = step_reader.OneShape()  # 获取整体模型
    
    # 3. 遍历提取实体（Solid）信息
    solid_list = []
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)  # 只遍历实体类型
    while explorer.More():
        solid = explorer.Current()
        # 计算实体的包围盒（获取坐标范围）
        bbox = Bnd_Box()
        bndlib_AddSolid(bbox, solid)
        x_min, y_min, z_min, x_max, y_max, z_max = bbox.Get()
        
        solid_info = {
            "shape": solid,
            "type": "Solid",
            "bounding_box": (x_min, y_min, z_min, x_max, y_max, z_max),
            "volume": solid.Volume() if hasattr(solid, 'Volume') else None  # 体积（可选）
        }
        solid_list.append(solid_info)
        
        explorer.Next()  # 遍历下一个实体
    
    print(f"成功读取STEP文件：{file_path}")
    print(f"提取到实体数量：{len(solid_list)}")
    for i, info in enumerate(solid_list):
        print(f"  实体{i+1} - 坐标范围：{info['bounding_box']}")
    
    return shape, solid_list

if __name__ == "__main__":
    # 替换为你的STEP文件路径
    step_file_path = "JAKA C 12-V2.0_20231016.STEP"  # 或 "your_model.stp"
    
    try:
        # 读取STEP文件
        model_shape, solids = read_step_file(step_file_path)
        
        # 4. 可视化模型（弹出窗口显示3D模型）
        display, start_display, add_menu, add_function_to_menu = init_display()
        display.DisplayShape(model_shape, update=True)
        display.FitAll()
        start_display()  # 启动可视化窗口
        
    except Exception as e:
        print(f"错误：{e}")