"""
设备选择工具模块

这个模块负责检测和选择最优的计算设备。
在AI模型训练和推理中，选择合适的设备（CPU/GPU/MPS）非常重要，
因为它直接影响计算速度和效率。

作者: AI助手
创建时间: 2024年
"""

# 导入PyTorch库，这是深度学习的主要框架
import torch

def get_optimal_device():
    """
    获取最优的计算设备。
    
    这个函数会按照以下优先级选择设备：
    1. MPS (Apple Silicon Mac的Metal Performance Shaders) - 最快
    2. CUDA (NVIDIA GPU) - 次快
    3. CPU - 最慢但兼容性最好
    
    返回值:
        tuple: (device, device_name)
            - device: torch.device对象，可以直接用于模型
            - device_name: 字符串，设备的描述信息
    
    使用示例:
        device, device_name = get_optimal_device()
        print(f"使用设备: {device_name}")
        model.to(device)  # 将模型移动到选定的设备
    """
    
    # 检查是否支持MPS (Apple Silicon Mac专用)
    # torch.backends.mps.is_available() 返回True表示支持MPS
    if torch.backends.mps.is_available():
        # 创建MPS设备对象
        device = torch.device("mps")
        return device, "Apple Silicon (MPS)"
    
    # 检查是否支持CUDA (NVIDIA GPU)
    # torch.cuda.is_available() 返回True表示有NVIDIA GPU且CUDA可用
    elif torch.cuda.is_available():
        # 创建CUDA设备对象
        device = torch.device("cuda")
        # 获取GPU名称，用于显示详细信息
        gpu_name = torch.cuda.get_device_name()
        return device, f"CUDA: {gpu_name}"
    
    # 如果都不支持，使用CPU
    else:
        # 创建CPU设备对象
        device = torch.device("cpu")
        return device, "CPU"

# 如果直接运行这个文件，会执行以下测试代码
if __name__ == "__main__":
    """
    测试代码 - 当直接运行这个文件时执行
    用于验证设备选择功能是否正常工作
    """
    print("🔍 测试设备选择功能...")
    
    # 获取最优设备
    device, device_name = get_optimal_device()
    
    # 打印结果
    print(f"✅ 选择的设备: {device}")
    print(f"📝 设备描述: {device_name}")
    
    # 测试设备是否可用
    if device.type == "mps":
        print("🍏 使用Apple Silicon MPS加速")
    elif device.type == "cuda":
        print("⚡ 使用NVIDIA CUDA加速")
    else:
        print("💻 使用CPU模式") 