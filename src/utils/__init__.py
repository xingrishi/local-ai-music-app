"""
工具包 - 包含各种辅助工具函数

这个包包含项目中使用的各种工具函数，比如设备选择、文件处理等。
这些工具函数被其他模块调用，提供通用的功能支持。

包含的模块：
- device.py: 设备选择相关工具

使用示例：
    from src.utils.device import get_optimal_device
    
    # 获取最优的计算设备
    device, device_name = get_optimal_device()
    print(f"使用设备: {device_name}")
"""

# 这个文件的存在告诉Python这个目录是一个包 