"""
AI音乐生成应用 - 源代码包

这个包包含了AI音乐生成应用的所有核心代码。
主要功能是根据文本描述生成音乐，支持small和medium两种模型。

包结构：
- models/: 模型相关代码
- utils/: 工具函数
- main.py: 主程序入口

使用示例：
    from src.models.musicgen import MusicGen
    from src.utils.device import get_optimal_device
    
    # 获取最优设备
    device, device_name = get_optimal_device()
    
    # 创建音乐生成器
    generator = MusicGen(model_size="small", device=device)
    
    # 生成音乐
    generator.generate("A peaceful piano melody")
"""

# 这个文件的存在告诉Python这个目录是一个包
# 当其他文件导入这个包时，Python会执行这个文件 