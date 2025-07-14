"""
模型包 - 包含所有AI模型相关的代码

这个包负责处理AI模型的加载、管理和使用。
目前包含MusicGen模型，用于文本到音乐的生成。

包含的模块：
- musicgen.py: MusicGen模型的实现

使用示例：
    from src.models.musicgen import MusicGen
    
    # 创建MusicGen实例
    generator = MusicGen(model_size="small")
    
    # 生成音乐
    generator.generate("A beautiful melody")
"""

# 这个文件的存在告诉Python这个目录是一个包 