"""
MusicGen模型模块

这个模块实现了Facebook的MusicGen模型，用于文本到音乐的生成。
MusicGen是一个基于Transformer的AI模型，可以根据文本描述生成音乐。

支持的模型大小：
- small: 300M参数，适合快速生成和测试
- medium: 1.5B参数，生成质量更高但需要更多资源

作者: AI助手
创建时间: 2024年
"""

# 导入必要的库
import time  # 用于计时
from transformers import AutoProcessor, MusicgenForConditionalGeneration  # Hugging Face的模型库
import torch  # PyTorch深度学习框架
import scipy.io.wavfile  # 用于保存音频文件

class MusicGen:
    """
    MusicGen模型类
    
    这个类封装了MusicGen模型的所有功能，包括：
    - 模型加载
    - 文本处理
    - 音乐生成
    - 音频保存
    """
    
    def __init__(self, model_size="small", device=None):
        """
        初始化MusicGen模型
        
        参数:
            model_size (str): 模型大小，可选 "small" 或 "medium"
                - small: 300M参数，加载快，内存需求少
                - medium: 1.5B参数，质量高，但需要更多内存和时间
            device (torch.device): 计算设备，如果为None则自动选择
        
        使用示例:
            # 创建small模型实例
            generator = MusicGen(model_size="small")
            
            # 创建medium模型实例，指定设备
            generator = MusicGen(model_size="medium", device=torch.device("mps"))
        """
        # 保存模型大小
        self.model_size = model_size
        
        # 构建模型名称（Hugging Face Hub上的模型ID）
        self.model_name = f"facebook/musicgen-{model_size}"
        
        # 设置计算设备，如果没有指定则使用CPU
        self.device = device or torch.device("cpu")
        
        # 初始化模型和处理器为None，延迟加载
        self.processor = None  # 文本处理器
        self.model = None      # 音乐生成模型

    def load_model(self):
        """
        加载模型和处理器
        
        这个方法会从Hugging Face Hub下载并加载模型文件。
        首次运行时会下载模型（small约2.5GB，medium约6-8GB），
        后续运行会使用本地缓存的模型。
        
        注意:
            - 需要网络连接来下载模型
            - 需要足够的磁盘空间存储模型文件
            - 需要足够的内存来加载模型
        """
        print(f"📥 正在加载模型: {self.model_name}")
        
        # 记录开始时间，用于计算加载耗时
        start_time = time.time()
        
        # 加载文本处理器（将文本转换为模型能理解的数字）
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        # 加载音乐生成模型
        self.model = MusicgenForConditionalGeneration.from_pretrained(self.model_name)
        
        # 将模型移动到指定的计算设备（CPU/GPU/MPS）
        self.model.to(self.device)
        
        # 计算并显示加载耗时
        load_time = time.time() - start_time
        print(f"✅ 模型加载完成 (耗时: {load_time:.2f}秒)")

    def get_default_max_tokens(self):
        """
        获取默认的最大token数
        
        返回值:
            int: 默认的token数量
                - small模型: 256 tokens
                - medium模型: 512 tokens
        
        说明:
            token数量决定了生成音乐的长度，token越多音乐越长
            但也会增加生成时间和内存使用
        """
        # medium模型使用更多token，生成更长的音乐
        return 512 if self.model_size == "medium" else 256

    def generate(self, prompt, max_tokens=None, output_path=None):
        """
        生成音乐
        
        这是核心方法，将文本描述转换为音乐文件。
        
        参数:
            prompt (str): 音乐描述文本，例如 "A peaceful piano melody"
            max_tokens (int, 可选): 最大生成token数，决定音乐长度
            output_path (str, 可选): 输出文件路径，如果为None则自动生成
        
        返回值:
            str: 生成的音频文件路径
        
        使用示例:
            # 基本使用
            generator.generate("A beautiful piano melody")
            
            # 指定参数
            generator.generate(
                prompt="An energetic rock song",
                max_tokens=1024,
                output_path="my_music.wav"
            )
        """
        # 如果模型还没加载，先加载模型
        if self.model is None:
            self.load_model()
        
        # 如果没有指定max_tokens，使用默认值
        max_tokens = max_tokens or self.get_default_max_tokens()
        
        # 如果没有指定输出路径，自动生成文件名
        if output_path is None:
            # 使用时间戳确保文件名唯一
            timestamp = int(time.time())
            output_path = f"music_{self.model_size}_{timestamp}.wav"
        
        # 显示生成信息
        print(f"🎵 生成音乐: '{prompt}'")
        print(f"📊 模型: {self.model_size}, 最大token数: {max_tokens}")
        
        # 使用处理器将文本转换为模型输入
        # padding=True: 自动填充到相同长度
        # return_tensors="pt": 返回PyTorch张量
        inputs = self.processor(
            text=[prompt],  # 文本列表，这里只有一个文本
            padding=True,   # 自动填充
            return_tensors="pt",  # 返回PyTorch张量
        ).to(self.device)  # 移动到指定设备
        
        # 开始生成音频
        print("🎼 正在生成音频...")
        start_time = time.time()
        
        # 使用torch.no_grad()禁用梯度计算，节省内存
        with torch.no_grad():
            # 调用模型生成音频
            audio_values = self.model.generate(**inputs, max_new_tokens=max_tokens)
        
        # 计算生成耗时
        generation_time = time.time() - start_time
        print(f"⏱️ 生成耗时: {generation_time:.2f}秒")
        
        # 从模型配置中获取采样率（通常是32000Hz）
        sampling_rate = self.model.config.audio_encoder.sampling_rate
        
        # 将音频张量转换为numpy数组并保存
        # .cpu(): 将张量从GPU移动到CPU
        # .numpy(): 转换为numpy数组
        # .squeeze(): 移除多余的维度
        audio_numpy = audio_values[0].cpu().numpy().squeeze()
        
        # 使用scipy保存为WAV文件
        scipy.io.wavfile.write(output_path, rate=sampling_rate, data=audio_numpy)
        
        # 计算音频时长
        duration = len(audio_numpy) / sampling_rate
        
        # 显示完成信息
        print(f"✅ 音乐生成完成! 保存位置: {output_path}")
        print(f"🎶 音频时长: {duration:.2f}秒, 采样率: {sampling_rate}Hz")
        
        # 返回输出文件路径
        return output_path

# 如果直接运行这个文件，会执行以下测试代码
if __name__ == "__main__":
    """
    测试代码 - 当直接运行这个文件时执行
    用于验证MusicGen模型是否正常工作
    """
    print("🧪 测试MusicGen模型...")
    
    try:
        # 创建small模型实例
        generator = MusicGen(model_size="small")
        
        # 生成测试音乐
        output_file = generator.generate(
            prompt="A simple piano melody",
            max_tokens=128  # 使用较少的token进行快速测试
        )
        
        print(f"🎉 测试成功! 生成的文件: {output_file}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print("请检查网络连接和依赖是否正确安装") 