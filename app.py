import torch
import argparse
import os
import time
from pathlib import Path
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile

class MusicGenerator:
    """音乐生成器类 - 支持small和medium模型"""
    
    def __init__(self, model_size="small"):
        self.model_size = model_size
        self.model_name = f"facebook/musicgen-{model_size}"
        self.device = self._get_optimal_device()
        self.processor = None
        self.model = None
        
    def _get_optimal_device(self):
        """获取最优计算设备"""
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("🍏 使用Apple Silicon (MPS) 加速")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"⚡ 使用CUDA加速: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            print("💻 使用CPU模式")
        return device
    
    def load_model(self):
        """加载模型和处理器"""
        print(f"📥 正在加载模型: {self.model_name}")
        start_time = time.time()
        
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = MusicgenForConditionalGeneration.from_pretrained(self.model_name)
        self.model.to(self.device)
        
        load_time = time.time() - start_time
        print(f"✅ 模型加载完成 (耗时: {load_time:.2f}秒)")
    
    def get_default_max_tokens(self):
        """获取默认的最大token数"""
        return 512 if self.model_size == "medium" else 256
    
    def generate(self, prompt, max_tokens=None, output_path=None):
        """生成音乐"""
        if self.model is None:
            self.load_model()
        
        # 设置默认参数
        max_tokens = max_tokens or self.get_default_max_tokens()
        
        if output_path is None:
            timestamp = int(time.time())
            output_path = f"music_{self.model_size}_{timestamp}.wav"
        
        print(f"🎵 生成音乐: '{prompt}'")
        print(f"📊 模型: {self.model_size}, 最大token数: {max_tokens}")
        
        # 处理输入
        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        # 生成音频
        print("🎼 正在生成音频...")
        start_time = time.time()
        
        with torch.no_grad():
            audio_values = self.model.generate(**inputs, max_new_tokens=max_tokens)
        
        generation_time = time.time() - start_time
        
        # 获取采样率
        sampling_rate = self.model.config.audio_encoder.sampling_rate
        
        # 保存音频
        audio_numpy = audio_values[0].cpu().numpy().squeeze()
        scipy.io.wavfile.write(output_path, rate=sampling_rate, data=audio_numpy)
        
        # 计算音频信息
        duration = len(audio_numpy) / sampling_rate
        
        print(f"✅ 音乐生成完成!")
        print(f"📁 保存位置: {output_path}")
        print(f"⏱️ 生成耗时: {generation_time:.2f}秒")
        print(f"🎶 音频时长: {duration:.2f}秒")
        print(f"📊 采样率: {sampling_rate}Hz")
        
        return output_path

def main():
    parser = argparse.ArgumentParser(description="AI音乐生成器 v2.0 - 支持small和medium模型")
    parser.add_argument(
        "--prompt",
        type=str,
        default="A calming piano melody with gentle rain in the background",
        help="音乐描述文本"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["small", "medium"],
        default="small",
        help="选择模型大小 (small/medium)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="最大生成token数"
    )
    
    args = parser.parse_args()
    
    print("🎵 AI音乐生成器 v2.0")
    print("=" * 50)
    
    # 创建生成器
    generator = MusicGenerator(args.model)
    
    # 生成音乐
    try:
        output_file = generator.generate(
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            output_path=args.output
        )
        print(f"\n🎉 音乐生成成功! 文件: {output_file}")
    except Exception as e:
        print(f"\n❌ 生成失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
