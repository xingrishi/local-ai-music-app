from flask import Flask, render_template, request, jsonify, send_file
import os
import time
import uuid
from pathlib import Path
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile

app = Flask(__name__)

# 配置上传文件夹
UPLOAD_FOLDER = 'static/generated'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
        if self.model is not None:
            return
            
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
    
    def generate(self, prompt, max_tokens=None):
        """生成音乐"""
        self.load_model()
        
        # 设置默认参数
        max_tokens = max_tokens or self.get_default_max_tokens()
        
        # 生成唯一文件名
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        output_path = os.path.join(UPLOAD_FOLDER, f"music_{self.model_size}_{timestamp}_{unique_id}.wav")
        
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
        
        return {
            'file_path': output_path,
            'filename': os.path.basename(output_path),
            'duration': duration,
            'generation_time': generation_time,
            'model': self.model_size
        }

# 全局生成器实例
generators = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_music():
    try:
        data = request.get_json()
        prompt = data.get('prompt', 'A calming piano melody')
        model_size = data.get('model', 'small')
        
        # 获取或创建生成器实例
        if model_size not in generators:
            generators[model_size] = MusicGenerator(model_size)
        
        generator = generators[model_size]
        
        # 生成音乐
        result = generator.generate(prompt)
        
        return jsonify({
            'success': True,
            'audio_url': f'/static/generated/{result["filename"]}',
            'filename': result['filename'],
            'duration': result['duration'],
            'generation_time': result['generation_time'],
            'model': result['model']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print("🎵 AI音乐生成器 Web版启动中...")
    print("🌐 访问地址: http://localhost:8080")
    app.run(debug=True, host='0.0.0.0', port=8080) 