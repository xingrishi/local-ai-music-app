"""
AI音乐生成器 - 主程序入口

这是整个应用的主入口文件，负责：
1. 解析命令行参数
2. 设置计算设备
3. 创建音乐生成器
4. 执行音乐生成

使用方法:
    python main.py --model small --prompt "A peaceful piano melody"
    python main.py --model medium --prompt "An energetic rock song" --max-tokens 1024

作者: AI助手
创建时间: 2024年
"""

# 导入标准库
import argparse  # 用于解析命令行参数

# 导入我们自己的模块
from models.musicgen import MusicGen  # 音乐生成模型
from utils.device import get_optimal_device  # 设备选择工具

def main():
    """
    主函数 - 程序的入口点
    
    这个函数负责：
    1. 解析用户输入的命令行参数
    2. 设置最优的计算设备
    3. 创建音乐生成器实例
    4. 执行音乐生成
    5. 处理可能的错误
    """
    
    # 创建命令行参数解析器
    # description参数会在用户输入 --help 时显示
    parser = argparse.ArgumentParser(description="AI音乐生成器 (模块化版)")
    
    # 添加 --prompt 参数，用于指定音乐描述
    parser.add_argument(
        "--prompt",  # 参数名
        type=str,   # 参数类型为字符串
        default="A calming piano melody with gentle rain in the background",  # 默认值
        help="音乐描述文本"  # 帮助信息
    )
    
    # 添加 --model 参数，用于选择模型大小
    parser.add_argument(
        "--model",
        type=str,
        choices=["small", "medium"],  # 只允许这两个选项
        default="small",  # 默认使用small模型
        help="选择模型大小 (small/medium)"  # 帮助信息
    )
    
    # 添加 --output 参数，用于指定输出文件路径
    parser.add_argument(
        "--output",
        type=str,
        default=None,  # 默认值为None，表示自动生成文件名
        help="输出文件路径"  # 帮助信息
    )
    
    # 添加 --max-tokens 参数，用于控制生成音乐的长度
    parser.add_argument(
        "--max-tokens",
        type=int,  # 参数类型为整数
        default=None,  # 默认值为None，表示使用模型默认值
        help="最大生成token数"  # 帮助信息
    )
    
    # 解析命令行参数
    # 如果用户输入了参数，args会包含这些值
    # 如果用户没有输入，会使用默认值
    args = parser.parse_args()
    
    # 获取最优的计算设备
    # get_optimal_device() 会返回一个元组：(device, device_name)
    device, device_name = get_optimal_device()
    print(f"使用设备: {device_name}")
    
    # 创建音乐生成器实例
    # MusicGen类是我们自定义的类，封装了模型的所有功能
    generator = MusicGen(model_size=args.model, device=device)
    
    # 执行音乐生成
    # generate() 方法会：
    # 1. 加载模型（如果还没加载）
    # 2. 处理文本输入
    # 3. 生成音频
    # 4. 保存为WAV文件
    generator.generate(
        prompt=args.prompt,        # 音乐描述
        max_tokens=args.max_tokens,  # 最大token数
        output_path=args.output    # 输出文件路径
    )

# 这是Python的特殊语法，表示"如果直接运行这个文件"
# 当用户执行 "python main.py" 时，这个条件为True
# 当这个文件被其他文件导入时，这个条件为False
if __name__ == "__main__":
    """
    程序入口点
    
    当直接运行这个文件时，会执行main()函数
    这样可以确保main()函数只在直接运行时执行，
    而不会在被导入时执行
    """
    main()

# 如果直接运行这个文件，会执行以下测试代码
if __name__ == "__main__":
    """
    测试代码 - 当直接运行这个文件时执行
    用于验证整个程序是否正常工作
    """
    print("🎵 AI音乐生成器启动...")
    print("=" * 50)
    
    try:
        # 调用主函数
        main()
        print("\n🎉 程序执行完成!")
        
    except KeyboardInterrupt:
        # 用户按Ctrl+C中断程序
        print("\n⏹️ 程序被用户中断")
        
    except Exception as e:
        # 捕获其他所有错误
        print(f"\n❌ 程序执行出错: {e}")
        print("请检查参数是否正确，网络是否连接")
        
        # 显示使用帮助
        print("\n💡 使用帮助:")
        print("python main.py --help  # 查看所有参数")
        print("python main.py --model small --prompt 'A piano melody'  # 基本使用")
        print("python main.py --model medium --max-tokens 1024 --prompt 'Rock music'  # 高级使用") 