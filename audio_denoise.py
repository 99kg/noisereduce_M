"""
noisereduce 高级音频降噪工具 - 支持指定噪声样本位置
用法: python audio_denoise_advanced.py [选项] <输入文件或目录>
"""

import argparse
import os
import sys
import time
import numpy as np
import noisereduce as nr
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import re

# 声音类型预设参数
PROFILE_SETTINGS = {
    "footsteps": {
        "stationary": True,
        "prop_decrease": 0.9,
        "n_fft": 512,
        "win_length": 512,
        "hop_length": 128,
        "n_std_thresh_stationary": 1.5
    },
    "rain": {
        "stationary": False,
        "prop_decrease": 0.8,
        "n_fft": 2048,
        "win_length": 2048,
        "hop_length": 512,
        "n_std_thresh_stationary": 2.0
    },
    "wind": {
        "stationary": False,
        "prop_decrease": 0.7,
        "n_fft": 1024,
        "win_length": 1024,
        "hop_length": 256,
        "n_std_thresh_stationary": 1.8
    },
    "voice": {
        "stationary": True,
        "prop_decrease": 1.0,
        "n_fft": 1024,
        "win_length": 1024,
        "hop_length": 256,
        "n_std_thresh_stationary": 1.5
    },
    "default": {
        "stationary": True,
        "prop_decrease": 0.95,
        "n_fft": 1024,
        "win_length": 1024,
        "hop_length": 256,
        "n_std_thresh_stationary": 1.7
    }
}

def parse_time_range(time_str, total_duration):
    """
    解析时间范围字符串，支持多种格式：
    - "start": 开头1秒
    - "end": 结尾1秒
    - "10.5-12.0": 从10.5秒到12.0秒
    - "5%": 总时长的5%作为噪声样本
    """
    try:
        # 处理特殊关键字
        if time_str.lower() == "start":
            return (0.0, 1.0)
        elif time_str.lower() == "end":
            return (total_duration - 1.0, total_duration)
        
        # 处理百分比格式
        if '%' in time_str:
            percent = float(time_str.strip('%')) / 100.0
            duration = total_duration * percent
            return (0.0, duration)  # 默认从开头取
        
        # 处理时间范围格式
        if '-' in time_str:
            start, end = map(float, time_str.split('-'))
            return (start, end)
        
        # 单一时间点
        position = float(time_str)
        return (position, position + 1.0)  # 默认取1秒
    
    except Exception:
        print(f"⚠️ 无法解析时间范围: {time_str}, 使用默认开头1秒")
        return (0.0, 1.0)

def extract_noise_clip(audio, sr, time_range, total_duration):
    """根据时间范围提取噪声片段"""
    start_time, end_time = time_range
    
    # 确保时间范围在有效范围内
    start_time = max(0.0, min(start_time, total_duration - 0.1))
    end_time = max(start_time + 0.1, min(end_time, total_duration))
    
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    
    return audio[start_sample:end_sample]

def process_file(input_path, output_path, profile="default", noise_position="start", noise_duration=1.0):
    """处理单个音频文件"""
    try:
        # 读取音频文件
        audio, sr = sf.read(input_path)
        total_duration = len(audio) / sr
        
        # 如果是立体声，转换为单声道
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # 解析噪声位置
        time_range = parse_time_range(noise_position, total_duration)
        
        # 提取噪声样本
        noise_clip = extract_noise_clip(audio, sr, time_range, total_duration)
        
        # 获取预设参数
        params = PROFILE_SETTINGS.get(profile, PROFILE_SETTINGS["default"])
        
        # 应用降噪
        cleaned_audio = nr.reduce_noise(
            y=audio,
            y_noise=noise_clip,
            sr=sr,
            **params
        )
        
        # 保存结果
        sf.write(output_path, cleaned_audio, sr)
        return True
    except Exception as e:
        print(f"  处理失败: {str(e)}")
        return False

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="高级音频降噪工具 - 支持指定噪声样本位置",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("-i", "--input", help="输入文件或目录路径")
    parser.add_argument("-o", "--output", help="输出目录路径", default="cleaned_audio")
    parser.add_argument("-p", "--profile", 
                        choices=["footsteps", "rain", "wind", "voice", "default"],
                        default="default",
                        help="声音类型预设")
    parser.add_argument("-np", "--noise-position", default="start",
                        help="噪声样本位置: 'start'(开头), 'end'(结尾), '10.5-12.0'(时间段), '5%'(百分比)")
    parser.add_argument("-d", "--noise-duration", type=float, default=1.0,
                        help="当使用特殊位置时的默认噪声时长(秒)")
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="递归处理子目录")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="显示详细处理信息")
    parser.add_argument("-t", "--test-noise", action="store_true",
                        help="仅保存噪声样本用于分析")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理单个文件
    if os.path.isfile(args.input):
        input_path = Path(args.input)
        
        if args.test_noise:
            output_path = output_dir / f"noise_sample_{input_path.name}"
        else:
            output_path = output_dir / f"cleaned_{input_path.name}"
        
        print(f"处理文件: {input_path.name}")
        print(f"噪声位置: {args.noise_position}")
        start_time = time.time()
        
        success = process_file(
            input_path, 
            output_path,
            profile=args.profile,
            noise_position=args.noise_position,
            noise_duration=args.noise_duration
        )
        
        if success:
            elapsed = time.time() - start_time
            if args.test_noise:
                print(f"✅ 噪声样本已提取! 保存到: {output_path}")
            else:
                print(f"✅ 降噪完成! 输出文件: {output_path}")
            print(f"   耗时: {elapsed:.2f}秒")
        else:
            print("❌ 处理失败")
    
    # 处理目录
    elif os.path.isdir(args.input):
        input_dir = Path(args.input)
        
        # 收集所有音频文件
        audio_files = []
        extensions = ['.flac', '.wav', '.mp3', '.ogg', '.aiff']
        
        if args.recursive:
            search_pattern = "**/*"
        else:
            search_pattern = "*"
        
        for ext in extensions:
            audio_files.extend(input_dir.glob(f"{search_pattern}{ext}"))
            audio_files.extend(input_dir.glob(f"{search_pattern}{ext.upper()}"))
        
        if not audio_files:
            print("❌ 未找到支持的音频文件")
            return
        
        print(f"找到 {len(audio_files)} 个音频文件")
        print(f"使用预设: {args.profile}")
        print(f"噪声位置: {args.noise_position}")
        
        # 处理所有文件
        success_count = 0
        for file in tqdm(audio_files, desc="处理进度", unit="文件"):
            if args.verbose:
                print(f"\n处理: {file.name}")
            
            if args.test_noise:
                output_filename = f"noise_sample_{file.name}"
            else:
                output_filename = f"cleaned_{file.name}"
            
            output_path = output_dir / file.relative_to(input_dir).parent / output_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            success = process_file(
                file, 
                output_path,
                profile=args.profile,
                noise_position=args.noise_position,
                noise_duration=args.noise_duration
            )
            
            if success:
                success_count += 1
                if args.verbose:
                    print(f"  保存到: {output_path}")
            elif args.verbose:
                print(f"  处理失败")
        
        print(f"\n🎉 处理完成! 成功: {success_count}/{len(audio_files)}")
        print(f"输出目录: {output_dir}")
    
    else:
        print(f"❌ 路径不存在: {args.input}")
        sys.exit(1)

if __name__ == "__main__":
    main()