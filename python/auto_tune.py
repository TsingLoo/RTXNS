import os
import re
import subprocess
import time

def modify_train_params(width, epochs, freq_bands):
    train_file = "d:/repo/RTXNS/python/train_sss.py"
    with open(train_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # regex replace width
    content = re.sub(r'width\s*=\s*\d+', f'width = {width}', content)
    # regex replace epochs
    content = re.sub(r'epochs\s*=\s*\d+', f'epochs = {epochs}', content)
    # regex replace freq bands
    content = re.sub(r'freq_bands\s*=\s*\d+', f'freq_bands={freq_bands}', content)
    
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write(content)
        
def modify_evaluate_params(width, freq_bands):
    eval_file = "d:/repo/RTXNS/python/evaluate_render.py"
    with open(eval_file, 'r', encoding='utf-8') as f:
        content = f.read()
    content = re.sub(r'width\s*=\s*\d+', f'width = {width}', content)
    content = re.sub(r'freq_bands\s*=\s*\d+', f'freq_bands={freq_bands}', content)
    with open(eval_file, 'w', encoding='utf-8') as f:
        f.write(content)

def auto_tune_loop():
    print("========================================")
    print("  Neural SSS Auto-Tuner Initialized")
    print("========================================")
    
    candidates = [
        {"width": 64, "freq": 2, "epoch": 50},
        {"width": 128, "freq": 4, "epoch": 80},
        {"width": 256, "freq": 6, "epoch": 100}
    ]
    
    psnr_file = "c:/tmp/jade_sss_data/psnr.txt"
    target_psnr = 30.0
    
    for c in candidates:
        print(f"\n[Auto-Tuner] 尝试模型配置: Capacity Width={c['width']}, Freq={c['freq']} ...")
        modify_train_params(c['width'], c['epoch'], c['freq'])
        modify_evaluate_params(c['width'], c['freq'])
        
        # 1. Train
        print("[Auto-Tuner] 正在启动后台训练进程...")
        subprocess.run(["python", "d:/repo/RTXNS/python/train_sss.py"])
        
        # 2. Evaluate
        print("[Auto-Tuner] 训练结束，正在执行推理验证与打分...")
        subprocess.run(["python", "d:/repo/RTXNS/python/evaluate_render.py"])
        
        # 3. Check PSNR
        if os.path.exists(psnr_file):
            with open(psnr_file, 'r') as f:
                psnr_str = f.read().strip()
                try:
                    psnr = float(psnr_str)
                    print(f"[Auto-Tuner] 当前配置质量评价 PSNR: {psnr:.2f} dB")
                    if psnr >= target_psnr:
                        print(f"[Auto-Tuner] 质量达标 (>= {target_psnr} dB)! 自动搜参成功完毕！")
                        return
                except:
                    pass
        else:
            print("[Auto-Tuner] 未能找到 PSNR 测试成绩！")
            
    print("[Auto-Tuner] 所有备选配置已耗尽，请检查网络结构上限或渲染集一致性。")

if __name__ == "__main__":
    auto_tune_loop()
