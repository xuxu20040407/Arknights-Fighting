import subprocess
from PIL import Image
import os
import cv2
import numpy as np
from src.core.image_recognition import load_templates, recognize_monsters

# 指定ADB路径
ADB_PATH = r"D:\Program Files\Netease\MuMu Player 12\shell\.\adb.exe"
MUMU_ADB_PORT = "16384"
OUTPUT_PATH = "./capture/screenshot.png"  # 截图保存路径
RESOLUTION = [2560, 1440]  # MuMu模拟器分辨率

def check_adb():
    """检查ADB是否可用"""
    try:
        result = subprocess.run([ADB_PATH, "version"], capture_output=True, text=True)
        return "Android Debug Bridge" in result.stdout
    except:
        return False

def connect_mumu():
    """连接MuMu模拟器"""
    try:
        subprocess.run([ADB_PATH, "kill-server"])
        subprocess.run([ADB_PATH, "connect", f"127.0.0.1:{MUMU_ADB_PORT}"])
        
        # 检查连接状态
        result = subprocess.run([ADB_PATH, "devices"], capture_output=True, text=True)
        if f"127.0.0.1:{MUMU_ADB_PORT}" in result.stdout:
            print("MuMu模拟器连接成功")
            return True
        else:
            print("连接失败：设备未列出")
            return False
    except Exception as e:
        print(f"连接失败: {e}")
        return False

def take_specific_screenshot(region=(0, 0, 100, 100)):
    """截取指定区域"""
    try:
        # 截图并保存到设备
        subprocess.run([ADB_PATH, "-s", f"127.0.0.1:{MUMU_ADB_PORT}", "shell", "screencap", "-p", "/sdcard/screen.png"])
        
        # 拉取截图到本地
        subprocess.run([ADB_PATH, "-s", f"127.0.0.1:{MUMU_ADB_PORT}", "pull", "/sdcard/screen.png", OUTPUT_PATH])
        
        # 裁剪并保存
        img = Image.open(OUTPUT_PATH)
        cropped_img = img.crop(region)
        cropped_img.save(OUTPUT_PATH)
        return cropped_img
    except Exception as e:
        print(f"截图失败: {e}")
        return None

def recognize_screenshot(screenshot_path):
    """识别截图中的内容"""
    # 加载模板
    templates_data = load_templates()
    if not templates_data:
        print("模板加载失败，无法进行识别")
        return

    # 加载截图
    screenshot = cv2.imread(screenshot_path)
    if screenshot is None:
        print("无法加载截图文件")
        return

    # 调用识别函数
    output_image, results = recognize_monsters(screenshot, templates_data)
    
    # 保存识别结果图像
    output_path = "./capture/recognized_screenshot.png"
    cv2.imwrite(output_path, output_image)
    print(f"识别完成，结果已保存到 {output_path}")
    print("识别结果:", results)

if __name__ == "__main__":
    if not check_adb():
        print("错误: ADB工具未找到，请安装并配置ADB")
        print("解决方案:")
        print("1. 下载ADB工具: https://developer.android.com/studio/releases/platform-tools")
        print("2. 解压后添加所在目录到系统PATH环境变量")
        print("3. 或者修改代码中的ADB_PATH为你的adb.exe完整路径")
        exit(1)
    
    if not connect_mumu():
        print("连接MuMu模拟器失败，请检查:")
        print(f"1. MuMu模拟器是否正在运行(尝试端口 {MUMU_ADB_PORT})")
        print("2. MuMu设置中是否启用了ADB(设置->其他->开启ADB)")
        print("3. 防火墙是否阻止了ADB连接")
        exit(1)

    screenshot = take_specific_screenshot(region=(RESOLUTION[0]/2-650, RESOLUTION[1]-230, RESOLUTION[0]/2+650, RESOLUTION[1]-70))
    if screenshot:
        screenshot.show()
        print("截图成功并已保存为cropped.png")
        recognize_screenshot(OUTPUT_PATH)