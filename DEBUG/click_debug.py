import subprocess
import time
import random
from PIL import Image, ImageDraw
import sys
import io
from datetime import datetime

# 配置区域 ==============================================
ADB_PATH = r"D:\Program Files\Netease\MuMu Player 12\shell\.\adb.exe"
MUMU_ADB_PORT = "16384"  # MuMu 12用16384，MuMu 6用7555

DEBUG_MODE = True  # 调试模式会显示更多信息

# 要点击的区域列表（left, top, right, bottom）
CLICK_TASKS  = [((2300, 1200, 2400, 1400), 10)
    # ((1800, 1150, 2400, 1300),2),   # 示例：第一个按钮区域
    # ((1800, 1100, 2000, 1200),2),   # 示例：第二个按钮区域
    # ((2300, 1100, 2400, 1200),2), 
    # ((1200, 870, 1350, 950),13) 
]

# 全局设置 ==============================================
MIN_INTERVAL = 0.5  # 任务间最小间隔（防检测）
MAX_RETRY = 2       # 失败重试次数

# 函数定义 ==============================================
def debug_log(message):
    """调试日志输出"""
    if DEBUG_MODE:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[DEBUG][{timestamp}] {message}")

def check_adb():
    """检查ADB是否可用"""
    try:
        result = subprocess.run([ADB_PATH, "version"], capture_output=True, text=True)
        if "Android Debug Bridge" in result.stdout:
            debug_log("ADB检测成功")
            return True
        debug_log("ADB检测失败：未找到ADB")
        return False
    except Exception as e:
        debug_log(f"ADB检测异常: {str(e)}")
        return False

def connect_mumu():
    """连接MuMu模拟器"""
    try:
        debug_log("尝试连接MuMu模拟器...")
        # 重启ADB服务
        subprocess.run([ADB_PATH, "kill-server"], timeout=5)
        subprocess.run([ADB_PATH, "start-server"], timeout=5)
        
        # 连接模拟器
        result = subprocess.run(
            [ADB_PATH, "connect", f"127.0.0.1:{MUMU_ADB_PORT}"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if "connected" in result.stdout:
            debug_log(f"成功连接到MuMu模拟器 (端口 {MUMU_ADB_PORT})")
            return True
        
        print(f"❌ 连接失败: {result.stdout.strip()}")
        return False
        
    except subprocess.TimeoutExpired:
        print("❌ ADB连接超时")
        return False
    except Exception as e:
        print(f"❌ 连接异常: {str(e)}")
        return False

def take_screenshot(save_path=None):
    """获取屏幕截图（可选保存）"""
    try:
        debug_log("正在截取屏幕...")
        img_data = subprocess.check_output(
            [ADB_PATH, "-s", f"127.0.0.1:{MUMU_ADB_PORT}", "exec-out", "screencap", "-p"],
            timeout=5
        )
        img = Image.open(io.BytesIO(img_data))
        if save_path:
            img.save(save_path)
            debug_log(f"截图已保存到: {save_path}")
        return img
    except Exception as e:
        debug_log(f"截图失败: {str(e)}")
        return None

def smart_wait(seconds, reason=""):
    """智能等待函数（支持随机范围和进度显示）"""
    start_time = time.time()
    
    # 处理随机等待
    if isinstance(seconds, (tuple, list)):
        seconds = random.uniform(seconds[0], seconds[1])
    
    if seconds <= 0:
        return
    
    # 带原因的等待提示
    if reason:
        print(f"⏳ 等待 {seconds:.1f}s | {reason}", end="", flush=True)
    
    while True:
        elapsed = time.time() - start_time
        remaining = max(0, seconds - elapsed)
        
        if DEBUG_MODE and reason:
            print(f"\r⏳ 等待 {remaining:.1f}s | {reason}", end="", flush=True)
        
        if elapsed >= seconds:
            break
        
        time.sleep(0.1)  # 降低CPU占用
    
    if reason:
        print()  # 换行

def click_region(region, attempt=1):
    """点击指定区域（带重试机制）"""
    left, top, right, bottom = region
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    
    debug_log(f"尝试点击 [{center_x},{center_y}] (第{attempt}次尝试)")
    
    try:
        subprocess.run(
            [ADB_PATH, "-s", f"127.0.0.1:{MUMU_ADB_PORT}", "shell", "input", "tap", str(center_x), str(center_y)],
            check=True,
            timeout=5,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        debug_log("点击成功")
        return True
    except subprocess.TimeoutExpired:
        debug_log("点击超时")
    except Exception as e:
        debug_log(f"点击异常: {str(e)}")
    
    return False

def visualize_tasks(screenshot):
    """可视化所有点击任务（调试用）"""
    draw = ImageDraw.Draw(screenshot)
    
    for i, (region, _) in enumerate(CLICK_TASKS, 1):
        left, top, right, bottom = region
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        
        # 绘制区域矩形
        draw.rectangle(region, outline="red", width=3)
        
        # 绘制中心点和序号
        draw.ellipse([(center_x-10, center_y-10), (center_x+10, center_y+10)], fill="blue")
        draw.text((center_x-5, center_y-7), str(i), fill="white")
    
    screenshot.show()
    debug_log("可视化标记已完成（显示5秒后自动关闭）")
    time.sleep(5)
    return screenshot

# 主程序 ==============================================
def main():
    print("\n" + "="*40)
    print(" MuMu模拟器自动化点击脚本")
    print(f" 调试模式: {'启用' if DEBUG_MODE else '禁用'}")
    print("="*40 + "\n")
    
    # 1. 环境检查
    if not check_adb():
        print("❌ 错误: ADB工具不可用")
        print("请检查:")
        print(f"1. ADB路径是否正确: {ADB_PATH}")
        print("2. MuMu模拟器是否安装了ADB驱动")
        sys.exit(1)
    
    # 2. 连接模拟器
    if not connect_mumu():
        print("❌ 错误: 无法连接模拟器")
        print("解决方案:")
        print(f"1. 确认MuMu正在运行 (端口 {MUMU_ADB_PORT})")
        print("2. 设置->其他->开启ADB")
        print("3. 尝试重启MuMu模拟器")
        sys.exit(1)
    
    # 3. 调试可视化
    if DEBUG_MODE and CLICK_TASKS:
        debug_log("启动调试可视化...")
        screenshot = take_screenshot("debug_preview.png")
        if screenshot:
            visualize_tasks(screenshot)
    
    # 4. 执行点击任务
    print(f"\n▶ 开始执行 {len(CLICK_TASKS)} 个任务")
    success_count = 0
    
    for task_num, (region, wait_time) in enumerate(CLICK_TASKS, 1):
        print(f"\n[{task_num}/{len(CLICK_TASKS)}] 任务准备...")
        debug_log(f"区域: {region} | 等待: {wait_time}s")
        
        # 等待指定时间
        smart_wait(wait_time, reason=f"任务 {task_num} 前置等待")
        
        # 带重试的点击
        for attempt in range(1, MAX_RETRY+1):
            if click_region(region, attempt):
                success_count += 1
                break
            elif attempt < MAX_RETRY:
                smart_wait(1, reason=f"第{attempt}次重试前等待")
        else:
            print(f"⚠️ 任务 {task_num} 失败（已达最大重试次数）")
            continue
        
        # 任务间间隔（防检测）
        if task_num < len(CLICK_TASKS):
            smart_wait(MIN_INTERVAL, reason="任务间安全间隔")
    
    # 结果统计
    print("\n" + "="*40)
    print(f"✅ 成功: {success_count} | ❌ 失败: {len(CLICK_TASKS)-success_count}")
    print("="*40)
    
    # 最终截图确认
    if DEBUG_MODE:
        take_screenshot("debug_result.png")
        debug_log("最终结果截图已保存: debug_result.png")

if __name__ == "__main__":
    main()