import subprocess
import time
import cv2
import numpy as np
from PIL import Image
import io
import sys

# 配置区域 ==============================================
ADB_PATH = r"D:\Program Files\Netease\MuMu Player 12\shell\.\adb.exe"
MUMU_ADB_PORT = "16384"
DEBUG_MODE = True
CHECK_INTERVAL = 1.5  # 检测间隔(秒)

# 模板配置 (需替换为实际模板路径)
TEMPLATE_A = "data\win-lose\Left.png"
TEMPLATE_B = "data\win-lose\Right.png"
THRESHOLD = 0.7  # 匹配阈值(0-1)

# 函数定义 ==============================================
def debug_log(message):
    if DEBUG_MODE:
        print(f"[DEBUG] {time.strftime('%H:%M:%S')} {message}")

def take_screenshot():
    """获取屏幕截图（返回OpenCV格式）"""
    try:
        img_data = subprocess.check_output(
            [ADB_PATH, "-s", f"127.0.0.1:{MUMU_ADB_PORT}", "exec-out", "screencap", "-p"],
            timeout=5
        )
        nparr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        debug_log(f"截图失败: {str(e)}")
        return None

def load_template(template_path):
    """加载模板图片"""
    try:
        template = cv2.imread(template_path, cv2.IMREAD_COLOR)
        if template is None:
            raise ValueError(f"无法加载模板: {template_path}")
        return template
    except Exception as e:
        debug_log(f"模板加载错误: {str(e)}")
        return None

def template_match(screenshot, template):
    """执行模板匹配"""
    try:
        result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return max_val
    except Exception as e:
        debug_log(f"模板匹配错误: {str(e)}")
        return 0

def monitor_templates():
    """持续监控模板出现"""
    # 预加载模板
    template_a = load_template(TEMPLATE_A)
    template_b = load_template(TEMPLATE_B)
    
    if template_a is None and template_b is None:
        print("❌ 错误: 没有可用的模板")
        return False
    
    print(f"🔍 开始监控模板 (间隔 {CHECK_INTERVAL}秒)...")
    debug_log(f"匹配阈值: {THRESHOLD}")
    
    while True:
        start_time = time.time()
        
        # 1. 截取屏幕
        screenshot = take_screenshot()
        if screenshot is None:
            time.sleep(CHECK_INTERVAL)
            continue
        
        # 2. 执行模板匹配
        matched = False
        
        if template_a is not None:
            score_a = template_match(screenshot, template_a)
            if score_a >= THRESHOLD:
                print(f"✅ 检测到模板A (匹配度: {score_a:.2f})")
                handle_template_a()  # 执行A模板对应的操作
                matched = True
        
        if not matched and template_b is not None:
            score_b = template_match(screenshot, template_b)
            if score_b >= THRESHOLD:
                print(f"✅ 检测到模板B (匹配度: {score_b:.2f})")
                handle_template_b()  # 执行B模板对应的操作
                matched = True
        
        # 3. 调试信息
        if DEBUG_MODE and not matched:
            debug_log("未检测到目标模板")
            cv2.imwrite("last_check.png", screenshot)
            debug_log("当前画面已保存为: last_check.png")
        
        # 4. 控制检测间隔
        elapsed = time.time() - start_time
        if elapsed < CHECK_INTERVAL:
            time.sleep(CHECK_INTERVAL - elapsed)

def handle_template_a():
    """模板A出现时的处理逻辑"""
    print("🛠️ 执行模板A对应操作...")
    # 示例：点击特定区域
    # click_region((x1,y1,x2,y2))
    # 或执行其他自定义操作

def handle_template_b():
    """模板B出现时的处理逻辑"""
    print("🛠️ 执行模板B对应操作...")
    # 示例：点击特定区域
    # click_region((x1,y1,x2,y2))
    # 或执行其他自定义操作

# 主程序 ==============================================
if __name__ == "__main__":
    # 检查ADB连接
    try:
        subprocess.run([ADB_PATH, "connect", f"127.0.0.1:{MUMU_ADB_PORT}"], check=True)
    except subprocess.CalledProcessError:
        print("❌ 无法连接MuMu模拟器")
        sys.exit(1)
    
    print("="*40)
    print(" MuMu模拟器模板监控系统")
    print(f" 监控模板: {TEMPLATE_A} 和 {TEMPLATE_B}")
    print("="*40)
    
    try:
        monitor_templates()
    except KeyboardInterrupt:
        print("\n🛑 用户中断监控")
    except Exception as e:
        print(f"❌ 监控异常: {str(e)}")
    finally:
        print("监控已停止")

