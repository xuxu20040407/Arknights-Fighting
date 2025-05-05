import subprocess
import time
import cv2
import numpy as np
from PIL import Image
import os
import sys
from src.core.image_recognition import load_templates, recognize_monsters
import json
from datetime import datetime

# 配置区域 ==============================================
ADB_PATH = r"D:\Program Files\Netease\MuMu Player 12\shell\.\adb.exe"
MUMU_ADB_PORT = "16384"

# 模板配置
TEMPLATE_A = "data\\win-lose\\Left.png"
TEMPLATE_B = "data\\win-lose\\Right.png"
TEMPLATE_THRESHOLD = 0.5  # 匹配阈值(0-1)
TEMPLATE_CHECK_INTERVAL = 0.8  # 检测间隔(秒)

# 点击任务配置
CLICK_TASKS = [
    ((1800, 1150, 2400, 1300), 2),
    ((1800, 1100, 2000, 1200), 2),
    ((2300, 1100, 2400, 1200), 2),
    ((1200, 870, 1350, 950), 13)  # 最后一个任务，将在检测到模板后重复执行
]
CLOSE_TASK = ((2300, 1200, 2400, 1400), 10)
MAX_RETRY = 2       # 失败重试次数
MAX_REPEAT = 10     # 最大重复次数
ITERATION=10

# 函数定义 ==============================================
def save_result_to_json(result, output_dir="./results"):
    """
    保存结果到带有时间戳的 JSON 文件
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取当前时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 构造文件名
    file_name = f"result_{timestamp}.json"
    file_path = os.path.join(output_dir, file_name)
    
    # 保存字典到 JSON 文件
    try:
        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(result, json_file, ensure_ascii=False, indent=4)
        print(f"结果已保存到文件: {file_path}")
    except Exception as e:
        print(f"保存结果到 JSON 文件时发生错误: {e}")

def take_screenshot_cv():
    """获取屏幕截图（返回OpenCV格式）"""
    try:
        img_data = subprocess.check_output(
            [ADB_PATH, "-s", f"127.0.0.1:{MUMU_ADB_PORT}", "exec-out", "screencap", "-p"],
            timeout=5
        )
        nparr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except:
        return None

def load_template(template_path):
    """加载模板图片"""
    try:
        template = cv2.imread(template_path, cv2.IMREAD_COLOR)
        if template is None:
            raise ValueError(f"无法加载模板: {template_path}")
        return template
    except:
        return None

def template_match(screenshot, template):
    """执行模板匹配（支持GPU加速）"""
    try:
        # 检查是否有可用的CUDA设备
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            # 将图像和模板上传到GPU
            gpu_screenshot = cv2.cuda_GpuMat()
            gpu_template = cv2.cuda_GpuMat()
            gpu_screenshot.upload(screenshot)
            gpu_template.upload(template)
            
            # 创建CUDA模板匹配器
            matcher = cv2.cuda.createTemplateMatching(
                cv2.CV_8UC3, cv2.TM_CCOEFF_NORMED
            )
            
            # 执行匹配
            gpu_result = matcher.match(gpu_screenshot, gpu_template)
            
            # 下载结果到CPU
            result = gpu_result.download()
            
            # 获取最大匹配值
            _, max_val, _, _ = cv2.minMaxLoc(result)
            return max_val
        else:
            # 没有CUDA设备时回退到CPU版本
            result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            return max_val
    except Exception as e:
        print(f"模板匹配错误: {e}")
        return 0

def click_region(region, attempt=1):
    """点击指定区域"""
    left, top, right, bottom = region
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    
    try:
        subprocess.run(
            [ADB_PATH, "-s", f"127.0.0.1:{MUMU_ADB_PORT}", "shell", "input", "tap", str(center_x), str(center_y)],
            check=True,
            timeout=5,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return True
    except:
        return False

def process_and_recognize_screenshot(screenshot_path="./capture/screenshot.png", recognized_output_path="./capture/recognized_screenshot.png"):
    """
    截图并识别内容，返回识别结果
    """
    # 截图
    screenshot = take_screenshot_cv()
    if screenshot is None:
        print("截图失败，无法进行识别")
        return None

    # 保存截图
    cv2.imwrite(screenshot_path, screenshot)

    # 加载模板
    templates_data = load_templates()
    if not templates_data:
        print("模板加载失败，无法进行识别")
        return None

    # 调用识别函数
    output_image, results = recognize_monsters(screenshot, templates_data)

    # 保存识别结果图像
    cv2.imwrite(recognized_output_path, output_image)
    print(f"识别完成，结果已保存到 {recognized_output_path}")
    print("识别结果:", results)

    return results

def monitor_templates(last_task_region,result):
    """监控模板出现并在检测到后执行最后一个任务"""
    template_a = load_template(TEMPLATE_A)
    template_b = load_template(TEMPLATE_B)
    
    if template_a is None and template_b is None:
        print("❌ 错误: 没有可用的模板")
        return False
    
    repeat_count = 0
    
    while repeat_count < MAX_REPEAT:
        loop_start = time.time()  # 记录循环开始时间
        
        screenshot = take_screenshot_cv()
        if screenshot is None:
            time.sleep(TEMPLATE_CHECK_INTERVAL)
            continue
        
        # 模板匹配开始时间
        match_start = time.time()
        matched = False
        
        if template_a is not None:
            score_a = template_match(screenshot, template_a)
            if score_a >= TEMPLATE_THRESHOLD:
                print(f"✅ 检测到模板A (匹配度: {score_a:.2f})")
                result['wl']='left'
                save_result_to_json(result)
                matched = True
        
        if not matched and template_b is not None:
            score_b = template_match(screenshot, template_b)
            if score_b >= TEMPLATE_THRESHOLD:
                print(f"✅ 检测到模板B (匹配度: {score_b:.2f})")
                result['wl']='right'
                save_result_to_json(result)
                matched = True
        
        match_time = time.time() - match_start  # 计算模板匹配耗时
        
        if matched:
            repeat_count += 1
            if repeat_count==10:
                last_task_region, wait_time =CLOSE_TASK
                time.sleep(wait_time)
                print("正在退出至主界面...")
                click_region(last_task_region)
                break
            print(f"🔄 重复执行最后一个任务 ({repeat_count}/{MAX_REPEAT})")
            
            last_task, wait_time = last_task_region
            
            # 计算剩余等待时间（扣除模板匹配时间）
            remaining_wait = max(0, wait_time - match_time)
            if remaining_wait > 0:
                time.sleep(remaining_wait)
            
            # 执行点击操作
            click_start = time.time()
            result=process_and_recognize_screenshot()

            click_region(last_task)
            click_time = time.time() - click_start
            
            # 计算剩余间隔时间（扣除整个操作耗时）
            operation_time = match_time + remaining_wait + click_time
        else:
            operation_time = match_time  # 仅模板匹配耗时
        
        # 计算并等待剩余间隔时间
        elapsed = time.time() - loop_start
        remaining_interval = max(0, TEMPLATE_CHECK_INTERVAL - elapsed)
        if remaining_interval > 0:
            time.sleep(remaining_interval)
    
    print(f"✅ 已完成所有 {MAX_REPEAT} 次重复任务")
    return

# 主程序 ==============================================
def main():
    # 执行点击任务
    for task_num, (region, wait_time) in enumerate(CLICK_TASKS[:-1], 1):
        print(f"[{task_num}/{len(CLICK_TASKS)}] 执行任务...")
        
        time.sleep(wait_time)
        
        for attempt in range(1, MAX_RETRY+1):
            if click_region(region, attempt):
                break
            elif attempt < MAX_RETRY:
                time.sleep(1)
    
    # 开始监控模板并重复最后一个任务
    if CLICK_TASKS:
        last_task = CLICK_TASKS[-1]
        last_task_region, wait_time = last_task
        print(f"[{len(CLICK_TASKS)}/{len(CLICK_TASKS)}] 执行任务...")
        time.sleep(wait_time)

        result=process_and_recognize_screenshot()
        
        click_region(last_task_region)

        print("\n" + "="*40)
        print("开始监控模板并重复最后一个任务")
        print("="*40)
        
        monitor_templates(last_task,result)
    

    

if __name__ == "__main__":
    # 连接模拟器
    try:
        subprocess.run([ADB_PATH, "connect", f"127.0.0.1:{MUMU_ADB_PORT}"], check=True)
    except:
        print("❌ 无法连接MuMu模拟器")
        sys.exit(1)
    
    print("="*40)
    print(" MuMu模拟器自动化收集斗蛐蛐数据脚本")
    print("="*40)
    for i in range(ITERATION):
        print(f"第{i+1}次开局")
        main()
        time.sleep(20)