import sys

import cv2
import numpy as np
import os
import traceback
from typing import List, Tuple, Dict, Optional, Literal, Union
import re # 用于解析 OCR 结果
from tqdm import tqdm # 导入 tqdm 用于进度条
from src.core.log import logger # 导入日志记录器
try:
    import ddddocr
except ImportError:
    logger.error(f"错误: 未找到 ddddocr 库。请运行 'pip install ddddocr' {traceback.format_exc()}")
    ddddocr = None # 设置为 None 以便后续检查

# Define the path relative to this file's location
# Goes up two levels (core -> src -> project root) then into data/image
TEMPLATE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'image'))

# Initialize ORB detector
# nfeatures: The maximum number of features to retain.
# scaleFactor: Pyramid decimation ratio, greater than 1.
# nlevels: The number of pyramid levels.
# edgeThreshold: This is size of the border where the features are not detected.
# firstLevel: The level of pyramid to put source image to.
# WTA_K: The number of points that produce each element of the oriented BRIEF descriptor.
# scoreType: The default HARRIS_SCORE means that Harris algorithm is used to rank features.
# patchSize: Size of the patch used by the oriented BRIEF descriptor.
# Initialize SIFT detector (more robust, slower than ORB)
# Note: May require opencv-contrib-python
try:
    sift = cv2.SIFT_create()
    logger.debug("SIFT detector initialized successfully.")
except AttributeError:
    logger.error("错误: cv2.SIFT_create() 不可用。请尝试安装 'opencv-contrib-python'")
    logger.error("pip install opencv-contrib-python")
    logger.warning("将回退到 ORB。")
    sift = None # Fallback or indicate error
    orb = cv2.ORB_create(nfeatures=1500, scoreType=cv2.ORB_FAST_SCORE) # Keep ORB as fallback if SIFT fails

# 初始化 ddddocr (如果库已导入)
ocr_instance = None
if ddddocr:
    try:
        ocr_instance = ddddocr.DdddOcr(show_ad=False)
        ocr_instance.set_ranges("0123456789x")
        logger.info("ddddocr OCR 实例初始化成功。")
    except Exception as e:
        logger.error(f"错误: 初始化 ddddocr 实例失败: {e}")
        ocr_instance = None # 确保实例为 None
else:
    logger.error("警告: ddddocr 库不可用，数字识别功能将无法使用。")

# Define type hint for loaded template data
# Tuple: (gray_image, keypoints, descriptors, average_bgr_color)
TemplateData = Dict[str, Tuple[np.ndarray, List[cv2.KeyPoint], np.ndarray, np.ndarray]]

def load_templates(template_dir: str = TEMPLATE_DIR) -> Optional[TemplateData]:
    """
    Loads all PNG template images, detects ORB keypoints and descriptors.

    Args:
        template_dir: The directory containing the template images.

    Returns:
        A dictionary where keys are template names (filename without extension)
        and values are tuples: (template_image_gray, keypoints, descriptors).
        Returns None if the directory doesn't exist or no valid templates are processed.
    """
    if not os.path.isdir(template_dir):
        logger.error(f"错误: 模板目录未找到于 {template_dir}")
        return None

    templates_data: TemplateData = {}
    logger.debug(f"正在从以下路径加载模板并计算 ORB 特征: {template_dir}")

    # 定义并创建用于存储灰度模板的目录 (位于项目根目录)
    template_gray_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'template_gray'))
    os.makedirs(template_gray_dir, exist_ok=True)
    logger.debug(f"灰度模板将保存到: {template_gray_dir}")

    processed_templates = 0
    skipped_templates = 0

    # 获取所有 PNG 文件列表以供 tqdm 使用
    png_files = [f for f in os.listdir(template_dir) if f.lower().endswith(".png")]
    if not png_files:
        logger.warning(f"警告: 在 {template_dir} 中未找到任何 PNG 模板文件。")
        return None

    # 使用 tqdm 包装文件列表迭代
    for filename in tqdm(png_files, desc="加载并处理模板", unit="个"):
        # 不再需要检查 endswith(".png")，因为 png_files 已经过滤过了
        template_name = os.path.splitext(filename)[0]
        filepath = os.path.join(template_dir, filename) # 修正缩进
        try:
            # Load in grayscale and color
                template_gray = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                template_color = cv2.imread(filepath, cv2.IMREAD_COLOR) # Load color version

                # 保存灰度模板图像
                if template_gray is not None:
                    gray_save_path = os.path.join(template_gray_dir, filename) # 使用原始文件名
                    try:
                        cv2.imwrite(gray_save_path, template_gray)
                        # print(f"  - 灰度模板已保存: {gray_save_path}") # 可选：取消注释以获取详细日志
                    except Exception as save_e:
                        logger.warning(f"警告: 无法保存灰度模板 {gray_save_path}: {save_e}")
                # 结束保存灰度模板

                if template_gray is None or template_color is None: # Check both
                    logger.warning(f"警告: 无法加载模板图片 (灰度或彩色): {filepath}")
                    skipped_templates += 1
                    continue

                # Detect SIFT keypoints and compute descriptors
                if sift:
                    kp, des = sift.detectAndCompute(template_gray, None)
                    detector_name = "SIFT"
                elif 'orb' in globals(): # Fallback to ORB if SIFT failed
                    kp, des = orb.detectAndCompute(template_gray, None)
                    detector_name = "ORB (Fallback)"
                else:
                    logger.error(f"错误: SIFT 和 ORB 检测器均未初始化。无法处理模板 '{template_name}'。")
                    skipped_templates += 1
                    continue

                if des is None or len(kp) < 5: # Require at least a few keypoints
                    logger.debug(f"警告: 在模板 '{template_name}' 中找到的 {detector_name} 特征点过少 ({len(kp) if kp is not None else 0})，已跳过。")
                    skipped_templates += 1
                    continue

                # Calculate average color of the color template
                # cv2.mean returns (B, G, R, Alpha) if Alpha exists, we only need BGR
                avg_color = np.array(cv2.mean(template_color)[:3])

                templates_data[template_name] = (template_gray, kp, des, avg_color)
                #print(f"  - 已处理模板: {filename} (名称: '{template_name}', 使用 {detector_name} 找到 {len(kp)} 个特征点, 平均BGR: {avg_color.round(1)})")
                processed_templates += 1
        except Exception as e: # 修正缩进，与 try 对齐
            logger.error(f"错误: 处理模板 {filename} 时出错: {e}")
            skipped_templates += 1

    if processed_templates == 0:
        logger.warning(f"警告: 在 {template_dir} 中未找到或处理任何有效的 PNG 模板图片。")
        return None

    logger.info(f"模板加载和特征提取完成。成功处理 {processed_templates} 个模板，跳过 {skipped_templates} 个。")
    return templates_data


# --- Feature-based Monster Recognition (SIFT or ORB Fallback) ---
def recognize_monsters(
    screenshot_crop: np.ndarray,
    templates_data: TemplateData,
    min_good_matches: int = 9, # Minimum number of good matches to consider it a detection
    lowe_ratio: float = 0.7,    # Lowe's ratio test threshold
    color_distance_threshold: float = 50.0 # Max allowed Euclidean distance between avg colors
) -> Tuple[np.ndarray, Dict[Literal['left', 'right'], Dict[str, Optional[int]]]]:
    """
    使用特征匹配、颜色校验和 OCR 识别截图中的怪物及其数量，并在图像上标记识别结果。

    Args:
        screenshot_crop: 裁剪后的截图图像 (BGR 或灰度图)。
        templates_data: 包含已加载模板数据的字典 (图像, 关键点, 描述符)。
        min_good_matches: 视为有效检测所需的最小优质特征匹配数。
        lowe_ratio: Lowe's ratio test 的比率阈值。
        color_distance_threshold: 平均颜色允许的最大欧氏距离，用于颜色校验。

    Returns:
        一个元组:
        - np.ndarray: 绘制了边界框和标签的图像副本 (BGR)。
        - Dict: 按左右 ('left'/'right') 分类检测到的模板名称，
                值为识别到的数量 (int) 或 None (如果 OCR 失败)。
    """
    results: Dict[Literal['left', 'right'], Dict[str, Optional[int]]] = {'left': {}, 'right': {}}

    # --- Pre-check and Image Preparation ---
    if screenshot_crop is None:
        logger.error("错误: 输入截图为空。")
        # 返回一个空的 BGR 图像和空结果
        return np.zeros((100, 100, 3), dtype=np.uint8), results
    if not templates_data:
        logger.error("错误: 模板数据未加载。")
        # 返回原始图像的 BGR 副本和空结果
        if len(screenshot_crop.shape) == 2:
            output_image = cv2.cvtColor(screenshot_crop, cv2.COLOR_GRAY2BGR)
        else:
            output_image = screenshot_crop.copy()
        return output_image, results # Return potentially modified/copied image

    # Ensure we have a BGR image for drawing and a grayscale for processing
    if len(screenshot_crop.shape) == 3:
        output_image = screenshot_crop.copy() # BGR copy for drawing
        gray_crop = cv2.cvtColor(screenshot_crop, cv2.COLOR_BGR2GRAY)
    elif len(screenshot_crop.shape) == 2:
        output_image = cv2.cvtColor(screenshot_crop, cv2.COLOR_GRAY2BGR) # Convert to BGR for drawing
        gray_crop = screenshot_crop # Already grayscale
    else:
        logger.error(f"错误: 输入截图的通道数无效 ({len(screenshot_crop.shape)})。")
        # 返回一个空的 BGR 图像和空结果
        return np.zeros((100, 100, 3), dtype=np.uint8), results

    if ocr_instance is None:
        logger.error("警告: ddddocr 未初始化，无法进行数量识别。仅进行模板匹配。")

    # --- ORB Feature Detection on Screenshot ---
    detector_name = "SIFT" if sift else "ORB (Fallback)"
    try:
        if sift:
            kp_crop, des_crop = sift.detectAndCompute(gray_crop, None)
        elif 'orb' in globals():
             kp_crop, des_crop = orb.detectAndCompute(gray_crop, None)
        else:
            logger.error(f"错误: SIFT 和 ORB 检测器均未初始化。无法处理截图。")
            return output_image, results

        if des_crop is None or len(kp_crop) == 0:
            logger.debug(f"调试 ({detector_name}): 在截图区域未检测到特征点。")
            return output_image, results # Return drawn image (no changes yet) and results
        logger.debug(f"调试 ({detector_name}): 在截图区域检测到 {len(kp_crop)} 个特征点。")
    except Exception as e:
        logger.error(f"错误: 在截图区域检测 {detector_name} 特征点时出错: {e}")
        return output_image, results # Return drawn image and results

    # Create BFMatcher object
    # Use NORM_L2 for SIFT (float descriptors), NORM_HAMMING for ORB (binary descriptors)
    if sift and des_crop is not None and des_crop.dtype == np.float32:
        matcher_norm = cv2.NORM_L2
        logger.debug("调试: 使用 NORM_L2 (SIFT) 进行匹配。")
    elif 'orb' in globals() and des_crop is not None and des_crop.dtype == np.uint8:
        matcher_norm = cv2.NORM_HAMMING
        logger.debug("调试: 使用 NORM_HAMMING (ORB Fallback) 进行匹配。")
    else:
         # Default or guess if detector/descriptor type is unclear, might lead to errors
         logger.debug("警告: 无法确定描述符类型，默认使用 NORM_L2。如果使用 ORB，这可能导致错误。")
         matcher_norm = cv2.NORM_L2 # Default guess

    bf = cv2.BFMatcher(matcher_norm, crossCheck=False) # crossCheck=False for ratio test

    # Get crop dimensions for side calculation
    h_crop, w_crop = gray_crop.shape[:2]
    midpoint_x_crop = w_crop // 2

    # Iterate through each template
    for template_name, (template_img_gray, kp_template, des_template, template_avg_color) in templates_data.items(): # Unpack avg color
        if des_template is None: continue

        try:
            if des_template.size == 0 or des_crop.size == 0: continue
            if des_template.dtype != des_crop.dtype:
                 # print(f"警告: 模板 '{template_name}' ({des_template.dtype}) 与截图 ({des_crop.dtype}) 的描述符类型不匹配。跳过。")
                 continue # ORB 应该是 CV_8U

            matches = bf.knnMatch(des_template, des_crop, k=2)
            # print(f"  调试 ({detector_name}): 模板 '{template_name}' - 初始匹配数: {len(matches)}")

        except cv2.error as e:
             logger.error(f"警告: 匹配模板 '{template_name}' 时出错 (knnMatch): {e}")
             continue
        except Exception as e:
             logger.error(f"错误: 匹配模板 '{template_name}' 时发生意外错误: {e}")
             continue

        # --- Ratio Test ---
        good_matches = []
        if len(matches) > 0 and len(matches[0]) == 2:
            for m, n in matches:
                if m.distance < lowe_ratio * n.distance:
                    good_matches.append(m)

        logger.debug(f"  调试 ({detector_name}): 模板 '{template_name}' - Ratio Test后优质匹配数: {len(good_matches)} (需要 >= {min_good_matches})")

        # --- 检测和 OCR ---
        if len(good_matches) >= min_good_matches:
            # 1. 计算 Homography 和边界框
            src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_crop[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            try:
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is None:
                    logger.debug(f"  调试 (Homography): 模板 '{template_name}' 无法计算 Homography。")
                    continue
            except cv2.error as e:
                logger.debug(f"  调试 (Homography): 模板 '{template_name}' 计算 Homography 时出错: {e}")
                continue

            h_template, w_template = template_img_gray.shape
            pts_template = np.float32([[0, 0], [0, h_template - 1], [w_template - 1, h_template - 1], [w_template - 1, 0]]).reshape(-1, 1, 2)

            try:
                pts_crop = cv2.perspectiveTransform(pts_template, M)
                # pts_crop 现在包含模板在截图中的四个角点坐标 [[x1, y1]], [[x2, y2]], ...
                # 计算边界框 (bounding box)
                x_coords = pts_crop[:, 0, 0]
                y_coords = pts_crop[:, 0, 1]
                x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
                y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
                center_x = (x_min + x_max) // 2

            except cv2.error as e:
                 logger.debug(f"  调试 (Transform): 模板 '{template_name}' 变换角点时出错: {e}")
                 continue

            # --- 颜色校验 ---
            # 提取截图上匹配区域的颜色信息
            # 确保边界框坐标在截图范围内
            roi_x_min = max(0, x_min)
            roi_y_min = max(0, y_min)
            roi_x_max = min(w_crop, x_max) # 使用截图宽度
            roi_y_max = min(h_crop, y_max) # 使用截图高度

            if roi_x_max > roi_x_min and roi_y_max > roi_y_min: # 确保 ROI 尺寸有效
                matched_roi_color = output_image[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
                if matched_roi_color.size > 0:
                    # 计算匹配区域的平均颜色
                    roi_avg_color = np.array(cv2.mean(matched_roi_color)[:3])
                    # 计算与模板平均颜色的距离
                    color_distance = np.linalg.norm(roi_avg_color - template_avg_color)
                    logger.debug(f"  调试 (颜色): 模板 '{template_name}' - ROI颜色: {roi_avg_color.round(1)}, 模板颜色: {template_avg_color.round(1)}, 距离: {color_distance:.2f} (阈值: {color_distance_threshold})")

                    # 如果颜色距离过大，则跳过此匹配
                    if color_distance > color_distance_threshold:
                        logger.debug(f"  警告 (颜色): 模板 '{template_name}' 颜色校验失败 (距离 {color_distance:.2f} > {color_distance_threshold})。跳过此匹配。")
                        continue # 跳到下一个模板
                else:
                    logger.debug(f"  警告 (颜色): 模板 '{template_name}' 提取的 ROI 区域为空。跳过颜色校验。")
            else:
                 logger.debug(f"  警告 (颜色): 模板 '{template_name}' 计算出的 ROI 区域无效 ({roi_x_min},{roi_y_min})-({roi_x_max},{roi_y_max})。跳过颜色校验。")
            # --- 颜色校验结束 ---


            # --- Draw Bounding Box and Label on Output Image ---
            # Draw the bounding box (red color, thickness 2)
            cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            # Prepare label text (template name)
            label = template_name
            # Put the label above the bounding box
            label_pos = (x_min, y_min - 10) # Adjust position as needed
            # Ensure label position is within image bounds (top)
            if label_pos[1] < 10: label_pos = (x_min, y_min + 15) # Move below if too close to top
            cv2.putText(output_image, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 1, cv2.LINE_AA) # Red text

            # 2. 判断左右侧
            side: Literal['left', 'right'] = 'left' if center_x < midpoint_x_crop else 'right'
            logger.debug(f"  === 检测到模板: '{template_name}' === ({len(good_matches)} 个优质匹配) -> 中心X: {center_x:.1f} -> {side}侧 [已在图像上标记]")

            # 3. 计算数量 ROI 并执行 OCR (如果 ocr_instance 可用)
            recognized_count: Optional[int] = None # 默认为 None
            if ocr_instance:
                # --- 使用模板边界框作为 OCR 的 ROI ---
                # 确保边界框坐标在截图范围内
                ocr_x_min = max(0, x_min)
                ocr_y_min = max(0, y_min)
                ocr_x_max = min(w_crop, x_max)
                ocr_y_max = min(h_crop, y_max)
                if side == 'left':
                    ocr_y_min = int(ocr_y_min + (ocr_y_max - ocr_y_min) * 0.8)
                    ocr_x_min = int(ocr_x_min + (ocr_x_max - ocr_x_min) * 0.5 )
                else: # right side
                    ocr_y_min = int(ocr_y_min + (ocr_y_max - ocr_y_min) * 0.8)
                    ocr_x_max = int(ocr_x_max - (ocr_x_max - ocr_x_min) * 0.5 )
                logger.debug(f"    > OCR ROI 坐标: ({ocr_x_min}, {ocr_y_min}) - ({ocr_x_max}, {ocr_y_max})")
                if ocr_x_max > ocr_x_min and ocr_y_max > ocr_y_min: # 确保 ROI 尺寸有效
                    # 直接从灰度图中提取边界框内的区域 (即模板本身)
                    roi_img = gray_crop[ocr_y_min:ocr_y_max, ocr_x_min:ocr_x_max]

                    

                    # --- (可选) ROI 预处理 ---
                    # 应用二值化预处理来增强数字对比度
                    _, roi_img_thresh = cv2.threshold(roi_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    roi_to_ocr = roi_img_thresh # 使用二值化后的图像进行 OCR
                    # roi_to_ocr = roi_img # 如果二值化效果不好，可以切换回原始 ROI
                    # --- 调试: 保存提取的 ROI (现在是模板区域) ---
                    # debug_roi_filename = f"debug_roi_bbox_{template_name}_{side}.png" # Changed filename slightly
                    # try:
                    #     cv2.imwrite(debug_roi_filename, roi_img_thresh)
                    #     logger.debug(f"    > 调试: BBox ROI 图像已保存至 {debug_roi_filename}")
                    # except Exception as imwrite_e:
                    #     logger.debug(f"    > 调试: 保存 BBox ROI 图像时出错: {imwrite_e}")
                    # --- End Debug ---
                    try:
                        # 将 NumPy 图像转为 bytes
                        img_bytes = cv2.imencode('.png', roi_to_ocr)[1].tobytes()
                        # 调用 ddddocr
                        ocr_result_str = ocr_instance.classification(img_bytes)
                        logger.debug(f"    > OCR ({side}侧 '{template_name}' 模板区域): 识别结果 '{ocr_result_str}'") # Updated log message

                        # 解析数字 (假设格式为 "xN" 或 "N", 或者模板内直接包含数字)
                        match = re.search(r'\d+', ocr_result_str)
                        if match:
                            recognized_count = int(match.group(0))
                            logger.debug(f"    > OCR 解析数量: {recognized_count}")
                        else:
                            # 尝试直接转换整个结果，看是否是纯数字
                            try:
                                recognized_count = int(ocr_result_str)
                                logger.debug(f"    > OCR 直接转换数量: {recognized_count}")
                            except ValueError:
                                logger.debug(f"    > OCR 警告: 未能从 '{ocr_result_str}' 中解析或转换出数字 (模板区域)。")
                                recognized_count = None # 确保是 None 如果无法解析

                    except Exception as ocr_e:
                        logger.warning(f"    > OCR 错误: 识别模板 '{template_name}' 区域时出错: {ocr_e}")
                        # 可以选择在这里保存失败的 ROI 图像用于调试
                        # cv2.imwrite(f"debug_ocr_fail_bbox_{template_name}_{side}.png", roi_to_ocr)

                else:
                    logger.debug(f"    > OCR 警告: 模板 '{template_name}' 的边界框无效或超出边界，无法进行 OCR。")

            # 4. 存储结果 (模板名称和识别到的数量)
            # 即使 OCR 失败 (recognized_count is None)，也记录模板被检测到
            if template_name in results[side]:
                 # 如果模板已存在（例如同一侧有多个相同模板），如何处理？
                 # 选项1：覆盖（只保留最后一个检测到的数量）
                 # 选项2：累加（如果需要总数，但这似乎不符合需求）
                 # 选项3：存储列表（允许多个实例及其数量）
                 # 当前实现：覆盖，并将之前的 None 替换为识别到的数字 (如果成功)
                 if recognized_count is not None or results[side][template_name] is None:
                      results[side][template_name] = recognized_count
                 logger.debug(f"    > 更新 '{template_name}' ({side}) 数量为: {results[side][template_name]}")
            else:
                 results[side][template_name] = recognized_count
                 logger.debug(f"    > 添加 '{template_name}' ({side}) 数量为: {recognized_count}")

    logger.info(f"({detector_name}+OCR): 识别完成。最终结果: {results}")
    return output_image, results


if __name__ == '__main__':
    detector_name_for_test = "SIFT" if 'sift' in globals() and sift is not None else "ORB" # Define for test block
    print(f"测试 {detector_name_for_test} 图像识别模块...") # Use test-specific name

    # 1. Load templates and compute features
    loaded_templates_data = load_templates()

    if loaded_templates_data:
        print(f"\n成功加载并处理了 {len(loaded_templates_data)} 个模板的特征。")

        # 2. Create/Load a dummy screenshot for testing
        # Find a template to use
        if not loaded_templates_data:
             print("没有加载的模板可用于测试。")
             sys.exit()

        test_template_name = next(iter(loaded_templates_data))
        test_template_img, _, _ = loaded_templates_data[test_template_name]
        if test_template_img is None:
            print(f"无法获取模板图像 '{test_template_name}' 进行测试。")
            sys.exit()

        th, tw = test_template_img.shape[:2]

        # Create a slightly larger black background
        screenshot_height = th + 100
        screenshot_width = tw + 150 # Make wider for side test
        dummy_screenshot_bgr = np.zeros((screenshot_height, screenshot_width, 3), dtype=np.uint8) # BGR

        # Convert test template to BGR if it's grayscale (for pasting)
        if len(test_template_img.shape) == 2:
            test_template_bgr = cv2.cvtColor(test_template_img, cv2.COLOR_GRAY2BGR)
        else:
            test_template_bgr = test_template_img

        # Place the template on the left side
        paste_x_left, paste_y = 20, 30
        if paste_y + th <= screenshot_height and paste_x_left + tw <= screenshot_width:
            dummy_screenshot_bgr[paste_y:paste_y + th, paste_x_left:paste_x_left + tw] = test_template_bgr
            print(f"\n创建了一个虚拟截图 ({screenshot_width}x{screenshot_height})")
            print(f" - 在左侧 ({paste_x_left}, {paste_y}) 放置了模板 '{test_template_name}'")

            # Place another instance on the right side
            paste_x_right = tw + 40
            if paste_x_right + tw <= screenshot_width:
                 dummy_screenshot_bgr[paste_y:paste_y + th, paste_x_right:paste_x_right + tw] = test_template_bgr
                 print(f" - 在右侧 ({paste_x_right}, {paste_y}) 放置了模板 '{test_template_name}'")

                 # Save the dummy screenshot
                 dummy_screenshot_path = os.path.join(os.path.dirname(__file__), '..', '..', f'dummy_{detector_name_for_test.lower()}_test.png') # Use test-specific name
                 try:
                     cv2.imwrite(dummy_screenshot_path, dummy_screenshot_bgr)
                     print(f"虚拟截图已保存至: {dummy_screenshot_path}")
                 except Exception as e:
                     print(f"无法保存虚拟截图: {e}")

                 # 3. Run recognition on the dummy screenshot (full image as crop for test)
                 print(f"\n运行 {detector_name_for_test} 识别...")
                 # --- 准备带数字的虚拟截图 ---
                 # 在模板下方添加数字文本
                 font = cv2.FONT_HERSHEY_SIMPLEX
                 font_scale = 0.6
                 font_color = (255, 255, 255) # White
                 thickness = 1
                 text_left = "x3"
                 text_right = "x12"

                 # 计算文本放置位置 (需要调整)
                 text_size_left, _ = cv2.getTextSize(text_left, font, font_scale, thickness)
                 text_size_right, _ = cv2.getTextSize(text_right, font, font_scale, thickness)

                 # 左侧模板的右下方
                 text_x_left = paste_x_left + tw + 5 # 模板右侧加偏移
                 text_y_left = paste_y + th + text_size_left[1] + 5 # 模板下方加偏移

                 # 右侧模板的左下方
                 text_x_right = paste_x_right - text_size_right[0] - 5 # 模板左侧减去文本宽度再加偏移
                 text_y_right = paste_y + th + text_size_right[1] + 5 # 模板下方加偏移

                 # 确保文本在图像内
                 if text_y_left < screenshot_height and text_x_left + text_size_left[0] < screenshot_width:
                      cv2.putText(dummy_screenshot_bgr, text_left, (text_x_left, text_y_left), font, font_scale, font_color, thickness, cv2.LINE_AA)
                      print(f" - 在左侧模板下方添加了文本 '{text_left}'")
                 else:
                      print(f" - 警告: 无法添加左侧文本（空间不足或计算错误）")

                 if text_y_right < screenshot_height and text_x_right > 0:
                      cv2.putText(dummy_screenshot_bgr, text_right, (text_x_right, text_y_right), font, font_scale, font_color, thickness, cv2.LINE_AA)
                      print(f" - 在右侧模板下方添加了文本 '{text_right}'")
                 else:
                      print(f" - 警告: 无法添加右侧文本（空间不足或计算错误）")

                 # 保存带文本的虚拟截图 (覆盖之前的)
                 try:
                     cv2.imwrite(dummy_screenshot_path, dummy_screenshot_bgr)
                     print(f"带文本的虚拟截图已更新至: {dummy_screenshot_path}")
                 except Exception as e:
                     print(f"无法保存带文本的虚拟截图: {e}")


                 # 3. Run recognition on the dummy screenshot
                 print(f"\n运行 {detector_name_for_test} + OCR 识别...")
                 # --- Run Recognition ---
                 # The function now returns the image with boxes drawn and the results dict
                 output_img_with_boxes, detected_results = recognize_monsters(
                     dummy_screenshot_bgr, # Pass the BGR image with text
                     loaded_templates_data,
                     min_good_matches=8, # Lower for potentially simple test case
                     lowe_ratio=0.75
                 )

                 print("\n识别结果 (左右分类及数量):")
                 print(detected_results)

                 # --- Save or Display the Output Image with Boxes ---
                 output_image_path = os.path.join(os.path.dirname(__file__), '..', '..', f'dummy_{detector_name_for_test.lower()}_test_output.png') # Use test-specific name
                 try:
                     cv2.imwrite(output_image_path, output_img_with_boxes) # Save the image with boxes
                     print(f"\n带有边界框和标签的输出图像已保存至: {output_image_path}")
                     # You might want to display it if running interactively:
                     # cv2.imshow("Recognition Result", output_img_with_boxes)
                     # cv2.waitKey(0)
                     # cv2.destroyAllWindows()
                 except Exception as e:
                     print(f"无法保存带标记的输出图像: {e}")


                 # --- Simple Validation (using detected_results) ---
                 expected_left_count = 3
                 expected_right_count = 12
                 actual_left_count = detected_results.get('left', {}).get(test_template_name)
                 actual_right_count = detected_results.get('right', {}).get(test_template_name)

                 print("\n测试验证:")
                 if actual_left_count == expected_left_count:
                     print(f"  - 左侧数量 ({actual_left_count}) 匹配预期 ({expected_left_count}) - 成功")
                 else:
                     print(f"  - 左侧数量 ({actual_left_count}) 不匹配预期 ({expected_left_count}) - 失败 或 OCR 未运行/失败")

                 if actual_right_count == expected_right_count:
                     print(f"  - 右侧数量 ({actual_right_count}) 匹配预期 ({expected_right_count}) - 成功")
                 else:
                     print(f"  - 右侧数量 ({actual_right_count}) 不匹配预期 ({expected_right_count}) - 失败 或 OCR 未运行/失败")

            else:
                print("\n无法在右侧放置第二个模板（空间不足）。")
        else:
             print("\n无法在左侧放置模板（空间不足）。")

    else:
        print("\n无法加载模板。识别测试已跳过。")
        print(f"请确保模板图片存在于: {TEMPLATE_DIR}")