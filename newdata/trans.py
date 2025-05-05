import csv
import json

def load_trans_mapping(trans_file_path):
    """加载 trans.txt 文件并构建映射字典"""
    mapping = {}
    with open(trans_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                key, value = line.strip().split(',')
                mapping[f"obj_{key}"] = f"obj_{value}"
    return mapping

def csv_to_json_array_with_mapping(csv_file_path, output_json_path, trans_file_path):
    # 加载映射字典
    trans_mapping = load_trans_mapping(trans_file_path)
    
    # 初始化结果列表
    result = []
    
    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        # 读取CSV文件
        reader = csv.reader(csv_file, delimiter='\t')
        headers = next(reader)  # 读取表头
        headers = headers[-1].split(',')
        
        for row in reader:
            if not row:  # 跳过空行
                continue
                
            # 分割最后一列的数据
            last_column = row[-1].split(',')
            wl = last_column[-1]  # 胜负标记是最后一个元素
            values = last_column[:-1]  # 前面的所有元素是数值
            
            # 初始化JSON结构
            json_data = {
                "left": {},
                "right": {},
                "wl": 'left' if wl == 'L' else 'right'  # 胜负标记
            }
            
            # 处理数值数据
            for i, header in enumerate(headers[:-1]):  # 排除最后一列（现在是复合数据）
                if not header:  # 跳过空列名
                    continue
                    
                value = int(values[i]) if values[i] else 0
                
                # 根据列名判断是左还是右
                if header.endswith('L'):
                    obj_key = f"obj_{header[:-1]}"  # 去掉'L'并添加'obj_'
                    obj_key = trans_mapping.get(obj_key, obj_key)  # 转换为映射后的 obj_key
                    if value != 0:
                        json_data["left"][obj_key] = value
                    
                elif header.endswith('R'):
                    obj_key = f"obj_{header[:-1]}"  # 去掉'R'并添加'obj_'
                    obj_key = trans_mapping.get(obj_key, obj_key)  # 转换为映射后的 obj_key
                    if value != 0:
                        json_data["right"][obj_key] = value
                    
            # 将当前行的数据添加到结果列表
            result.append(json_data)
    
    # 写入JSON文件
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(result, json_file, indent=4, ensure_ascii=False)
    
    print(f"转换完成，结果已保存到: {output_json_path}")

# 使用示例
csv_file_path = r'arknights.csv'  # 替换为你的CSV文件路径
output_json_path = r'output.json'  # 输出JSON文件路径
trans_file_path = r'data\trans.txt'  # 替换为你的trans.txt文件路径
csv_to_json_array_with_mapping(csv_file_path, output_json_path, trans_file_path)