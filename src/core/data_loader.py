# Loads monster data from monster.csv
import csv
import os
from typing import List, Dict, Any
from src.core.log import logger
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
MONSTER_CSV_PATH = os.path.join(DATA_DIR, 'monster.csv')

def load_monster_data() -> List[Dict[str, Any]]:
    """
    Loads monster data from the CSV file.

    Returns:
        A list of dictionaries, where each dictionary represents a monster
        and contains its attributes. Returns an empty list if the file
        is not found or is empty.
    """
    monsters = []
    if not os.path.exists(MONSTER_CSV_PATH):
        logger.error(f"Error: Monster data file not found at {MONSTER_CSV_PATH}")
        return monsters

    try:
        with open(MONSTER_CSV_PATH, mode='r', encoding='utf-8-sig') as csvfile: # Use utf-8-sig to handle potential BOM
            reader = csv.DictReader(csvfile)
            # Basic validation: Check if essential columns exist
            # Use the actual Chinese column names from the CSV
            required_columns = {'ID', '名称'} # Check for essential columns
            if not required_columns.issubset(reader.fieldnames or []): # Handle case where fieldnames might be None
                logger.error(f"错误：CSV 文件缺少必需的列。需要：{required_columns}，找到：{reader.fieldnames}")
                return monsters

            for row in reader:
                # Ensure ID exists and is not empty, as it's crucial for linking to images
                if not row.get('ID'):
                    logger.warning(f"警告：跳过缺少或为空的 'ID' 的行：{row}")
                    continue
                # Type conversion/validation will be handled in the Monster model or UI layer
                # e.g., converting health, attack to integers
                monsters.append(dict(row))
    except Exception as e:
        logger.error(f"Error reading or parsing CSV file {MONSTER_CSV_PATH}: {e}")
        return [] # Return empty list on error

    return monsters

if __name__ == '__main__':
    # Example usage when running this module directly
    # Create a dummy data directory and CSV for testing
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    # Update dummy data to reflect the actual CSV structure (using Chinese keys)
    # Note: Using Chinese keys directly in code might cause issues depending on environment/encoding.
    # It's often safer to map them to English variable names after loading.
    # However, for this dummy creation, we'll match the expected input format.
    dummy_data = [
        {'ID': '999', '名称': '测试虫', '生命值': '100', '攻击力': '10', '防御力': '5', '法术抗性': '0', '攻击间隔': '2', '移动速度': '1', '攻击范围半径': '', '法伤': '0', '特殊能力': ''},
        {'ID': '998', '名称': '测试狗', '生命值': '150', '攻击力': '20', '防御力': '0', '法术抗性': '0', '攻击间隔': '1.5', '移动速度': '1.5', '攻击范围半径': '', '法伤': '0', '特殊能力': '跑得快'}
    ]
    try:
        # Use utf-8-sig for writing to ensure BOM for Excel compatibility if needed
        with open(MONSTER_CSV_PATH, mode='w', newline='', encoding='utf-8-sig') as f:
            # Use the keys from the first dummy data entry as fieldnames
            if dummy_data:
                 fieldnames = dummy_data[0].keys()
            else:
                 # Define default fieldnames if dummy_data is empty
                 fieldnames = ['ID', '名称', '生命值', '攻击力', '防御力', '法术抗性', '攻击间隔', '移动速度', '攻击范围半径', '法伤', '特殊能力']

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(dummy_data)
        logger.info(f"Created dummy {MONSTER_CSV_PATH} for testing.")
    except Exception as e:
        logger.error(f"Could not create dummy CSV: {e}")

    loaded_monsters = load_monster_data()
    if loaded_monsters:
        logger.info("\nSuccessfully loaded monster data:")
        for monster in loaded_monsters:
            logger.info(monster)
    else:
        logger.error("\nFailed to load monster data or file is empty/invalid.")

    # Clean up dummy file
    # try:
    #     os.remove(MONSTER_CSV_PATH)
    #     print(f"\nRemoved dummy {MONSTER_CSV_PATH}.")
    # except OSError as e:
    #     print(f"Error removing dummy file: {e}")