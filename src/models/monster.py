from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Monster:
    """
    Represents a monster with its attributes.
    Uses Optional for attributes that might be missing in the CSV.
    Provides default values where appropriate.
    """
    # Field names matching CSV columns (using English for robustness in code)
    id: str # From 'ID' column
    name: str # From '名称' column
    template_name: str = field(init=False) # Generated: obj_{id}

    # Attributes loaded from CSV, potentially optional or needing type conversion
    health: Optional[int] = None        # From '生命值'
    attack: Optional[int] = None        # From '攻击力'
    defense: Optional[int] = None       # From '防御力'
    resistance: Optional[float] = None  # From '法术抗性'
    attack_interval: Optional[float] = None # From '攻击间隔'
    move_speed: Optional[float] = None      # From '移动速度'
    attack_range: Optional[float] = None    # From '攻击范围半径'
    is_magic_damage: Optional[bool] = None  # From '法伤' (0 or 1)
    special_ability: Optional[str] = None   # From '特殊能力'


    def __post_init__(self):
        """Calculate template_name based on id after initialization."""
        # Ensure id is a string before formatting
        self.template_name = f"obj_{str(self.id)}" if self.id else "obj_unknown"

    @classmethod
    def from_dict(cls, data: dict):
        """
        Factory method to create a Monster instance from a dictionary
        (e.g., a row read from CSV by csv.DictReader using Chinese keys).
        Handles potential missing keys and basic type conversions.
        Maps Chinese keys to English attribute names.
        """
        # Helper function for safe type conversion
        def safe_int(value):
            try:
                # Handle empty strings as None before conversion
                return int(value) if value is not None and value != '' else None
            except (ValueError, TypeError):
                return None

        def safe_float(value):
            try:
                # Handle empty strings as None before conversion
                return float(value) if value is not None and value != '' else None
            except (ValueError, TypeError):
                return None

        def safe_bool_from_int(value):
            # Assumes '1' means True (magic damage), '0' or other means False
            try:
                return int(value) == 1 if value is not None and value != '' else None
            except (ValueError, TypeError):
                return None # Or False as default? Depends on requirements

        # Map Chinese keys from CSV dict to English dataclass fields
        return cls(
            id=data.get('ID', 'unknown_id'), # Use Chinese key 'ID'
            name=data.get('名称', '未知名称'), # Use Chinese key '名称'
            health=safe_int(data.get('生命值')),
            attack=safe_int(data.get('攻击力')),
            defense=safe_int(data.get('防御力')),
            resistance=safe_float(data.get('法术抗性')),
            attack_interval=safe_float(data.get('攻击间隔')),
            move_speed=safe_float(data.get('移动速度')),
            attack_range=safe_float(data.get('攻击范围半径')),
            is_magic_damage=safe_bool_from_int(data.get('法伤')),
            special_ability=data.get('特殊能力', '') # Keep as string, handle empty
        )

# Example Usage (can be run directly for testing)
if __name__ == "__main__":
    # Example dictionary simulating a row from CSV using Chinese keys
    csv_row_1 = {
        'ID': '36', '名称': '阿咬', '生命值': '1500', '攻击力': '435', '防御力': '120',
        '法术抗性': '10', '攻击间隔': '2', '移动速度': '1.1', '攻击范围半径': '', '法伤': '0', '特殊能力': ''
    }
    csv_row_2 = {
        'ID': '29', '名称': '宿主流浪者', '生命值': '5500', '攻击力': '975', '防御力': '200',
        '法术抗性': '30', '攻击间隔': '3', '移动速度': '0.65', '攻击范围半径': '', '法伤': '0', '特殊能力': '250/s生命恢复'
    }
    csv_row_3 = { # Missing values example
        'ID': '999', '名称': '测试怪', '生命值': '', '攻击力': None, '防御力': 'abc',
        '法术抗性': '10', '攻击间隔': '', '移动速度': '', '攻击范围半径': '', '法伤': '', '特殊能力': None
    }


    monster1 = Monster.from_dict(csv_row_1)
    monster2 = Monster.from_dict(csv_row_2)
    monster3 = Monster.from_dict(csv_row_3)


    print("怪物 1:", monster1)
    print("怪物 2:", monster2)
    print("怪物 3:", monster3)

    print(f"\n怪物 1 模板名称: {monster1.template_name}") # Should be obj_36
    print(f"怪物 1 生命值: {monster1.health}")
    print(f"怪物 2 特殊能力: {monster2.special_ability}")
    print(f"怪物 3 防御力: {monster3.defense}") # Should be None
    print(f"怪物 3 生命值: {monster3.health}") # Should be None
    print(f"怪物 3 模板名称: {monster3.template_name}") # Should be obj_999