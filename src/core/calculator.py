import math
from typing import Tuple, Union, Optional, TYPE_CHECKING
from src.core.log import logger
# Use TYPE_CHECKING to avoid circular import issues if Monster needs Calculator later
if TYPE_CHECKING:
    from src.models.monster import Monster

def calculate_damage(attacker: 'Monster', defender: 'Monster') -> Tuple[Union[float, str], float, str, float]:
    """
    Calculates DPH, DPS, and DPH percentage of defender's health.

    Args:
        attacker: The attacking Monster object.
        defender: The defending Monster object.

    Returns:
        A tuple containing:
        - dph (Union[float, str]): Damage per hit. "不破防" or float.
        - dps (float): Damage per second.
        - damage_type (str): "物理" or "法术".
        - dph_percent (float): DPH as a percentage of defender's health (0-100).
    """
    dph: Union[float, str] = 0.0
    dps: float = 0.0
    damage_type: str = "物理" # Default to physical

    # --- Input Validation and Conversion ---
    try:
        attack = float(attacker.attack) if attacker.attack is not None and str(attacker.attack).strip() != '' else 0.0
    except (ValueError, TypeError):
        attack = 0.0
        logger.warninga(f"警告: 攻击者 {attacker.name} 攻击值无效 ({attacker.attack})")

    try:
        defense = float(defender.defense) if defender.defense is not None and str(defender.defense).strip() != '' else 0.0
    except (ValueError, TypeError):
        defense = 0.0
        logger.warning(f"警告: 防御者 {defender.name} 防御值无效 ({defender.defense})")

    try:
        # Resistance is often 0-100, convert to 0.0-1.0
        resistance = float(defender.resistance) / 100.0 if defender.resistance is not None and str(defender.resistance).strip() != '' else 0.0
        if not (0.0 <= resistance <= 1.0):
            logger.warning(f"警告: 防御者 {defender.name} 法抗值 ({defender.resistance}) 超出预期范围 (0-100)，已修正为有效范围。")
            resistance = max(0.0, min(1.0, resistance))
    except (ValueError, TypeError):
        resistance = 0.0
        logger.warning(f"警告: 防御者 {defender.name} 法抗值无效 ({defender.resistance})")

    try:
        attack_interval = float(attacker.attack_interval) if attacker.attack_interval is not None and str(attacker.attack_interval).strip() != '' else 0.0
    except (ValueError, TypeError):
        attack_interval = 0.0
        logger.warning(f"警告: 攻击者 {attacker.name} 攻击间隔无效 ({attacker.attack_interval})")

    try:
        defender_health = float(defender.health) if defender.health is not None and str(defender.health).strip() != '' else 0.0
        if defender_health <= 0:
             logger.warning(f"警告: 防御者 {defender.name} 生命值无效或为零 ({defender.health})，百分比计算将为 0。")
             defender_health = 0.0 # Treat as 0 for calculation safety
    except (ValueError, TypeError):
        defender_health = 0.0
        logger.warning(f"警告: 防御者 {defender.name} 生命值无效 ({defender.health})")


    # --- Damage Calculation ---
    if attacker.is_magic_damage:
        damage_type = "法术"
        dph = attack * (1.0 - resistance)
    else:
        damage_type = "物理"
        if attack <= defense:
            dph = "不破防"
        else:
            dph = attack - defense

    # --- DPS Calculation ---
    if isinstance(dph, (int, float)) and dph > 0 and attack_interval > 0:
        dps = dph / attack_interval
    else:
        # If "不破防" or DPH is zero/negative, or interval is invalid, DPS is 0
        dps = 0.0

    # Ensure DPH is non-negative if it's a number
    if isinstance(dph, (int, float)):
        dph = max(0.0, dph)

    # --- DPH Percentage Calculation ---
    dph_percent: float = 0.0
    if isinstance(dph, (int, float)) and dph > 0 and defender_health > 0:
        percentage = (dph / defender_health) * 100.0
        dph_percent = min(percentage, 100.0) # Cap at 100%
    # If dph is "不破防" or 0, or defender_health is 0, percentage remains 0.0

    return dph, dps, damage_type, dph_percent