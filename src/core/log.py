import logging.handlers
import os
import sys

import colorlog

# --- 配置 ---
LOG_DIR = 'log'  # 日志文件存放目录
LOG_FILE = 'app.log' # 日志文件名
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024  # 单个日志文件最大大小 (10MB)
LOG_FILE_BACKUP_COUNT = 5 # 保留的备份文件数量
LOG_LEVEL = logging.DEBUG # 日志记录级别

# --- 创建日志目录 ---
if not os.path.exists(LOG_DIR):
    try:
        os.makedirs(LOG_DIR)
    except OSError as e:
        print(f"无法创建日志目录 '{LOG_DIR}': {e}")
        # 如果无法创建目录，可以考虑退出或使用默认行为
        # 这里我们仅打印错误，日志文件处理器会失败

# --- 日志格式 ---
# 控制台彩色日志格式
log_colors = {
    'DEBUG':    'cyan',
    'INFO':     'green',
    'WARNING':  'yellow',
    'ERROR':    'red',
    'CRITICAL': 'bold_red',
}
console_log_formatter = colorlog.ColoredFormatter(
    '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(log_color)s%(message)s%(reset)s',
    datefmt='%m-%d %H:%M:%S',
    log_colors=log_colors,
    reset=True,
    style='%'
)

# 文件日志格式 (无颜色)
file_log_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d - %(funcName)s] - %(message)s',
    datefmt='%m-%d %H:%M:%S',
    style='%'
)

# --- 配置控制台处理器 ---
stream_handler = colorlog.StreamHandler(sys.stdout)
stream_handler.setFormatter(console_log_formatter)
stream_handler.setLevel(logging.INFO) # 控制台也输出所有级别的日志

# --- 配置文件处理器 ---
log_file_path = os.path.join(LOG_DIR, LOG_FILE)
try:
    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path,
        maxBytes=LOG_FILE_MAX_BYTES,
        backupCount=LOG_FILE_BACKUP_COUNT,
        encoding='utf-8' # 明确指定编码
    )
    file_handler.setFormatter(file_log_formatter)
    file_handler.setLevel(LOG_LEVEL) # 文件记录所有级别的日志
except Exception as e:
    print(f"无法配置日志文件处理器 '{log_file_path}': {e}")
    file_handler = None # 标记处理器配置失败

# --- 获取并配置根日志记录器 ---
# 使用 getLogger() 而不是 getLogger(__name__) 以便全局共享
logger = logging.getLogger('ArknightALL')
logger.setLevel(LOG_LEVEL)

# --- 清除旧的处理器并添加新的处理器 ---
# 防止重复添加处理程序
if logger.hasHandlers():
    logger.handlers.clear()

logger.addHandler(stream_handler)
if file_handler: # 仅当文件处理器成功创建时才添加
    logger.addHandler(file_handler)

# --- 示例日志消息 (可选) ---
# def example_function():
#     logger.debug('这是一个 debug 消息')
#     logger.info('这是一个 info 消息')
#     logger.warning('这是一个 warning 消息')
#     logger.error('这是一个 error 消息')
#     logger.critical('这是一个 critical 消息')

# if __name__ == '__main__':
#     example_function()