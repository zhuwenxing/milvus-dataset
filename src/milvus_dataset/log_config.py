from loguru import logger
import sys

# 移除默认的处理器
logger.remove()
# 添加控制台处理器
logger.add(sys.stderr, format= "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{thread.name}</cyan> |"
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO")



def configure_logger(level="INFO", **kwargs):
    """
    配置日志器。

    :param level: 日志级别，默认为 "INFO"
    :param kwargs: 其他 Loguru 配置参数
    """
    logger.remove()
    logger.add(sys.stderr, level=level, format=format, **kwargs)
