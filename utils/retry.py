import time
import functools
import logging
import random

# 配置日志（可选，但强烈建议用于记录重试行为）
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def retry(max_attempts=3, delay=1, exceptions=(Exception,), backoff=1):
    """
    一个用于函数执行的重试装饰器。

    Args:
        max_attempts (int): 最大重试次数。
        delay (int): 初始等待延迟（秒）。
        exceptions (tuple): 触发重试的异常类型。
        backoff (int): 延迟的乘数因子（例如，如果 backoff=2，延迟将是 1s, 2s, 4s, ...）。
    """

    def decorator(func):
        # 使用 @functools.wraps 确保装饰后的函数保留原函数的名称、文档字符串和元数据
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            
            # 内部变量，用于跟踪当前尝试次数和延迟时间
            current_delay = delay
            
            for attempt in range(1, max_attempts + 1):
                try:
                    # 尝试执行被装饰的函数
                    result = func(*args, **kwargs)
                    logging.info(f"Function '{func.__name__}' succeeded on attempt {attempt}.")
                    return result
                    
                except exceptions as e:
                    # 如果捕获到指定异常
                    if attempt < max_attempts:
                        logging.warning(
                            f"Function '{func.__name__}' failed on attempt {attempt}/{max_attempts} "
                            f"with error: {type(e).__name__}. Retrying in {current_delay:.2f} seconds..."
                        )
                        # 等待
                        time.sleep(current_delay)
                        
                        # 更新延迟时间（指数退避）
                        current_delay *= backoff
                    else:
                        # 达到最大尝试次数，抛出原始异常
                        logging.error(f"Function '{func.__name__}' failed after {max_attempts} attempts.")
                        raise e
            
            # 理论上不会到达这里，但在某些极端情况下为了代码完整性保留
            return None 
        
        return wrapper
        
    return decorator