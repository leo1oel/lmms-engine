from contextlib import redirect_stdout

import torch.distributed as dist
from loguru import logger


class Logging:
    @staticmethod
    def info(msg: str):
        if dist.is_initialized():
            if dist.get_rank() == 0:
                logger.info(msg)
        else:
            logger.info(msg)

    @staticmethod
    def error(msg: str):
        if dist.is_initialized():
            if dist.get_rank() == 0:
                logger.error(msg)
        else:
            logger.error(msg)

    @staticmethod
    def warning(msg: str):
        if dist.is_initialized():
            if dist.get_rank() == 0:
                logger.warning(msg)
        else:
            logger.warning(msg)

    @staticmethod
    def debug(msg: str):
        if dist.is_initialized():
            if dist.get_rank() == 0:
                logger.debug(msg)
        else:
            logger.debug(msg)

    @staticmethod
    def null_logging(msg):
        with redirect_stdout(None):
            print(msg)
