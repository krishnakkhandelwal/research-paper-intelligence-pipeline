"""
Shared logging utility. Every pipeline stage (ingestion, parsing,
enrichment, embedding, storage) wraps its entry function with
@log_stage(...) so we get consistent, structured logs like:

2026-07-19 10:03:11 | INFO | table_extraction | SUCCESS stage='table_extraction' paper_id='p001' duration=1.42s

This is what backs the JD's "Monitoring and Maintenance" requirement.
"""

import logging
import sys
import time
from functools import wraps


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def log_stage(stage_name: str):
    """Decorator: logs start/success/failure and duration of a pipeline stage."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(stage_name)
            paper_id = kwargs.get("paper_id", "unknown")
            start = time.time()
            logger.info(f"START stage='{stage_name}' paper_id='{paper_id}'")
            try:
                result = func(*args, **kwargs)
                duration = round(time.time() - start, 2)
                logger.info(
                    f"SUCCESS stage='{stage_name}' paper_id='{paper_id}' duration={duration}s"
                )
                return result
            except Exception as e:
                duration = round(time.time() - start, 2)
                logger.error(
                    f"FAILED stage='{stage_name}' paper_id='{paper_id}' duration={duration}s error={e}"
                )
                raise

        return wrapper

    return decorator
