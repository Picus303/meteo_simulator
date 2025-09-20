from __future__ import annotations

import sys
import time
import json
import logging
from typing import Any

    
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": _iso8601_time(record.created),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }

        # Attach extras
        if "extra" in record.__dict__ and isinstance(record.__dict__["extra"], dict):
            payload.update(record.__dict__["extra"])
        # Add exception info if any
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def _iso8601_time(created: float) -> str:
    # time.gmtime -> UTC; we can adjust or keep UTC for logs.
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(created))


def get_logger(name: str = "meteo", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_kv(logger: logging.Logger, msg: str, **kv: Any) -> None:
    logger.info(msg, extra={"extra": kv})