import logging
from typing import Any

from Custom.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
logger.info("App started (LangChain version).")