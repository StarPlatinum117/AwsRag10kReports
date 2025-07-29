import logging
from rich.logging import RichHandler

def setup_logging(level: int = logging.INFO) -> None:
    FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    logging.basicConfig(
        level=level,
        format=FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[RichHandler(markup=True)]
    )
