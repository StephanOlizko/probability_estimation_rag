import logging
import sys
from config import Config

config = Config()

def setup_logger(level=logging.DEBUG):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.config_path, encoding="utf-8"),
        ]
    )
