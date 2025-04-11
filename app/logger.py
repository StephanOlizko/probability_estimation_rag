import logging

def setup_logger():
    logger = logging.getLogger("retrieval_pipeline")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    fh = logging.FileHandler("pipeline.log")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.info("Logger setup complete")
    return logger
