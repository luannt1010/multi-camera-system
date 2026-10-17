import logging
import sys

import yaml
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from src.logging_config import configure_logging


logger = logging.getLogger("src.yolo.train")
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "yolo_config.yaml"


def main():

    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model = YOLO(config["model"]["weights"])

    # Train
    model.train(
        data=config["data"]["path"],
        **config["train"]
    )

    # Valid
    metrics = model.val(
        data=config["data"]["path"],
        **config["validation"]
    )

    logger.info("mAP50: %.4f", metrics.box.map50)
    logger.info("mAP50-95: %.4f", metrics.box.map)

if __name__ == "__main__":
    with configure_logging():
        main()
