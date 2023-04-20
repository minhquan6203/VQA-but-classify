import argparse

from get_config import get_config
from task import OpenEndedTask
from logging_utils import setup_logger

logger = setup_logger()

parser = argparse.ArgumentParser()
parser.add_argument("--config-file", type=str, required=True)

args = parser.parse_args()

config = get_config(args.config_file)

task = ViTmBERTGeneration(config)
task.start()
task.get_predictions()
logger.info("Task done.")