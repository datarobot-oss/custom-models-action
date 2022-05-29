#!/usr/bin/env python
import argparse
import logging
import os
import sys

from custom_inference_model import CustomInferenceModel
from custom_inference_deployment import CustomInferenceDeployment
from common.exceptions import GenericException

logger = logging.getLogger()


def argparse_options(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Whether a action should be manage custom inference models or deployments.",
    )
    parser.add_argument("--webserver", required=True, help="DataRobot frontend webserver URL.")
    parser.add_argument(
        "--api-token", required=True, help="DataRobot public API authentication token."
    )
    parser.add_argument("--branch", required=True, help="The branch against which PRs take action.")
    parser.add_argument("--root-dir", required=True, help="The workspace root directory.")

    return parser.parse_args(args)


def setup_log_configuration():
    log_level = os.environ.get("LOGLEVEL", "INFO").upper()
    log_format = "%(asctime)s [%(levelname)s]  %(message)s"
    try:
        logging.basicConfig(format=log_format, level=log_level)
    except ValueError:
        logging.basicConfig(format=log_format, level=logging.INFO)


def main(options):
    setup_log_configuration()

    try:
        if options.deploy:
            CustomInferenceDeployment(options).run()
        else:
            CustomInferenceModel(options).run()
    except GenericException as e:
        # Avoid printing the stacktrace
        logger.error(str(e))
        sys.exit(e.code)


if __name__ == "__main__":
    main(argparse_options())
