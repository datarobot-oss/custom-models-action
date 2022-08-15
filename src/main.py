#!/usr/bin/env python

#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

"""
The main entry point to the GitHub actions implementation. It contains implementation for two
different GitHub actions - custom inference model and custom inference model deployment.
"""

import argparse
import logging
import os
import sys

from common.exceptions import GenericException
from custom_inference_deployment import CustomInferenceDeployment
from custom_inference_model import CustomInferenceModel

logger = logging.getLogger()


def argparse_options(args=None):
    """
    Retrieve command line arguments.

    Parameters
    ----------
    args : list or None
        A list of arguments.

    Returns
    -------
    argparse.Namespace,
        The command line argument values.
    """

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
    parser.add_argument(
        "--skip-cert-verification",
        action="store_true",
        help="Whether a request to an HTTPS URL will be made without a certificate verification.",
    )
    parser.add_argument("--branch", required=True, help="The branch against which PRs take action.")
    parser.add_argument("--root-dir", required=True, help="The workspace root directory.")
    parser.add_argument(
        "--allow-model-deletion",
        action="store_true",
        help="Whether to detect local deleted model definitions and consequently delete them in "
        "DataRobot.",
    )
    parser.add_argument(
        "--allow-deployment-deletion",
        action="store_true",
        help="Whether to detect local deleted deployment definitions and consequently "
        "delete them in DataRobot.",
    )

    options = parser.parse_args(args)
    logger.debug("Command line args: %s", options)

    return options


def setup_log_configuration():
    """
    Setup logging configuration.
    """

    log_level = os.environ.get("LOGLEVEL", "INFO").upper()
    log_format = "%(asctime)s [%(levelname)s]  %(message)s"
    try:
        logging.basicConfig(format=log_format, level=log_level)
    except ValueError:
        logging.basicConfig(format=log_format, level=logging.INFO)


def main(args=None):
    """
    The main entry point method to the GitHub actions. The method makes sure to catch
    any exception and to exit the program with a proper exit code. The exit is being called
    only when the method is being called as a standalone program (from a command line).
    Otherwise, the exception is just re-raised for handling in higher layers.

    Parameters
    ----------
    args : list or None
        An optional list of command line arguments.
    """

    setup_log_configuration()
    options = argparse_options(args)

    try:
        if options.deploy:
            CustomInferenceDeployment(options).run()
        else:
            CustomInferenceModel(options).run()
            print(
                "::set-output name=message::"
                "Custom inference model GitHub action completed with success.\n"
            )
    except GenericException as ex:
        # Avoid printing the stacktrace
        logger.error(str(ex))
        print(f"::set-output name=message::{str(ex)}")
        if args:
            # It was called from the functional tests
            raise ex
        # It was called from the GitHub action
        sys.exit(ex.code)


if __name__ == "__main__":
    main()
