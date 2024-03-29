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
from common.github_env import GitHubEnv
from common.namepsace import Namespace
from custom_models_action import CustomModelsAction

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
    parser.add_argument("--webserver", required=True, help="DataRobot Web server URL.")
    parser.add_argument(
        "--api-token", required=True, help="DataRobot public API authentication token."
    )
    parser.add_argument(
        "--branch", required=True, help="The branch against which the program will function."
    )
    parser.add_argument(
        "--namespace", help="It is used to group and isolate models and deployments."
    )
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
    parser.add_argument(
        "--models-only",
        action="store_true",
        help="Whether to handle custom inference models only, without deployments.",
    )
    parser.add_argument(
        "--skip-cert-verification",
        action="store_true",
        help="Whether a request to an HTTPS URL will be made without a certificate verification.",
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
    Namespace.init(options.namespace)
    try:
        CustomModelsAction(options).run()
        GitHubEnv.set_output_param(
            "message", "Custom inference model GitHub action completed with success."
        )
    except GenericException as ex:
        # Avoid printing the stacktrace
        logger.error(str(ex))
        GitHubEnv.set_output_param("message", str(ex))
        if args:
            # It was called from the functional tests
            raise ex
        # It was called from the GitHub action
        sys.exit(ex.code)


if __name__ == "__main__":
    main()
