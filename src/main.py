#!/usr/bin/env python
import argparse
from custom_inference_model import CustomInferenceModel
from custom_inference_deployment import CustomInferenceDeployment


def argparse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--deploy',
        action='store_true',
        help='Whether a action should be manage custom inference models or deployments ',
    )
    parser.add_argument('--webserver', required=True, help='DataRobot frontend webserver URL')
    parser.add_argument(
        '--api-token', required=True, help='DataRobot public API authentication token'
    )
    parser.add_argument('--branch', required=True, help='The branch against which PRs take action')
    parser.add_argument(
        '--root-dir', required=True, help='The workspace root directory'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = argparse_options()
    if args.deploy:
        CustomInferenceDeployment(args).run()
    else:
        CustomInferenceModel(args).run()
