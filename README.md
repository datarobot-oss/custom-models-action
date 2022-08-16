# Custom Inference Models GitHub Actions Core Implementation
This repository contains the core implementation of the DataRobot GitHub actions to manage
Datarobot custom inference models and deployments with CI/CD GitHub workflows. On its own, this
repository will not be used directly by customers in their workflows, but will be referred to,
from the GitHub actions that will be created in separate repositories.

## The Repository Structure
The top level folders include the following:
* `.github` - contains a GitHub workflow that executes the following jobs:
  * A linter and code style check.
  * An execution of the unit-tests.
  * An execution of the custom inference model GitHub action.
  * An execution of the custom inference model deployment GitHub action.
* `actions` - contains two actions:
  * "Custom Inference Models" - an action to manage custom inference models in DataRobot.
  * "Custom Inference Model Deployments" - an action to manage custom inference model
    deployments in DataRobot.
* `deps` - contains Python requirements for development of this repository.
* `src` - contains the source code that implements the related GitHub actions.
* `tests` - contains code source and resources to test the implementation. It includes the
  following:
  * `datasets` - the datasets that are used by the tests.
  * `deployments` - contains a deployment definition that is used by tests.
  * `functional` - contains the functional tests source code.
  * `models` - contains a model definition and source code, which are used by the
    tests.
  * `unit` - contains the unit-tests source code.

## The Main Entry Point
The GitHub actions eventually will call the Python program entry point with different input 
arguments, according to the desired functionality.

The main entry point resides in `src/main.py` and supports the following input arguments:

* `--allow-deployment-deletion` - whether to detect local deleted deployment definitions and
     consequently delete them in DataRobot. It applies to Custom Inference Model Deployment GitHub
     action only (Optional. Default: false).
* `--allow-model-deletion` - whether to detect local deleted model definitions and consequently
     delete them in DataRobot. It applies to Custom Inference Model GitHub action only
     (Optional. Default: false).
* `--api-token` - DataRobot public API authentication token.
* `--branch` - the branch against which the program will function.
* `--deploy` - determines whether to manage custom inference models or deployments (Optional.
     Default: false).
* `--root-dir`- the workspace root directory.
* `--skip-cert-verification` - whether a request to an HTTPS URL will be made without a certificate
     verification (Optional. Default: false).
* `--webserver` - DataRobot Web server URL.

#### Output Statistics
The program supports the following statistic attributes, which can later be used by other steps
in a GitHub job:
* `total-affected-models` - the total number of models that were affected by this action.
* `total-created-models` - the total number of new models that were created by this action.
* `total-deleted-models` - the total number of models that were deleted by this action.
* `total-created-model-versions` - the total number of new model versions that were created by
  this action.

These are printed to the standard output in a pre-defined pattern, as described in GitHub
"[Setting an output parameter](
https://docs.github.com/en/actions/using-workflows/workflow-commands-for-github-actions#setting-an-output-parameter
)".

## Functional Tests
The functional tests are written on top of the main entry point, thus simulating the GitHub
actions execution. In order to enable communication with a real DataRobot system, there should
be two environment variables that are expected to be set:

* `DATAROBOT_WEBSERVER` - the DataRobot web server URL, which can be accessed publicly.
* `DATAROBOT_API_TOKEN` - the API token that is used to validate credentials with DataRobot system.

In the current repository there is a definition of one model under `tests/models/py3_sklearn/`,
and one deployment under `tests/deployments`, which are used by the functional test.

## Development Workflow
Changes in this repository should be submitted as a pull requests. When a pull request is
created the associated GitHub workflow will be triggered and the following jobs will be executed
sequentially:
* A linter and code style.
* Execution of the unit-tests.
* Execution of a single functional test

**NOTE:** in order to enable the full execution of the functional test, the two related environment
variables (`DATAROBOT_WEBSERVER` & `DATAROBOT_API_TOKEN`) should be set in the "Secrets" section
in the GitHub repository.
