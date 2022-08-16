# Custom Inference Models GitHub Actions Implementation
This repository contains the core implementation of the DataRobot GitHub actions to manage
Datarobot custom inference models and deployments with CI/CD GitHub workflows.

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

## Development Prerequisites
There are two mandatory environment variables that must be set in the GitHub repository
**Secrets** section. These are used to communicate with the DataRobot system, which is required
during execution of the associated GitHub actions. The environment variables are:
* `DATAROBOT_WEBSERVER` - the DataRobot web server URL, which can be accessed publicly.
* `DATAROBOT_API_TOKEN` - the API token that is used to validate credentials with DataRobot system.

This means that a dedicated account should always be maintained in DataRobot for the sake of a
development of this repository.

## Development Workflow
Any change to this repository should be submitted as a pull request. When a pull request is
created the associated GitHub workflow will be executed and the following jobs will be executed
sequentially:
* A linter and code style.
* Execution of the unit-tests.
* Execution of the "[Custom Inference Model](#model_action_ref)" GitHub action.
* Execution of the "[Custom Inference Model Deployment](#deployment_action_ref)" GitHub action.

## The GitHub Actions

### Custom Inference Model <a name="model_action_ref"></a>
The GitHub action is supposed to be integrated into a GitHub workflow, which is expected to be
triggered by pull request and push events. When it is executed, it scans the repository for
model definitions and carries out the relevant model actions in the configured DataRobot system.

#### Input Arguments
This action supports the following input arguments:
* `allow-model-deletion` - whether a model deletion can take place. (Default: `false`)
* `api-token` - an API token from DataRobot.
* `main-branch` - the main branch for which pull request and push events will trigger the action.
* `webserver` - the URL to DataRobot application (.e.g https://app.datarobot.com).
* `skip-cert-verification` - whether a request to an HTTPS URL will be made without a certificate
                             verification (Default: `false`).

#### Output Statistics
The action support the following statistic attributes, which can be used by other steps in the
GitHub job:
* `total-affected-models` - the total number of models that were affected by this action.
* `total-created-models` - the total number of new models that were created by this action.
* `total-deleted-models` - the total number of models that were deleted by this action.
* `total-created-model-versions` - the total number of new model versions that were created by
                                   this action.

In the current repository there is a definition of one model under `tests/models/py3_sklearn/`,
which will result in a creation of a custom inference model in DataRobot.

### Custom Inference Model Deployment <a name="deployment_action_ref"></a>
The GitHub action is supposed to be integrated into a GitHub workflow, which is expected to be
triggered by pull request and push events. When it is executed, it scans the repository for
deployment definitions and carries out the relevant deployment actions in the configured DataRobot
system.

#### Input Arguments
This action supports the following input arguments:
* `allow-deployment-deletion` - whether a deployment deletion can take place. (Default: `false`)
* `api-token` - an API token from DataRobot.
* `release-branch` - the release branch for which pull request and push events will trigger the
                     action.
* `webserver` - the URL to DataRobot application (.e.g https://app.datarobot.com).
* `skip-cert-verification` - whether a request to an HTTPS URL will be made without a certificate
                             verification (Default: `false`).

#### Output Statistics
The action support the following statistic attributes, which can be used by other steps in the
GitHub job:
* `total-affected-deployments` - the total number of deployments that were affected by this action.
* `total-created-deployments` - the total number of new deployments that were created by this
                                action.
* `total-deleted-deployments` - the total number of deployments that were deleted by this action.

In the current repository there is a definition of one deployment under `tests/deployments/`,
which will result in a creation of a custom inference model deployment in DataRobot.
