# Custom Models GitHub Action

(**NOTE: this repository is still a work in progress**)

This repository contains the DataRobot GitHub action to manage Datarobot custom inference models
and deployments via CI/CD GitHub workflows. It enables users to create/delete and change settings of
models and deployments.

The control over models and deployments are done via metadata YAML files, which in general, can be
located at any folder under the repository. The YAML files are searched, collected and tested
against a specific schema to identify whether they contain the related definitions for each of these
entities.

## Datasets
Datasets that are referenced in this definition YAML files are expected to exist in DataRobot
catalogue before doing any GitHub step. The user is required to upload these datasets
upfront, to the DataRobot catalog via its UI or any other client.

## Drop-In Environments
Environments that are referenced in this definition YAML files are expected to exist in DataRobot
Custom Models environments, before doing any GitHub action. The user is required to validate
the existence of such drop-in environments upfront and use any ID of these environments. The user
can also install a new drop-in environment and use its ID. For more information please reference
DataRobot documentation.

## The GitHub Action's Input Arguments <a id="input-arguments"/>
The GitHub action is implemented as a Python program, which is being called with specific arguments
that are provided by the user, in the GitHub workflow.

### Mandatory Input Arguments
The action supports the following mandatory input arguments:

* `--api-token` - DataRobot public API authentication token.
* `--branch` - the branch against which the program will function.
* `--webserver` - DataRobot Web server URL.

### Optional Input Arguments
The action supports the following optional input arguments:

* `--allow-deployment-deletion` - whether to detect local deleted deployment definitions and
     consequently delete them in DataRobot (Default: false).
* `--allow-model-deletion` - whether to detect local deleted model definitions and consequently
     delete them in DataRobot (Default: false).
* `--models-only` - determines whether to manage custom inference models only or also deployments
     (Default: false).
* `--skip-cert-verification` - whether a request to an HTTPS URL will be made without a certificate
     verification (Default: false).

## The GitHub Action's Output Statistics
The GitHub action supports the following output arguments, which can be later used by follow-up
steps in the same GitHub job (refer to the workflow example blow):

* `total-affected-models` - the total number of models that were affected by this action.
* `total-created-models` - the total number of new models that were created by this action.
* `total-deleted-models` - the total number of models that were deleted by this action.
* `total-created-model-versions` - the total number of new model versions that were created by
   this action.
* `total-affected-deployments` - the total number of deployments that were affected by this action.
* `total-created-deployments` - the total number of new deployments that were created by this action.
* `total-deleted-deployments` - the total number of deployments that were deleted by this action.
* `message` - the output message from the GitHub action.

## Model Definition
The user is required to provide the model's metadata in a YAML file. The model's full schema is
defined in
[this source code block](
https://github.com/datarobot/custom-models-action/blob/62b9df9e8895becabd7592e65c0ed52252690498/src/schema_validator.py#L271
)

A model metadata YAML file may contain a schema of a single model's definition (as specified above),
or a schema of multiple models' definition.

The **multiple models' schema** is defined [here](
https://github.com/datarobot/custom-models-action/blob/62b9df9e8895becabd7592e65c0ed52252690498/src/schema_validator.py#L351
).

The single model's definition YAML file is required to be located inside the model's root directory.
The multiple models' definition YAML file can be located anywhere in the repository.

For examples please refer to the [model definition examples section](#model-examples) below.

**Notes:**
* A model is first created during a pull request, whenever a new definition is detected.
* A model is actually deleted during merging to the main branch, upon missing of the associated
  model's definition. This can happen if the definition's YAML file was deleted or the user
  changed its unique ID.
* Changes to the models in DataRobot are done during a pull request to the configured main
  branch. These include changes to settings as well as to creation of new custom inference model
  versions.
* A new model version is created upon changes to the model's code or the fields under the `version`
  section.

### Model Definition Sections

* At the top level, there are attributes which cannot be changed once the model is created.
* `settings` - changes to the fields under this section will result in changes to the model's
               settings, without creation of a new version.
* `version` - changes to the fields under this section will result in creation of a new version.
* `test` - contains attributes that control the custom inference model testing. If omitted, a
           test will not be executed.

## Deployment Definition
The user is required to provide the deployment's metadata in a YAML file. The deployment's full
schema is defined in [this source code block](
https://github.com/datarobot/custom-models-action/blob/62b9df9e8895becabd7592e65c0ed52252690498/src/schema_validator.py#L639
).

A deployment metadata YAML file may contain a schema of a single deployment's definition (as
specified above), or a schema of multiple deployments' definition.

The **multiple deployments' schema** is defined [here](
https://github.com/datarobot/custom-models-action/blob/62b9df9e8895becabd7592e65c0ed52252690498/src/schema_validator.py#L679
).

The deployment's definition YAML file (either a single or multiple) can be located anywhere in
the repository.

For examples please refer to the [deployment definition examples section](#deployment-examples)
below.

**Notes:**:
 * Changes to deployments in DataRobot are done only upon merging a pull request
   to the configured main branch. During a pull request, the GitHub action only performs
   integrity checks.
 * Every new version of the associated custom inference model will result in a new challenger
   or a model's replacement in the deployment. It depends on the deployment's configuration, which
   can be controlled from the YAML file. The default is creation of a new challenger.

### Deployment Definition Sections

* At the top level, there are attributes which are not supposed to be changed once the
  deployment is created. An exceptional is the `user_provided_model_id` field, which associates
  a model definition to the given deployment. A change in this field will trigger a model
  replacement or challenger, depending on the deployment's configuration.
* `settings` - changes to the fields under this section will result in changes to the deployment's
               settings.

## GitHub Workflow
A GitHub workflow is a configurable process made up of one or more jobs. It is defined in a YAML
file, which is supposed to be located under `.github/workflows` in the repository
(for more information please refer to
  [Using Workflows in GitHub](https://docs.github.com/en/actions/using-workflows)
)

In order to successfully use the DataRobot custom models GitHub action the following should be
embedded in the GitHub workflow definition:

1. The action should be run upon two events: `pull_request` and `push`. Therefore, the
   following should be set::
        ```yaml
        on:
          pull_request:
            branches: [ master ]
          push:
            branches: [ master ]
        ```
2. Use the DataRobot custom models action in a workflow job as follows:

```yaml
    jobs:
        datarobot-custom-models-action:
            # Run this job on any action of a PR, but skip the job upon merging to the main branch.
            # This will be taken care of by the push event.
            if: ${{ github.event.pull_request.merged != true }}

            runs-on: ubuntu-latest

            steps:
              - uses: actions/checkout@v3
                with:
                  fetch-depth: 0

              - name: DataRobot Custom Models Action
                id: datarobot-custom-models-action
                uses: datarobot/custom-models-action@v1.1.0
                with:
                  api-token: ${{ secrets.DATAROBOT_API_TOKEN }}
                  webserver: ${{ secrets.DATAROBOT_WEBSERVER }}
                  branch: master
                  allow-model-deletion: true
                  allow-deployment-deletion: true
```
   **Notes:**
   - `if: ${{ github.event.pull_request.merged != true }}` - this is an important condition that is
     needed in order to skip the action's execution upon merging. The action will be triggered
     by the 'push' event.
   - The action scans the repository files therefore it is required to use the `actions/checkout@v3`
     action in a step before DataRobot action.
   - The `custom-models-action@1.1.0` link refers to a specific historic release. You might want
     to look at newer versions in the [RELEASES.md](RELEASES.md).
   - There are two actions' input arguments that are used to establish a communication with
     DataRobot system. They should reside in the repository "Secrets" section:
     - `DATAROBOT_API_TOKEN` - the API token that is used to validate credentials with DataRobot
                               system.
     - `DATAROBOT_WEBSERVER` - the DataRobot web server URL, which can be accessed publicly.
   - For the full possible input arguments to the action please refer to the
     [input arguments section](#input-arguments) above.
   - For a complete example please refer to the [workflow example](#workflow-example) below.

## Development Information

### The Repository Structure
The top level folders include the following:
* `action.yaml` - the definition of the DataRobot custom models GitHub action.
* `.github` - contains a GitHub workflow that executes the following jobs:
    * A linter and code style check.
    * An execution of the unit-tests.
    * An execution of the functional tests.
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

### Functional Tests
- The functional tests are written on top of the main entry point, thus simulating the GitHub
  actions execution. In order to enable communication with a real DataRobot system, there should
  be two environment variables that are expected to be set:

  * `DATAROBOT_WEBSERVER` - the DataRobot web server URL, which can be accessed publicly.
  * `DATAROBOT_API_TOKEN` - the API token that is used to validate credentials with DataRobot system.

- In the current repository there is a definition of one model under `tests/models/py3_sklearn/`,
  and one deployment under `tests/deployments`, which are used by the functional test.

### Development Workflow
Changes in this repository should be submitted as a pull requests. When a pull request is
created the associated GitHub workflow will be triggered and the following jobs will be executed
sequentially:
* A linter and code style.
* Execution of the unit-tests.
* Execution of a single functional test

**Note:** in order to enable the full execution of the functional test, the two related
variables (`DATAROBOT_WEBSERVER` & `DATAROBOT_API_TOKEN`) should be set in the
"Secrets" section in the GitHub repository. These are read by the workflow, which sets
the proper environment variables.


## Examples

### Metadata Definition Examples

#### Model Examples <a id="model-examples"/>

##### A Minimal Single Model Definition
Here is an example of a minimal model's definition, which includes only mandatory fields:

```yaml
user_provided_model_id: any-model-unique-id-1
target_type: Regression
settings:
  name: My Awsome GitHub Model 1 [GitHub CI/CD]
  target_name: Grade 2014

version:
  # Make sure this is the environment ID is in your system.
  # This one is the '[DataRobot] Python 3 Scikit-Learn Drop-In' environment
  model_environment_id: 5e8c889607389fe0f466c72d
```
##### A Full Single Model Definition
Here is an example of a full model's definition, which includes both mandatory and optional fields:

```yaml
user_provided_model_id: any-model-unique-id-1
target_type: Binary
settings:
  name: My Awsome GitHub Model 1 [GitHub CI/CD]
  description: My awesome model
  target_name: Grade 2014
  holdout_dataset_id: 627790ca5621558b55c78d78
  language: Python
  negative_class_label: '0'
  positive_class_label: '1'
  training_dataset_id: 627790ba56215587b3021632

version:
  # Make sure this is the environment ID is in your system.
  # This one is the '[DataRobot] Python 3 Scikit-Learn Drop-In' environment
  model_environment_id: 5e8c889607389fe0f466c72d
  exclude_glob_pattern:
    - README.md
    - out/
  include_glob_pattern:
    - ./
  memory: 100Mi
  replicas: 3

test:
  memory: 100Mi
  skip: false
  test_data_id: 62779143562155aa34a3d65b
  checks:
    null_value_imputation:
      block_deployment_if_fails: true
      enabled: true
    performance:
      block_deployment_if_fails: false
      enabled: true
      max_execution_time: 100
      maximum_response_time: 50
      number_of_parallel_users: 3
    prediction_verification:
      block_deployment_if_fails: false
      enabled: true
      match_threshold: 0.9
      output_dataset_id: 627791f5562155d63f367b05
      passing_match_rate: 85
      predictions_column: Grade 2014
    side_effects:
      block_deployment_if_fails: true
      enabled: true
    stability:
      block_deployment_if_fails: true
      enabled: true
      maximum_payload_size: 1000
      minimum_payload_size: 100
      number_of_parallel_users: 1
      passing_rate: 95
      total_prediction_requests: 50
```

**NOTE**: the patterns used in the `exclude_glob_pattern` & `include_glob_pattern` fields are an
extension to the common glob rules. A path that ends with `/` (slash), which means a directory,
will automatically be regarded as suffixed with `**`. This means that the directory will be
scanned recursively.

##### A Multi Models Definition
Here is an example of a multi-models definition, which includes only mandatory fields:

```yaml
datarobot_models:
  - model_path: ./models/model_1
    model_metadata:
      user_provided_model_id: any-model-unique-id-1
      target_type: Regression
      settings:
        name: My Awsome GitHub Model 1 [GitHub CI/CD]
        target_name: Grade 2014

      version:
        # Make sure this is the environment ID is in your system.
        # This one is the '[DataRobot] Python 3 Scikit-Learn Drop-In' environment
      model_environment_id: 5e8c889607389fe0f466c72d

  - model_path: ./models/model_2
    model_metadata:
      user_provided_model_id: any-model-unique-string-2
      target_type: Regression
      settings:
        name: My Awsome GitHub Model 2 [GitHub CI/CD]
        target_name: Grade 2014

      version:
      # Make sure this is the environment ID is in your system.
      # This one is the '[DataRobot] Python 3 Scikit-Learn Drop-In' environment
      model_environment_id: 5e8c889607389fe0f466c72d
```

#### Deployment Examples <a id="deployment-examples"/>

##### A Minimal Single Deployment Definition
Here is an example of a minimal deployment's definition, which includes only mandatory fields:

```yaml
user_provided_deployment_id: my-awesome-deployment-id
user_provided_model_id: any-model-unique-id-1
```

##### A Full Single Deployment Definition
Here is an example of a full deployment's definition, which includes both mandatory and optional
fields:

```yaml
user_provided_deployment_id: my-awesome-deployment-id
user_provided_model_id: any-model-unique-string-2
prediction_environment_name: "https://eks-test.orm.company.com"
settings:
  label: "My Awesome Deployment (model-2)"
  description: "This is a more detailed description."
  importance: LOW
  association:
    prediction_id: Animal
    required_in_pred_request: true
    actuals_id: Animal
    actuals_dataset_id: 6d8c889607389fe0f466c72e
  enable_target_drift: true
  enable_feature_drift: true
  enable_predictions_collection: true
  enable_challenger_models: true
  segment_analysis:
    enabled: true
    attributes:
      - Host-IP
      - Remote-IP
```

##### A Multi Deployments Definition
Here is an example of a multi-deployments definition, which includes only mandatory fields:

```yaml
- user_provided_deployment_id: any-deployment-unique-id-1
  user_provided_model_id: any-model-unique-id-1

- user_provided_deployment_id: any-deployment-unique-id-2
  user_provided_model_id: any-model-unique-string-2

- user_provided_deployment_id: any-deployment-unique-id-3
  user_provided_model_id: any-model-unique-id-3
```

### GitHub Workflow Example <a id="workflow-example"/>
Here is an example for a GitHub workflow definition, which should be located under: `.
github/workflows/workflow.yaml`:

```yaml
name: Workflow CI/CD

on:
  pull_request:
    branches: [ master ]
  push:
    branches: [ master ]

  # Allows you to run this workflow manually from the Actions tab
  workflow_dispatch:

jobs:
  datarobot-custom-models-action:
    # Run this job on any action of a PR, but skip the job upon merging to the main branch. This
    # will be taken care of by the push event.
    if: ${{ github.event.pull_request.merged != true }}

    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0

      - name: DataRobot Custom Models Action
        id: datarobot-custom-models-action
        uses: datarobot/custom-models-action@v1.1.0
        with:
          api-token: ${{ secrets.DATAROBOT_API_TOKEN }}
          webserver: ${{ secrets.DATAROBOT_WEBSERVER }}
          branch: master
          allow-model-deletion: true
          allow-deployment-deletion: true

      - name: DataRobot Custom Models Action Results
        run: |
          echo "Total affected models: ${{ steps.datarobot-custom-models-action.outputs.total-affected-models }}"
          echo "Total created models: ${{ steps.datarobot-custom-models-action.outputs.total-created-models }}"
          echo "Total deleted models: ${{ steps.datarobot-custom-models-action.outputs.total-deleted-models }}"
          echo "Total created model versions: ${{ steps.datarobot-custom-models-action.outputs.total-created-model-versions }}"

          echo "Total affected deployments: ${{ steps.datarobot-custom-models-action.outputs.total-affected-deployments }}"
          echo "Total created deployments: ${{ steps.datarobot-custom-models-action.outputs.total-created-deployments }}"
          echo "Total deleted deployments: ${{ steps.datarobot-custom-models-action.outputs.total-deleted-deployments }}"

          echo "Message: ${{ steps.datarobot-custom-models-action.outputs.message }}"
```

# Copyright and License

Custom Models GitHub Action is Copyright 2022 DataRobot, Inc.  All rights reserved.
Licensed under a Modified 3-Clause BSD License (the "License").  See the LICENSE file. You may not
use this software except in compliance with the License.

Software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT ANY EXPRESS OR
IMPLIED WARRANTIES OF ANY KIND AND WITHOUT ANY LICENSE TO ANY PATENTS OR TRADEMARKS. See the
LICENSE file for the specific language governing permissions and limitations under the License.