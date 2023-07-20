# Custom Models GitHub Action

> **Note**: This repository is still a work in progress

The custom models action manages custom inference models and their associated deployments in
DataRobot via GitHub CI/CD workflows.
These workflows allow you to create or delete models and deployments and modify settings. Metadata
defined in YAML files enables the custom model action's control over models and deployments. Most
YAML files for this action can reside in any folder within your custom model's repository. The YAML
is searched, collected, and tested against a schema to determine if it contains the entities used
in these workflows.

## Prerequisites
The following feature flags must be enabled in DataRobot:
* ENABLE_MLOPS
* ENABLE_CUSTOM_INFERENCE_MODEL
* ENABLE_CUSTOM_MODEL_GITHUB_CI_CD

(Please contact [DataRobot support](mailto:support@datarobot.com) for more information)

## Custom Model Action Quick Start

This quickstart example uses a [Python Scikit-Learn model template](https://github.com/datarobot/datarobot-user-models/tree/master/model_templates/python3_sklearn) 
from the [datarobot-user-model repository](https://github.com/datarobot/datarobot-user-models/tree/master/model_templates). 
To set up a custom models action that will create a custom inference model and deployment in
DataRobot from a custom model repository in GitHub, take the following steps:

1. In the `.github/workflows` directory of your custom model repository, create a YAML file
   (with any filename) containing the following:

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
      datarobot-custom-models:
        # Run this job on any action of a PR, but skip the job upon merging to the main branch. This
        # will be taken care of by the push event.
        if: ${{ github.event.pull_request.merged != true }}

        runs-on: ubuntu-latest

        steps:
          - uses: actions/checkout@v3
            with:
              fetch-depth: 0

          - name: DataRobot Custom Models Step
            id: datarobot-custom-models-step
            uses: datarobot-oss/custom-models-action@v1.6.0
            with:
              api-token: ${{ secrets.DATAROBOT_API_TOKEN }}
              webserver: https://app.datarobot.com/
              branch: master
              allow-model-deletion: true
              allow-deployment-deletion: true
    ```
    
    Configure the following fields:

    - `branches`: Provide the name of your repository's main branch (usually either `master`
      or `main`) for `pull_request` and `push`.
      If you created your repository in GitHub, you likely need to update these fields to `main`. 
      While `master` and `main` are the most common branch names, you can target any branch; 
      for example, you could run the workflow on a `release` branch or a `test` branch.

    - `api-token`: Provide a value for the `${{ secrets.DATAROBOT_API_TOKEN }}` variable by creating
      an [encrypted secret for GitHub Actions](https://docs.github.com/en/actions/security-guides/encrypted-secrets#creating-encrypted-secrets-for-a-repository)
      containing your [DataRobot API key](https://docs.datarobot.com/en/docs/platform/account-mgmt/acct-settings/api-key-mgmt.html#api-key-management).
      Alternatively, you can set the token string directly to this field; however, this method is
      highly discouraged because your API key is extremely sensitive data. If you use this method, 
      anyone with access to your repository can access your API key.

    - `webserver`: Provide your DataRobot webserver value here if it isn't the default DataRobot 
      US server (`https://app.datarobot.com/`).

    - `branch`: Provide the name of your repository's main branch (usually either `master` or
      `main`). If you created your repository in GitHub, you likely need to update this field to
      `main`. While `master` and `main` are the most common branch names, you can target any branch; 
      for example, you could run the workflow on a `release` branch or a `test` branch.

2. Commit the workflow YAML file and push it to the remote. After you complete this step, any
   push to the remote (or merged pull request) triggers the action.

3. In the folder for your DataRobot custom model, add a model definition YAML file
   (e.g., `model.yaml`) containing the following YAML and update the field values according to your
   model's characteristics:

    ```yaml
    user_provided_model_id: user/model-unique-id-1
    target_type: Regression
    settings:
      name: My Awesome GitHub Model 1 [GitHub CI/CD]
      target_name: Grade 2014

    version:
      # Make sure this is the environment ID is in your system.
      # This one is the '[DataRobot] Python 3 Scikit-Learn Drop-In' environment
      model_environment_id: 5e8c889607389fe0f466c72d
    ```
    Configure the following fields:

    - `user_provided_model_id`: Provide any descriptive and unique string value.
      A good practice could be this kind of pattern: `<user>/<model-unique-id>`.
      Please note that, by default, this ID will reside in a unique namespace, which is the GitHub
      repository ID. Alternatively, the namespace can be configured as an input argument to the
      custom models action.
    - `target_type`: Provide the correct target type for your custom model.
    - `target_name`: Provide the correct target name for your custom model.
    - `model_environment_id`: Provide the DataRobot execution environment required for your custom
      model. You can find these environments in the DataRobot application under
      [**Model Registry** > **Custom Model Workshop** > **Environments**](https://docs.datarobot.com/en/docs/mlops/deployment/custom-models/custom-env.html).

4. In any directory in your repository, add a deployment definition YAML file (with any filename)
   containing the following YAML:

    ```yaml
    user_provided_deployment_id: user/my-awesome-deployment-id
    user_provided_model_id: user/model-unique-id-1
    ```

    Configure the following fields:

    - `user_provided_deployment_id`: Provide any descriptive and unique string value.
      A good practice could be this kind of pattern: `<user>/<deployment-unique-id>`.
      Please note that, by default, this ID will reside in a unique namespace, which is the GitHub
      repository ID. Alternatively, the namespace can be configured as an input argument to the
      custom models action.
    - `user_provided_model_id`: Provide the exact `user_provided_model_id` you set in the model
      definition YAML file.

5. Commit these changes and push to the remote:

    - Navigate to your custom model repository in GitHub and click the `Actions` tab. You'll notice
      that the action is being executed.

    - Navigate to the DataRobot application. You'll notice that a new custom model was
      created along with an associated deployment. This action can take a few minutes.

> **Note**: Creating two commits (or merging two pull requests) in quick succession can result in
> a `ResourceNotFoundError`. For example, you add a model definition with a training dataset, make
> a commit, and push to the remote. Then, you immediately delete the model definition, make a
> commit, and push to the remote. The training data upload action may begin after model deletion,
> resulting in an error. To avoid this scenario, wait for an action's execution to complete before
> pushing new commits or merging new pull requests to the remote repository.

## Custom Model Action Commit Information in DataRobot

After your workflow creates a model and a deployment in DataRobot, you can access the commit
information from the model's version info and the deployment's overview:

### Model Version Info

1. In the **Model Registry**, click **Custom Model Workshop**.

2. On the **Models** tab, click a GitHub-sourced model from the list and then click the
   **Versions** tab.

3. Under **Manage Versions**, click the version you want to view the commit for.

4. Under **Version Info**, find the **Git Commit Reference** and then click the commit hash
   (or commit ID) to open the commit in GitHub that created the current version.

### Model Package Info

  1. In the **Model Registry**, click **Model Packages**.
  
  2. On the **Model Packages** tab, click a GitHub-sourced model package from the list.

  3. Under **Package Info**, review the model information provided by your workflow, find
     the **Git Commit Reference**, and then click the commit hash (or commit ID) to open the commit
     that created the current model package.

### Deployment overview

1. In the **Deployments** inventory, click a GitHub-sourced deployment from the list.

2. On the deployment's **Overview** tab, review the model and deployment information provided by
   your workflow.

3. In the **Content** group box, find the **Git Commit Reference** and click the commit hash
   (or commit ID) to open the commit that created the deployment.

## Custom Model Action Reference

### Datasets

Datasets referenced in custom models action YAML files are expected to exist in the DataRobot AI
catalog before configuring the action in GitHub. You should upload these datasets to the DataRobot
AI catalog (via the UI or any other client) prior to configuring the GitHub action.

### Drop-In Environments

Environments referenced in custom models action YAML files are expected to exist in DataRobot before
configuring the action in GitHub. You should validate the existence of the required drop-in
environments prior to configuring the GitHub action. In addition, you can install new drop-in
environments. For more information, see the [Custom model environments documentation](https://docs.datarobot.com/en/docs/mlops/deployment/custom-models/custom-env.html).

### The GitHub Action's Input Arguments <a id="input-arguments"/>

This GitHub action is implemented as a Python program, called with specific arguments
provided in the GitHub workflow.

#### Mandatory Input Arguments 

This action requires the following input arguments:

|    Argument   |                   Description                  |
|---------------|------------------------------------------------|
| `--api-token` | Your DataRobot public API authentication key.  |
| `--branch`    | The branch on which the program will function. |
| `--webserver` | Your DataRobot instance's web server URL.      |


#### Optional Input Arguments 

The action supports the following optional input arguments:

| Argument                      | Description                                                                                                               |
|-------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| `--namespace`                 | Determines the namespace under which models and deployments will be created, updated and deleted.                         |
| `--allow-deployment-deletion` | Determines whether to detect local deleted deployment definitions and delete them in DataRobot. <br> **Default**: `false` |
| `--allow-model-deletion`      | Determines whether to detect local deleted model definitions and delete them in DataRobot <br> **Default**: `false`       |
| `--models-only`               | Determines whether to manage custom inference models only or also deployments <br> **Default**: `false`                   |
| `--skip-cert-verification`    | Determines whether a request to an HTTPS URL is made without a certificate verification. <br> **Default**: `false`        |

### A Namespace (Optional)

A namespace is a unique string that can be provided as an input argument to the action. The
purpose is to guarantee that the custom model action handles models and deployments that only exist
in the configured namespace. Any other models and deployments that are not in the configured
namespace will remain untouched.

If not provided, the GitHub repository ID will be used as the namespace (see: `GITHUB_REPOSITORY_ID`
in this [link](https://docs.github.com/en/actions/learn-github-actions/variables#default-environment-variables)).
It means that the custom models action will process models and deployments that were created from
this repository only.

If, for instance, users would like to work with the same model and deployment definitions from
different branches and still make sure that different entities will be created in DataRobot,
they can simply configure a different namespace to the custom models action in the GitHub workflow.

Please note, that a change of the namespace input argument to the custom models action in a GitHub
workflow, will result in new models and deployments in DataRobot. The existing ones will remain in
DataRobot without a control from the GitHub action.

### The GitHub Action's Output Metrics

The GitHub action supports the following output arguments, which can later be used by follow-up
steps in the same GitHub job (refer to the workflow example below):

| Argument                              | Description                                              |
|---------------------------------------|----------------------------------------------------------|
| `models--total-affected`              | The number of models affected by this action.            |
| `models--total-created`               | The number of new models created by this action.         |
| `models--total-deleted`               | The number of models deleted by this action.             |
| `models--total-updated-settings`      | The number of models whose settings were updated.        |
| `models--total-created-versions`      | The number of new model versions created by this action. |
| `deployments--total-affected`         | The number of deployments affected by this action.       |
| `deployments--total-created`          | The number of new deployments created by this action.    |
| `deployments--total-deleted`          | The number of deployments deleted by this action.        |
| `deployments--total-updated-settings` | The number of deployments whose settings were updated.   |
| `message`                             | The output message from the GitHub action.               |

### Model Definition

The GitHub action requires the model's metadata in a YAML file. The model's full schema is defined
by `MODEL_SCHEMA`, which can be found in [here](src/schema_validator.py).

A model metadata YAML file may contain the schema of a single model's definition
(as specified above) or the schema of multiple models' definitions.

The **multiple models' schema** is defined by `MULTI_MODELS_SCHEMA`, which can be found in
[here](src/schema_validator.py).

The single model's definition YAML file must be located inside the model's root directory.
The multiple models' definition YAML file can be located anywhere in the repository.

For examples, please refer to the [model definition examples section](#model-examples) below.

<details><summary>Notes</summary>

* A model is first created during a pull request whenever a new definition is detected.
* A model is deleted during a merge to the main branch if the associated model's definition 
  is missing. This can happen if the model definition's YAML file is deleted or if the model's 
  unique ID is changed.
* Changes to the models in DataRobot are made during a pull request to the configured main
  branch. These include changes to settings as well as the creation of new custom inference model
  versions.
* A new model version is created upon changes to the model's code or the fields under the `version`
  section.

</details>

#### Model Definition Sections

At the top level, there are attributes you cannot change after a model is created:

* `settings`: Changes to the fields under this section result in changes to the model's
              settings without creating a new version.
* `version`: Changes to the fields under this section result in a new version.
* `test`: Contains attributes that control the custom inference model testing. If omitted, a
          test will not be executed.

### Deployment Definition

The user is required to provide the deployment's metadata in a YAML file. The deployment's full
schema is defined by `DEPLOYMENT_SCHEMA`, which can be found in [here](src/schema_validator.py).

A deployment metadata YAML file may contain the schema of a single deployment's definition (as
specified above) or the schema of multiple deployments' definitions.

The **multiple deployments' schema** is defined by `MULTI_DEPLOYMENTS_SCHEMA`, which can be
found in [here](src/schema_validator.py).

The deployment definition YAML file (single or multiple) can be located anywhere in
the repository.

For examples, please refer to the [deployment definition examples section](#deployment-examples)
below.

<details><summary>Notes</summary>

 * Changes to deployments in DataRobot are made upon making a commit or merging a pull request
   to the configured main branch. During a pull request, the GitHub action only performs
   integrity checks.
 * Every new version of the associated custom inference model will result in a new challenger
   or a model's replacement in the deployment. It depends on the deployment's configuration, which
   can be controlled from the YAML file. The default is the creation of a new challenger.

</details>

#### Deployment Definition Sections

At the top level, some attributes shouldn't be changed once the deployment is created: 

* `user_provided_model_id`: An exception that associates a model definition to the given deployment.
  A change in this field triggers model replacement or challenger creation, depending on the
  deployment's configuration.
* `settings`: Changes to the fields in this section will result in changes to the deployment's
  settings.

### GitHub Workflow

A GitHub workflow is a configurable process of one or more jobs. It is defined in a YAML
file located under `.github/workflows` in the repository. For more information, refer to
[Using Workflows in GitHub](https://docs.github.com/en/actions/using-workflows).

To use the Custom Models Action, the following YAML should be included in the 
GitHub workflow definition:

1. The action should run on two events: `pull_request` and `push`. Therefore, the
   following should be defined:

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
      datarobot-custom-models:
        # Run this job on any action of a PR, but skip the job upon merging to the main branch.
        # This will be taken care of by the push event.
        if: ${{ github.event.pull_request.merged != true }}

        runs-on: ubuntu-latest

        steps:
          - uses: actions/checkout@v3
            with:
              fetch-depth: 0

          - name: DataRobot Custom Models Step
            id: datarobot-custom-models-step
            uses: datarobot-oss/custom-models-action@v1.6.0
            with:
              api-token: ${{ secrets.DATAROBOT_API_TOKEN }}
              webserver: ${{ secrets.DATAROBOT_WEBSERVER }}
              branch: master
              allow-model-deletion: true
              allow-deployment-deletion: true
    ```
  <details><summary>Notes</summary>

  - `if: ${{ github.event.pull_request.merged != true }}`: An important condition that is
    needed in order to skip the action's execution upon merging. The action will be triggered
    by the 'push' event.
  - `actions/checkout@v3`: The action scans the repository files; therefore, it requires the
    checkout action a step before the DataRobot action.
  - `custom-models-action@1.1.4`: This link refers to a specific historic release. You might want
    to look at newer versions in the [RELEASES.md](RELEASES.md).
  - Two input arguments are used to establish communication with DataRobot. 
    These arguments should be defined in the repository **Secrets** section:
    - `DATAROBOT_API_TOKEN`: The API token used to validate credentials with DataRobot.
    - `DATAROBOT_WEBSERVER`: The publicly accessible DataRobot web server URL.
      For the full possible input arguments to the action, refer to the
      [input arguments section](#input-arguments) above.

  </details>

  For a complete example, refer to the [workflow example](#workflow-example) below.

### Development Information

#### The Repository Structure

The top-level files and directories include the following:

* `action.yaml`: The YAML file containing the definition of the DataRobot custom models GitHub action.
* `.github`: The directory containing a GitHub workflow that executes the following jobs:
    * Linter 
    * Code style checks
    * Unit-tests
    * Functional tests
* `deps`: The directory containing Python requirements for the development of this repository.
* `src`: The directory containing the source code that implements the related GitHub actions.
* `tests`: The directory containing the code source and resources to test the implementation. 
  It includes the following:
    * `datasets`: The directory containing datasets used by the tests.
    * `deployments`: The directory containing a deployment definition that is used by tests.
    * `functional`: The directory containing the functional tests' source code.
    * `models`: The directory containing the model definition and source code used by the tests.
    * `unit`: The directory containing the unit-test source code.

#### Functional Tests

Functional tests are written on top of the main entry point, simulating the GitHub actions execution. 
To enable communication with DataRobot, you must set two important environment variables:

  * `DATAROBOT_WEBSERVER`: The DataRobot web server URL, which can be accessed publicly.
  * `DATAROBOT_API_TOKEN`: The API key used to validate credentials with the DataRobot system.

In the current repository, there is a definition of one model under `tests/models/py3_sklearn/`
and one deployment under `tests/deployments` used by the functional test.

#### Development Workflow

Changes in this repository should be submitted as pull requests. When a pull request is
created, the associated GitHub workflow is triggered, and the following jobs are executed
sequentially:

* Linter
* Code style checks
* Unit-tests.
* Functional test(s).

> **Note**: To enable the full execution of the functional test, the two related
> variables (`DATAROBOT_WEBSERVER` and `DATAROBOT_API_TOKEN`) were set in the
> **Secrets** section of the GitHub repository. These are read by the workflow, which 
> sets the proper environment variables.

### Metadata Definition Examples

#### Model Examples <a id="model-examples"/>

<details><summary>A Minimal Single Model Definition</summary>

Below is an example of a minimal model's definition, which includes only mandatory fields:

```yaml
user_provided_model_id: user/any-model-unique-id-1
target_type: Regression
settings:
  name: My Awsome GitHub Model 1 [GitHub CI/CD]
  target_name: Grade 2014

version:
  # Make sure this is the environment ID is in your system.
  # This one is the '[DataRobot] Python 3 Scikit-Learn Drop-In' environment
  model_environment_id: 5e8c889607389fe0f466c72d
```

</details>

<details><summary>Full Single Model Definition</summary>

Below is an example of a full model's definition, which includes both mandatory and optional fields
(for the full scheme please refer to `MODEL_SCHEMA` in [here](src/schema_validator.py)):

```yaml
user_provided_model_id: user/any-model-unique-id-1
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
  memory: 256Mi
  replicas: 2
  egress_network_policy: NONE
  model_replacement_reason: DATA_DRIFT

test:
  memory: 256Mi
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

> **Note**: The patterns used in the `exclude_glob_pattern` & `include_glob_pattern` fields are an
> extension to the common glob rules. A path that ends with `/` (slash), which means a directory,
> will automatically be regarded as suffixed with `**`. This means that the directory will be
> scanned recursively.

</details>

<details><summary>Multi Models Definition</summary>

Below is an example of a multi-models definition, which includes only mandatory fields:

```yaml
datarobot_models:
  - model_path: ./models/model_1
    model_metadata:
      user_provided_model_id: user/any-model-unique-id-1
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
      user_provided_model_id: user/any-model-unique-string-2
      target_type: Regression
      settings:
        name: My Awsome GitHub Model 2 [GitHub CI/CD]
        target_name: Grade 2014

      version:
        # Make sure this is the environment ID is in your system.
        # This one is the '[DataRobot] Python 3 Scikit-Learn Drop-In' environment
        model_environment_id: 5e8c889607389fe0f466c72d
```

</details>

#### Deployment Examples <a id="deployment-examples"/>

<details><summary>Minimal Single Deployment Definition</summary>

Below is an example of a minimal deployment's definition, which includes only mandatory fields:

```yaml
user_provided_deployment_id: user/my-awesome-deployment-id
user_provided_model_id: user/any-model-unique-id-1
```

</details>

<details><summary>Full Single Deployment Definition</summary>

Below is an example of a full deployment's definition, which includes both mandatory and optional
fields (for the full schema please refer to `DEPLOYMENT_SCHEMA` in [here](src/schema_validator.py):

```yaml
user_provided_deployment_id: user/my-awesome-deployment-id
user_provided_model_id: user/any-model-unique-string-2
prediction_environment_name: "https://eks-test.orm.company.com"
settings:
  label: "My Awesome Deployment (model-2)"
  description: "This is a more detailed description."
  importance: LOW  # NOTE: a higher importance value than "LOW" will trigger a review process
                   # for any operation, such as 'create', 'update', 'delete', etc. So, practically
                   # the user will need to wait for approval from a reviewer in order to be able
                   # to apply new changes and merge them to the main branch.
  association:
    association_id_column: id
    required_in_pred_request: true
    actual_values_column: Animal
    actuals_dataset_id: 6d8c889607389fe0f466c72e
  enable_target_drift: true
  enable_feature_drift: true
  enable_predictions_collection: true
  enable_challenger_models: true
  segment_analysis:
    enabled: true
    # NOTE: the 'segment_analysis' may contain an 'attributes' section, where users can specify
    # attributes that are categorical features in the associated model.
    # Be aware that if you enabled segment analysis, without specifying attribute, you can still
    # access various statistics by segment of built-in attributes in DataRobot.
    #
    # attributes:
    # - <categorical-attr-1>
    # - <categorical-attr-2>
```

</details>

<details><summary>Multi Deployments Definition</summary>

Below is an example of a multi-deployments definition, which includes only mandatory fields:

```yaml
- user_provided_deployment_id: user/any-deployment-unique-id-1
  user_provided_model_id: user/any-model-unique-id-1

- user_provided_deployment_id: user/any-deployment-unique-id-2
  user_provided_model_id: user/any-model-unique-string-2

- user_provided_deployment_id: user/any-deployment-unique-id-3
  user_provided_model_id: user/any-model-unique-id-3
```

</details>

#### GitHub Workflow Example <a id="workflow-example"/>

This is an example of a GitHub workflow definition. The YAML file should be located 
at the following location: `.github/workflows/workflow.yaml`. 

The YAML file should contain the following:

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
  datarobot-custom-models:
    # Run this job on any action of a PR, but skip the job upon merging to the main branch. This
    # will be taken care of by the push event.
    if: ${{ github.event.pull_request.merged != true }}

    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0

      - name: DataRobot Custom Models Step
        id: datarobot-custom-models-step
        uses: datarobot-oss/custom-models-action@v1.6.0
        with:
          api-token: ${{ secrets.DATAROBOT_API_TOKEN }}
          webserver: ${{ secrets.DATAROBOT_WEBSERVER }}
          branch: master
          allow-model-deletion: true
          allow-deployment-deletion: true

      - name: DataRobot Custom Models Action Metrics
        run: |
          echo "Total affected models: ${{ steps.datarobot-custom-models-step.outputs.models--total-affected }}"
          echo "Total created models: ${{ steps.datarobot-custom-models-step.outputs.models--total-created }}"
          echo "Total deleted models: ${{ steps.datarobot-custom-models-step.outputs.models--total-deleted }}"
          echo "Total models whose settings were updated: ${{ steps.datarobot-custom-models-step.outputs.models--total-updated-settings }}"
          echo "Total created model versions: ${{ steps.datarobot-custom-models-step.outputs.models--total-created-versions }}"

          echo "Total affected deployments: ${{ steps.datarobot-custom-models-step.outputs.deployments--total-affected }}"
          echo "Total created deployments: ${{ steps.datarobot-custom-models-step.outputs.deployments--total-created }}"
          echo "Total deleted deployments: ${{ steps.datarobot-custom-models-step.outputs.deployments--total-deleted }}"
          echo "Total deployments whose settings were updated: ${{ steps.datarobot-custom-models-step.outputs.deployments--total-updated-settings }}"

          echo "Message: ${{ steps.datarobot-custom-models-step.outputs.message }}"
```

## Copyright and License

Custom Models GitHub Action is Copyright 2022 DataRobot, Inc.  All rights reserved.
Licensed under a Modified 3-Clause BSD License (the "License").  See the LICENSE file. You may not
use this software except in compliance with the License.

Software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT ANY EXPRESS OR
IMPLIED WARRANTIES OF ANY KIND AND WITHOUT ANY LICENSE TO ANY PATENTS OR TRADEMARKS. See the
LICENSE file for the specific language governing permissions and limitations under the License.
