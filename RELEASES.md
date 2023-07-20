# RELEASES

## 1.6.0 [2023-07-20]
* Add support for training/holdout dataset assignment as model version level
* Internal improvements
* Bug fixes

## 1.5.0 [2023-07-16]
* Add support for LLM (text generation) custom models
* Add support for egress network policy for a custom model
* Bug fix

## 1.4.2 [2023-06-06]
* Bug fix

## 1.4.1 [2023-06-01]
* Bug fixes

## 1.4.0 [2023-05-11]
  * Enable users to specify a reason for a model replacement in a deployment
  * Separately track model's settings from its versions

## 1.3.0 [2023-04-27]
  * Fetch and print deployment's log in case of a deployment's creation failure
  * Add a missing mapping of Anomaly target type
  * Internal improvements
  * Bug fixes

## 1.2.2 [2023-03-21]
  * Improve segment analysis functionality and update README.md
  * Bug fix

## 1.2.1 [2023-03-12]
  * Fix examples in the README.md to comply with the defaults in DataRobot
  * Internal improvement

## 1.2.0 [2023-02-20]
  * Add support for namespaces
  * Add a new metric to count models and deployment whose settings were changed
  * Internal improvements

## 1.1.8 [2023-01-31]
  * Internal improvements
  * Bug fixes

## 1.1.7 [2023-01-11]
  * Update README.md
  * Internal improvements for local utility
  * Bug fixes

## 1.1.6 [2023-01-05]
  * Change attribute names in a deployment schema:
    * `prediction_id` ==> `association_id_column`
    * `actuals_id` ==> `actual_values_column`
  * Update README.md regarding `importance` attribute in a deployment.
  * Update README.md regarding the user's provided ID.
  * Bug fixes.

## 1.1.5 [2022-12-08]
  * Improve documentation in the README.md file

## 1.1.4 [2022-12-05]
  * Internal improvements to the development environment
  * Bug fixes

## 1.1.3 [2022-11-23]
  * Add a 'Quick Start' section to the README.md.

## 1.1.2 [2022-11-23]
  * Update the action's name and add branding section.

## 1.1.1 [2022-11-22]
  * Update the README.md about referencing specific releases.

## 1.1.0 [2022-11-13]
  * Add support for custom model versions' dependencies.
  * Add a script to create a release.
  * Bug fixes

## 1.0.2 [2022-11-07]
  * Bug fix

## 1.0.1 [2022-11-06]
  * Update the way statistics are emitted by the action.
  * Functional tests are executed against DataRobot US production system.

## 1.0.0 [2022-09-13]
  * The first release.
