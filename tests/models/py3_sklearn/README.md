## Python Sklearn Inference Model Template


This model is intended to work with the
[Python 3 Scikit-Learn Drop-In Environment](https://github.com/datarobot/datarobot-user-models/tree/master/public_dropin_environments/python3_sklearn).

The supplied pkl file is a scikit-learn model trained on
[juniors_3_year_stats_regression.csv](../../datasets/juniors_3_year_stats_regression_structured_training_with_holdout.csv)
with a `Grade 2014` as the target (regression), though any binary or regression model trained using the libraries
outlined in [Python 3 Scikit-Learn Drop-In Environment](https://github.com/datarobot/datarobot-user-models/tree/master/public_dropin_environments/python3_sklearn) will work.

For this sample model, custom.py contains additional data pre-processing that the model itself lacks.

## Instructions
Create a new custom model with these files and use the Python Drop-In Environment with it
