#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

from typing import Any
from typing import Dict

import pandas as pd

"""
A super simple example of custom model body with a single 'transform' hook.
"""


def load_model(code_dir: str) -> Any:
    """
    Can be used to load supported models if your model has multiple artifacts, or for loading
    models that **drum** does not natively support

    Parameters
    ----------
    code_dir : is the directory where model artifact and additional code are provided, passed in

    Returns
    -------
    If used, this hook must return a non-None value
    """
    return "dummy"


def transform(data, model):
    """
    Note: This hook may not have to be implemented for your model.
    In this case implemented for the model used in the example.

    Modify this method to add data transformation before scoring calls. For example, this can be
    used to implement one-hot encoding for models that don't include it on their own.

    Parameters
    ----------
    data: pd.DataFrame
    model: object, the deserialized model

    Returns
    -------
    pd.DataFrame
    """
    # Execute any steps you need to do before scoring
    # Remove target and id columns if they're in the dataset
    for unused_column in ["Grade 2014", "Species", "id"]:
        if unused_column in data:
            data.pop(unused_column)
    data = data.fillna(0)
    return data


def score(data: pd.DataFrame, model: Any, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    """
    This hook is only needed if you would like to use **drum** with a framework not natively
    supported by the tool.

    Note: While best practice is to include the score hook, if the score hook is not present
    DataRobot will add a score hook and call the default predict method for the library
    See https://github.com/datarobot/datarobot-user-models#built-in-model-support for details

    This dummy implementation returns a dataframe with all rows having value 42 in the
    "Predictions" column, regardless of the provided input dataset.

    Parameters
    ----------
    data : is the dataframe to make predictions against. If `transform` is supplied,
    `data` will be the transformed data.
    model : is the deserialized model loaded by **drum** or by `load_model`, if supplied
    kwargs : additional keyword arguments to the method

    Returns
    -------
    This method should return predictions as a dataframe with the following format:
      Regression: must have a single column called `Predictions` with numerical values
    """
    preds = pd.DataFrame([42 for _ in range(data.shape[0])], columns=["Predictions"])
    return preds
