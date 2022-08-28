#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

"""
A super simple example of custom model body with a single 'transform' hook.
"""

from common import conversion


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
    # Remove target columns if they're in the dataset
    for target_col in ["Grade 2014", "Species"]:
        if target_col in data:
            data.pop(target_col)
    data = data.fillna(0)

    inches = 10
    centimeters = conversion.inch_to_cm(inches)
    print(f"Inches: {inches}, Centimeters: {centimeters}")

    return data
