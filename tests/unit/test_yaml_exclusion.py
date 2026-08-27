#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

"""Unit tests for YAML file exclusion functionality."""

import os
from pathlib import Path

import pytest
import yaml

from model_controller import ModelController


@pytest.mark.parametrize(
    "exclude_pattern,expected_files",
    [
        (None, ["model1.yaml", "model2.yaml"]),  # No exclude pattern
        ("test/", ["model1.yaml"]),  # Simple exclude
        (".*test.*\.yaml$", ["model1.yaml"]),  # Complex regex
        ("", ["model1.yaml", "model2.yaml"]),  # Empty pattern
        ("model[12]\\.yaml", []),  # All models
    ],
)
def test_yaml_file_exclusion(options, exclude_pattern, expected_files, workspace_path):
    """Test that YAML files are properly excluded based on the exclude pattern."""
    options.exclude = exclude_pattern
    options.workspace_path = workspace_path  # Set workspace path in options
    
    # Create test files with valid YAML content
    model1_content = {
        "user_provided_model_id": "test/model1",
        "target_type": "Regression",
        "settings": {
            "name": "Test Model 1",
            "target_name": "target"
        },
        "version": {
            "model_environment_id": "5e8c889607389fe0f466c72d"
        }
    }
    
    model2_content = {
        "user_provided_model_id": "test/model2",
        "target_type": "Regression",
        "settings": {
            "name": "Test Model 2",
            "target_name": "target"
        },
        "version": {
            "model_environment_id": "5e8c889607389fe0f466c72d"
        }
    }
    
    # Create directories and write YAML files
    os.makedirs(workspace_path / "test", exist_ok=True)
    with open(workspace_path / "model1.yaml", "w") as f:
        yaml.dump(model1_content, f)
    with open(workspace_path / "test/model2.yaml", "w") as f:
        yaml.dump(model2_content, f)
    
    controller = ModelController(options, None)
    
    # Collect processed files
    processed_files = [path for path, _ in controller._next_yaml_content_in_repo()]
    processed_files = [Path(p).name for p in processed_files]
    
    assert set(processed_files) == set(expected_files) 
