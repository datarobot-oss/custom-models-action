#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

# pylint: disable=protected-access

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
        (r".*test.*\.yaml$", ["model1.yaml"]),  # Complex regex
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
        "settings": {"name": "Test Model 1", "target_name": "target"},
        "version": {"model_environment_id": "5e8c889607389fe0f466c72d"},
    }

    model2_content = {
        "user_provided_model_id": "test/model2",
        "target_type": "Regression",
        "settings": {"name": "Test Model 2", "target_name": "target"},
        "version": {"model_environment_id": "5e8c889607389fe0f466c72d"},
    }

    # Create directories and write YAML files
    os.makedirs(workspace_path / "test", exist_ok=True)
    with open(workspace_path / "model1.yaml", "w", encoding="utf-8") as f:
        yaml.dump(model1_content, f)
    with open(workspace_path / "test/model2.yaml", "w", encoding="utf-8") as f:
        yaml.dump(model2_content, f)

    controller = ModelController(options, None)

    # Collect processed files
    processed_files = [path for path, _ in controller._next_yaml_content_in_repo()]
    processed_files = [Path(p).name for p in processed_files]

    assert set(processed_files) == set(expected_files)


def test_yaml_scan_reports_a_clear_error_for_unparsable_file(options, workspace_path):
    """A repo may contain .yaml/.yml files that were never meant to be scanned as model
    metadata - e.g. a Helm chart template using Go-template syntax (`{{- ... -}}`), which
    isn't valid standalone YAML. The scan must still fail (not silently skip: a silently
    skipped file's user_provided_id is untracked, so handle_deleted_models would treat a
    previously-synced model whose metadata merely has a typo as "deleted from the repo" -
    risking an actual deletion when allow_model_deletion is set). But the failure must be a
    clear, actionable error pointing at the file and at --exclude, not a bare traceback."""
    options.exclude = None
    options.workspace_path = workspace_path

    model_content = {
        "user_provided_model_id": "test/model1",
        "target_type": "Regression",
        "settings": {"name": "Test Model 1", "target_name": "target"},
        "version": {"model_environment_id": "5e8c889607389fe0f466c72d"},
    }
    with open(workspace_path / "model1.yaml", "w", encoding="utf-8") as f:
        yaml.dump(model_content, f)

    os.makedirs(workspace_path / "chart/templates", exist_ok=True)
    with open(workspace_path / "chart/templates/job.yaml", "w", encoding="utf-8") as f:
        f.write('{{- include "base-chart.baseJob" (list . "install-job") -}}\nname: install-job\n')

    controller = ModelController(options, None)

    with pytest.raises(yaml.YAMLError, match=r"(?s)job\.yaml.*--exclude"):
        list(controller._next_yaml_content_in_repo())


def test_yaml_scan_excluded_chart_file_does_not_raise(options, workspace_path):
    """The same unparsable file, once covered by --exclude, must be skipped cleanly - this
    is the actual fix for a repo like global-envs-models that ships its own Helm chart."""
    options.exclude = r"(^|/)chart/"
    options.workspace_path = workspace_path

    model_content = {
        "user_provided_model_id": "test/model1",
        "target_type": "Regression",
        "settings": {"name": "Test Model 1", "target_name": "target"},
        "version": {"model_environment_id": "5e8c889607389fe0f466c72d"},
    }
    with open(workspace_path / "model1.yaml", "w", encoding="utf-8") as f:
        yaml.dump(model_content, f)

    os.makedirs(workspace_path / "chart/templates", exist_ok=True)
    with open(workspace_path / "chart/templates/job.yaml", "w", encoding="utf-8") as f:
        f.write('{{- include "base-chart.baseJob" (list . "install-job") -}}\nname: install-job\n')

    controller = ModelController(options, None)
    processed_files = [Path(path).name for path, _ in controller._next_yaml_content_in_repo()]

    assert processed_files == ["model1.yaml"]
