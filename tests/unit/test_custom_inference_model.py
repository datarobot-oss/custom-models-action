from argparse import Namespace
from bson import ObjectId
import os
from pathlib import Path
import pytest
import uuid
from tempfile import TemporaryDirectory
import yaml

from custom_inference_model import CustomInferenceModel


class TestCustomInferenceModel:
    @staticmethod
    def _single_model_metadata():
        return {
            "model_git_id": str(uuid.uuid4()),
            "target_type": "Regression",
            "target_name": "Grade 2014",
            "version": {"model_environment": str(ObjectId())},
        }

    @pytest.fixture(scope="session")
    def models_root(self):
        with TemporaryDirectory() as models_root:
            yield Path(models_root)

    @pytest.fixture(scope="session")
    def common_file_path(self, models_root):
        common_dir = models_root / "common"
        os.mkdir(common_dir)
        common_filename = "common.py"
        common_file_path = common_dir / common_filename
        with open(common_file_path, "w") as f:
            f.write("# common.py")
        return common_file_path.relative_to(models_root)

    @pytest.fixture(scope="session")
    def common_module(self, common_file_path):
        module_name = common_file_path.with_suffix("")
        module_name = str(module_name).replace("/", ".")
        return module_name

    @pytest.fixture
    def single_model(self, models_root, common_module):
        model_path = models_root / "single-model"
        os.makedirs(model_path)
        with open(model_path / "custom.py", "w") as f:
            f.write(f"import {common_module}")

        with open(model_path / "model.yaml", "w") as f:
            f.write(yaml.dump(self._single_model_metadata()))

        return model_path.relative_to(models_root)

    @pytest.fixture
    def two_models_in_a_single_yaml(self, models_root, common_file_path, common_module):
        # Multiple models in a single metadata
        multi_models_yaml_content = {"models": []}
        for model in ["model-1", "model-2"]:
            model_path = models_root / model
            os.mkdir(model_path)
            open(model_path / "custom.py", "w").write(f"import {common_module}")
            open(model_path / "README.md", "w").write(f"# README (to be excluded)")

            single_model_metadata = self._single_model_metadata()
            single_model_metadata["include_glob_pattern"] = [
                f"/{model_path.relative_to(models_root)}/**",
                f"/{common_file_path.parent}/**",
            ]
            single_model_metadata["exclude_glob_pattern"] = [
                f"/{model_path.relative_to(models_root)}/README.md"
            ]
            multi_models_yaml_content["models"].append(single_model_metadata)

        with open(models_root / "models.yml", "w") as f:
            f.write(yaml.dump(multi_models_yaml_content))

    @pytest.fixture(scope="session")
    def options(self, models_root):
        return Namespace(root_dir=models_root.absolute())

    @pytest.mark.usefixtures("single_model")
    @pytest.mark.usefixtures("two_models_in_a_single_yaml")
    def test_scan_and_load_model_metadata(self, options):
        custom_inference_model = CustomInferenceModel(options)
        custom_inference_model._scan_and_load_models_metadata()
        assert len(custom_inference_model._models_metadata) == 3
