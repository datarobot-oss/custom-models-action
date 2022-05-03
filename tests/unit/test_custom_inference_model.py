from argparse import Namespace
from bson import ObjectId
import os
from pathlib import Path
import pytest
import uuid
from tempfile import TemporaryDirectory
import yaml

from custom_inference_model import CustomInferenceModel
from schema_validator import ModelSchema


class TestCustomInferenceModel:
    @staticmethod
    def _single_model_metadata():
        return {
            ModelSchema.MODEL_ID_KEY: str(uuid.uuid4()),
            "target_type": "Regression",
            "target_name": "Grade 2014",
            "version": {"model_environment": str(ObjectId())},
        }

    @pytest.fixture
    def models_root(self):
        with TemporaryDirectory() as models_root:
            os.makedirs(Path(models_root) / "dir-1" / "dir-11")
            os.makedirs(Path(models_root) / "dir-2")
            yield Path(models_root)

    @pytest.fixture
    def no_models(self, models_root):
        yield models_root

    @pytest.fixture
    def common_file_path(self, models_root):
        common_dir = models_root / "common"
        os.mkdir(common_dir)
        common_filename = "common.py"
        common_file_path = common_dir / common_filename
        with open(common_file_path, "w") as f:
            f.write("# common.py")
        return common_file_path.relative_to(models_root)

    @pytest.fixture
    def common_module(self, common_file_path):
        module_name = common_file_path.with_suffix("")
        module_name = str(module_name).replace("/", ".")
        return module_name

    @pytest.fixture
    def single_model_factory(self, models_root, common_module):
        def _inner(model_name):
            model_path = models_root / model_name
            os.makedirs(model_path)
            with open(model_path / "custom.py", "w") as f:
                f.write(f"import {common_module}")
            with open(model_path / "non-datarobot-model.yaml", "w") as f:
                f.write("model_id: 1234")

            with open(model_path / "model.yaml", "w") as f:
                f.write(yaml.dump(self._single_model_metadata()))

            return model_path.relative_to(models_root)

        return _inner

    @pytest.fixture
    def multi_models_in_a_single_yaml_factory(
        self, models_root, common_file_path, common_module
    ):
        def _inner(num_models):
            # Multiple models in a single metadata
            multi_models_yaml_content = {ModelSchema.MULTI_MODELS_KEY: []}
            for index in range(1, 1 + num_models):
                model_name = f"model-from-multi-def-{index}"
                model_path = models_root / model_name
                os.mkdir(model_path)
                with open(model_path / "custom.py", "w") as f:
                    f.write(f"import {common_module}")
                with open(model_path / "README.md", "w") as f:
                    f.write(f"# README (to be excluded)")
                with open(model_path / "non-datarobot-model.yml", "w") as f:
                    f.write(f"models: []")

                single_model_metadata = self._single_model_metadata()
                # noinspection PyTypeChecker
                single_model_metadata["version"]["include_glob_pattern"] = [
                    f"/{model_path.relative_to(models_root)}/**",
                    f"/{common_file_path.parent}/**",
                ]
                # noinspection PyTypeChecker
                single_model_metadata["version"]["exclude_glob_pattern"] = [
                    f"/{model_path.relative_to(models_root)}/README.md"
                ]
                multi_models_yaml_content[ModelSchema.MULTI_MODELS_KEY].append(
                    single_model_metadata
                )

            with open(models_root / "models.yml", "w") as f:
                f.write(yaml.dump(multi_models_yaml_content))

        return _inner

    @pytest.fixture
    def options(self, models_root):
        return Namespace(root_dir=models_root.absolute())

    @pytest.mark.usefixtures("no_models")
    def test_scan_and_load_no_models(self, options):
        custom_inference_model = CustomInferenceModel(options)
        custom_inference_model._scan_and_load_datarobot_models_metadata()
        assert len(custom_inference_model._models_metadata) == 0

    @pytest.mark.parametrize("num_models", [1, 2, 3])
    def test_scan_and_load_models_from_multi_separate_yaml_files(
        self, options, single_model_factory, num_models
    ):
        for counter in range(1, num_models + 1):
            single_model_factory(f"model-{counter}")
        custom_inference_model = CustomInferenceModel(options)
        custom_inference_model._scan_and_load_datarobot_models_metadata()
        assert len(custom_inference_model._models_metadata) == num_models

    @pytest.mark.parametrize("num_models", [0, 1, 3])
    def test_scan_and_load_models_from_one_multi_models_yaml_file(
        self, options, multi_models_in_a_single_yaml_factory, num_models
    ):
        multi_models_in_a_single_yaml_factory(num_models)
        custom_inference_model = CustomInferenceModel(options)
        custom_inference_model._scan_and_load_datarobot_models_metadata()
        assert len(custom_inference_model._models_metadata) == num_models

    @pytest.mark.parametrize("num_single_models", [0, 1, 3])
    @pytest.mark.parametrize("num_multi_models", [0, 2])
    def test_scan_and_load_models_from_both_multi_and_single_yaml_files(
        self,
        options,
        num_single_models,
        num_multi_models,
        single_model_factory,
        multi_models_in_a_single_yaml_factory,
    ):
        multi_models_in_a_single_yaml_factory(num_multi_models)
        for counter in range(1, num_single_models + 1):
            single_model_factory(f"model-{counter}")

        custom_inference_model = CustomInferenceModel(options)
        custom_inference_model._scan_and_load_datarobot_models_metadata()
        assert len(custom_inference_model._models_metadata) == (
            num_multi_models + num_single_models
        )
