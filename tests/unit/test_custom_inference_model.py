from argparse import Namespace

from bson import ObjectId
import os
from pathlib import Path
from mock import patch
import pytest
import uuid
from tempfile import TemporaryDirectory

from mock.mock import PropertyMock, Mock

from custom_inference_model import CustomInferenceModel
from custom_inference_model import ModelInfo
from exceptions import ModelMainEntryPointNotFound, SharedAndLocalPathCollision
from schema_validator import ModelSchema


class TestCustomInferenceModel:
    @pytest.fixture
    def no_models(self, common_path):
        yield common_path

    @pytest.mark.usefixtures("no_models")
    def test_scan_and_load_no_models(self, options):
        custom_inference_model = CustomInferenceModel(options)
        custom_inference_model._scan_and_load_datarobot_models_metadata()
        assert len(custom_inference_model._models_info) == 0

    @pytest.mark.parametrize("num_models", [1, 2, 3])
    def test_scan_and_load_models_from_multi_separate_yaml_files(
        self, options, single_model_factory, num_models
    ):
        for counter in range(1, num_models + 1):
            single_model_factory(f"model-{counter}", write_metadata=True)
        custom_inference_model = CustomInferenceModel(options)
        custom_inference_model._scan_and_load_datarobot_models_metadata()
        assert len(custom_inference_model._models_info) == num_models

    @pytest.mark.parametrize("num_models", [0, 1, 3])
    def test_scan_and_load_models_from_one_multi_models_yaml_file(
        self, options, models_factory, num_models
    ):
        models_factory(num_models, is_multi=True)
        custom_inference_model = CustomInferenceModel(options)
        custom_inference_model._scan_and_load_datarobot_models_metadata()
        assert len(custom_inference_model._models_info) == num_models

    @pytest.mark.parametrize("num_single_models", [0, 1, 3])
    @pytest.mark.parametrize("num_multi_models", [0, 2])
    def test_scan_and_load_models_from_both_multi_and_single_yaml_files(
        self,
        options,
        num_single_models,
        num_multi_models,
        single_model_factory,
        models_factory,
    ):
        models_factory(num_multi_models, is_multi=True)
        for counter in range(1, num_single_models + 1):
            single_model_factory(f"model-{counter}")

        custom_inference_model = CustomInferenceModel(options)
        custom_inference_model._scan_and_load_datarobot_models_metadata()
        assert len(custom_inference_model._models_info) == (
            num_multi_models + num_single_models
        )


class TestGlobPatterns:
    @pytest.mark.parametrize("num_models", [1, 2, 3])
    @pytest.mark.parametrize("is_multi", [True, False], ids=["multi", "single"])
    @pytest.mark.parametrize(
        "with_include_glob", [True, False], ids=["with-include-glob", "without-include-glob"]
    )
    @pytest.mark.parametrize(
        "with_exclude_glob", [True, False], ids=["with-exclude-glob", "without-exclude-glob"]
    )
    def test_glob_patterns(
        self,
        models_factory,
        common_path,
        excluded_src_path,
        options,
        num_models,
        is_multi,
        with_include_glob,
        with_exclude_glob,
    ):
        models_factory(num_models, is_multi, with_include_glob, with_exclude_glob)
        custom_inference_model = CustomInferenceModel(options)
        custom_inference_model.run()

        assert len(custom_inference_model.models_info) == num_models

        for index in range(num_models):
            model_path = custom_inference_model.models_info[index].model_path
            readme_file_path = model_path / "README.md"
            assert readme_file_path.is_file()

            model_file_paths = custom_inference_model.models_info[index].model_file_paths
            assert excluded_src_path not in model_file_paths

            if with_include_glob:
                assert common_path in model_file_paths
            else:
                assert common_path not in model_file_paths

            if with_exclude_glob:
                assert readme_file_path.absolute() not in model_file_paths
            else:
                assert readme_file_path.absolute() in model_file_paths

    @pytest.mark.parametrize(
        "included_paths, excluded_paths, expected_num_model_files",
        [
            ({"./custom.py", "./bbb.py", "score/bbb.md"}, {"bbb.py"}, 2),
            ({"./custom.py", ".//bbb.py", "score/bbb.md"}, {"bbb.py"}, 2),
            ({"./custom.py", "bbb.py", "score/bbb.py"}, {"bbb.py"}, 2),
            ({"./custom.py", "bbb.py", "score/bbb.py"}, {"bbb.sh"}, 3),
            ({"./custom.py", "bbb.py", "score/./bbb.py"}, {"score/bbb.py"}, 2),
            ({"./custom.py", "bbb.py", "score//bbb.py"}, {"score/bbb.py"}, 2),
            ({"./custom.py", ".//bbb.py", "score/./bbb.py"}, {"score/bbb.py"}, 2),
            ({"./custom.py", "score/../bbb.py"}, {"score/bbb.py"}, 2),
            ({"./custom.py", "score/../bbb.py"}, {"bbb.py"}, 2),
            ({"./custom.py", "//score/bbb.py"}, {"/score/bbb.py"}, 1),
            ({"./custom.py", "//score/./bbb.py"}, {"/score/bbb.py"}, 1),
        ]
    )
    def test_filtered_model_paths(self, included_paths, excluded_paths, expected_num_model_files):
        model_info = ModelInfo("yaml-path", "model-path", None)
        CustomInferenceModel._set_filtered_model_paths(model_info, included_paths, excluded_paths)
        assert len(model_info.model_file_paths) == expected_num_model_files
        for excluded_path in excluded_paths:
            assert Path(excluded_path) not in model_info.model_file_paths

    @pytest.mark.parametrize("is_multi", [True, False], ids=["multi", "single"])
    def test_missing_main_program(self, models_factory, common_path, options, is_multi):
        models_factory(1, is_multi, include_main_prog=False)
        custom_inference_model = CustomInferenceModel(options)
        with pytest.raises(ModelMainEntryPointNotFound):
            custom_inference_model.run()

    @pytest.mark.parametrize(
        "local_paths, shared_paths, collision_expected",
        [
            (["/repo/model/custom.py", "/repo/model/util.py"], ["/repo/util.py"], True),
            (
                ["/repo/model/custom.py", "/repo/model/common/util.py"],
                ["/repo/common/util.py"],
                True,
            ),
            (["/repo/model/common/convert.py"], ["/repo/common/util.py"], True),
            (["/repo/model/common/util/convert.py"], ["/repo/common/util.py"], True),
            (
                ["/repo/model/custom.py", "/repo/model/score.py"],
                ["/repo/common/util.py", "/repo/common/common.py"],
                False,
            ),
        ],
        ids=[
            "file-collision",
            "package-collision1",
            "package-collision2",
            "package-collision3",
            "no-collision",
        ]
    )
    def test_local_and_shared_collisions(self, local_paths, shared_paths, collision_expected):
        options = Namespace(root_dir="/repo")
        with patch.object(
            CustomInferenceModel, "models_info", new_callable=PropertyMock
        ) as mock_models_info_property, patch.object(
            ModelInfo, "model_file_paths", new_callable=PropertyMock
        ) as mock_model_file_paths_property:
            model_path = Path("/repo/model")
            model_info = ModelInfo("yaml-path", model_path, None)
            mock_models_info_property.return_value = model_info
            mock_model_file_paths_property.return_value = (
                [Path(p) for p in local_paths] + [Path(p) for p in shared_paths]
            )
            custom_inference_model = CustomInferenceModel(options)
            if collision_expected:
                with pytest.raises(SharedAndLocalPathCollision):
                    custom_inference_model._validate_collision_between_local_and_shared(model_info)
            else:
                custom_inference_model._validate_collision_between_local_and_shared(model_info)
