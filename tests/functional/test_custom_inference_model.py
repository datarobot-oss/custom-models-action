from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
import contextlib
import logging
import os
import shutil

from bson import ObjectId
from git import Repo
import pytest
import yaml

from common.convertors import MemoryConvertor
from common.exceptions import DataRobotClientError
from dr_client import DrClient
from main import main
from schema_validator import ModelSchema


def webserver_accessible():
    webserver = os.environ.get("DATAROBOT_WEBSERVER")
    api_token = os.environ.get("DATAROBOT_API_TOKEN")
    if webserver and api_token:
        return DrClient(webserver, api_token).is_accessible()
    return False


@contextlib.contextmanager
def env_set(env_key, env_value):
    does_exist = env_key in os.environ
    if does_exist:
        old_value = os.environ[env_key]
    os.environ[env_key] = env_value
    yield
    if does_exist:
        os.environ[env_key] = old_value


@pytest.fixture
def repo_root_path():
    with TemporaryDirectory() as repo_tree:
        path = Path(repo_tree)
        yield path


@pytest.fixture
def git_repo(repo_root_path):
    repo = Repo.init(repo_root_path)
    repo.config_writer().set_value("user", "name", "functional-test-user").release()
    repo.config_writer().set_value("user", "email", "functional-test@company.com").release()
    logging.basicConfig()
    logging.root.setLevel(logging.INFO)
    type(repo.git).GIT_PYTHON_TRACE = "full"
    return repo


@pytest.fixture
def build_repo_for_testing(repo_root_path, git_repo):
    # 1. Copy models from source tree
    models_src_root_dir = Path(__file__).parent / ".." / "models"
    shutil.copytree(models_src_root_dir, repo_root_path / models_src_root_dir.name)

    # 2. Add files to repo
    os.chdir(repo_root_path)
    git_repo.git.add("--all")
    git_repo.git.commit("-m", "Initial commit", "--no-verify")


@pytest.fixture
@pytest.mark.usefixtures("build_repo_for_testing")
def model_metadata_yaml_file(repo_root_path, git_repo):
    model_yaml_file = next(repo_root_path.rglob("**/model.yaml"))
    with open(model_yaml_file) as f:
        yaml_content = yaml.safe_load(f)
        yaml_content[ModelSchema.MODEL_ID_KEY] = f"my-awesome-model-{str(ObjectId())}"

    with open(model_yaml_file, "w") as f:
        yaml.safe_dump(yaml_content, f)

    git_repo.git.add(model_yaml_file)
    git_repo.git.commit("--amend", "--no-edit")

    return model_yaml_file


@pytest.fixture
def model_metadata(model_metadata_yaml_file):
    with open(model_metadata_yaml_file) as f:
        return yaml.safe_load(f)


@pytest.fixture
def main_branch_name():
    return "master"


@pytest.fixture
def feature_branch_name():
    return "feature"


@pytest.fixture
def merge_branch_name():
    return "merge-feature-branch"


@pytest.fixture
def dr_client():
    webserver = os.environ.get("DATAROBOT_WEBSERVER")
    api_token = os.environ.get("DATAROBOT_API_TOKEN")
    return DrClient(webserver, api_token)


@pytest.fixture
def cleanup(dr_client, model_metadata):
    yield

    try:
        dr_client.delete_custom_model_by_git_model_id(model_metadata[ModelSchema.MODEL_ID_KEY])
    except DataRobotClientError:
        pass


@pytest.mark.skipif(not webserver_accessible(), reason="DataRobot webserver is not accessible")
class TestCustomInferenceModel:
    class Change(Enum):
        INCREASE_MEMORY = 1
        ADD_FILE = 2
        REMOVE_FILE = 3

    @pytest.mark.usefixtures("build_repo_for_testing", "cleanup")
    def test_e2e_pull_request(
        self,
        dr_client,
        repo_root_path,
        git_repo,
        model_metadata,
        model_metadata_yaml_file,
        main_branch_name,
        feature_branch_name,
        merge_branch_name,
    ):
        files_to_add_and_remove = [
            model_metadata_yaml_file.parent / "some_new_file_1.py",
            model_metadata_yaml_file.parent / "some_new_file_2.py",
        ]
        changes = [self.Change.INCREASE_MEMORY, self.Change.ADD_FILE, self.Change.REMOVE_FILE]
        # Ensure that the `INCREASE_MEMORY` is always first
        assert changes[0] == self.Change.INCREASE_MEMORY
        # Ensure that the `REMOVE_FILE` is always last
        assert changes[-1] == self.Change.REMOVE_FILE

        # 1. Create feature branch
        feature_branch = git_repo.create_head(feature_branch_name)

        # 2. Make two changes, one at a time on a feature branch
        for change in [self.Change.INCREASE_MEMORY, self.Change.ADD_FILE, self.Change.REMOVE_FILE]:
            # 3. Checkout feature branch
            feature_branch.checkout()

            # 4. Make a change and commit it
            if change == self.Change.INCREASE_MEMORY:
                new_memory = self._increase_model_memory_by_1mb(model_metadata_yaml_file)
                git_repo.git.add(model_metadata_yaml_file)
                git_repo.git.commit("-m", f"Increase memory to {new_memory}")
            elif change == self.Change.ADD_FILE:
                for filepath in files_to_add_and_remove:
                    with open(filepath, "w") as f:
                        f.write("# New file for testing")
                    git_repo.git.add(filepath)
                git_repo.git.commit("-m", "Add new files.")
            elif change == self.Change.REMOVE_FILE:
                for filepath in files_to_add_and_remove:
                    os.remove(filepath)
                    git_repo.git.add(filepath)
                git_repo.git.commit("-m", f"Remove the files.")

            # 5. Create merge branch from master and check it out
            merge_branch = git_repo.create_head(merge_branch_name, main_branch_name)
            git_repo.head.reference = merge_branch
            git_repo.head.reset(index=True, working_tree=True)

            # 6. Merge feature branch --no-ff
            git_repo.git.merge(feature_branch, "--no-ff")

            # 7. Run GitHub pull request action
            self._run_pull_request_action(
                repo_root_path, git_repo, main_branch_name, merge_branch_name
            )

            # 8. Validation
            cm_version = dr_client.fetch_custom_model_latest_version_by_git_model_id(
                model_metadata[ModelSchema.MODEL_ID_KEY]
            )
            # Assuming `INCREASE_MEMORY` always first
            assert cm_version["maximumMemory"] == MemoryConvertor.to_bytes(new_memory)
            if change == self.Change.ADD_FILE:
                for filepath in files_to_add_and_remove:
                    assert filepath.name in [item["filePath"] for item in cm_version["items"]]
            elif change == self.Change.REMOVE_FILE:
                for filepath in files_to_add_and_remove:
                    assert filepath.name not in [item["filePath"] for item in cm_version["items"]]

            # 9. Checkout the main branch
            git_repo.heads.master.checkout()
            if change != self.Change.REMOVE_FILE:
                # 10. Delete the merge branch
                git_repo.delete_head(merge_branch, "--force")

        # 11. Merge changes from the merge branch into the main branch
        git_repo.git.merge(merge_branch, "--squash")
        git_repo.git.add("--all")
        git_repo.git.commit("-m", "Changes from merged feature branch")
        head_commit_sha = git_repo.head.commit.hexsha
        self._run_push_action(repo_root_path, git_repo, main_branch_name, head_commit_sha)

        # 12. Validation
        cm_version = dr_client.fetch_custom_model_latest_version_by_git_model_id(
            model_metadata[ModelSchema.MODEL_ID_KEY]
        )
        # Assuming 'INCREASE_MEMORY` change took place
        assert cm_version["maximumMemory"] == MemoryConvertor.to_bytes(new_memory)

    @staticmethod
    def _increase_model_memory_by_1mb(model_yaml_file):
        with open(model_yaml_file) as f:
            yaml_content = yaml.safe_load(f)
            memory = ModelSchema.get_value(
                yaml_content, ModelSchema.VERSION_KEY, ModelSchema.MEMORY_KEY
            )
            memory = memory if memory else "256Mi"
            num_part, unit = MemoryConvertor._extract_unit_fields(memory)
            new_memory = f"{num_part+1}{unit}"
            yaml_content[ModelSchema.VERSION_KEY][ModelSchema.MEMORY_KEY] = new_memory

        with open(model_yaml_file, "w") as f:
            yaml.safe_dump(yaml_content, f)

        return new_memory

    @staticmethod
    def _run_pull_request_action(repo_root_path, git_repo, main_branch_name, merge_branch_name):
        with env_set("GITHUB_EVENT_NAME", "pull_request"), env_set(
            "GITHUB_SHA", git_repo.commit(merge_branch_name).hexsha
        ), env_set("GITHUB_BASE_REF", main_branch_name):
            main(
                [
                    "--webserver",
                    os.environ.get("DATAROBOT_WEBSERVER"),
                    "--api-token",
                    os.environ.get("DATAROBOT_API_TOKEN"),
                    "--branch",
                    main_branch_name,
                    "--root-dir",
                    str(repo_root_path),
                ]
            )

    @staticmethod
    def _run_push_action(repo_root_path, git_repo, main_branch_name, main_branch_head_sha):
        with env_set("GITHUB_EVENT_NAME", "push"), env_set(
            "GITHUB_SHA", git_repo.commit(main_branch_head_sha).hexsha
        ), env_set("GITHUB_BASE_REF", main_branch_name):
            main(
                [
                    "--webserver",
                    os.environ.get("DATAROBOT_WEBSERVER"),
                    "--api-token",
                    os.environ.get("DATAROBOT_API_TOKEN"),
                    "--branch",
                    main_branch_name,
                    "--root-dir",
                    str(repo_root_path),
                ]
            )

    @pytest.mark.usefixtures("build_repo_for_testing", "cleanup")
    def test_e2e_push(self, repo_root_path, git_repo, model_metadata_yaml_file, main_branch_name):
        # 1. Make three changes, one at a time on the main branch
        for _ in range(3):
            # 2. Make a change and commit it
            new_memory = self._increase_model_memory_by_1mb(model_metadata_yaml_file)
            git_repo.git.add(model_metadata_yaml_file)
            git_repo.git.commit("-m", f"Increase memory to {new_memory}")

            # 3. Run GitHub pull request action
            head_commit_sha = git_repo.head.commit.hexsha
            self._run_push_action(repo_root_path, git_repo, main_branch_name, head_commit_sha)
