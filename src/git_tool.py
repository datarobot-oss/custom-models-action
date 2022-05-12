from git import Repo
from pathlib import Path


class GitTool:
    repo = None

    def __init__(self, git_repo_path):
        self._repo = Repo.init(git_repo_path)
        self._repo_path = Path(git_repo_path)

    @property
    def repo(self):
        return self._repo

    @property
    def repo_path(self):
        return self._repo_path

    def num_commits(self, ref="HEAD"):
        try:
            return int(self.repo.git.rev_list("--count", ref))
        except TypeError:
            return 0

    def find_changed_files(self, to_commit_sha, from_commit_sha=None):
        to_commit = self.repo.commit(to_commit_sha)
        if from_commit_sha:
            diff = to_commit.diff(from_commit_sha)

            changed_files = []
            for git_index in diff:
                changed_files.append(self.repo_path / git_index.a_path)
        else:
            changed_files = [self.repo_path / p for p in to_commit.stats.files.keys()]

        return changed_files

    def is_parent_of(self, parent_commit_sha, child_commit_sha):
        parent_sha1 = self.repo.commit(child_commit_sha).parents[0].hexsha
        parent_sha2 = self.repo.commit(parent_commit_sha).hexsha
        return parent_sha1 == parent_sha2
