import logging

from git import Repo
from pathlib import Path

from common.exceptions import NoCommonAncestor
from common.exceptions import NonMergeCommitError

logger = logging.getLogger()


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
            from_commit_sha = self.repo.commit(from_commit_sha)
            diff = from_commit_sha.diff(to_commit)

            changed_or_new_files = []
            deleted_files = []
            for git_index in diff:
                a_path = self.repo_path / git_index.a_path
                b_path = self.repo_path / git_index.b_path
                if git_index.change_type in ["M", "A"]:
                    changed_or_new_files.append(self.repo_path / a_path)
                elif git_index.change_type in ["D"]:
                    deleted_files.append(self.repo_path / a_path)
                elif git_index.change_type in ["R", "T"]:
                    deleted_files.append(self.repo_path / a_path)
                    changed_or_new_files.append(self.repo_path / b_path)
            return changed_or_new_files, deleted_files
        else:
            return self._categorize_changed_files(to_commit.stats.files)

    def _categorize_changed_files(self, files_stats):
        changed_or_new_files = []
        deleted_files = []
        for relative_path, stats in files_stats.items():
            full_file_path = self.repo_path / relative_path
            if stats["deletions"] == stats["lines"]:
                if full_file_path.exists():
                    changed_or_new_files.append(full_file_path)
                else:
                    deleted_files.append(full_file_path)
                pass
            else:
                changed_or_new_files.append(full_file_path)
        return changed_or_new_files, deleted_files

    def merge_base_commit_sha(self, main_branch, pr_commit):
        ancestor_commits = self.repo.merge_base(main_branch, pr_commit)
        if not ancestor_commits:
            raise NoCommonAncestor(f"No common ancestor between {main_branch} and {pr_commit}.")
        return ancestor_commits[0].hexsha

    def is_ancestor_of(self, ancestor_commit_sha, top_commit_sha):
        ancestor_commit = self.repo.commit(ancestor_commit_sha)
        top_commit = self.repo.commit(top_commit_sha)
        return self.repo.is_ancestor(ancestor_commit, top_commit)

    def print_pretty_log(self):
        print(
            self.repo.git.log(
                "--color",
                "--graph",
                "--name-only",
                "--pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) "
                "%C(bold blue)<%an>%Creset'",
                "--abbrev-commit",
            )
        )

    def feature_branch_top_commit_sha_of_a_merge_commit(self, commit_sha):
        commit = self.repo.commit(commit_sha)
        if len(commit.parents) < 2:
            raise NonMergeCommitError(f"The given commit '{commit_sha}' is not a merge commit.")
        feature_branch_top_commit = commit.parents[1]
        return feature_branch_top_commit.hexsha
