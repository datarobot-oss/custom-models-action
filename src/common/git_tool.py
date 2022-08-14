#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

"""A module that provides high level access to local (and remote) git repositories."""

import logging
import os

from git import Repo
from pathlib import Path

from common.exceptions import NoCommonAncestor
from common.exceptions import NonMergeCommitError

logger = logging.getLogger()


class GitTool:
    """A wrapper around git.Repo, which provides high level interface to git operations."""

    GITHUB_COMMIT_URL_PATTERN = "https://github.com/{user_and_project}/commit/{sha}"

    def __init__(self, git_repo_path):
        os.environ["GIT_PYTHON_TRACE"] = "false"  # "full"
        self._repo = Repo.init(git_repo_path)
        self._repo_path = Path(git_repo_path)

    @property
    def repo(self):
        """Returns the git.Repo entity."""
        return self._repo

    @property
    def repo_path(self):
        """Return the repository file path."""
        return self._repo_path

    def num_commits(self, ref="HEAD"):
        """Return the number of commits for a given ref."""
        try:
            return int(self.repo.git.rev_list("--count", ref))
        except TypeError:
            return 0

    def num_remotes(self):
        """Return the number of configured remotes of a local repository."""
        return len(self.repo.remotes)

    def remote_name(self):
        """Return the remote name if exists, otherwise None."""
        return self.repo.remote().name if self.num_remotes() else None

    def find_changed_files(self, to_commit_sha, from_commit_sha=None):
        """
        Find the files that were changes in some way from one commit to another. If the 'from'
        commit is not provided than it refers to a single commit, which is provided as
        the first argument.

        Parameters
        ----------
        to_commit_sha : str
            A string representing the 'to' commit. It can be SHA string or ref name.
        from_commit_sha : str or None
            A string representing the 'from' commit. It can be SHA string or ref name or None.
            If None, then it is ignored.

        Returns
        -------
        tuple,
            A tuple of two arguments - a list of the changes/new files and a list of deleted files.
        """

        to_commit = self.repo.commit(to_commit_sha)
        if from_commit_sha:
            from_commit_sha = self.repo.commit(from_commit_sha)
            diff = from_commit_sha.diff(to_commit)

            changed_or_new_files = []
            deleted_files = []
            for git_index in diff:
                a_path = self.repo_path / git_index.a_path
                if git_index.change_type in ["M", "A"]:
                    changed_or_new_files.append(a_path)
                elif git_index.change_type in ["D"]:
                    deleted_files.append(a_path)
                elif git_index.change_type in ["R", "T"]:
                    deleted_files.append(a_path)
                    changed_or_new_files.append(self.repo_path / git_index.b_path)
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
        """
        Find the best common ancestor(s) between two commits.

        Parameters
        ----------
        main_branch : str
            A ref name or commit of the main branch.
        pr_commit : str
            A commit from a pull request feature branch.

        Returns
        -------
        str,
            The commit SHA of the common ancestor.
        """

        ancestor_commits = self.repo.merge_base(main_branch, pr_commit)
        if not ancestor_commits:
            raise NoCommonAncestor(f"No common ancestor between {main_branch} and {pr_commit}.")
        return ancestor_commits[0].hexsha

    def is_ancestor_of(self, ancestor_commit_sha, top_commit_sha):
        """
        Whether a given commit is an ancestor of another.

        Parameters
        ----------
        ancestor_commit_sha : str
            A commit SHA (or ref name) to be checked if it is an ancestor.
        top_commit_sha : str
            A commit SHA (or ref name) that is considered as the top in the tree.

        Returns
        -------
        bool,
            Whether one commit is an ancestor of the other.

        """

        try:
            if not (ancestor_commit_sha and top_commit_sha):
                raise ValueError("Invalid None input arguments.")
            ancestor_commit = self.repo.commit(ancestor_commit_sha)
            top_commit = self.repo.commit(top_commit_sha)
            return self.repo.is_ancestor(ancestor_commit, top_commit)
        except ValueError as ex:
            logger.debug(f"Ancestor commit could not be found. Error: {str(ex)}")
            return False

    def print_pretty_log(self):
        """Log a pretty git-log using a debug severity, up to 7 commits."""

        git_logs = self.repo.git.log(
            "--color",
            "--graph",
            "--name-only",
            "--pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) "
            "%C(bold blue)<%an>%Creset'",
            "--abbrev-commit",
            "-n",
            "7",
        )
        logger.debug(f"\n{git_logs}")

    def feature_branch_top_commit_sha_of_a_merge_commit(self, commit_sha):
        """
        The top commit in a merge branch, which was created from a feature branch as part of
        a pull request.

        Parameters
        ----------
        commit_sha : str
            A commit SHA string.

        Returns
        -------
        str,
            A commit SHA string.
        """

        commit = self.repo.commit(commit_sha)
        if len(commit.parents) < 2:
            raise NonMergeCommitError(f"The given commit '{commit_sha}' is not a merge commit.")
        feature_branch_top_commit = commit.parents[1]
        return feature_branch_top_commit.hexsha
