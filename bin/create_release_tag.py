#!/usr/bin/env python
"""A script to create a release."""

import argparse
import logging
import os
import re
import sys
from datetime import date
from pathlib import Path

from git import Repo

logger = logging.getLogger()


# pylint: disable=too-few-public-methods
class ReleaseCreator:
    """The release creator handles the logic to create a release."""

    def __init__(self, args):
        self._args = args
        self._git_workspace_path = Path(sys.path[0]) / ".."
        self._repo = Repo.init(self._git_workspace_path.resolve())

    def run(self):
        """The main method to drive the functionality to create a release."""

        self._validate_integrity()

        if self._tag_already_exists():
            logger.info("Tag already exists, tag: %s", self._args.tag)
            self._verify_override_conditions()
            self._remove_tag()
        else:
            self._verify_valid_docs()

        self._create_tag()
        self._push_to_remote()

    def _validate_integrity(self):
        if self._repo.is_dirty():
            raise Exception("There is probably an un-committed work. The repository is dirty.")

        if self._repo.active_branch.name != "master":
            raise Exception("A release can only be created from the 'master' branch.")

    def _tag_already_exists(self):
        return self._args.tag in self._repo.tags

    def _verify_override_conditions(self):
        if not self._args.force_override:
            raise ValueError("A release already exists with the given tag!")

        user_response = input(
            f"A release already exists with the given tag: {self._args.tag}.\n"
            "Do you want to override it? [Yes] "
        )
        if user_response != "Yes":
            sys.exit(0)

        self._verify_valid_docs()

    def _verify_valid_docs(self):
        self._verify_releases_history()
        self._verify_tag_reference_in_readme()

    def _verify_releases_history(self):
        releases_filepath = self._git_workspace_path / "RELEASES.md"
        today_date = date.today().strftime("%Y-%m-%d")
        expected_release_report_line = f"## {self._args.tag[1:]} [{today_date}]"
        with open(releases_filepath, encoding="utf-8") as releases_file:
            for line in releases_file.readlines():
                if line.startswith("##"):  # the latest reported release
                    line = line.strip()
                    if line != expected_release_report_line:
                        raise Exception(
                            "There's no valid report in RELEASES.md. "
                            f"Expecting: '{expected_release_report_line}'"
                        )
                    break
        logger.info("Release history is valid (RELEASES.md).")

    def _verify_tag_reference_in_readme(self):
        readme_filepath = self._git_workspace_path / "README.md"
        with open(readme_filepath, encoding="utf-8") as fd:
            content = fd.read()

        unexpected_tags_pattern = f"datarobot-oss/custom-models-action@((?!{self._args.tag})\\S+)"
        unexpected_tags = re.findall(unexpected_tags_pattern, content)
        if unexpected_tags:
            print(f"An invalid tag(s) are referenced in the README.md: {unexpected_tags}")
            user_response = input(f"Do you want to replace with new tag: {self._args.tag}? [yY] ")
            if user_response in ["y", "Y"]:
                with open(readme_filepath, "w", encoding="utf-8") as fd:
                    content = re.sub(
                        unexpected_tags_pattern,
                        f"datarobot-oss/custom-models-action@{self._args.tag}",
                        content,
                    )
                    fd.write(content)
                    print("Replacement done. Please commit and try again ...")
                    sys.exit(0)

    def _remove_tag(self):
        logger.info("Removing tag: %s", self._args.tag)
        if not self._args.dry_run:
            self._repo.delete_tag(self._args.tag)

    def _create_tag(self):
        logger.info("Creating a tag: %s", self._args.tag)
        if not self._args.dry_run:
            self._repo.create_tag(self._args.tag)

    def _push_to_remote(self):
        user_response = input("Do you want to push the tags to remote? [yY] ")
        if user_response in ["y", "Y"]:
            logger.info("Pushing branch and tags to remote ...")
            if not self._args.dry_run:
                self._repo.remotes.origin.push(self._args.tag, force=True)


def configure_logging():
    """Setup logging configuration."""

    log_level = os.environ.get("LOGLEVEL", "INFO").upper()
    log_format = "%(asctime)s [%(levelname)s]  %(message)s"
    logging.basicConfig(format=log_format, level=log_level)


def get_cli_args():
    """Read and validate command line arguments."""

    parser = argparse.ArgumentParser(
        prog=os.path.basename(sys.argv[0]),
        description="Create a new release by tagging it with a provided tag.",
    )

    parser.add_argument(
        "tag", help="The release version to use for tagging in the form of 'vX.Y.Z'."
    )
    parser.add_argument(
        "--force-override",
        action="store_true",
        help="Whether to delete and recreate an already existing release tag "
        "(This is a dangerous action).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not actually make changes.")

    args = parser.parse_args()

    if not re.match(r"^v[0-9]+(\.[0-9]+){2}$", args.tag):
        print("Invalid tag pattern. Expecting: 'vX.Y.Z'.")
        sys.exit(-1)

    return args


if __name__ == "__main__":
    configure_logging()
    cli_args = get_cli_args()
    release_creator = ReleaseCreator(cli_args)
    release_creator.run()
