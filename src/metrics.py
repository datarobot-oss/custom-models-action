#  Copyright (c) 2023. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

"""
This module holds model and deployment's metrics, which are collected during the GitHub action's
execution.
"""

from common.constants import Label
from common.github_env import GitHubEnv


class Metric:
    """Holds a single metric related attributes."""

    def __init__(self, label, value=0):
        self.label = label
        self.value = value

    def user_facing_name(self, entity_label):
        """
        Returns a user-facing name for the metric.

        Parameters
        ----------
        entity_label : common.constants.Label
            The entity label for which to return the metric name.

        Returns
        -------
        str:
            A user-facing name for the metric.
        """

        return f"{entity_label}--{self.label}"

    def snake_case(self):
        """
        Returns a snake case name for the metric.

        Returns
        -------
        str:
            A snake case name for the metric.
        """

        return self.label.replace("-", "_")


class Metrics:
    """Contains metric attributes that will be exposed by the GitHub actions."""

    def __init__(self, entity_label):
        self._entity_label = entity_label

        # Both models & deployment
        self.total_affected = Metric("total-affected")
        self.total_created = Metric("total-created")
        self.total_deleted = Metric("total-deleted")
        self.total_updated_settings = Metric("total-updated-settings")

        if entity_label == Label.MODELS:
            self.total_created_versions = Metric("total-created-versions")

    def metric_labels(self):
        """
        Returns the all the supported metric user-facing names

        Returns
        -------
        set:
            A set of metric labels
        """

        return {metric.user_facing_name(self._entity_label.value) for metric in self._get_metrics()}

    def _get_metrics(self):
        metrics = [
            self.total_affected,
            self.total_created,
            self.total_deleted,
            self.total_updated_settings,
        ]
        if self._entity_label == Label.MODELS:
            metrics.append(self.total_created_versions)
        return metrics

    def save(self):
        """Save the metrics to the GitHub environment."""

        for metric in self._get_metrics():
            GitHubEnv.set_output_param(
                metric.user_facing_name(self._entity_label.value), metric.value
            )
