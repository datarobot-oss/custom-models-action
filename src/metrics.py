#  Copyright (c) 2023. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

"""
This module holds model and deployment's metrics, which are collected during the GitHub action's
execution.
"""

import dataclasses
from dataclasses import dataclass

from common.constants import Label
from common.github_env import GitHubEnv


@dataclass
class Metric:
    """A data class that holds a single metric related attributes."""

    label: str
    value: int


@dataclass
class Metrics:
    """Contains metric attributes that will be exposed by the GitHub actions."""

    # Both models & deployment
    total_affected: Metric = Metric("total-affected", 0)
    total_created: Metric = Metric("total-created", 0)
    total_deleted: Metric = Metric("total-deleted", 0)
    total_updated_settings: Metric = Metric("total-updated-settings", 0)

    # Model only
    total_created_versions: Metric = Metric("total-created-versions", 0)

    def __init__(self, entity_label):
        self._entity_label = entity_label
        for metric in self._get_metrics(entity_label):
            metric_attr = getattr(self, metric.name)
            metric_attr.value = 0

    @classmethod
    def metric_labels(cls, entity_label):
        """
        Returns the all the supported metric names for models or deployments.

        Parameters
        ----------
        entity_label : common.constants.Label
            The entity label for which to return the metric names.

        Returns
        -------
        set:
            A set of metric labels
        """

        return {
            cls.metric_label(entity_label, metric.default.label)
            for metric in cls._get_metrics(entity_label)
        }

    @classmethod
    def metric_label(cls, entity_label, raw_metric_label):
        """
        Returns a formatted metric label.

        Parameters
        ----------
        entity_label : common.constants.Label
            An entity label (e.g. Label.MODELS, Label.DEPLOYMENTS).
        raw_metric_label : str
            The base metric name.

        Returns
        -------
        str:
            A formatted metric label.
        """

        return f"{entity_label.value}--{raw_metric_label}"

    @classmethod
    def _get_metrics(cls, entity_label):
        metrics = set()
        for metric in dataclasses.fields(cls):
            if entity_label == Label.DEPLOYMENTS and metric.name == "total_created_versions":
                continue
            metrics.add(metric)
        return metrics

    def save(self):
        """Save the metrics to the GitHub environment."""

        for metric in self._get_metrics(self._entity_label):
            metric_attr = getattr(self, metric.name)
            GitHubEnv.set_output_param(
                self.metric_label(self._entity_label, metric.default.label), metric_attr.value
            )
