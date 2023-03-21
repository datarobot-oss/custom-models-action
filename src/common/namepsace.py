#  Copyright (c) 2023. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

"""A module that contains methods to handle a namespace."""
import logging

from common.exceptions import NamespaceAlreadySet
from common.exceptions import NamespaceNotInitialized
from common.github_env import GitHubEnv

logger = logging.getLogger()


class Namespace:
    """Provides access and handling methods to namespace."""

    _namespace = None

    @classmethod
    def namespace(cls):
        """Return the namespace."""

        return cls._namespace

    @staticmethod
    def default_namespace():
        """A default namespace is used if the user does not provide his own namespace."""

        return f"{GitHubEnv.repository_id()}/"

    @classmethod
    def init(cls, namespace=None):
        """
        Initialize the namespace. This method should be called only once. The input
        namespace argument is optional and if not provided then a default namespace is used.

        Parameters
        ----------
        namespace : str or None
            A namespace string. If None, a default namespace will be used.
        """

        if namespace:
            if not namespace.endswith("/"):
                namespace += "/"
        else:
            namespace = cls.default_namespace()

        logger.info("Set namespace: %s", namespace)

        if namespace == cls.namespace():
            return

        if cls.namespace():
            raise NamespaceAlreadySet(
                "A namespace has already been set. "
                f"Existing namespace: {cls.namespace()}, new: {namespace}."
            )

        cls._namespace = namespace

    @classmethod
    def uninit(cls):
        """Un-init the namespace."""

        logger.info("Un-set namespace.")
        cls._namespace = None

    @classmethod
    def is_in_namespace(cls, user_provided_id):
        """
        Whether a given user provided ID belongs to the namespace if exists.

        Parameters
        ----------
        user_provided_id : str
            A user provided ID.
        """

        if not cls.namespace():
            raise NamespaceNotInitialized("A namespace was not initialized.")
        return user_provided_id.startswith(cls.namespace())

    @classmethod
    def namespaced(cls, user_provided_id):
        """
        Returns the input argument's user provided ID with a prefixed namespace (if exists).
        If the input user_provided_id already starts with the namespace, it will be returned
        untouched.

        Parameters
        ----------
        user_provided_id : str
            A user provided ID.

        Returns
        -------
        str:
            The user provided ID with a prefixed namespace if exists.

        """
        if cls.is_in_namespace(user_provided_id):
            return user_provided_id
        return f"{cls.namespace()}{user_provided_id}"

    @classmethod
    def un_namespaced(cls, user_provided_id):
        """
        Removes the prefixed namespace from the input argument's user provided ID (if exists).
        If the input user_provided_id does not start with the namespace, it will be returned
        untouched.

        Parameters
        ----------
        user_provided_id : str
            A user provided ID.

        Returns
        -------
        str:
            The user provided ID with a prefixed namespace if exists.

        """
        if cls.is_in_namespace(user_provided_id):
            namespace_len = len(cls.namespace())
            return user_provided_id[namespace_len:]
        return user_provided_id
