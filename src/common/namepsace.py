#  Copyright (c) 2023. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

"""A module that contains methods to handle a namespace."""

from common.exceptions import InvalidEmptyArgument
from common.exceptions import NamespaceAlreadySet


class Namespace:
    """Provides access and handling methods to namespace."""

    _namespace = None

    @classmethod
    def namespace(cls):
        """Return the namespace."""

        return cls._namespace

    @classmethod
    def set_namespace(cls, namespace, force_override=False):
        """
        Set a namespace. The namespace can be set only once.

        Parameters
        ----------
        namespace : str
            A non-empty string.
        force_override : bool
            Whether to override an already configured namespace.
        """

        if not namespace:
            raise InvalidEmptyArgument("Namespace must be a valid string.")

        if namespace == cls._namespace:
            # Happens during the functional tests
            return

        if cls._namespace and not force_override:
            raise NamespaceAlreadySet(
                f"A namespace has already been set. Existing: {cls._namespace}, new: {namespace}."
            )

        cls._namespace = namespace

    @classmethod
    def unset_namespace(cls):
        """Unset the namespace."""

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

        if cls._namespace:
            return user_provided_id.startswith(f"{cls._namespace}/")
        return True

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
        if not cls._namespace or cls.is_in_namespace(user_provided_id):
            return user_provided_id
        return f"{cls._namespace}/{user_provided_id}"
