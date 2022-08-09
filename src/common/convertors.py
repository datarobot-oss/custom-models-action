#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

"""A module that contains conversion classes."""

import re

from common.exceptions import InvalidMemoryValue


# pylint: disable=too-few-public-methods
class MemoryConvertor:
    """A memory convertor."""

    UNIT_TO_BYTES = {
        "m": 0.001,
        "k": 10**3,
        "M": 10**6,
        "G": 10**9,
        "T": 10**12,
        "P": 10**15,
        "E": 10**18,
        "Ki": 2**10,
        "Mi": 2**20,
        "Gi": 2**30,
        "Ti": 2**40,
        "Pi": 2**50,
        "Ei": 2**60,
    }

    @classmethod
    def to_bytes(cls, memory):
        """
        Convert a given memory to bytes. If the input memory argument is of type int, it is assumed
        to be converted already. Otherwise, the expected input is a string, which contains two
        parts - a number and unit. The pattern follows the patterns in K8S.

        Parameters
        ----------
        memory : int or str
            The memory to be converted into bytes.

        Returns
        -------
        int,
            The memory in bytes.
        """

        if isinstance(memory, int):
            return memory

        num, unit = cls._extract_unit_fields(memory)
        if not unit:
            return int(num)

        return int(num) * cls.UNIT_TO_BYTES[unit]

    @classmethod
    def _extract_unit_fields(cls, memory):
        match = re.match(r"^([0-9]+)(m|k|M|G|T|P|E|Ki|Mi|Gi|Ti|Pi|Ei){,1}$", memory)
        if not match:
            raise InvalidMemoryValue(f"The memory value format is invalid: {memory}")

        num, unit = match.groups()
        return int(num), unit
