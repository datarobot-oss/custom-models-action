import re

from exceptions import InvalidMemoryValue


class MemoryConvertor:
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
        if isinstance(memory, int):
            return memory

        num, unit = cls._extract_unit_fields(memory)
        if not unit:
            return int(num)

        return num * cls.UNIT_TO_BYTES[unit]

    @classmethod
    def _extract_unit_fields(cls, memory):
        match = re.match(r"^([0-9]+)(m|k|M|G|T|P|E|Ki|Mi|Gi|Ti|Pi|Ei){,1}$", memory)
        if not match:
            raise InvalidMemoryValue(f"The memory value format is invalid: {memory}")
        return match.groups()
