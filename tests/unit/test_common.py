import pytest

from common.convertors import MemoryConvertor
from common.exceptions import InvalidMemoryValue


class TestConvertor:
    def test_to_bytes_success(self):
        for unit in MemoryConvertor.UNIT_TO_BYTES.keys():
            configured_memory = f"3{unit}"
            num_bytes = MemoryConvertor.to_bytes(configured_memory)
            assert 3 * MemoryConvertor.UNIT_TO_BYTES[unit] == num_bytes

    @pytest.mark.parametrize("invalid_configured_memory", ["3a", "3aM", "b3", "b3M", "1.2M", "3.3"])
    def test_to_bytes_failure(self, invalid_configured_memory):
        with pytest.raises(InvalidMemoryValue) as ex:
            MemoryConvertor.to_bytes(invalid_configured_memory)
        assert "The memory value format is invalid" in str(ex)
