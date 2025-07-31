from logging import getLogger
from typing import Literal

from numpy import ascontiguousarray, complex64, complex128
from numpy.typing import DTypeLike, NDArray

from rustfft._core import (
    Direction,  # type: ignore
)
from rustfft._core import (
    FftPlannerF32 as _FftPlannerF32,  # type: ignore
)
from rustfft._core import (
    FftPlannerF64 as _FftPlannerF64,  # type: ignore
)


class FftPlanner:
    """Prepared FFT plan for a given size, direction, and data type."""

    _inner: _FftPlannerF32 | _FftPlannerF64
    _dtype: DTypeLike

    def __init__(
        self,
        len: int,
        direction: Literal["forward", "inverse"] = "forward",
        dtype: Literal["c64", "c128"] = "c128",
    ):
        """Prepare an FFT plan for a given size, direction, and data type.

        Args:
            len: Number of elements in the input
            direction: Forward FFT or inverse.
                       Defaults to "forward".
            dtype: Whether to use 32-bit floats (c64) or 64-bit floats (c128)
                   for each part of a complex number.
                   Defaults to "c128".

        Raises:
            ValueError: If inputs do not match available options
        """

        dl = direction.lower()
        if dl == "forward":
            d = Direction.Forward
        elif dl == "inverse":
            d = Direction.Inverse
        else:
            raise ValueError(f"Direction must be either `forward` or `inverse`; received `{direction}`")

        if dtype.lower() == "c64":
            self._inner = _FftPlannerF32(len, d)
            self._dtype = complex64  # 64 bits total
        elif dtype.lower() == "c128":
            self._inner = _FftPlannerF64(len, d)
            self._dtype = complex128  # 128 bits total
        else:
            raise ValueError(f"dtype must be either `c64` or `c128`; received `{dtype}`")

    def process(self, buffer: NDArray[complex64] | NDArray[complex128]) -> NDArray:
        """Run the FFT, converting the values in the buffer in-place if possible,
        and allocating a new array if densification or type conversion is required.

        Args:
            buffer: 1D array of complex numbers matching the initialized dtype
        """
        # Reallocate or convert type if necessary
        buffer_maybe_new = ascontiguousarray(buffer.astype(self._dtype))
        if id(buffer_maybe_new) != id(buffer):
            getLogger().warning(
                "Reallocating input buffer for FFT, either due to discontiguous data or incorrect data type."
            )

        # Run the actual FFT
        self._inner.process(buffer_maybe_new)

        return buffer_maybe_new

    def __call__(self, buffer: NDArray[complex64] | NDArray[complex128]) -> NDArray:
        return self.process(buffer)
