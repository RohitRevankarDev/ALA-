"""
vector.py
AME 5151 - Applied Linear Algebra Lab, Assignment 1

A minimal Vector class built on top of a plain Python list, extended with
the statistical methods `mean`, `demean` and `std`.
"""

import math
from typing import Iterable, Iterator, Union

Number = Union[int, float]


class Vector:
    """An n-dimensional real vector."""

    def __init__(self, entries: Iterable[Number]) -> None:
        self.entries = [float(e) for e in entries]

    # ------------------------------------------------------------------
    # Basic container behaviour
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> float:
        return self.entries[index]

    def __iter__(self) -> Iterator[float]:
        return iter(self.entries)

    def __repr__(self) -> str:
        return f"Vector({self.entries})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return NotImplemented
        return self.entries == other.entries

    # ------------------------------------------------------------------
    # Vector arithmetic
    # ------------------------------------------------------------------
    def _check_same_length(self, other: "Vector") -> None:
        if len(self) != len(other):
            raise ValueError(
                f"Vectors must have the same length ({len(self)} != {len(other)})."
            )

    def __add__(self, other: "Vector") -> "Vector":
        if isinstance(other, Vector):
            self._check_same_length(other)
            return Vector(a + b for a, b in zip(self, other))
        return NotImplemented

    def __sub__(self, other: "Vector") -> "Vector":
        if isinstance(other, Vector):
            self._check_same_length(other)
            return Vector(a - b for a, b in zip(self, other))
        return NotImplemented

    def __neg__(self) -> "Vector":
        return Vector(-a for a in self)

    def __mul__(self, scalar: Number) -> "Vector":
        if isinstance(scalar, (int, float)):
            return Vector(a * scalar for a in self)
        return NotImplemented

    __rmul__ = __mul__

    def dot(self, other: "Vector") -> float:
        """Inner product of two vectors."""
        self._check_same_length(other)
        return sum(a * b for a, b in zip(self, other))

    def norm(self) -> float:
        """Euclidean norm."""
        return math.sqrt(self.dot(self))

    def rms(self) -> float:
        """Root-mean-square value of the entries."""
        if len(self) == 0:
            raise ValueError("rms is undefined for an empty vector.")
        return self.norm() / math.sqrt(len(self))

    # ------------------------------------------------------------------
    # Assignment 1 methods
    # ------------------------------------------------------------------
    def mean(self) -> float:
        """Return the mean: (sum of entries) / (number of entries)."""
        if len(self) == 0:
            raise ValueError("mean is undefined for an empty vector.")
        return sum(self.entries) / len(self)

    def demean(self) -> "Vector":
        """Return a NEW vector with the mean subtracted from every entry."""
        m = self.mean()
        return Vector(a - m for a in self)

    def std(self) -> float:
        """
        Return the (population) standard deviation: the square root of the
        average of the squared deviations from the mean.
        Uses `demean` for the deviations.
        """
        deviations = self.demean()
        return math.sqrt(sum(d * d for d in deviations) / len(self))
