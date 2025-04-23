import math
from typing import Any

import numpy as np


class Vector:
    coord: np.array[int]
    dim: int
    primitive: bool = False

    def __init__(self, coord: list[int]):
        self.coord = np.array(coord)
        self.dim = len(coord)
        self.primitive = math.gcd(*coord) == 1

    def __call__(self, *args: Any, **kwargs: Any) -> np.array[int]:
        """Calls the generators, receive the coordinates."""
        return self.coord

    def is_primitive_with(self, other: "Vector") -> bool:
        scal = np.cross(self.coord, other.coord)
        return scal.tolist() == [0] * self.dim


class Cone:
    dim: int
    tangent_planes_vect: list[Vector]  # normal vectors to tangent planes
    origin: Vector

    def __init__(self, vectors: list[Vector], origin: Vector | None, dim: int = 3):
        self.dim = dim
        self.tangent_planes_vect = vectors
        if origin is None:
            origin = Vector([0, 0, 0])
        self.origin = origin

    def contain_signed_vectors(self, vectors: list[Vector]) -> bool:
        vectors_array = np.stack([v.coord for v in vectors])
        normal_vectors_array = np.stack([v.coord for v in self.tangent_planes_vect])
        return np.all(np.einsum("ij,ij->i", vectors_array, normal_vectors_array) <= 0)
