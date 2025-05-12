from type import Cone, Vector
from visualisation_zonotope import ordering_generators, zonotope_from_generator


def test_create_zonogone() -> None:
    assert True


def test_create_3Dzonotope() -> None:
    generators = [Vector([1, 0, 0]), Vector([0, 1, 0]), Vector([0, 0, 1])]
    cube = zonotope_from_generator(generators)
    assert cube == cube


def test_visualise_zonotope() -> None:
    assert True


def test_create_faces_plane() -> None:
    assert True


class TestVisualisationZonotopes:
    def setup_method(self):
        self.cone = Cone(
            vectors=[Vector([0, 0, -1]), Vector([0, -1, 0]), Vector([-1, 0, 0])],
            origin=Vector([0, 0, 0]),
            dim=3,
        )

    def test_ordering_generators(self) -> None:
        vectors = [Vector([0, 1, 1]), Vector([0, 0, 1]), Vector([1, 1, 1])]
        assert ordering_generators(vectors, self.cone) == [
            vectors[1],
            vectors[0],
            vectors[2],
        ]

        # Here is the test for merging vector when there is colinearity
        vectors.append(Vector([0, 5, 5]))
        sorted_vectors = ordering_generators(vectors, self.cone)
        assert sorted_vectors == [
            vectors[1],
            Vector([0, 6, 6]),
            vectors[2],
        ]

    def test_multiple_colinear_vector_ordering(self) -> None:
        vectors = [
            Vector([1, 0, 1]),
            Vector([1, 0, 1]),
            Vector([1, 0, 1]),
            Vector([1, 0, 1]),
        ]
        assert ordering_generators(vectors, self.cone) == [Vector([4, 0, 4])]
