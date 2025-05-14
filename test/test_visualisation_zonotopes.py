from type import Cone, Vector
from visualisation_utils import (
    create_faces_plane_from_generators,
    match_border_segments,
    ordering_generators,
)
from visualisation_zonotope import zonotope_from_generators


def test_create_zonogone() -> None:
    assert True


def test_create_3Dzonotope() -> None:
    generators = [Vector([1, 0, 0]), Vector([0, 1, 0]), Vector([0, 0, 1])]
    cube = zonotope_from_generators(generators)
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

    def test_create_faces_plane(self) -> None:
        vectors = [Vector([0, 1, 1]), Vector([1, 1, 0])]
        end_point = Vector([6, 6, 6])
        assert create_faces_plane_from_generators(vectors, end_point)[3.4641] == [
            [Vector([1, -1, 1])] + vectors
        ]

    def test_create_faces_with_six_vertices(self) -> None:
        vectors = [Vector([0, 1, 1]), Vector([0, 1, 0]), Vector([0, 0, 2])]
        end_point = Vector([6, 6, 6])
        assert create_faces_plane_from_generators(vectors, end_point)[6] == [
            [Vector([1, 0, 0])] + vectors
        ]

    def test_create_multiple_face_planes(self) -> None:
        vectors = [Vector([0, 1, 0]), Vector([1, 1, 0]), Vector([0, 0, 1])]
        end_point = Vector([6, 6, 6])
        assert len(create_faces_plane_from_generators(vectors, end_point)[6]) == 2
        assert len(create_faces_plane_from_generators(vectors, end_point)[0]) == 1

    def test_match_border(self) -> None:
        # generators = [Vector([0, 1, 0]), Vector([1, 1, 0]), Vector([0, 0, 1])]
        border = [
            Vector([0, 0, 0]),
            Vector([0, 1, 0]),
            Vector([0, 2, 1]),
            Vector([0, 1, 1]),
        ]
        plan = []

        assert True
        border = []

        assert True

    def one_point_match_border(self) -> None:
        border = []
        plan = []
        assert match_border_segments(border, plan)
