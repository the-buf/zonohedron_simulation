import numpy as np
import vtk

from type import Cone, Vector


def zonotope_from_generator(generator: list[Vector]) -> vtk.vtkPolyhedron:
    zonotope = vtk.vtkPolyhedron()
    return zonotope


def faces_plane_from_generators(generators: list[Vector]) -> list[list[int]]:
    return


def ordering_generators(generators: list[Vector], cone: Cone) -> list[Vector]:
    """We re-order the list of Vectors in order to process them later, and to aggregate
    colinear vectors pairwise.

    Args:
        generators (list[Vector]): list of the generators of the futur zonotope
        cone (Cone): The cone from which we are generating the zonotopes

    Returns:
        list[Vector]: re-ordered list of vectors of the zonotope.
    """
    vectors_sorted = [
        [
            ((v.coord - cone.origin.coord) / (np.linalg.norm(v.coord))).dot(
                cone.direction.coord
            ),
            v,
        ]
        for v in generators
    ]
    vectors_sorted = sorted(vectors_sorted, key=lambda v: v[0])
    colinear_test = 0
    # colinear_test runs through all values except the last one
    while colinear_test < len(vectors_sorted) - 1:
        print(
            colinear_test,
            vectors_sorted[colinear_test][1].coord,
            vectors_sorted[colinear_test + 1][1].coord,
        )
        if (
            vectors_sorted[colinear_test][0] == vectors_sorted[colinear_test + 1][0]
        ) and (
            np.array_equal(
                np.cross(
                    vectors_sorted[colinear_test][1].coord,
                    vectors_sorted[colinear_test + 1][1].coord,
                ),
                np.array([0, 0, 0]),  # the vectors are colinear
            )
        ):
            print("is colinear")
            new_vector = vectors_sorted[colinear_test][1].add(
                vectors_sorted[colinear_test + 1][1]
            )
            vectors_sorted[colinear_test][1] = new_vector
            vectors_sorted.pop(colinear_test + 1)
            print(len(vectors_sorted))
        else:
            colinear_test += 1
            print(len(vectors_sorted))

    return [vector[1] for vector in vectors_sorted]
