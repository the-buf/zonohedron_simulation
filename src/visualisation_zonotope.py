import vtk

from .type import Cone, Vector


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
    return
