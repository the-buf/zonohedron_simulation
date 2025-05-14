import logging
import math

import numpy as np

from type import Cone, Vector

logger = logging.getLogger(__name__)


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
    logger.info("Vectors are sorted with respect to ", cone.direction)
    vectors_sorted = sorted(vectors_sorted, key=lambda v: v[0])
    colinear_test = 0
    # colinear_test runs through all values except the last one
    while colinear_test < len(vectors_sorted) - 1:
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
            logger.info(
                "Colinearity between two vectors: ",
                vectors_sorted[colinear_test][1].coord,
                vectors_sorted[colinear_test + 1][1].coord,
            )
            new_vector = vectors_sorted[colinear_test][1].add(
                vectors_sorted[colinear_test + 1][1]
            )
            vectors_sorted[colinear_test][1] = new_vector
            vectors_sorted.pop(colinear_test + 1)
        else:
            colinear_test += 1
    logger.info("Number of generators removed:", len(generators) - len(vectors_sorted))
    return [vector[1] for vector in vectors_sorted]


def create_faces_plane_from_generators(
    generators: list[Vector], end_point: Vector
) -> dict[float : list[list[Vector]]]:
    """Creating lists that coincide with the faces of the zonotopes. The lists are
    composed of the normal (primitive) vector of the plane, and the generators that are
    in the plane and that will contribute in this on face.
    These lists are ranked in a dict by the value between the normal vector and the
    barycenter.
    For each pair of generators, we look for existing faces and then we either create a
    new face plane either add the generators to an existing face plane.

    Args:
        generators (list[Vector]): the ordered generators of the zonotope
        barycenter (Vector): the barycenter of the zonotope
        origin (Vector): the origin of the cone in which the zonotope is created


    Returns:
        list[list[Vectors]]: List of face planes. The first vector is the vector normal
        to the plane (not re-normalized), and then all the generators of the face.
    """
    logger.info("Find all the planes bearing a face of the zonotope...")
    tangent_planes = {}
    for i in range(len(generators)):
        for next in generators[(i + 1) :]:
            normal = np.cross(generators[i].coord, next.coord)

            if np.all(normal == np.array([0, 0, 0])):
                logger.warning(
                    "There are colinear generators at the stage of find face_planes! They are ",
                    generators[i].coord,
                    next.coord,
                )
            else:
                normal = normal / math.gcd(*normal)
                if np.vdot(end_point.coord, normal) < 0:
                    normal = (
                        -1 * normal
                    )  # now the normal vector is primitive and so unique.
            normal_vect = Vector(normal)
            projected_value = (
                math.floor(
                    np.vdot(end_point.coord, (normal / np.sqrt(np.sum(normal**2))))
                    * 1e5
                )
                / 1e5
            )  # We approximate the projected value for equality testing concern.
            if tangent_planes.get(projected_value, None):
                flag_added = False
                for v_list in tangent_planes[projected_value]:
                    if np.allclose(normal, v_list[0].coord):
                        # generator[i] is before next in the list so it must be in the list already.
                        next not in v_list and v_list.append(next)
                        flag_added = True
                if not flag_added:
                    tangent_planes[projected_value].append(
                        [normal_vect, generators[i], next]
                    )
            else:
                tangent_planes[projected_value] = [[normal_vect, generators[i], next]]
    return tangent_planes


def create_faces(border: list[Vector], plan: list[Vector]) -> list[list[float]]:
    """

    Args:
        border (list[Vector]): the border of the conex faces already created
        plan (list[Vector]): List of face planes. The first vector is a normal vector,
        and then all the generators of the face.

    Returns:
        list[list[Vectors]]:
    """
    # matching des cotes
    orgin_face = match_border_segments()
    # faire la face
    zonogon = create_zonogon(plan)
    face, vertices = None  ## TODO
    return vertices, face, orgin_face, border


def match_border_segments(border: list[Vector], plan: list[Vector]) -> Vector:
    """Finds the point from which the face generated by the generators in the list 'plan'
    will start.

    Args:
        border (list[Vector]): the ordered list of the
        plan (list[Vector]): _description_

    Returns:
        Vector: The point of beginning of the face generated by plan.
    """
    ## TODO
    return


def create_zonogon(plan: list[Vector]) -> list[Vector]:
    """_summary_

    Args:
        plan (list[Vector]): _description_

    Returns:
        list[Vector]: _description_
    """
    return
