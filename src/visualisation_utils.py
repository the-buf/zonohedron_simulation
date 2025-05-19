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
            new_vector = (
                vectors_sorted[colinear_test][1]
                + (vectors_sorted[colinear_test + 1][1])
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
                normal = normal / math.gcd(*map(int, normal))
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


def create_faces(
    border: list[Vector], plan: list[Vector], end_point: Vector, vertices: list[Vector]
) -> tuple[list[Vector], list[int], list[Vector]]:
    """

    Args:
        border (list[Vector]): the border of the conex faces already created
        plan (list[Vector]): List of face planes. The first vector is a normal vector,
        and then all the generators of the face.

    Returns:
        list[list[Vectors]]:
    """
    origin_index = match_border_segments(border, plan, end_point)

    if border[origin_index + 1 - len(border)] - border[origin_index] in plan[1:]:
        first_generator = border[origin_index + 1 - len(border)] - border[origin_index]
    else:  # the other connex segment is a generator
        first_generator = border[origin_index - 1] - border[origin_index]
    zonogon = create_zonogon(plan, border[origin_index], first_generator)

    border, new_vertices = update_border(border, zonogon, origin_index)

    vertices += new_vertices  # mauvais new vertices d'après ce que je viens de voir
    face = [vertices.index(z_vert) for z_vert in zonogon]

    return vertices, face, border


def match_border_segments(
    border: list[Vector], plan: list[Vector], end_point: Vector
) -> int:
    """Finds the point from which the face generated by the generators in the list 'plan'
    will start. The steps are:
    - find the point that is surrounded by segments that correspond to the plan
    generators.
    - verify that we are at the right place in order to ensure convexity of the zonotope.

    Args:
        border (list[Vector]): the ordered list of the border of the zonotope faces
        already created.
        plan (list[Vector]): the list composed of the normal vector of the face plane
        and the generators that generate the face.

    Returns:
        int: The index of the point of beginning of the face generated by plan in the
        border.
    """
    i = 0
    border_i = 0
    n = len(border)
    # NEED TO MATHICALLY PROVE THAT 1 ATTACH POINT CANT HAPPEN.
    while i < n:
        i += 1
        border_i = border_i - n if border_i + 2 >= n else border_i
        segment = border[border_i + 1] - border[border_i]
        if (segment in plan[1:]) or (-segment in plan[1:]):
            if (
                not np.vdot(
                    (end_point.coord / 2 - border[border_i].coord), plan[0].coord
                )
                >= 0
            ):
                border_i += int(n / 2)
            else:
                next_segment = border[border_i + 2] - border[border_i + 1]
                prev_segment = border[border_i] - border[border_i]
                if (segment in plan[1:]) and (
                    -prev_segment in plan[1:] or prev_segment not in plan[1:]
                ):
                    return border_i
                elif (
                    next_segment not in plan[1:] and -next_segment not in plan[1:]
                ):  # -segment is a generator
                    return border_i + 1
                elif prev_segment in plan[1:]:
                    border_i -= 1
                else:
                    border_i += 1
        else:
            border_i += 1
    raise ValueError(
        "The common generator was not found between the border ",
        border,
        " and face's generators",
        plan[1:],
    )


def create_zonogon(
    plan: list[Vector], origin_point: Vector, generator: Vector
) -> list[Vector]:
    """Create the zonogon from the origin point and the list of generators in the plane
    face.

    Args:
        plan (list[Vector]): the first vector is the normal vector to the plane of the
        zonogon. The rest are the generators.
        origin_point (Vector): the point from which we can add generators to have the
        zonogon.
        generator (Vector): generator that match the neighbor edge in the border.

    Returns:
        list[Vector]: List of the vertices of the zonogon
    """
    zonogon = [origin_point]
    sorted_generators = angle_sorting(generator, plan[1:])

    for i in range(len(sorted_generators)):
        zonogon.append(zonogon[-1] + sorted_generators[i])
    for i in range(len(sorted_generators) - 1):
        zonogon.append(zonogon[-1] - sorted_generators[i])
    return zonogon


def angle_sorting(ref_vector: Vector, generators: list[Vector]) -> list[Vector]:
    """Takes a list of coplanar vectors and a reference vector and return the list
    ordered in the trigonometric order starting with the reference vector, with
    potential resigning to have all the vectors in a max pi angle from the ref vector.
    """

    def angle(ref_vector: Vector, vect: Vector) -> np.float64:
        ref_vector_unit = ref_vector.coord / np.linalg.norm(ref_vector.coord)
        v_unit = vect.coord / np.linalg.norm(vect.coord)
        angle = np.arctan2(
            np.linalg.norm(np.cross(ref_vector_unit, v_unit)),
            np.dot(ref_vector_unit, v_unit),
        )
        return angle

    sorted_vectors = sorted(generators, key=lambda vect: angle(ref_vector, vect))
    i = sorted_vectors.index(ref_vector)
    return (
        sorted_vectors[i:] + [-gen for gen in sorted_vectors[:i]]
    )  # we order the generators to start with the chosen one. For this aim, we potentially resign generators.


def update_border(
    border: list[Vector], zonogon: list[Vector], origin_point_index: int
) -> tuple[list[Vector], list[Vector]]:
    """update the border.

    Args:
        border (list[Vector]): previous list of all the points that are at the border of
        the connex body composed of the previous faces of the zonotope.
        zonogon (list[Vector]): the new face.
        origin_point_index (Vector): the index in the border of the first element of the
        list zonogon.

    Raises:
        Exception: If the index search has any mistake, an error is raised to find the
        issue quickly and improve the code.

    Returns:
        tuple[list[Vector], list[Vector]]: the first term is the new border and the
        second list is the new vertices that will be added to the list of vertices of
        the polyhedron.
    """
    l_index = origin_point_index
    r_index = origin_point_index
    i = 0
    while i < len(border) + 1:
        if border[r_index] in zonogon:
            r_index += 1
            if r_index >= len(border):
                r_index -= len(border)
        elif border[l_index] in zonogon:
            l_index -= 1
            if l_index < -1:
                l_index += len(border)
        else:
            break

    if (
        border[(origin_point_index + 1) % len(border)] != zonogon[1]
    ):  # need to know if the lists are ordered the same way
        if border[origin_point_index - 1] == zonogon[1]:
            zonogon.reverse()
            zonogon.insert(0, zonogon[-1])
            zonogon.pop()

    # Realign the new border in zonogon and the border
    left = zonogon.index(border[l_index + 1])
    right = zonogon.index(border[r_index - 1])
    if left > right:
        zonogon.reverse()
        left = len(zonogon) - 1 - left
        right = len(zonogon) - 1 - right

    # Different config of indexes in border
    if (l_index <= origin_point_index) and (origin_point_index <= r_index):
        return (
            border[: l_index + 1] + zonogon[left : right + 1] + border[r_index:],
            zonogon[left : right + 1],
        )
    elif (origin_point_index <= r_index) and (r_index <= l_index):
        return (
            zonogon[left : right + 1] + border[r_index:l_index],
            zonogon[left : right + 1],
        )
    elif (r_index <= l_index) and (l_index <= origin_point_index):
        return (
            border[r_index : l_index + 1] + zonogon[left : right + 1],
            zonogon[left : right + 1],
        )

    raise Exception(
        "Cant find the right update for the border ",
        border,
        " and the zonogon ",
        zonogon,
    )
