import logging

import numpy as np
import vtk

from temp import Generators
from type import Cone, Vector
from visualisation_utils import (
    create_faces,
    create_faces_plane_from_generators,
    create_zonogon,
    ordering_generators,
)

logger = logging.getLogger(__name__)


def main():
    cone = Cone(
        vectors=[Vector([0, 0, -1]), Vector([0, -1, 0]), Vector([-1, 0, 0])],
        origin=Vector([0, 0, 0]),
        dim=3,
    )
    generators = [Vector(g) for g in Generators]  # to be updated
    zonohedron = zonotope_from_generators(generators, cone)

    # Visualize
    colors = vtk.vtkNamedColors()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(zonohedron.GetPolyData())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(colors.GetColor3d("PaleGoldenrod"))

    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetWindowName("Zonohedron")
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    renderer.AddActor(actor)
    renderer.SetBackground(colors.GetColor3d("CadetBlue"))
    renderer.GetActiveCamera().Azimuth(30)
    renderer.GetActiveCamera().Elevation(30)

    renderer.ResetCamera()

    renderWindow.Render()
    renderWindowInteractor.Start()


def zonotope_from_generators(generators: list[Vector], cone: Cone) -> vtk.vtkPolyhedron:
    # Clean the generators set
    ordered_generators = ordering_generators(generators, cone)

    # List the faces of the polytope with the generators associated
    end_point = Vector(
        (
            np.sum(np.array([g.coord for g in ordered_generators]), axis=0)
            - cone.origin.coord
        )
    )
    faces_plans = create_faces_plane_from_generators(ordered_generators, end_point)
    keys = sorted(faces_plans.keys())
    first_plan = faces_plans[keys[0]].pop(0)

    # Build the zonotope
    zonotope = vtk.vtkPolyhedron()
    vertices = create_zonogon(first_plan, cone.origin, first_plan[1])
    faces = [vertices]
    border = vertices.copy()

    for k in keys:
        for plan in faces_plans[k]:
            vertices, face, border = create_faces(border, plan, end_point, vertices)
            faces.append(face)

    # TODO the central symetry

    # initialize the zonotope
    for i in range(len(vertices)):
        zonotope.GetPointIds().InsertNextId(i)
        zonotope.GetPoints().InsertNextPoint(
            vertices[i][0], vertices[i][1], vertices[i][2]
        )

    summary_faces = [len(vertices)]
    for f in faces:
        summary_faces.append(len(f))
        for i in f:
            summary_faces.append(i)

    zonotope.SetFaces(summary_faces)
    zonotope.Initialize()

    return zonotope


if __name__ == "__main__":
    main()
