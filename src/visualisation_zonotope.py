import logging

import numpy as np
import vtk

from tests.temp import MakeZonoheddron
from type import Cone, Vector
from visualisation_utils import (
    create_faces,
    create_faces_plane_from_generators,
    ordering_generators,
)

logger = logging.getLogger(__name__)


def main():
    colors = vtk.vtkNamedColors()

    zonohedron = MakeZonoheddron()
    # Visualize
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
        (np.sum(np.array([g.coord for g in ordered_generators]), axis=0) - cone.origin)
    )
    faces_plans = create_faces_plane_from_generators(ordered_generators, end_point)

    # Build the zonotope
    zonotope = vtk.vtkPolyhedron()
    vertices = []
    faces = []
    border = []
    for plan in faces_plans:
        create_faces(border, plan)

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
