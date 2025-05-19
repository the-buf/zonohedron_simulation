import math as math

import numpy as np
import vtk

# from matplotlib.ticker import LinearLocator
# from scipy.spatial import ConvexHull
# from numpy.core.defchararray import array
# from numpy.core.overrides import verify_matching_signatures
# from numpy import diff
from scipy.stats import *  # noqa: F403

Generators = [
    [7.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [10.0, 1.0, 9.0],
    [5.0, 1.0, 2.0],
    [0.0, 7.0, 3.0],
    # [6.0, 1.0, 8.0], [2.0, 3.0, -4.0], [1.0, 7.0, 3.0], [1.0, -2.0, -4.0], [0.0, -5.0, -1.0],
    # [16.0, -2.0, 7.0],
    # [1.0, 8.0, 2.0],
    # [5.0, 0.0, -10.0],
    # [7.0, 5.0, 1.0],
    # [1.0, 8.0, 1.0],
    # # [2.0, 4.0, -1.0], [1.0, 3.0, -3.0], [1.0, 5.0, -3.0], [6.0, 9.0, -2.0], [6.0, 10.0, 5.0],
    # [7.0, 3.0, 2.0],
    # [4.0, -5.0, 0.0],
    # [3.0, 1.0, -2.0],
    # [4.0, -1.0, 2.0],
    # [1.0, 4.0, 2.0],
    # # [1.0, -10.0, 0.0], [3.0, -1.0, -2.0], [2.0, 5.0, 2.0], [1.0, 1.0, 11.0], [2.0, -15.0, 6.0],
    # [0.0, -3.0, 4.0],
    # [1.0, 0.0, 3.0],
    # [8.0, -6.0, 7.0],
    # [0.0, -2.0, -7.0],
    # [1.0, 0.0, -8.0],
    # [0.0, 10.0, -1.0],
    # [3.0, 1.0, -3.0],
    # [0.0, 6.0, -1.0],
    # [3.0, -4.0, 6.0],
    # [3.0, 0.0, -1.0],
    # [4.0, -9.0, 2.0],
    # [8.0, -1.0, -3.0],
    # [0.0, 0.0, 1.0],
    # [0.0, 2.0, -11.0],
    # [1.0, -3.0, -1.0],
    # # [0.0, -1.0, -6.0], [4.0, 7.0, 2.0], [7.0, 10.0, -6.0], [1.0, -2.0, 5.0], [0.0, 3.0, -1.0],
    # [0.0, -1.0, 7.0],
    # [3.0, 0.0, -1.0],
    # [11.0, 1.0, -1.0],
    # [7.0, -0.0, -2.0],
    # [4.0, -8.0, -6.0],
    # [0.0, -3.0, -5.0],
    # [8.0, 0.0, -1.0],
    # [0.0, 2.0, -5.0],
    # [1.0, 0.0, -3.0],
    # [3.0, 9.0, 1.0],
    # [1.0, -5.0, 1.0],
    # [2.0, -5.0, -2.0],
    # [0.0, 1.0, -2.0],
    # [4.0, 5.0, 1.0],
    # [1.0, 1.0, 7.0],
    # # [2.0, 3.0, -8.0], [1.0, 5.0, -4.0], [0.0, -4.0, -1.0], [4.0, -6.0, -2.0], [6.0, 1.0, 2.0],
    # [0.0, 1.0, -22.0],
    # [1.0, -1.0, -5.0],
    # [1.0, -10.0, -2.0],
    # [6.0, -7.0, -1.0],
    # [0.0, 8.0, -1.0],
    # [1.0, 2.0, -6.0],
    # [0.0, -4.0, 1.0],
    # [2.0, 0.0, 3.0],
    # [3.0, 6.0, -2.0],
    # [3.0, -1.0, 2.0],
    # [8.0, -0.0, 10.0],
    # [4.0, 0.0, -6.0],
    # [2.0, -4.0, 0.0],
    # [0.0, 3.0, -3.0],
    # [2.0, -2.0, 0.0],
]
# Generators = [[abs(i) for i in g ] for g in Generators]

print_flag = True


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


def MakeZonoheddron() -> vtk.vtkPolyhedron:
    aZonohedron = vtk.vtkPolyhedron()
    pts, Faces = ListVertices()
    pts = np.array(pts)

    for i in range(len(pts)):
        aZonohedron.GetPointIds().InsertNextId(i)

    for j in range(len(pts)):
        aZonohedron.GetPoints().InsertNextPoint(pts[j][0], pts[j][1], pts[j][2])

    faces = [len(pts)]
    for f in Faces:
        faces.append(len(f))
        for i in f:
            faces.append(i)

    print("faces ", faces)
    aZonohedron.SetFaces(faces)
    aZonohedron.Initialize()

    return aZonohedron


# generators -> Vertices
def ListVertices():
    # On remet tous les générateurs dans le même sens:
    for g_i in range(len(Generators)):
        g = Generators[g_i]
        if (g[0] < 0) or (g[0] == 0 and (g[1] < 0 or (g[1] == 0 and g[2] < 0))):
            Generators[g_i] = [-g[i] for i in range(len(g))]

    # On reverifie qu'il n'y a pas de colineaires
    list_to_remove = []
    for g_i in range(len(Generators) - 1):
        g = Generators[g_i]
        for gg in Generators[g_i + 1 :]:
            normal = np.cross(np.array(g), np.array(gg))
            if normal.tolist() == [0, 0, 0]:
                g = (np.array(g) + np.array(gg)).tolist()
                list_to_remove.append(gg)
    for i in range(len(list_to_remove)):
        Generators.remove(list_to_remove[i])
    print(Generators)

    # barycentre du Zonotope
    bary = np.sum(np.array(Generators), axis=0) / 2
    bary1 = np.sum(np.array(Generators), axis=0) / 2
    if True:
        print("Barycentre", bary, "\n")

    # liste des faces
    PlanesTangents = []

    # Faire les plans de face, avec un vecteur normal unitaire,
    # dans une liste ordonnée par produit scalaire avec le Bary.
    # Generateurs -> Liste des plans ___en O(nb_gene * ln(nb_plans_tangents))
    for g in range(len(Generators)):
        gg = Generators[g]
        for h in Generators[(g + 1) :]:
            vect1 = np.array(gg)
            vect2 = np.array(h)
            normal = np.cross(vect1, vect2)
            if np.all(normal == np.array([0, 0, 0])):
                print(normal, vect1, vect2)
            normal = normal / np.sqrt(np.sum(normal**2))
            if np.vdot(bary, normal) < 0:
                normal = -1 * normal
            L_vect = [np.vdot(bary, normal), normal, gg, h]
            # find dichotomique, les Elements sont
            PlanesTangents = InsertElement(PlanesTangents, L_vect)
    # Listes de plans -> faces, points ___en O(nb_plans_tangents)
    Faces = []
    if print_flag:
        for p in PlanesTangents[:20]:
            print("\n", "PlansTangents", p, "\n")

    # Liste de vertex en frontières, c'est un cycle, des points qui sont la frontière.
    Vertices = MakeFirstZonogone(PlanesTangents[0], [0, 0, 0])
    if print_flag:
        print(
            "Element",
            PlanesTangents[0],
            "\nFrontiere",
            Vertices,
            "\nnombre de plans tangents",
            len(PlanesTangents),
            "\n",
        )
    Frontiere = Vertices
    Faces.append([i for i in range(len(Frontiere))])
    translation_bary = []
    for Element in PlanesTangents[1:]:  #
        for v_f in range(len(Frontiere) - 1):
            if np.vdot(
                np.array(bary - np.array(Frontiere[v_f + 1])), Element[1]
            ) > 0.0001 + np.vdot(np.array(bary - np.array(Frontiere[v_f])), Element[1]):
                Frontiere = Frontiere[v_f:] + Frontiere[:v_f]
                break
        # Produit scalaire entre vecteur direct et (pt-bary).
        max = [np.vdot(np.array(bary - np.array(Frontiere[0])), Element[1]), 0, False]

        indice_front = len(Frontiere) - 1
        for v_ind in range(len(Frontiere)):
            v = Frontiere[v_ind]
            valeur = np.vdot(np.array(bary - np.array(v)), Element[1])
            if max[0] + 0.00001 <= valeur:
                max = [valeur, v_ind, False]
            elif max[0] - 0.00001 <= valeur:  # On a deja vu le max
                if max[1] == v_ind - 1:  # on a vu le max chez le voisin d'avant
                    max = [valeur, v_ind, True]
                elif not max[
                    2
                ]:  # si on a vu le max longtemps avant et TRUE, alors on fait rien
                    max = [valeur, v_ind, False]
            elif max[1] == v_ind - 1 and valeur < max[0] and max[2]:
                indice_front = v_ind - 1
        if not max[2]:
            indice_front = max[1]
        v_ind = indice_front
        valeur = np.vdot(
            np.array(bary - np.array(Frontiere[v_ind % len(Frontiere)])), Element[1]
        )
        if True:
            for v in Frontiere:
                print(np.vdot(np.array(bary - np.array(v)), Element[1]), v, Element[1])
        while True:
            v = Frontiere[v_ind]
            flag = 0
            for i in range(len(Element[2:])):
                if (np.array(v) - np.array(Element[(2 + i)])).tolist() == Frontiere[
                    (v_ind - 1) % len(Frontiere)
                ]:
                    print("Le point", v, "vient du vecteur", Element[2 + i])
                    flag = 1
                    v_ind -= 1
            v_ind = v_ind % len(Frontiere)
            if flag == 0:
                break

        # ici jusqu'a maintenant on est à peu près sûr d'avoir le bon point de départ.
        i_ngbr1 = (v_ind + 1) % len(Frontiere)
        i_ngbr2 = (v_ind - 1) % len(Frontiere)
        if print_flag:
            print(
                "\n",
                v,
                np.vdot(np.array(bary - np.array(v)), Element[1]),
                Frontiere[i_ngbr1],
                np.vdot(np.array(bary - np.array(Frontiere[i_ngbr1])), Element[1]),
                Frontiere[i_ngbr2],
                np.vdot(np.array(bary - np.array(Frontiere[i_ngbr2])), Element[1]),
            )

        # On veut renverser les générateurs le cas echeant pour avoir une enveloppe convexe.
        # On fait la liste des faces adjacentes (normalement 1 par géné max)
        Gene_non_commun = Element[2:]
        Faces_adja = []
        val1 = np.vdot(np.array(bary - np.array(Frontiere[i_ngbr1])), Element[1])
        val2 = np.vdot(np.array(bary - np.array(Frontiere[i_ngbr2])), Element[1])

        # On choisi le voisin pour faire ensuite le zonotope
        if np.allclose(val1, valeur):
            v_ngb = Frontiere[i_ngbr1]
        elif np.allclose(
            val2, valeur
        ):  # Si il n'y a pas de valeurs suivante, c'est que celui d'avant est le bon voisin pour
            v_ngb = Frontiere[i_ngbr2]
        else:
            v_ngb = v

        # si il n'y a aucun cote en commun, il faut les faces adjacentes qui ont ce sommet
        if not np.allclose(val1, valeur) and not np.allclose(val2, valeur):
            for f in Faces:
                if Vertices.index(v) in f:
                    Faces_adja.append(f)
        while np.allclose(val1, valeur) or np.allclose(val2, valeur):
            if np.allclose(val1, valeur):
                if (
                    np.array(Frontiere[i_ngbr1])
                    - np.array(Frontiere[(i_ngbr1 - 1) % len(Frontiere)])
                ).tolist() in Element[2:]:
                    g_ind = Element[2:].index(
                        (
                            np.array(Frontiere[i_ngbr1])
                            - np.array(Frontiere[(i_ngbr1 - 1) % len(Frontiere)])
                        ).tolist()
                    )
                elif (
                    (
                        -np.array(Frontiere[i_ngbr1])
                        + np.array(Frontiere[(i_ngbr1 - 1) % len(Frontiere)])
                    ).tolist()
                    in Element[2:]
                ):  # normalement ca ne peut pas arriver parce que sinon on ne serait pas remontés dans la boucle précédente
                    g_ind = Element[2:].index(
                        (
                            -np.array(Frontiere[i_ngbr1])
                            + np.array(Frontiere[(i_ngbr1 - 1) % len(Frontiere)])
                        ).tolist()
                    )
                else:
                    print(
                        "bizarre, deux point sont sur le meme plan mais ne corresponde a aucun générateur",
                        Frontiere[i_ngbr1],
                        Element[2:],
                        np.array(Frontiere[i_ngbr1])
                        - np.array(Frontiere[(i_ngbr1 - 1) % len(Frontiere)]),
                    )
                for f in Faces:
                    if (
                        Vertices.index(Frontiere[i_ngbr1]) in f
                        and Vertices.index(Frontiere[(i_ngbr1 - 1) % len(Frontiere)])
                        in f
                    ):
                        Faces_adja.append(f)
                        break
                i_ngbr1 = (i_ngbr1 + 1) % len(Frontiere)
                Gene_non_commun[g_ind] = 0
                val1 = np.vdot(
                    np.array(bary - np.array(Frontiere[i_ngbr1])), Element[1]
                )
            if np.allclose(val2, valeur):
                inter_val = np.array(Frontiere[i_ngbr2]) - np.array(
                    Frontiere[(i_ngbr2 + 1) % len(Frontiere)]
                )
                print(
                    "Ok on remonte cest bien, la on part de",
                    Frontiere[(i_ngbr2 + 1) % len(Frontiere)],
                    "on va a ",
                    Frontiere[i_ngbr2],
                    (inter_val).tolist(),
                )
                if inter_val.tolist() in Element[2:]:
                    g_ind = Element[2:].index(inter_val.tolist())
                elif (-inter_val).tolist() in Element[2:]:
                    g_ind = Element[2:].index((-inter_val).tolist())
                    Element[2 + g_ind] = [
                        -Element[2 + g_ind][gii]
                        for gii in range(len(Element[2 + g_ind]))
                    ]
                    # bary = bary - np.array(Element[2 + g_ind])
                    print("on a retourné ", Element[2 + g_ind])
                else:
                    print(
                        "bizarre, deux point sont sur le meme plan mais ne corresponde a aucun générateur",
                        inter_val,
                        Element[2:],
                    )
                for f in Faces:
                    if (
                        Vertices.index(Frontiere[i_ngbr2]) in f
                        and Vertices.index(Frontiere[(i_ngbr2 + 1) % len(Frontiere)])
                        in f
                    ):
                        Faces_adja.append(f)
                        break
                i_ngbr2 = (i_ngbr2 - 1) % len(Frontiere)
                Gene_non_commun[g_ind] = 0
                val2 = np.vdot(
                    np.array(bary - np.array(Frontiere[i_ngbr2])), Element[1]
                )

        if print_flag:
            print(
                "Faces adjacentes",
                Faces_adja,
                [Vertices[v] for v in Faces_adja[0]],
                "Générateurs restants",
                Gene_non_commun,
                "\n",
                Element,
            )

        # il faut que chaque face soit du meme cote que le barycentre par rapport aux autres faces. (ie dans le meme demi espace que Bary).

        for f in Faces_adja:
            vect_1 = np.array(Vertices[f[3]]) - np.array(Vertices[f[1]])
            vect_2 = np.array(Vertices[f[2]]) - np.array(Vertices[f[0]])
            f_normal = np.cross(vect_2, vect_1)
            for g in Gene_non_commun:
                if g != 0:
                    if (
                        np.vdot(np.array(g), f_normal)
                        * np.vdot(f_normal, np.array(bary) - np.array(Vertices[f[0]]))
                        < 0
                    ):
                        for n, ii in enumerate(Element[2:]):
                            if ii == g:
                                Element[2 + n] = [-gi for gi in g]
                                bary = bary + np.array([-gi for gi in g])
                                translation_bary.append([-gi for gi in g])
                                print(
                                    "Le générateur",
                                    g,
                                    " passait derrière la face",
                                    [Vertices[v] for v in f],
                                )
        flag_first_zono = False
        if v_ngb == v:
            vv = MakeFirstZonogone(Element, v)
            print(
                "\nLes deux valeurs conexes sont differentes en voila la preuve",
                np.vdot(
                    np.array(bary - np.array(Frontiere[(v_ind + 1) % len(Frontiere)])),
                    Element[1],
                ),
                valeur,
                np.vdot(
                    np.array(bary - np.array(Frontiere[(v_ind - 1) % len(Frontiere)])),
                    Element[1],
                ),
            )
            flag_first_zono = True
        else:
            vv = MakeZonogone(Element, v, v_ngb)
        if print_flag:
            print(
                "point de départ",
                v,
                v_ngb,
                "\n",
                "numero de face",
                len(Faces) + 1,
                "Element",
                Element,
                "\n Nouvelle face",
                vv,
            )  # ,'\n Frontiere', Frontiere, '\n Vertices', Vertices)

        # MaJ liste Frontiere, de Vertices, et , v_ind est le dernier des points communs
        Frontiere, Vertices, f = MiseAJour(
            Frontiere,
            Vertices,
            vv,
            indice_front,
            v_ind % len(Frontiere),
            flag_first_zono,
            bary,
        )
        if False:
            print("\n Frontiere", Frontiere, "\n")
        Faces.append(f)

    # # Deuxième moitié de zonotope :
    # #Pour chaque face, il y a une face symetrique, sachant que le sym d'un pt Frontiere est un pt Frontiere : Nouveaux points : len(Vertices)- len(Fron)
    # #On duplique la liste et pour chaque point, on change son nom, en se basant sur le fait qu'il est dans la Frontiere ou non.
    # Faces_sym = [[i for i in f] for f in Faces]
    # Additional_points = []
    # # avec l'étape de modification des géné, on a modifié le barycentre
    # nv_bary  = np.array(bary)
    # # nv_bary  = np.array(bary1)
    # print( np.array(nv_bary) - np.array(bary))
    # for i in range (len(Vertices)) :
    #     ind = len(Vertices) + len(Additional_points)
    #     Additional_points.append((2* nv_bary - np.array(Vertices[i]) ).tolist())
    #     for f in range (len(Faces)):
    #         for j in range (len(Faces[f])):
    #             if (Faces[f][j] == i):
    #                 Faces_sym[f][j] = ind
    # if (False):
    #     print(Faces_sym, Faces)
    # #si on fait ça on est en Vertice^2*len(Fronti).
    # Vertices = Vertices + Additional_points
    # Faces = Faces + Faces_sym

    print(len(Faces))
    return Vertices, Faces


# On insert un plan avec deux générateurs dans la liste des plans : On les classe en produit scalaire avec bary décroissant.
# Liste de Liste, pas de np.array normalement
def InsertElement(Liste, Element):
    if len(Liste) == 0:
        return [Element]
    elif len(Liste) < 5:
        for l_indi in range(len(Liste)):
            l_vect = Liste[l_indi]
            if l_vect[0] + 0.00001 < Element[0]:
                return Liste[:l_indi] + [Element] + Liste[l_indi:]
            elif np.abs(l_vect[0] - Element[0]) < 0.00001:
                if (np.abs(l_vect[1] - Element[1]) < 0.00001).all():
                    marqueur = [2, 3]
                    for i in range(2, len(Element)):
                        for j in range(2, len(l_vect)):
                            if np.all(
                                Element[i] == l_vect[j]
                            ):  # Comparaison de tous les elmts d'arrays.
                                marqueur.remove(i)
                    for i in marqueur:
                        l_vect.append(Element[i])
                    return Liste
        return Liste + [Element]
    else:
        indice = int(len(Liste) / 2)
        if Liste[indice][0] > Element[0] + 0.00001:
            return Liste[: indice + 1] + InsertElement(Liste[indice + 1 :], Element)
        else:
            return InsertElement(Liste[: indice + 1], Element) + Liste[indice + 1 :]


def angle_sort(A, direction):
    sin = np.vdot(np.array(direction), np.array(A)) / np.sqrt(np.sum(np.array(A) ** 2))
    return sin


# renvoie les Sommets avec le point d'appui en premier.
def MakeZonogone(Element, pt1, pt2):
    vertices = [pt1]
    vector = np.array(pt2) - np.array(pt1)

    direction = np.array(
        [sum([Element[j + 2][i] for j in range(len(Element) - 2)]) for i in range(3)]
    )
    # Normalement la direction ne peut être colinéaire au premier vecteur !
    direction_perpendic = (
        np.vdot(direction, vector) / (np.sum(direction**2)) * direction - vector
    )
    if print_flag:
        print(
            "\n Les deux points",
            pt1,
            pt2,
            "Direction",
            direction,
            "direction perpendiculaire",
            direction_perpendic,
        )

    # en fonction de si le premier vect est direct ou contraire, on affiche toujours sa famille en prem's,
    # mais dans les deux cas, les contraires doivent etre triés selon direction et pas direction_perpen
    Gen_direct = []
    Gen_contraire = []
    for g in Element[2:]:
        if (
            np.vdot(np.array(direction), np.array(g))
            / np.sqrt(np.sum(np.array(g) ** 2))
            >= 0
        ):
            Gen_direct.append(g)
        else:
            Gen_contraire.append(g)

    # Les tris sont du plus petit au plus grand
    Gen_direct = sorted(Gen_direct, key=lambda A: angle_sort(A, direction_perpendic))
    Gen_contraire = sorted(
        Gen_contraire, key=lambda A: angle_sort(A, -direction_perpendic)
    )
    if np.vdot(vector, direction) < 0:
        Gen_direct_f = Gen_contraire
        Gen_contraire_f = Gen_direct
    else:
        Gen_contraire_f = Gen_contraire
        Gen_direct_f = Gen_direct

    # print('les generateurs dans le bon sens', Gen_direct_f,'\n les géné quon inverse', Gen_contraire_f)
    for g in Gen_direct_f:
        vertices.append([vertices[len(vertices) - 1][i] + g[i] for i in range(3)])
    for g in Gen_contraire_f:
        vertices.append([vertices[len(vertices) - 1][i] + g[i] for i in range(3)])
    # on ne rajoute pas le dernier point qui est déjà pt1.
    if len(Gen_contraire_f) == 0:
        for g in Gen_direct_f[: len(Gen_direct_f) - 1]:
            vertices.append([vertices[len(vertices) - 1][i] - g[i] for i in range(3)])
    else:
        for g in Gen_direct_f:
            vertices.append([vertices[len(vertices) - 1][i] - g[i] for i in range(3)])
        for g in Gen_contraire_f[: len(Gen_contraire_f) - 1]:
            vertices.append([vertices[len(vertices) - 1][i] - g[i] for i in range(3)])

    return vertices


# renvoie les Sommets avec le point d'appui en premier.
def MakeFirstZonogone(Element, pt1):
    vertices = [pt1]
    direction = np.array(
        [sum([Element[j + 2][i] for j in range(len(Element) - 2)]) for i in range(3)]
    )
    if not np.all(np.cross(direction, np.array(Element[2])) == 0):
        direction_perpendic = (
            np.array(Element[2])
            - np.vdot(direction, np.array(Element[2]))
            / (np.sum(direction**2))
            * direction
        )
    else:
        direction_perpendic = (
            np.array(Element[3])
            - np.vdot(direction, np.array(Element[3]))
            / (np.sum(direction**2))
            * direction
        )
    Gen = sorted(Element[2:], key=lambda A: angle_sort(A, direction_perpendic))
    for g in Gen:
        vertices.append([vertices[len(vertices) - 1][i] + g[i] for i in range(3)])
    for g in Gen[: len(Gen) - 1]:
        vertices.append([vertices[len(vertices) - 1][i] - g[i] for i in range(3)])
    return vertices


# tous les sommets sont alignées modulo un cycle dans Front et new_v. indic est l'indice du dernier commun dans
def MiseAJour(
    Frontiere,
    Vertices,
    new_vertices,
    indice_Front,
    indice_Premier_pt,
    flag_first_zono,
    bary,
):
    # Pour que les deux listes commencent par le même, (mais il peut y avoir plusieurs fois ce point).
    Frontiere = Frontiere[indice_Premier_pt:] + Frontiere[:indice_Premier_pt]

    if Frontiere[-1] == new_vertices[1] or new_vertices[-1] == Frontiere[1]:
        nv = new_vertices[1:]
        nv.reverse()
        new_vertices = [new_vertices[0]] + nv

    # Pour qu'ils commencent tous les deux par le premier commun.
    while Frontiere[-1] == new_vertices[-1]:
        new_vertices = [new_vertices[-1]] + new_vertices[: len(new_vertices) - 1]
        Frontiere = [Frontiere[-1]] + Frontiere[: len(Frontiere) - 1]
    if False:
        print(
            "\n Frontiere",
            Frontiere,
            "\n",
            "angrage",
            Frontiere[indice_Front],
            "\n",
            "Premier point",
            Frontiere[indice_Premier_pt],
            "\n",
            "Nouvelle face",
            new_vertices,
            "\n",
        )

    # MAJ face et Vertices
    pts_commun = 0
    f = []
    while new_vertices[pts_commun] == Frontiere[pts_commun]:
        f.append(Vertices.index(new_vertices[pts_commun]))
        pts_commun += 1
    for i in range(len(new_vertices) - pts_commun):
        f.append(len(Vertices))
        Vertices.append(new_vertices[pts_commun + i])
    # MAJ Frontiere
    nvl_frontiere = new_vertices[pts_commun:]
    nvl_frontiere.reverse()
    # print(nvl_frontiere, 'la nouvelle frontiere', flag_first_zono)
    if flag_first_zono:
        # il faut mettre le bon coté truc les arêtes dans le cas ou il n'y qu'un point d'accroche on prend par ex nvl_fronti_1, par rapport à Frontiere[-1] et Frontiere[point_comm].
        point0 = np.array(Frontiere[0])
        normal_1 = np.cross(
            np.array(nvl_frontiere[0]) - point0, np.array(Frontiere[-1]) - point0
        )
        normal_2 = np.cross(
            np.array(nvl_frontiere[-1]) - point0, np.array(Frontiere[-1]) - point0
        )
        normal_1 = normal_1 / np.sqrt(np.sum(normal_1**2))
        normal_2 = normal_2 / np.sqrt(np.sum(normal_2**2))
        if abs(np.vdot(bary, normal_1)) < abs(np.vdot(bary, normal_2)):
            nvl_frontiere.reverse()
        print(
            "\n on ajoute une nouvelle face qui ne tient quà un fil, et on la met dans le bon ordre qui sera lui ",
            abs(np.vdot(bary, normal_1)),
            abs(np.vdot(bary, normal_2)),
            nvl_frontiere,
            "\n",
        )
    Frontiere = [Frontiere[0]] + nvl_frontiere + Frontiere[pts_commun - 1 :]
    if False:
        print("\n\nNouvelle Frontiere")
        for verrt in Frontiere:
            print(verrt)
    return Frontiere, Vertices, f


if __name__ == "__main__":
    main()

# Probleme :
# La face est du mauvais sens. 135
# Dernier probleme irrésolu : Face qui partent d'un seul sommet sont rajoutés dans le mauvais sens.

"""
Nouvelle face [[36.0, -35.0, 33.0], [36.0, -28.0, 36.0], [40.0, -33.0, 36.0],
[40.0, -40.0, 33.0]]
26.21829364838383 [40.0, -40.0, 33.0] [ 0.7787612  -0.57109154 -0.25958707]
30.994695659574546 [40.0, -33.0, 36.0] [ 0.7787612  -0.57109154 -0.25958707]
36.965198173562946 [36.0, -28.0, 36.0] [ 0.7787612  -0.57109154 -0.25958707]
38.98997728700249 [37.0, -24.0, 38.0] [ 0.7787612  -0.57109154 -0.25958707]
42.83186586122111 [35.0, -20.0, 38.0] [ 0.7787612  -0.57109154 -0.25958707]
49.840716638511836 [34.0, -10.0, 40.0] [ 0.7787612  -0.57109154 -0.25958707]
54.149861931216506 [35.0, -2.0, 42.0] [ 0.7787612  -0.57109154 -0.25958707]
57.31682413428862 [31.0, -1.0, 40.0] [ 0.7787612  -0.57109154 -0.25958707]
61.36638236116771 [32.0, 7.0, 41.0] [ 0.7787612  -0.57109154 -0.25958707]
69.10207692277007 [28.0, 16.0, 39.0] [ 0.7787612  -0.57109154 -0.25958707]
69.67316846758635 [28.0, 17.0, 39.0] [ 0.7787612  -0.57109154 -0.25958707]
73.04780032331892 [27.0, 22.0, 38.0] [ 0.7787612  -0.57109154 -0.25958707]
76.11092770006078 [30.0, 31.0, 39.0] [ 0.7787612  -0.57109154 -0.25958707]
81.56225608239802 [30.0, 41.0, 38.0] [ 0.7787612  -0.57109154 -0.25958707]
85.87140137510269 [30.0, 49.0, 37.0] [ 0.7787612  -0.57109154 -0.25958707]
88.25960238069806 [27.0, 50.0, 35.0] [ 0.7787612  -0.57109154 -0.25958707]
91.42656458377017 [27.0, 56.0, 34.0] [ 0.7787612  -0.57109154 -0.25958707]
93.45134369720971 [27.0, 60.0, 33.0] [ 0.7787612  -0.57109154 -0.25958707]
101.2908730851423 [19.0, 66.0, 26.0] [ 0.7787612  -0.57109154 -0.25958707]
101.2908730851423 [23.0, 71.0, 27.0] [ 0.7787612  -0.57109154 -0.25958707]
104.35400046188417 [20.0, 75.0, 21.0] [ 0.7787612  -0.57109154 -0.25958707]
107.98821938344233 [12.0, 75.0, 11.0] [ 0.7787612  -0.57109154 -0.25958707]
107.98821938344233 [8.0, 70.0, 10.0] [ 0.7787612  -0.57109154 -0.25958707]
108.92273282041442 [8.0, 73.0, 7.0] [ 0.7787612  -0.57109154 -0.25958707]

109.70149401789118 [6.0, 73.0, 4.0] [ 0.7787612  -0.57109154 -0.25958707]
110.27258556270746 [9.0, 79.0, 2.0] [ 0.7787612  -0.57109154 -0.25958707]
110.94751193385397 [9.0, 82.0, -2.0] [ 0.7787612  -0.57109154 -0.25958707]
110.99942934701909 [9.0, 83.0, -4.0] [ 0.7787612  -0.57109154 -0.25958707]

110.99942934701909 [13.0, 88.0, -3.0] [ 0.7787612  -0.57109154 -0.25958707]
110.94751193385397 [13.0, 87.0, -1.0] [ 0.7787612  -0.57109154 -0.25958707]
110.27258556270745 [13.0, 84.0, 3.0] [ 0.7787612  -0.57109154 -0.25958707]
109.49382436523071 [15.0, 84.0, 6.0] [ 0.7787612  -0.57109154 -0.25958707]
108.92273282041442 [12.0, 78.0, 8.0] [ 0.7787612  -0.57109154 -0.25958707]
107.98821938344233 [12.0, 75.0, 11.0] [ 0.7787612  -0.57109154 -0.25958707]
107.98821938344233 [8.0, 70.0, 10.0] [ 0.7787612  -0.57109154 -0.25958707]
108.76698058091907 [6.0, 70.0, 7.0] [ 0.7787612  -0.57109154 -0.25958707]

109.70149401789118 [6.0, 73.0, 4.0] [ 0.7787612  -0.57109154 -0.25958707]
110.27258556270746 [9.0, 79.0, 2.0] [ 0.7787612  -0.57109154 -0.25958707]
110.94751193385397 [9.0, 82.0, -2.0] [ 0.7787612  -0.57109154 -0.25958707]
110.99942934701909 [9.0, 83.0, -4.0] [ 0.7787612  -0.57109154 -0.25958707]

110.99942934701909 [8.0, 83.0, -7.0] [ 0.7787612  -0.57109154 -0.25958707]
110.84367710752373 [8.0, 85.0, -12.0] [ 0.7787612  -0.57109154 -0.25958707]
109.1304024730749 [8.0, 87.0, -23.0] [ 0.7787612  -0.57109154 -0.25958707]
107.88438455711209 [8.0, 88.0, -30.0] [ 0.7787612  -0.57109154 -0.25958707]
106.27494474899349 [7.0, 87.0, -37.0] [ 0.7787612  -0.57109154 -0.25958707]
105.0808442461958 [8.0, 89.0, -43.0] [ 0.7787612  -0.57109154 -0.25958707]
99.94102034284927 [8.0, 90.0, -65.0] [ 0.7787612  -0.57109154 -0.25958707]
99.68143327702369 [8.0, 90.0, -66.0] [ 0.7787612  -0.57109154 -0.25958707]
96.82597555294228 [9.0, 90.0, -74.0] [ 0.7787612  -0.57109154 -0.25958707]
93.86668300253064 [9.0, 88.0, -81.0] [ 0.7787612  -0.57109154 -0.25958707]
90.85547303895387 [9.0, 85.0, -86.0] [ 0.7787612  -0.57109154 -0.25958707]
90.2843814941376 [6.0, 79.0, -84.0] [ 0.7787612  -0.57109154 -0.25958707]
87.63659342271666 [7.0, 78.0, -89.0] [ 0.7787612  -0.57109154 -0.25958707]
85.61181430927712 [6.0, 74.0, -91.0] [ 0.7787612  -0.57109154 -0.25958707]
82.54868693253523 [3.0, 65.0, -92.0] [ 0.7787612  -0.57109154 -0.25958707]
80.99116453758174 [4.0, 65.0, -95.0] [ 0.7787612  -0.57109154 -0.25958707]
76.68201924487708 [3.0, 57.0, -97.0] [ 0.7787612  -0.57109154 -0.25958707]
71.90561723368636 [3.0, 50.0, -100.0] [ 0.7787612  -0.57109154 -0.25958707]
67.85605900680727 [2.0, 42.0, -101.0] [ 0.7787612  -0.57109154 -0.25958707]
61.36638236116771 [7.0, 42.0, -111.0] [ 0.7787612  -0.57109154 -0.25958707]
54.357531583876984 [8.0, 32.0, -113.0] [ 0.7787612  -0.57109154 -0.25958707]
53.7864400390607 [8.0, 31.0, -113.0] [ 0.7787612  -0.57109154 -0.25958707]
48.33511165672347 [8.0, 21.0, -112.0] [ 0.7787612  -0.57109154 -0.25958707]
44.0259663640188 [8.0, 13.0, -111.0] [ 0.7787612  -0.57109154 -0.25958707]
40.85900416094669 [8.0, 7.0, -110.0] [ 0.7787612  -0.57109154 -0.25958707]
31.617704617555958 [12.0, -1.0, -116.0] [ 0.7787612  -0.57109154 -0.25958707]
29.592925504116412 [12.0, -5.0, -115.0] [ 0.7787612  -0.57109154 -0.25958707]
26.841302606365232 [13.0, -8.0, -116.0] [ 0.7787612  -0.57109154 -0.25958707]
21.90914835567916 [15.0, -13.0, -118.0] [ 0.7787612  -0.57109154 -0.25958707]
18.53451649994659 [16.0, -18.0, -117.0] [ 0.7787612  -0.57109154 -0.25958707]
14.692627925727969 [18.0, -22.0, -117.0] [ 0.7787612  -0.57109154 -0.25958707]
13.758114488755872 [18.0, -25.0, -114.0] [ 0.7787612  -0.57109154 -0.25958707]
6.022419927153514 [22.0, -34.0, -112.0] [ 0.7787612  -0.57109154 -0.25958707]
12.512096572793075 [17.0, -34.0, -102.0] [ 0.7787612  -0.57109154 -0.25958707]
11.837170201646561 [17.0, -37.0, -98.0] [ 0.7787612  -0.57109154 -0.25958707]
13.031270704444239 [16.0, -39.0, -92.0] [ 0.7787612  -0.57109154 -0.25958707]
4.101475640044203 [22.0, -46.0, -93.0] [ 0.7787612  -0.57109154 -0.25958707]
-1.8690268739441898 [26.0, -51.0, -93.0] [ 0.7787612  -0.57109154 -0.25958707]
-1.9209442871093092 [26.0, -52.0, -91.0] [ 0.7787612  -0.57109154 -0.25958707]
-1.7651920476139633 [26.0, -54.0, -86.0] [ 0.7787612  -0.57109154 -0.25958707]
-0.20766965266046888 [25.0, -54.0, -83.0] [ 0.7787612  -0.57109154 -0.25958707]
-2.9073751372465235 [27.0, -56.0, -83.0] [ 0.7787612  -0.57109154 -0.25958707]
-5.970502513988398 [30.0, -60.0, -77.0] [ 0.7787612  -0.57109154 -0.25958707]
-4.257227879539551 [30.0, -62.0, -66.0] [ 0.7787612  -0.57109154 -0.25958707]
-3.011209963576757 [30.0, -63.0, -59.0] [ 0.7787612  -0.57109154 -0.25958707]
-0.15575223949534966 [29.0, -63.0, -51.0] [ 0.7787612  -0.57109154 -0.25958707]
4.984071663851181 [29.0, -64.0, -29.0] [ 0.7787612  -0.57109154 -0.25958707]
5.243658729676764 [29.0, -64.0, -28.0] [ 0.7787612  -0.57109154 -0.25958707]
-2.595870658255831 [37.0, -70.0, -21.0] [ 0.7787612  -0.57109154 -0.25958707]
0.051917413165112336 [36.0, -69.0, -16.0] [ 0.7787612  -0.57109154 -0.25958707]
0.051917413165112336 [37.0, -69.0, -13.0] [ 0.7787612  -0.57109154 -0.25958707]
1.6613572212837262 [38.0, -68.0, -6.0] [ 0.7787612  -0.57109154 -0.25958707]
4.620649771695364 [38.0, -66.0, 1.0] [ 0.7787612  -0.57109154 -0.25958707]
3.8418885742186166 [40.0, -66.0, 4.0] [ 0.7787612  -0.57109154 -0.25958707]
1.4536875686232573 [43.0, -67.0, 6.0] [ 0.7787612  -0.57109154 -0.25958707]
-2.1805313529348993 [51.0, -67.0, 16.0] [ 0.7787612  -0.57109154 -0.25958707]
0.8306786106418578 [51.0, -64.0, 21.0] [ 0.7787612  -0.57109154 -0.25958707]
-2.3362835924302416 [55.0, -65.0, 23.0] [ 0.7787612  -0.57109154 -0.25958707]
6.90501595096049 [51.0, -57.0, 29.0] [ 0.7787612  -0.57109154 -0.25958707]
15.834811015360529 [45.0, -50.0, 30.0] [ 0.7787612  -0.57109154 -0.25958707]
20.7669652660466 [43.0, -45.0, 32.0] [ 0.7787612  -0.57109154 -0.25958707]
23.466670750632655 [41.0, -43.0, 32.0] [ 0.7787612  -0.57109154 -0.25958707]
Le point [13.0, 88.0, -3.0] vient du vecteur [4.0, 5.0, 1.0]
point de départ [9.0, 83.0, -4.0] [13.0, 88.0, -3.0]
 numero de face 455 Element [67.18113263566076, array([ 0.7787612 , -0.57109154, -0.25958707]), [1.0, 0.0, 3.0], [4.0, 5.0, 1.0]]
 Nouvelle face [[9.0, 83.0, -4.0], [13.0, 88.0, -3.0], [14.0, 88.0, 0.0], [10.0, 83.0, -1.0]]
455
"""
