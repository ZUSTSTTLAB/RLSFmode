import pymesh
import numpy
"""
read_ply.py: Save a ply file to disk using pymesh and load the attributes used by RMSIF. 
"""


def save_ply(
    filename,
    vertices,
    faces=[],
    normals=None,
    charges=None,
    vertex_theta=None,
    dists=None,
    hphob=None,
    iface=None,
    normalize_charges=False,
):
    """ Save vertices, mesh in ply format.
        vertices: coordinates of vertices
        faces: mesh
    """
    mesh = pymesh.form_mesh(vertices, faces)
    if normals is not None:
        n1 = normals[:, 0]
        n2 = normals[:, 1]
        n3 = normals[:, 2]
        mesh.add_attribute("vertex_nx")
        mesh.set_attribute("vertex_nx", n1)
        mesh.add_attribute("vertex_ny")
        mesh.set_attribute("vertex_ny", n2)
        mesh.add_attribute("vertex_nz")
        mesh.set_attribute("vertex_nz", n3)
    if charges is not None:
        mesh.add_attribute("charge")
        mesh.set_attribute("charge", charges)
    if dists is not None:
        mesh.add_attribute("dists")
        mesh.set_attribute("dists", dists)
    if vertex_theta is not None:
        mesh.add_attribute("vertex_theta")
        mesh.set_attribute("vertex_theta", vertex_theta)
    if iface is not None:
        mesh.add_attribute("vertex_iface")
        mesh.set_attribute("vertex_iface", iface)

    pymesh.save_mesh(
        filename, mesh, *mesh.get_attribute_names(), use_float=True, ascii=True
    )

