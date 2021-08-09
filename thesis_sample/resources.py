import os
import topogenesis as tg
import pyvista as pv
import trimesh as tm
import numpy as np
import pandas as pd

# ------------------------------------------------------ #
# convert trimesh to PyVista mesh
# ------------------------------------------------------ #
def tri_to_pv(tri_mesh: tm.Trimesh):
    """
    Takes in a trimesh (triangulated) mesh,
    returns the same mesh as a pyvista (PolyData) mesh

    Args:
        tri_mesh (tm.base.Trimesh): Trimesh

    Returns:
        pv.core.pointset.PolyData: PyVista mesh
    """    
    faces = np.pad(tri_mesh.faces, ((0, 0),(1,0)), 'constant', constant_values=3)
    pv_mesh = pv.PolyData(tri_mesh.vertices, faces)
    return pv_mesh
    
# function for creating transformation matrix between origin [0,0,0] and a point
def transform_mat(point):
    mat = np.identity(4)
    mat[:3,-1] = np.array(point)
    return mat

# ---------------------------------------------- #
# create cuboid meshes on voxel positions
# ---------------------------------------------- #
def cuboids_from_voxels(voxelized_envelope:tg.lattice) -> tm.Trimesh:
    """
    Takes in a voxelized envelope,
    returns cuboid meshes in the position of the voxels
    and a mesh of cuboids combined

    Args:
        voxelized_envelope (tg.lattice)

    Returns:
        vox_cuboids: cuboid meshes representing the voxels seperately
        combined_voxels: a mesh of all the cuboid meshes combined
    """
    # voxel centroids
    vox_cts = voxelized_envelope.centroids

    # voxel size
    unit = voxelized_envelope.unit

    # voxel cuboid meshes
    vox_cuboids = [tm.creation.box(unit*0.99, transform=transform_mat(ct)) for ct in vox_cts]

    # combine voxels into one mesh
    combined_voxels = tm.util.concatenate(vox_cuboids)

    return vox_cuboids, combined_voxels

# ---------------------------------------------- #
# construct intervisibilities graph
# ---------------------------------------------- #   
def construct_graph(ref_vectors:np.ndarray, hitface_id:list, ray_id:list, envelope_lattice:tg.lattice, faces_number:int) -> np.ndarray:   
    """Given the intersections of rays with a set of voxels,
    construct the graph that describes their interdependencies according to these rays.

    Args:
        ref_vectors (np.ndarray): reference vectors corresponding to the visibility target
        hitface_id (list): list of face ids where intersection was detected
        ray_id (list): list of ray ids that had at least one intersection
        envelope_lattice (tg.lattice): topogenesis lattice of the envelope for which the intervisibilities are computed
        faces_number (int): number of faces which each cuboid mesh consists of

    Returns:
        np.ndarray: the intervisibilities graph
    """    
    # voxel centroids
    vox_cts = envelope_lattice.centroids

    # initialize array for inter-visibilities of voxels
    G = np.zeros((len(vox_cts),len(vox_cts),len(ref_vectors)), dtype=bool)

    # how many faces each ray hits
    unq_rays, unq_counts = np.unique(ray_id, return_counts = True)

    f0 = 0 # first face id

    # iterate through the rays
    for ray in unq_rays:

        # the faces that this ray hits
        faces = hitface_id[f0 : f0 + unq_counts[ray]]
        f0 += unq_counts[ray] # first face_id hit by the next ray

        # voxel from which the ray originates
        v_id = np.floor(ray/len(ref_vectors)).astype(int)
        
        # ray direction to which the ray corresponds
        r_dir = ray - v_id*len(ref_vectors)

        # find to which voxel each hit face belongs
        voxels = np.floor(faces/faces_number).astype(int)

        if len(voxels)>1:
            # remove duplicates
            unq_voxs = np.unique(voxels)

            # index of source voxel
            source_id = np.where(unq_voxs == v_id)[0]

            # remove source voxel
            blocking_voxs = np.delete(unq_voxs, source_id)

            # store the blocks for this voxel
            G[blocking_voxs, v_id, r_dir] = 1
    
    return G

# ---------------------------------------------- #
# construct visible rays matrix
# ---------------------------------------------- #   
def construct_visiblerays(ray_sources:np.ndarray, ray_dirs:np.ndarray, context_mesh: tm.Trimesh, envelope_lattice:tg.lattice):
    """Takes in the reference rays (sources and directions), the context mesh that causes possible obstruction of these rays
    and the envelope lattice for which visibilities are computed,
    returns an array inidicating which rays are visible from which voxels given a partial obstruction from a context.

    Args:
        ray_sources (np.ndarray): sources of rays to compute visibilities for
        ray_dirs (np.ndarray): directions of rays to compute visibilities for
        context_mesh (tm.Trimesh): mesh of the context for which rays obstruction is checked
        envelope_lattice (tg.lattice): lattice for which visibilities are computed

    Returns:
        np.ndarray: voxels x rays array indicating which rays are visible from which voxels
    """    
    # intersection of rays from voxel centroids to visibility target with context mesh
    _, ray_id = context_mesh.ray.intersects_id(ray_origins=ray_sources, ray_directions=ray_dirs, multiple_hits=False)

    # number of voxels
    v_numb = len(envelope_lattice.centroids)

    # number of rays
    r_numb = len(ray_sources)
    
    # intialize rays visibility matrix
    U = np.ones((r_numb), dtype=bool)

    # turn rays that had an intersection to 0
    U[ray_id] = 0

    # reshape array according to voxels number
    U = U.reshape((v_numb, -1)).transpose()

    return U

# ------------------------------------------------------ #
# cost index calculation
# ------------------------------------------------------ #
def cost_index_calculation(G, U, x, w):
    G_T = np.transpose(G, axes = [2,0,1])
    G_TT = np.transpose(G, axes = [2,1,0])
    annoying = np.dot(G_T, x)
    annoyed = np.dot(G_TT, x)
    # discard obstructed rays
    c1 = np.multiply((annoying - annoyed), U)
    # calculate cost function
    c = np.dot(c1.transpose(),w)
    return c

# ------------------------------------------------------ #
# reshape and store values into lattice
# (according to envelope shape)
# ------------------------------------------------------ #
def reshape_and_store_to_lattice(values_list, envelope_lattice):
    env_all_vox_id = envelope_lattice.indices.flatten()
    env_all_vox = envelope_lattice.flatten() # envelope inclusion condition: True-False
    env_in_vox_id = env_all_vox_id[env_all_vox] # keep in-envelope voxels (True)

    # initialize array
    values_array = np.full(env_all_vox.shape, 0.0)
    
    # store values for the in-envelope voxels
    values_array[env_in_vox_id] = values_list

    # reshape to lattice shape
    values_array_3d = values_array.reshape(envelope_lattice.shape)

    # convert to lattice
    values_lattice = tg.to_lattice(values_array_3d, envelope_lattice)

    return values_lattice

