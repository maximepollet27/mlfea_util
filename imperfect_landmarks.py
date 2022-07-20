import os
import math
import pickle
import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
from compas.datastructures import Mesh
from compas.geometry import Point, Line, Plane, Circle, is_point_in_circle, intersection_line_plane

def load_mesh(path, get_points=False):

    # Load mid, top, and bot points and elem_numbers
    mid_3d = np.loadtxt(path.joinpath('full_grid_mid.csv'), delimiter=',')
    top_3d = np.loadtxt(path.joinpath('full_grid_top.csv'), delimiter=',')
    bot_3d = np.loadtxt(path.joinpath('full_grid_bot.csv'), delimiter=',')
    mid_2d = np.zeros(mid_3d.shape)
    top_2d = np.zeros(mid_3d.shape)
    bot_2d = np.zeros(mid_3d.shape)
    mid_2d[:, :2] = mid_3d[:, :2]
    top_2d[:, :2] = top_3d[:, :2]
    bot_2d[:, :2] = bot_3d[:, :2]
    # Import the corners of each element
    elem_corner1 = np.loadtxt(path.joinpath('elem_corner1.csv'), dtype=np.int16)
    elem_corner2 = np.loadtxt(path.joinpath('elem_corner2.csv'), dtype=np.int16)
    elem_corner3 = np.loadtxt(path.joinpath('elem_corner3.csv'), dtype=np.int16)
    elem_corner4 = np.loadtxt(path.joinpath('elem_corner4.csv'), dtype=np.int16)
    # Gather the elements' corner points in a list of lists
    corners_list = []
    for i in range(len(elem_corner1)):
        corners_list.append([elem_corner1[i], elem_corner2[i], elem_corner3[i], elem_corner4[i]])

    # Build 2d and 3d (mid) meshes
    mesh_2d = Mesh.from_vertices_and_faces(mid_2d, corners_list)
    mesh_3d = Mesh.from_vertices_and_faces(mid_3d, corners_list)

    if get_points:
        mtop_3d = Mesh.from_vertices_and_faces(top_3d, corners_list)
        mbot_3d = Mesh.from_vertices_and_faces(bot_3d, corners_list)
        return mesh_2d, mesh_3d, mtop_3d, mbot_3d
    else:
        return mesh_2d, mesh_3d

def find_face(span, pt, mesh_2d):

    # build circle to investigate
    radius = 1.5 * math.sqrt(span * span / 8000)
    circle = Circle(Plane(pt[:2], [0, 0, 1]), radius)

    # Find points from 2d mesh that are within the circle
    potential_vertices = [vert for vert in mesh_2d.vertices() if is_point_in_circle(mesh_2d.vertex_coordinates(vert), circle)]

    # Find faces of interest
    potential_faces = []
    for potential_vert in potential_vertices:
        pot_faces = [fac for fac in mesh_2d.faces() if potential_vert in mesh_2d.face_vertices(fac)]
        potential_faces += pot_faces
    potential_faces = list(set(potential_faces))

    # Loop over potential faces and find which is the closest
    min_dist = math.inf
    closest_face = None
    for fac in potential_faces:
        centroid = mesh_2d.face_centroid(fac)
        dist = np.linalg.norm(np.array(centroid[:2]) - pt[:2])
        if dist < min_dist:
            closest_face = fac
            min_dist = dist
    
    return closest_face

def get_landmarks(isample):

    # Inputs
    span = float(isample.split('_')[0][1:])
    height = float(isample.split('_')[1][1:])
    thickness = float(isample.split('_')[2][1:])
    psample = 'span{:.6f}_height{:.6f}_radius{:.6f}_thick{:.6f}'.format(span, height, span/10, thickness)
    ROOT_PATH = Path().cwd().parent

    # Load perfect meshes
    #pmesh_2d, pmesh_3d = load_mesh(ROOT_PATH.joinpath('data_backup/training_data/{:s}'.format(psample)))
    pmesh_2d, pmesh_3d = load_mesh(ROOT_PATH.joinpath('./mlfea_util/data/perfect_shells/{:s}'.format(psample)))

    # Load imperfect meshes
    #imesh_2d, imesh_3d = load_mesh(ROOT_PATH.joinpath('data_backup/training_data_def/{:s}'.format(isample)))
    imesh_2d, imesh_3d, itop_3d, ibot_3d = load_mesh(ROOT_PATH.joinpath('./mlfea_util/data/imperfect_shells/{:s}'.format(isample)), get_points=True)

    # load landmarks field and scale it
    ref_landmarks = np.loadtxt(ROOT_PATH.joinpath('./mlfea_util/data/landmarks/ref_landmarks_6417.csv'), delimiter=',')
    landmarks = ref_landmarks * span/2

    # Loop over all landmarks
    mesh_weights = {}
    mesh_defs = {}
    for ind, pt in enumerate(landmarks):

        # find relevant elem
        closest_face = find_face(span, pt, pmesh_2d)
        
        # Get perfect point
        pface_plane = Plane(pmesh_3d.face_plane(closest_face)[0], pmesh_3d.face_plane(closest_face)[1])
        perfect_pt = intersection_line_plane(Line([pt[0], pt[1], 0], [pt[0], pt[1], 1000]), pface_plane)
        # Create a line along the normal that passes through the perfect_pt
        normal = pmesh_3d.face_plane(closest_face)[1]
        start_point = Point(perfect_pt[0]+normal[0], perfect_pt[1]+normal[1], perfect_pt[2]+normal[2])
        end_point = Point(perfect_pt[0]-normal[0], perfect_pt[1]-normal[1], perfect_pt[2]-normal[2])

        # get imperfect point
        iface_plane = Plane(imesh_3d.face_plane(closest_face)[0], imesh_3d.face_plane(closest_face)[1])
        imperfect_pt = intersection_line_plane(Line(start_point, end_point), iface_plane)

        # Calc distance between perfect and imperfect
        distance = perfect_pt.distance_to_point(imperfect_pt)
        if imperfect_pt[2] < perfect_pt[2]:
            distance *= -1

        # Calc thickness -> find top and bot points and calc dist
        def_normal = imesh_3d.face_plane(closest_face)[1]
        def_start_point = Point(imperfect_pt[0]+def_normal[0], imperfect_pt[1]+def_normal[1], imperfect_pt[2]+def_normal[2])
        def_end_point = Point(imperfect_pt[0]-def_normal[0], imperfect_pt[1]-def_normal[1], imperfect_pt[2]-def_normal[2])
        itop_face_plane = Plane(itop_3d.face_plane(closest_face)[0], itop_3d.face_plane(closest_face)[1])
        it_pt = intersection_line_plane(Line(def_start_point, def_end_point), itop_face_plane)
        ibot_face_plane = Plane(ibot_3d.face_plane(closest_face)[0], ibot_3d.face_plane(closest_face)[1])
        ib_pt = intersection_line_plane(Line(def_start_point, def_end_point), ibot_face_plane)
        def_thick = it_pt.distance_to_point(ib_pt)

        # Calc weights of each node for imperfect stress
        # Get the other points
        i_mid, j_mid, k_mid, l_mid = imesh_3d.face_vertices(closest_face)
        i = np.array(imesh_3d.vertex_coordinates(i_mid))
        j = np.array(imesh_3d.vertex_coordinates(j_mid))
        k = np.array(imesh_3d.vertex_coordinates(k_mid))
        l = np.array(imesh_3d.vertex_coordinates(l_mid))

        # Calc weights
        wi = 1/np.linalg.norm(imperfect_pt - i)
        wj = 1/np.linalg.norm(imperfect_pt - j)
        wk = 1/np.linalg.norm(imperfect_pt - k)
        wl = 1/np.linalg.norm(imperfect_pt - l)

        # Store results in dicts
        mesh_defs[ind] = (distance, def_thick)

        mesh_weights[ind] = {i_mid: wi,
                            j_mid: wj,
                            k_mid: wk,
                            l_mid: wl}
        
    # Save temporary results in input folders
    with ROOT_PATH.joinpath('data_backup/training_data_def/{:s}/weights.pickle'.format(isample)).open('wb') as f:
        pickle.dump(mesh_weights, f)
    with ROOT_PATH.joinpath('data_backup/training_data_def/{:s}/landmarks_defs.pickle'.format(isample)).open('wb') as f:
        pickle.dump(mesh_defs, f)

    return (isample, mesh_defs, mesh_weights)

if __name__ == '__main__':

    # root path
    ROOT_PATH = Path().cwd().parent

    # Load all samples
    #all_isamples = os.listdir(ROOT_PATH.joinpath('./data_backup/training_data_def'))
    all_isamples = os.listdir(ROOT_PATH.joinpath('./mlfea_util/data/imperfect_shells'))

    for sample in all_isamples:
        results = get_landmarks(sample)

    # Distribute nalyses over 15 cores
    # with Pool(14) as pool:
    #     results = list(tqdm.tqdm(pool.imap(get_landmarks, all_isamples), total=len(all_isamples)))
    
    # Save full results in relevant folder
    with Path.cwd().joinpath('./data/weights_and_defs.pickle').open('wb') as f:
        pickle.dump(results, f)
