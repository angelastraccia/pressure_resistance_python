# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 12:24:51 2022

@author: GALADRIEL_GUEST

This script make a cleaning of the slices that are made in tecplot.
It extract the slices according to the set of points and vectors in input.
For each slice, the x,y,z coordinate of the points enclosed are extracted, and 
a projection of the slice in the slice plan if effectuated, to get a set of 2d points.
Then, the convex hull area is calculated from scipy, and the approximate area is computed from the 
alphashape module. From these two values, the convexity of the slice is computed.
The circularity of the slice is also extracted.
To avoid any irregular slice, all slices which present a product of circularity by convexity inferior to 0.9 are removed.
After the cleaning, the new set of points and vectors are returned.
"""

import importlib
import glob
import os as os
import xml.etree.cElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
import re
import os
from os import listdir
from os.path import isfile, join
import importlib
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull


import tecplot as tp
from tecplot.exception import *
from tecplot.constant import *
from tecplot.constant import PlotType
from tecplot.constant import PlotType, SliceSource
import scipy
import scipy.io
from scipy.interpolate import interp1d
import logging
import skg
from skg import nsphere
import pickle
from tqdm import tqdm
import alphashape
from descartes import PolygonPatch

import math
from itertools import combinations

#%%
os.chdir("L:/vasospasm/pressure_resistance_calculation/vasospasm_resistance")
import geometry_slice as geom
importlib.reload(geom)

#%%


def data_coor(data_file,pinfo,case):
    """
    

    Parameters
    ----------
    data_file : data_file created when loading data on tecplot

    Returns
    -------
    coordinates_fluent : coordinates of all the points in the fluent simulation

    """

    zone_name = pinfo + "_" + case + ".walls"

    cx = data_file.zone(zone_name).values("X")[:]
    cy = data_file.zone(zone_name).values("Y")[:]
    cz = data_file.zone(zone_name).values("Z")[:]
    x_base = np.asarray(cx)
    y_base = np.asarray(cy)
    z_base = np.asarray(cz)

    coordinates_fluent = np.array([x_base, y_base, z_base]).T

    return coordinates_fluent

def find_closest(data_file,pinfo,case, origin, vessel_name):
    """


    Parameters
    ----------
    origin : coordinates of the centerline point on which one want to work

    Returns
    -------
    coordinates of the closest point in the Fluent data to the centerline point, by minimazation with the euclidean distance

    """

    L = []
    coordinates_fluent = data_coor(data_file,pinfo,case)
    for i in range(coordinates_fluent.shape[0]):
        b = np.linalg.norm(coordinates_fluent[i, :] - origin)
        L.append(b)

    lmin = np.min(L)
    imin = L.index(lmin)

    return coordinates_fluent[imin, :]


def find_slice(slices, origin):
    """


    Parameters
    ----------
    slices : type : generator, slice generator.
    origin : coordinates of the origin of the slice searched.

    Returns
    -------
    dict_slice : dictionary of the slices from the generator.
    imin : index of the searched slice.

    """
    L = []
    dict_slice = {}
    i = 0
    
    # Loop through the slices and determine the average x,y,z values
    for s in slices:
        dict_slice["{}".format(i)] = s
        (x, y, z) = (
            np.mean(s.values("X")[:]),
            np.mean(s.values("Y")[:]),
            np.mean(s.values("Z")[:]),
        )
        L.append(np.array([x, y, z]))
        i += 1

    # Calculate the distance between the centerline point and the slice origin
    Lnorm = [np.linalg.norm(x - origin) for x in L]
    
    # Identify which slice is closest to the centerline point and return the corresponding index
    mini = min(Lnorm)
    imin = Lnorm.index(mini)

    return dict_slice, imin


def get_slice(data_file,origin, normal, vessel_name,pinfo,case,i_vessel):
    """
    Compute the pressure in a given slice.
    Assuming that every data that could be needed from fluent are loaded.


    Parameters
    ----------
    origin : (3,1) array of coordinates for the centerline point
    vectors : (3,1) array for normal vector to slice

    Returns
    -------
    Slice_array: array of all x,y,z coordinates to the slice
    origin_slice: (3,1) array for Fluent coordinates closest to the centerline point
    """

    frame = tp.active_frame()
    plot = frame.plot()

    plot.show_slices = True
    slices = plot.slices(0)
    slices.show = True

    # Identifies the closest point in the Fluent data to the centerline point
    origin_s = find_closest(data_file,pinfo,case,origin, vessel_name)
    origin_slice = (origin_s[0], origin_s[1], origin_s[2])
    normal = (normal[0], normal[1], normal[2])

    # Extract slice at centerline point with normal from Tecplot
    slices_0 = tp.data.extract.extract_slice(
        mode=ExtractMode.OneZonePerConnectedRegion,
        origin=origin_slice,
        normal=normal,
        dataset=data_file,
    )
    
    # Identify which part of the slice is closest to the centerline point
    dict_slice, n_iteration = find_slice(slices_0, origin_slice)

    # Define the final slice as the one closest to the centerline point
    final_slice = dict_slice.get("{}".format(n_iteration))
    X_array = np.asarray(final_slice.values("X")[:])
    Y_array = np.asarray(final_slice.values("Y")[:])
    Z_array = np.asarray(final_slice.values("Z")[:])
    Slice_array= np.array((X_array,Y_array,Z_array))
    
    # Save figures with slices through vessel in Tecplot
    pathused = 'L:/vasospasm/' + pinfo + '/' + case + '/4-results/pressure_resistance/'
    image_name = pathused + 'slices_vessel_' + vessel_name + '.png'
    print(image_name)
    tp.export.save_png(image_name,
         width=1049,
         region=ExportRegion.AllFrames,
         supersample=1,
         convert_to_256_colors=False)
    
    return Slice_array,origin_slice




def orthogonal_projection(Slice_array,origin,normal):
    slice_rev = Slice_array.T
    
    normal = normal/ np.linalg.norm(normal)
    
    U = np.array([-normal[1],normal[0],0])
    u = U/np.linalg.norm(U)
    
    V = np.array([-normal[0]*normal[2],-normal[1]*normal[2],normal[0]*normal[0]+normal[1]*normal[1]])
    v = V/np.linalg.norm(V)
    
    slice_proj = np.zeros((slice_rev.shape[0],2))
    for i in range(slice_rev.shape[0]):
        
        xprime = np.dot(U,slice_rev[i,:]-origin)
        yprime = np.dot(V,slice_rev[i,:]-origin)
        
        slice_proj[i,0] = xprime
        slice_proj[i,1] = yprime
        
    return slice_proj


def compute_on_slice_convex(data_file,i_vessel, dpoints, dvectors,pinfo,case):
    """


    Parameters
    ----------
    i_vessel :index of the vessel.

    Returns
    -------
    Lslice : list of area, circularity, and convexity of the slice

    """

    vessel_name = dpoints.get("points{}".format(i_vessel))[0]

    points = dpoints.get("points{}".format(i_vessel))[1]
    vectors = dvectors.get("vectors{}".format(i_vessel))[1]
    dslice = {}
    num_points = points.shape[0]
    Lslice = np.zeros((num_points,3))
    if num_points > 2:
        print(
            "### Compute on ",
            dpoints.get("points{}".format(i_vessel))[0],
            " ### Vessel ",
            i_vessel,
            "/",
            len(dpoints),
            "\n",
        )

        # Progress bar
        # Cycle through all of the slices
        for j in tqdm(range(len(points))):
            
            origin = points[j, :]
            normal = vectors[j, :]
            
            
            Slice_array,origin_slice = get_slice(
                data_file,origin, normal, vessel_name,pinfo,case,i_vessel
            )
            
            # Projects slice coordinates onto a 2D plane
            one_rev = orthogonal_projection(Slice_array,origin,normal)
            
            # Convex Hull : Perimeter and Area
            hull = ConvexHull(one_rev)
            HullP = hull.area # Perimeter of the convex hull (N-1 dimension)
            HullArea = hull.volume # Area of the convex hull 
            
            
            # Alpha shape
            Alpha = alphashape.alphashape(one_rev,500)
            Area = Alpha.area
            P = Alpha.length
            
            # Define morphological metrics :
            if P != 0:
                circularity = 4*np.pi*Area/(P*P)
            else:
                circularity = 0
            if HullArea !=0:
                convexity = Area/HullArea
            else: 
                convexity = 0
            
            
            Lslice[j,0] = Area
            Lslice[j,1] = circularity
            Lslice[j,2] = convexity
            
            if P!=0 and circularity !=0 :
                fig,ax = plt.subplots(figsize=(7, 7))
                ax.scatter(one_rev[:,0],one_rev[:,1],marker = 'o')
                
                for simplex in hull.simplices:
                    ax.plot(one_rev[simplex, 0], one_rev[simplex, 1],'k-')
                
                
                ax.add_patch(PolygonPatch(Alpha, alpha = 0.2))
                plt.show()
            print('\n')
            print("   $$ Control point ", j, "Circularity :  = ", circularity)
            print("   $$ Control point ", j, "Convexity :  = ",convexity)
            #print("   $$ Control point ", j, "Area :  = ", Area)
            
            

            
    else:
        Lslice = [0] * (len(points))
        

    return Lslice

def get_list_files_dat(pinfo, case, num_cycle):
    """


    Parameters
    ----------
    pinfo : str, patient information, composed of pt/vsp + number.
    num_cycle : int, number of the cycle computed
    case : str, baseline or vasospasm

    Returns
    -------
    onlyfiles : list of .dat files for the current patient, for the case and cycle wanted.
    """

    num_cycle = str(num_cycle)

    pathwd = "L:/vasospasm/" + pinfo + "/" + case + "/3-computational/hyak_submit"

    os.chdir(pathwd)
    onlyfiles = []
    for file in glob.glob("*.dat"):
        if pinfo + "_" + case + "_cycle" + num_cycle in file:
            onlyfiles.append(file)
    indices = [l[13:-4] for l in onlyfiles]

    return onlyfiles, indices,pathwd



def compute_radius(pinfo,case,num_cycle, dpoints,dvectors,i_vessel):
    
    os.chdir("L:/vasospasm/pressure_resistance_calculation/vasospasm_resistance")

    import geometry_slice as geom

    importlib.reload(geom)


    dslice={}
    
    onlydat, indices_dat,pathwd = get_list_files_dat(pinfo, case, num_cycle) # Get the dat files for the patient and its case
    print(pathwd)

    filename = onlydat[0]

    # print(
    #     " ############  Step 2 : Connection to Tecplot  ############ Time step : ",
    #     i,
    #     "\n",
    # )
    logging.basicConfig(level=logging.DEBUG)

    # To enable connections in Tecplot 360, click on:
    #   "Scripting" -> "PyTecplot Connections..." -> "Accept connections"

    tp.session.connect()
    tp.new_layout()
    frame = tp.active_frame()

    dir_file = pathwd + '/' + filename
   
    data_file = tp.data.load_fluent(
        case_filenames=[
            pathwd + '/' 
            + pinfo
            + "_"
            + case
            + ".cas"
        ],
        data_filenames=[dir_file],
    )

    tp.macro.execute_command("$!RedrawAll")
    
    # # Change view
    # tp.active_frame().plot().view.theta=-175.22
    # tp.active_frame().plot().view.position=(0.035059,
    #     tp.active_frame().plot().view.position[1],
    #     tp.active_frame().plot().view.position[2])
    # tp.active_frame().plot().view.position=(tp.active_frame().plot().view.position[0],
    #     0.515035,
    #     tp.active_frame().plot().view.position[2])
    # tp.active_frame().plot().view.position=(tp.active_frame().plot().view.position[0],
    #     tp.active_frame().plot().view.position[1],
    #     -0.190815)
    # tp.active_frame().plot().view.width=0.110197
    # tp.active_frame().plot().view.psi=23.894
    # tp.active_frame().plot().view.theta=-169.736
    # tp.active_frame().plot().view.alpha=-7.00612
    # tp.active_frame().plot().view.position=(0.035059,
    #     tp.active_frame().plot().view.position[1],
    #     tp.active_frame().plot().view.position[2])
    # tp.active_frame().plot().view.position=(tp.active_frame().plot().view.position[0],
    #     0.223477,
    #     tp.active_frame().plot().view.position[2])
    # tp.active_frame().plot().view.position=(tp.active_frame().plot().view.position[0],
    #     tp.active_frame().plot().view.position[1],
    #     0.0692605)
    # tp.active_frame().plot().view.width=0.110197

    # Use translucency
    tp.active_frame().plot(PlotType.Cartesian3D).use_translucency=True
    tp.active_frame().plot().fieldmaps(2).effects.surface_translucency=90
    print("Translucency enabled")
    # Set contour to pressure
    tp.active_frame().plot().rgb_coloring.red_variable_index = 3
    tp.active_frame().plot().rgb_coloring.green_variable_index = 3
    tp.active_frame().plot().rgb_coloring.blue_variable_index = 13
    tp.active_frame().plot().contour(0).variable_index = 3
    tp.active_frame().plot().contour(1).variable_index = 4
    tp.active_frame().plot().contour(2).variable_index = 5
    tp.active_frame().plot().contour(3).variable_index = 6
    tp.active_frame().plot().contour(4).variable_index = 7
    tp.active_frame().plot().contour(5).variable_index = 8
    tp.active_frame().plot().contour(6).variable_index = 9
    tp.active_frame().plot().contour(7).variable_index = 10
    tp.active_frame().plot().show_contour = True
    tp.macro.execute_command("$!RedrawAll")
    # Turn off contour for walls
    tp.active_frame().plot().fieldmaps(1).contour.show=False
    # Slice plotting
    tp.active_frame().plot(PlotType.Cartesian3D).show_slices = True
    slices = frame.plot().slices()
    slices.contour.show = True
    frame = tp.active_frame()
    plot = frame.plot()
    plot.show_slices = True
    slices = plot.slices(0)
    slices.show = True
    tp.macro.execute_command("$!RedrawAll")
    
    return compute_on_slice_convex(data_file,i_vessel,dpoints,dvectors,pinfo,case)



def get_dCS(pinfo,case,num_cycle,dpoints,dvectors):
    
    # Gets slices
    dCS = {}    
    
    # Cycle through all vessels
    for i_vessel in range(len(dpoints)):
        vessel_name = dpoints.get("points{}".format(i_vessel))[0]
        dCS['slice{}'.format(i_vessel)] = vessel_name, compute_radius(pinfo,case,num_cycle, dpoints, dvectors, i_vessel)
    return dCS

    
    
def morphometric_cleaning(dCS,dpoints,dvectors,ddist,dradii):
    # dCS structure  : 
        #dict of the slices of the vessel concerned by the pressure compute 
        # Inside a point of a vessel : 3,1 list : Area, convexity, circularity:
        # identify the slices which have a weak convexity*circularity criterion
        # Create new dpoints and dvectors with only the points and normal which are reliable
        # Use these two to compute pressure.
        
        
    dCS_cleaned,dpoints_cleaned,dvectors_cleaned,ddist_cleaned,dradii_cleaned = {},{},{},{},{}
    
    # Determines which slices to remove based on circularity and convexity
    for i_vessel in range(len(dCS)):
        
        # Identify the slice
        vessel_name = dCS.get("slice{}".format(i_vessel))[0]
        vessel_slice = dCS.get("slice{}".format(i_vessel))[1]
        
        # Remove slices that do not match the circularity/convexity criterion
        indices_to_remove = []
        num_slices = vessel_slice.shape[0]
        for j in range(num_slices):
            criterion = vessel_slice[j][1] * vessel_slice[j][2]
            if criterion < 0.95:
                #print(i_vessel,j)
                print(vessel_name)
                print('Slice #' + str(j))
                print(vessel_slice[j][1])
                print(vessel_slice[j][2])

                indices_to_remove.append(j)
        
        # Create new arrays with removed slices
        num_points = dpoints.get("points{}".format(i_vessel))[1].shape[0]
        indices_to_keep = [i for i in range(num_points) if i not in indices_to_remove]
        cleaned_num_points = num_points-len(indices_to_remove)
        cleaned_points = np.zeros((cleaned_num_points,3))
        cleaned_vectors =  np.zeros((cleaned_num_points,3))
        cleaned_CS =  np.zeros((cleaned_num_points,3))
        cleaned_distance =  np.zeros((cleaned_num_points))
        cleaned_radii =  np.zeros((cleaned_num_points))
        
        save_index = 0
        for k in indices_to_keep:
            cleaned_points[save_index,:] = dpoints.get("points{}".format(i_vessel))[1][k,:]
            cleaned_vectors[save_index,:] = dvectors.get("vectors{}".format(i_vessel))[1][k,:]
            cleaned_CS[save_index,:] = dCS.get("slice{}".format(i_vessel))[1][k,:]
            cleaned_distance[save_index] = ddist.get("distance{}".format(i_vessel))[1][k]
            cleaned_radii[save_index] = dradii.get("radii{}".format(i_vessel))[1][k]
            
            save_index += 1
        
        # Create new dictionaries with removed slices
        dCS_cleaned['slice{}'.format(i_vessel)] = vessel_name,cleaned_CS
        dpoints_cleaned['points{}'.format(i_vessel)] =  vessel_name, cleaned_points
        dvectors_cleaned['vectors{}'.format(i_vessel)] =  vessel_name,cleaned_vectors
        ddist_cleaned['distance{}'.format(i_vessel)] =  vessel_name,cleaned_distance
        dradii_cleaned['radii{}'.format(i_vessel)] =  vessel_name,cleaned_radii
        
    return dCS_cleaned,dpoints_cleaned,dvectors_cleaned,ddist_cleaned,dradii_cleaned



            
            

    
    