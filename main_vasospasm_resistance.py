# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 13:17:42 2022

@author: Angela and Francois


"""


# %% Imports

import glob
import os as os
import numpy as np
import tecplot as tp
from tecplot.constant import PlotType
import pickle
from tqdm import tqdm
import pandas as pd
import time

from tecplot.constant import PlotType, SliceSource
from tecplot.exception import *
from tecplot.constant import *
import scipy
import scipy.io
from scipy.interpolate import interp1d
import logging
import skg
from skg import nsphere
import matplotlib.pyplot as plt
import re
import os
from os import listdir
from os.path import isfile, join
import importlib
import xml.etree.cElementTree as ET

# Old library
#import geometry_slice as geom 

os.chdir("L:/vasospasm/calculation_resistance_dissipation/pressure_resistance_python")
import get_cross_section as cross_section


#%% Functions

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

    pathwd = "L:/vasospasm/" + pinfo + "/" + case + "/3-computational/hyak_submit/"

    os.chdir(pathwd)
    onlyfiles = []
    for file in glob.glob("*.dat"):
        if pinfo + "_" + case + "_cycle" + num_cycle in file:
            onlyfiles.append(file)
    indices = [l[13:-4] for l in onlyfiles]

    return onlyfiles, indices,pathwd


def save_dict(dico, name):
    """


    Parameters
    ----------
    dico : dictionary one wants to save
    name : str. path + name of the dictionary

    Returns
    -------
    None.

    """

    with open(name + ".pkl", "wb") as f:
        pickle.dump(dico, f)


def load_dict(name):
    """


    Parameters
    ----------
    name : str. path + name of the dictionary one wants to load

    Returns
    -------
    b : the loaded dictionary

    """
    with open(name + ".pkl", "rb") as handle:
        b = pickle.load(handle)
    return b



# %% Functions Tecplot

def data_coor(data_file,pinfo,case):
    """


    Parameters
    ----------
    data_file : data_file created when loading data on tecplot

    Returns
    -------
    coordinates_fluent : coordinates of all the points or the fluent simulation

    """

    name = pinfo + "_" + case + ".walls"

    # print('chosen file : ', name)

    cx = data_file.zone(name).values("X")[:]
    cy = data_file.zone(name).values("Y")[:]
    cz = data_file.zone(name).values("Z")[:]
    x_base = np.asarray(cx)
    y_base = np.asarray(cy)
    z_base = np.asarray(cz)

    coordinates_fluent = np.array([x_base, y_base, z_base]).T

    return coordinates_fluent


def find_closest(data_file,pinfo,case, origin, name):
    """


    Parameters
    ----------
    data_file : tecplot data_file
    pinfo,case : str of the patient info
    origin :coordinates of the control point on which one want to work

    Returns
    -------
    coordinates of the closest point to the control points, by minimazation with the euclidean distance

    """

    
    # Return x,y,z coordinates from the Tecplot data
    coordinates_fluent = data_coor(data_file,pinfo,case)
    
    # Calculate the distance between the coordiantes and the centerline point 
    L = []
    for i in range(coordinates_fluent.shape[0]):
        b = np.linalg.norm(coordinates_fluent[i, :] - origin)
        L.append(b)

    # Return the coordinates closet to the centerline point
    min_val = np.min(L)
    min_index = L.index(min_val)

    return coordinates_fluent[min_index, :]


def find_slice(slices, origin):
    """


    Parameters
    ----------
    slices : type : generator, slice generator.
    origin : coordinates of the origin of the slice searched.

    Returns
    -------
    dict_slice : dictionnary of the slices form the generator.
    min_index : index of the searched slice.

    """
    L = []
    dict_slice = {}
    i = 0
    for s in slices:
        dict_slice["{}".format(i)] = s
        (x, y, z) = (
            np.mean(s.values("X")[:]),
            np.mean(s.values("Y")[:]),
            np.mean(s.values("Z")[:]),
        )
        L.append(np.array([x, y, z]))
        i += 1

    Lnorm = [np.linalg.norm(x - origin) for x in L]
    min_val = min(Lnorm)
    min_index = Lnorm.index(min_val)

    return dict_slice, min_index


def get_pressure(data_file,origin, normal, name,pinfo,case):
    """
    Compute the pressure in a given slice.
    Assuming that every data that could be needed from fluent are loaded.


    Parameters
    ----------
    data_file: current tecplot data_file
    origin : (3,1) array of the origin coordinates of the slice.
    vectors : (3,1) array of the normal vector coordinate
    name : str of the name of the vessel
    pinfo : str (ex : 'pt7')
    case : str (ex : 'baseline')

    Returns
    -------
    min_pressure : list, minimum pressure value in the slice
    avg_pressure :list, averaage pressure value in the slice
    max_pressure : list, maximum pressure value in the slice

    """

    frame = tp.active_frame()
    plot = frame.plot()

    plot.show_slices = True
    slices = plot.slices(0)
    slices.show = True

    # Find the closest coordinates to the centerline point
    origin_s = find_closest(data_file,pinfo,case,origin, name)
    origin_slice = (origin_s[0], origin_s[1], origin_s[2])
    normal = (normal[0], normal[1], normal[2])

    slices_0 = tp.data.extract.extract_slice(
        mode=ExtractMode.OneZonePerConnectedRegion,
        origin=origin_slice,
        normal=normal,
        dataset=data_file,
    )

    dict_slice, which_slice_index = find_slice(slices_0, origin_slice)

    final_slice = dict_slice.get("{}".format(which_slice_index))
    min_pressure = np.min(final_slice.values("Pressure")[:])
    avg_pressure = np.mean(final_slice.values("Pressure")[:])
    max_pressure = np.max(final_slice.values("Pressure")[:])

    return min_pressure, avg_pressure, max_pressure


def compute_along(data_file,i_vessel, dpoints, dvectors,pinfo,case):
    """


    Parameters
    ----------
    data_file : tecplot datafile
    i_vessel :index of the vessel.
    dpoints: dictionary of the control points
    dvectors : dictionary of the normal vectors
    pinfo : str, ex : 'pt2'
    case : str, ex : 'baseline'

    Returns
    -------
    Lpressure : numpy array of the min/avg and max pressure along the vessel.

    """

    name = dpoints.get("points{}".format(i_vessel))[0]

    Lavg, Lmin, Lmax = [], [], []

    points = dpoints.get("points{}".format(i_vessel))[1]
    vectors = dvectors.get("vectors{}".format(i_vessel))[1]
    dslice = {}
    num_points = points.shape[0]
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

        for j in tqdm(range(num_points)):
            
            origin = points[j, :]
            normal = vectors[j, :]
            min_pressure, avg_pressure, max_pressure = get_pressure(
                data_file,origin, normal, name,pinfo,case
            )
            print("   $$ Control point ", j, " : Pressure = ", avg_pressure)
            Lmin.append(min_pressure)
            Lavg.append(avg_pressure)
            Lmax.append(max_pressure)

            Lpressure = np.array([Lmin, Lavg, Lmax])

    else:
        L = [0] * (num_points)
        Lpressure = np.array([L, L, L])

    return Lpressure



def save_pressure(data_file,i_vessel, dpoints, dvectors,pinfo,case):
    
    """


    Parameters
    ----------
    data_file : tecplot datafile
    i : vessel index.
    dpoints : dictionary of the control points
    dvectors : dictionary of the normal vectors
    pinfo : str, ex : 'pt2'
    case: str, ex: 'baseline'

    Returns
    -------
    dpressure : dictionnary of the pressure in the vessel i.

    """
    dpressure = {}
    Lpress = compute_along(data_file,i_vessel, dpoints, dvectors,pinfo,case)
    # Inverts the pressure array so pressure is decreasing along the vessel
    pressure_array = invert_array(np.array(Lpress))
    vessel_name = dpoints.get("points{}".format(i_vessel))[0]
    dpressure["pressure{}".format(i_vessel)] = vessel_name, pressure_array
    
    # Inverts point and vector arrays to match descending pressure
    
    
    # Inverts distance array and recalculates
    

    return dpressure


def save_pressure_all(data_file,dpoints, dvectors,pinfo,case):
    """


    Returns
    -------
    dpressure : dictionary of the pressure in all the vessels

    """
    dpressure = {}
    num_vessels = len(dpoints)
    for i_vessel in range(num_vessels):
        Lpress = compute_along(data_file,i_vessel, dpoints, dvectors,pinfo,case)
        # Inverts the pressure array so pressure is decreasing along the vessel
        pressure_array = invert_array(np.array(Lpress))
        vessel_name = dpoints.get("points{}".format(i_vessel))[0]
        dpressure["pressure{}".format(i_vessel)] = vessel_name, pressure_array

    return dpressure

def invert_array(arr):
    """
    invert the order of a numpy array if its first value is inferior to its last

    Parameters
    ----------
    arr :numpy array 

    Returns
    -------
    int
        1 if the array has been inverted, 0 if not
    arr : numpy array , inverted or not
        

    """
    new_arr = np.ones_like(arr)
    # If the mean value of the pressure in the first slice is less than the
    # mean value of the pressure at the last slice, invert the array
    if arr[1, 0] < arr[1, arr.shape[1] - 1]:
        for i in range(3):
            new_arr[i, :] = arr[i][::-1]
        return 1, new_arr
    else:
        return 0, arr

# %% Main

# pinfo = input('Patient number -- ') 
# case = input ('Condition -- ')
# num_cycle = 2
# select_file = 'a'
# select_vessel = 'a'

def main(pinfo,case,num_cycle,select_file,select_vessel):
    
    """
    
    The purpose of this script is to compute and plot the pressure along the vessels
    of the circle of Willis of a specific patient for certain points. 
    
    The method used is to create slices and extract the minimum, maximum, and average pressure along the vessel.
    
    This program
    --> Step 1 : Imports dictionaries created by read_centerlines_vtp.py for the points, normal vectors, radii and distances along the vessel.
    
    --> Step 2 : Clean the slices.
    The tecplot case file is loaded with the data of the first timestep as a test. The slices are extracted respectfully to the dict
    of points and vectors. All the x,y,z coordinates of the slice are extracted, and the slice is described through geometric/morphologic 
    descriptors such as convexity and circularity. If the values are not satisfying, the slice is removed. This step returns a new set
    of points and vectors of only the clean data.
    
    --> Step 3 : Compute pressure with tecplot
    
    Step 5.1 : Selecting the good case (patient, case, num_cycle, .dat file), and load data into Tecplot
    Step 5.2 : Find the list of the closest points in fluent to the control points.
    Step 5.3 : Make a slice for each control point and normal vector
    Step 5.4 : Find the subslice that correspond to the vessel in which one is interested
    Step 5.5 : Compute the min, max and average pressure in the subslice
    Step 5.6 : Save the pressures in a dictionnary,containing every cycle/.dat file/ vessel.
    Step 5.7 : change the order of the vessel if necessary to have a decreasing plot of pressure
    
    
    """
       
    
    Infos = {'patient informations': [pinfo],'cycle': [num_cycle]}
    context = pd.DataFrame(data = Infos)
    context.to_csv('L:/vasospasm/' + pinfo + '/' + case + '/4-results/pressure_resistance/infos.csv')
    results_path = 'L:/vasospasm/' + pinfo + '/' + case + '/4-results/pressure_resistance/'
    
    # Get the dat files for the patient and its case
    onlydat, indices_dat,pathwd = get_list_files_dat(pinfo, case, num_cycle) 
       
    print(
    " ############  Step 1 : Import vessel points and normal vectors  ############",
    "\n",
    )
       
    os.chdir("L:/vasospasm/pressure_resistance_calculation/vasospasm_resistance")
    
    dpoints_original = load_dict(results_path +"points_original_" + pinfo + "_" + case)
    dvectors_original = load_dict(results_path +"vectors_original_" + pinfo + "_" + case)
    ddist_original = load_dict(results_path +"dist_original_" + pinfo + "_" + case)
    dradii_original = load_dict(results_path +"radii_original_" + pinfo + "_" + case)
    
    #%% FROM THESE CONTROL POITNS AND VECTORS, FIRST LOADING INTO TECPLOT TO CLEAN THE DATA (REMOVE THE IRREGULAR SLICES)
    
    print(
    " ############  Step 2 : First connection to tecplot :  Slice cleaning  ############",
    "\n",
    )
    
    #dCS_original = load_dict(results_path +"dCS_original_" + pinfo + "_" + case)
    
    
    print('get dCS')
    tic = time.perf_counter()
    dCS_original = cross_section.get_dCS(pinfo, case, num_cycle, dpoints_original, dvectors_original)
    dCS_cleaned,dpoints_cleaned,dvectors_cleaned,ddist_cleaned,dradii_cleaned = cross_section.morphometric_cleaning(dCS_original, dpoints_original, dvectors_original,ddist_original,dradii_original)
    
    toc = time.perf_counter()
    time_minutes = (toc-tic)/60
    print(f"Slice cleaning took {time_minutes:0.4f} minutes")
    
    tic = time.perf_counter()
    
    dpressure = {}
    dpressure["Informations"] = pinfo, case
    
    if select_file == "a":
        # Replace by 30 and 60 to compute on the second period
        cycle_start = 0
        cycle_stop = 30
    else:
        i_file = int(select_file)
        cycle_start = i_file
        cycle_stop = i_file + 1
    
    for i_data in range(cycle_start, cycle_stop):
       
        # LOAD THE TECPLOT DATA
        
        filename = onlydat[i_data]
        print(filename)
        
        print(
            " ############  Step 3 : Connection to Tecplot  ############ Time step : ",
            i_data,
            "\n",
        )
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
        tp.active_frame().plot().fieldmaps(2).contour.show=False
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
    
        print(" ############  Step 4 : Compute pressure  ############\n")
        
        if select_vessel == "a":
            dpressure["{}".format(indices_dat[i_data])] = save_pressure_all(
                data_file,dpoints_cleaned, dvectors_cleaned,pinfo,case
            )
            
        
        else:
            i_vessel = int(select_vessel)
            dpress = save_pressure(data_file,i_vessel, dpoints_cleaned, dvectors_cleaned,pinfo,case)
            dpressure["{}".format(indices_dat[i_data])] = dpress
    
    
    toc = time.perf_counter()
    
    time_hours = (toc-tic)/3600
    print(f"Extracting pressure data took {time_minutes:0.4f} hours")
    
    #%% Finalize points, vectors, and distances based on the order of descending pressure
    
    
    dpoints_final, dvectors_final, ddist_final, dradii_final = {},{},{},{}
    
    num_vessels = len(dpoints_cleaned)
    
    for i_vessel in range(num_vessels):
    
        # Checks if the flag generated during the invert_array set of save_pressure is 1 (array was reversed)
        reverse_flag = dpressure.get("{}".format(indices_dat[i_data])).get("pressure{}".format(i_vessel))[1][0]
        
        if reverse_flag == 1:
            # Reverse the order of the points and radii, and the order and direction of the normal vectors
            points_reversed = dpoints_cleaned.get("points{}".format(i_vessel))[1][::-1] 
            vectors_reversed = np.multiply(dvectors_cleaned.get("vectors{}".format(i_vessel))[1][::-1],-1)
            radii_reversed = dradii_cleaned.get("radii{}".format(i_vessel))[1][::-1]
            
            # Recalculate distances
            distances_flipped = ddist_cleaned.get("distance{}".format(i_vessel))[1][::-1] 
            
            num_points = len(points_reversed)
            difference_between_points = np.subtract(distances_flipped[:-1],distances_flipped[1:])
            distances_reversed = [np.sum(difference_between_points[0:i]) for i in range(num_points)]
            
            # Assign to final dictionary
            vessel_name = dpoints_cleaned.get("points{}".format(i_vessel))[0]
            dpoints_final["points{}".format(i_vessel)] = vessel_name, points_reversed
            dvectors_final["vectors{}".format(i_vessel)] = vessel_name, vectors_reversed
            ddist_final["distance{}".format(i_vessel)] = vessel_name, distances_reversed
            dradii_final["radii{}".format(i_vessel)] = vessel_name, radii_reversed
        
        # Otherwise store the data in the final dictionaries
        else:
            dpoints_final["points{}".format(i_vessel)] = dpoints_cleaned["points{}".format(i_vessel)]
            dvectors_final["vectors{}".format(i_vessel)] = dvectors_cleaned["vectors{}".format(i_vessel)]
            ddist_final["distance{}".format(i_vessel)] = ddist_cleaned["distance{}".format(i_vessel)]
            dradii_final["radii{}".format(i_vessel)] = dradii_cleaned["radii{}".format(i_vessel)]
        
           
        #    # Save the data into the patient folder :
           
           
    save_dict(dCS_cleaned, results_path +'cross_section_' + pinfo + '_' + case)
    save_dict(ddist_final,results_path + "dist_" + pinfo + "_" + case)
    save_dict(dpoints_final, results_path + 'points_' + pinfo + '_' + case)
    save_dict(dvectors_final, results_path + 'vectors_' + pinfo + '_' + case)
    save_dict(dradii_final, results_path + 'radii_' + pinfo + '_' + case)
    save_dict(dpressure,results_path + 'pressure_' + pinfo + '_' + case)
    
    print('Saved data to dictionaries')
    
    return dCS_cleaned,ddist_final,dpoints_final,dvectors_final,dradii_final,dpressure
   
#return dpressure,ddist,n_dcross_section,dpoints_u,dvectors_u

#%%

dCS_cleaned_bas,ddist_final_bas,dpoints_final_bas,dvectors_final_bas,dradii_final_bas, dpressure_bas = main('pt39','baseline',2,"a","a")
dCS_cleaned_bas,ddist_final_bas,dpoints_final_bas,dvectors_final_bas,dradii_final_bas, dpressure_bas = main('pt40','vasospasm',2,"a","a")

