
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 15:47:24 2022

@author: GALADRIEL_GUEST

This script makes all the plots from the data already computed : 
    plots of global/local resistance, pressure and cross section in every vessel for baseline/vasospasm
    the heatmap of final values and rates for pressure/resistance/flowrate
"""

# %% Imports


import scipy.io
import scipy
import math
import pandas as pd
import glob
import csv
import numpy as np
import pickle
import importlib
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import seaborn as sns
import shutil

os.chdir("L:/vasospasm/calculation_resistance_dissipation/pressure_resistance_python")




# %% Functions & Main


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
    data_indices = [l[13:-4] for l in onlyfiles]

    return onlyfiles, data_indices, pathwd

# %%


def plot_area(ddist, dradii, i_vessel, case, ax):

    # Return radii and distance for the vessel
    radii = dradii.get("radii{}".format(i_vessel))[1][:]
    dist = ddist.get("distance{}".format(i_vessel))[1][:]

    # Calculate the area in m^2
    area = np.multiply(math.pi, np.power(radii, 2))

    # Plot distance vs radii
    ax.plot(dist, area, label="Cross-sectional area | " + case)

    plt.grid()

    return ax


def get_Q_final(pinfo, case, dpoints, num_cycle):

    # Identifies the directory with the case_info.mat file
    dinfo = scipy.io.loadmat(
        "L:/vasospasm/" + pinfo + "/" + case + "/3-computational/case_info.mat"
    )

    # Identifies the inlet/outlet vessels specified in Q_final
    vessel_names = ['L_MCA', 'R_MCA', 'L_A2', 'R_A2',
                    'L_P2', 'R_P2', 'BAS', 'L_TICA', 'R_TICA']

    # Get the vessel names from the points dictionary
    vessel_names_dict = [dpoints.get('points{}'.format(i_vessel))[
        0] for i_vessel in range(len(dpoints))]

    # Create 30 integers spread over 0-100
    L_indexes = [int(x) for x in np.linspace(0, 100, 30)]

    # Extract the variable Q_final from the case_info.mat
    dqfinal = dinfo.get("Q_final")

    # Creates an array of flow rates for the 11 inlets/outlets at 30 time instances
    Q_arr = np.zeros((30, dqfinal.shape[1]))
    for i in range(dqfinal.shape[1]):
        for k in range(30):
            Q_arr[k, i] = dqfinal[L_indexes[k], i]

    # Determines which data_indices correspond between the results dictionary names
    # and the Qfinal names
    Verity = np.zeros((len(vessel_names), len(vessel_names_dict)))

    for i in range(len(vessel_names)):
        for j in range(len(vessel_names_dict)):
            # verity matrix
            Verity[i, j] = (vessel_names[i] in vessel_names_dict[j])

    L_ind = []
    for i in range(len(vessel_names)):
        for j in range(len(vessel_names_dict)):
            if Verity[i, j] == 1:
                L_ind.append(j)

    # Creates a dictionary of flow rates
    dQ = {}
    for k in range(len(L_ind)):

        L_to_extract = Verity[:, L_ind[k]]

        for i in range(len(vessel_names)):

            if L_to_extract[i] == 1:
                ind = vessel_names.index(vessel_names_dict[L_ind[k]])
                dQ['Q_{}'.format(vessel_names_dict[L_ind[k]])] = Q_arr[:, ind]

    # Extract the flow rates from the collateral arteries and add them to the
    # dictionary of flow rates
    dQ_collateral = get_Q_collateral(pinfo, case, num_cycle)
    dQ.update(dQ_collateral)
    
    # Assign collateral flow rates when .out file is not available
    variation_input = dinfo.get("variation_input")
    
    print(variation_input)
    
    # Missing Acom
    if variation_input == 2:
        dQ['Q_{}'.format("L_A1")] = dQ.get('Q_{}'.format("L_A2"))[:]
        dQ['Q_{}'.format("R_A1")] = dQ.get('Q_{}'.format("R_A2"))[:]
    
    # Missing left P1
    elif variation_input == 3:
        dQ['Q_{}'.format("L_PCOM")] = dQ.get('Q_{}'.format("L_P2"))[:]
    
    # Missing right P1
    elif variation_input == 4:
        dQ['Q_{}'.format("R_PCOM")] = dQ.get('Q_{}'.format("R_P2"))[:]
    
    # Missing left Pcom
    elif variation_input == 7:
        dQ['Q_{}'.format("L_P1")] = dQ.get('Q_{}'.format("L_P2"))[:]
        
    # Missing right Pcom
    elif variation_input == 8:
        dQ['Q_{}'.format("R_P1")] = dQ.get('Q_{}'.format("R_P2"))[:]
        
    else:
        print('No modifications to collateral flow rates')       
        
    save_dict(dQ, "L:/vasospasm/" + pinfo + "/" + case + "/4-results/pressure_resistance/flow_rates_" + pinfo + "_" + case)

    return dQ, list(dQ.keys())


def get_Q_collateral(pinfo, case, num_cycle):
    """
    Inputs : pinfo (str), case(str), num_cycle(the id of the cycle, usually 2)
    returns a dictionary of the flowrate at the end of LA1,RA1,LP1 and RP1

    """
    # Return data about the Fluent results
    onlyfiles, data_indices, pathwd = get_list_files_dat(
        pinfo, case, num_cycle)

    # Change the directory to the hyak_submit folder
    os.chdir(pathwd)

    # Determine which time steps the data files are stored at
    cycle_start = 0
    cycle_stop = 30

    time_steps = []
    for i_data in range(cycle_start, cycle_stop):
        print(i_data)
        print(data_indices[i_data])
        data_string = data_indices[i_data].split('-')
        data_index = int(data_string[2])
        time_steps.append(data_index)

    dQ = {}

    # Cycle through the .out files
    names = ['la1', 'lp1', 'ra1', 'rp1','lpcom','rpcom']
    for name in glob.glob('*.out'):
        name_trunc = name[:-4]
        if name_trunc in names:
            filename = name_trunc + '.out'

            # Read .out data
            outfile_data = pd.read_csv(filename, skiprows=3, sep=' ', names=[
                                       'Time_step', 'Flow', 'Velmax', 'flow_time'])

            # Append the flow rate for a given time step
            Q_outfile = []
            for t in time_steps:
                # Returns flow rate at time steps from .dat files
                # Note: t-1 to correct for Python indexing starting at 0
                Q_outfile.append(outfile_data.at[t-1, 'Flow'])

            # Converts la1 to L_A1, etc. and saves it to the dictionary
            final_name = (name_trunc[0] + '_' + name_trunc[1:]).upper()
            dQ['Q_{}'.format(final_name)] = np.array(Q_outfile)

    return dQ


def plot_R(dpressure, ddist, dpoints, i_vessel, pinfo, case, num_cycle, ax, ax2):

    # Get Fluent data file information
    onlydat, data_indices, pathwd = get_list_files_dat(pinfo, case, num_cycle)

    # Name of the vessel
    name_vessel = dpoints.get("points{}".format(i_vessel))[0]
    print(name_vessel)

    # Determine how many pressure entries there are for that vessel
    len_vessel = (
        dpressure.get("{}".format(data_indices[0]))
        .get("pressure{}".format(i_vessel))[1][1]
        .shape[1]
    )

    # Find the average distance between the two points
    Ldist = []
    dist = ddist.get("distance{}".format(i_vessel))[1]

    for i in range(0, len(dist)-1):
        Ldist.append(float((dist[i] + dist[i + 1]) / 2))

    # Return the flow rates over time for the inlets/outlets and collateral pathways
    Q_final, list_name = get_Q_final(pinfo, case, dpoints, num_cycle)

    # Calculate the mean flow rate over time
    # Take the absolute value so global resistance is positive
    Q_mean = np.abs(np.mean(Q_final.get('Q_{}'.format(name_vessel))[:]))
    print('Flow rate: ' + str(Q_mean))

    len_cycle = 30

    # min_pressure_env = []
    # max_pressure_env = []

    # # Finds the mean of the minimum and maximum pressures
    # for i_data in range(len_cycle):
    #     # Averages the minimum pressures across the slices for each time step
    #      min_pressure_one_dt = np.mean([
    #          dpressure.get("{}".format(data_indices[i_data])).get("pressure{}".format(i_vessel))[
    #              1
    #          ][1][0, i]
    #          for i in range(len_vessel)
    #      ])
    #      min_pressure_env.append(min_pressure_one_dt )

    #      # Averages the maximum pressures across the slices for each time step
    #      max_pressure_one_dt = np.mean([
    #          dpressure.get("{}".format(data_indices[i_data])).get("pressure{}".format(i_vessel))[
    #              1
    #          ][1][2, i]
    #          for i in range(len_vessel)
    #      ])
    #      max_pressure_env.append(max_pressure_one_dt)

    # # Identifies the index where the pressure is minimum/maximum in time
    # index_min_pressure = min_pressure_env.index(min(min_pressure_env))
    # index_max_pressure = max_pressure_env.index(max(max_pressure_env))

    # # Finding the time step with the min variation of pressure

    # min_max_pressure_envelope = np.zeros((2, len_vessel))
    mean_pressure = np.zeros(len_vessel)

    for i_slice in range(len_vessel):
        # Lmin = [
        #     dpressure.get("{}".format(data_indices[k])).get("pressure{}".format(i_vessel))[
        #         1
        #     ][1][0, i]
        #     for k in range(len_cycle)
        # ]
        Lmean = [
            dpressure.get("{}".format(data_indices[k])).get("pressure{}".format(i_vessel))[
                1
            ][1][1, i_slice]
            for k in range(len_cycle)
        ]
        # Lmax = [
        #     dpressure.get("{}".format(data_indices[k])).get("pressure{}".format(i_vessel))[
        #         1
        #     ][1][2, i]
        #     for k in range(len_cycle)
        # ]

        # # Define an array with the minimum and maximum pressures across the slices
        # min_max_pressure_envelope[0]=  [ dpressure.get("{}".format(data_indices[index_min_pressure])).get("pressure{}".format(i_vessel))[1][1][0, i_slice]]
        # min_max_pressure_envelope[1]=  [ dpressure.get("{}".format(data_indices[index_max_pressure])).get("pressure{}".format(i_vessel))[1][1][0, i_slice]]

        mean_pressure[i_slice] = sum(Lmean) / len(Lmean)

    # Calculate global resistance by comparing the first slice to each subsequent slice
    global_resistance = np.zeros(len_vessel-1)
    for i_slice in range(0, len_vessel-1):
        global_resistance[i_slice] = (
            mean_pressure[0] - mean_pressure[i_slice+1]) / Q_mean

    # Calculate the local resistance by comparing two adjacent slices
    local_resistance = np.zeros(len_vessel-1)
    for i_slice in range(len_vessel - 1):
        local_resistance[i_slice] = (
            mean_pressure[i_slice] - mean_pressure[i_slice+1]) / Q_mean

    # Line plots of local and global resistance
    ax.plot(Ldist, local_resistance, label="Local resistance | " + case)
    ax2.plot(Ldist, global_resistance, "-",
             label="Global resistance | " + case)

    # # Code for envelope of resistance
    # ax.fill_between(Ldist[:-1],
    #                        local_resistance[:,2],
    #                        local_resistance[:,0],
    #                        alpha=0.2
    #                        )

    # ax2.fill_between(Ldist[1:],
    #                        global_resistance[:,
    #                                      0],
    #                        global_resistance[:,
    #                                      2],
    #                        alpha=0.2
    #                        )

    plt.grid()

    #fig = plt.figure(figsize=(14.4, 10.8))
    #ax = fig.add_subplot(111)

    return ax, Q_final


def plot_pressure_vs_distance(dpressure, ddist, i_vessel, pinfo, case, num_cycle, ax):
    """


    Parameters
    ----------
    dpressure : dictionary of the pressure along all the vessels, during all timesteps
    ddist : dictionary of the distance along all the vessels
    i_vessel : index of the vessel
    pinfo : str, patient information : ex : 'pt2'
    case : str, ex : 'baseline'
    num_cycle : int, number of the cycle, usually 2
    ax : matplotlib ax object

    Returns
    -------
    ax : matplotlib ax object

    """

    # Identify Fluent data
    onlydat, data_indices, pathwd = get_list_files_dat(pinfo, case, num_cycle)

    # Determine how many slices are associated with that vessel
    len_vessel = (
        dpressure.get("{}".format(data_indices[0]))
        .get("pressure{}".format(i_vessel))[1][1]
        .shape[1]
    )
    name_vessel = dpressure.get("{}".format(data_indices[0])).get(
        "pressure{}".format(i_vessel)
    )[0]

    # Define the distance
    dist = ddist.get("distance{}".format(i_vessel))[1]

    # Ldist = []
    # for i in range(0, dist.shape[0] - 1):
    #     Ldist.append(float((dist[i] + dist[i + 1]) / 2))

    pressure = np.zeros((len_vessel, 3))
    pressure_offset = np.zeros((len_vessel, 3))

    len_cycle = 30

    # Loop over slices and identify the minimum, mean and maximum pressures
    for i_slice in range(len_vessel):
        Lmin = [
            dpressure.get("{}".format(data_indices[k])).get(
                "pressure{}".format(i_vessel))[1][1][0, i_slice]
            for k in range(len_cycle)
        ]

        Lmean = [
            dpressure.get("{}".format(data_indices[k])).get(
                "pressure{}".format(i_vessel))[1][1][1, i_slice]
            for k in range(len_cycle)
        ]
        Lmax = [
            dpressure.get("{}".format(data_indices[k])).get(
                "pressure{}".format(i_vessel))[1][1][2, i_slice]
            for k in range(len_cycle)
        ]

        # Find the average in time of the minimum, mean and maximum pressures
        pressure[i_slice, 0] = sum(Lmin) / len(Lmin)
        pressure[i_slice, 1] = sum(Lmean) / len(Lmean)
        pressure[i_slice, 2] = sum(Lmax) / len(Lmax)

        # Offset the pressure so it starts at the origin
        pressure_offset[i_slice, 0] = pressure[i_slice, 0] - pressure[0, 1]
        pressure_offset[i_slice, 1] = pressure[i_slice, 1] - pressure[0, 1]
        pressure_offset[i_slice, 2] = pressure[i_slice, 2] - pressure[0, 1]

    # Plot the average pressure as a line and shaded region with min/max pressures
    ax.plot(dist, pressure_offset[:, 1], "-",
            label="Average pressure over time | " + case)
    ax.fill_between(
        dist,
        pressure_offset[:, 0],
        pressure_offset[:, 2],
        alpha=0.2
    )

    plt.grid()
    return ax

#%% Heat map

def find_matching_distance(ddist_bas, ddist_vas, i_vessel):

    distance_bas = ddist_bas.get("distance{}".format(i_vessel))[1][:]
    distance_vas = ddist_vas.get("distance{}".format(i_vessel))[1][:]

    distance_bas_max = np.max(distance_bas)
    distance_vas_max = np.max(distance_vas)

    if distance_bas_max > distance_vas_max:
        distance_difference = np.abs(
            np.subtract(distance_bas, distance_vas_max))
        bas_end_index = np.argmin(distance_difference)
        vas_end_index = len(distance_vas)-1
    else:
        distance_difference = np.abs(
            np.subtract(distance_vas, distance_bas_max))
        vas_end_index = np.argmin(distance_difference)
        bas_end_index = len(distance_bas)-1

    return bas_end_index, vas_end_index


def calculate_resistance(pinfo, case, i_vessel, vessel_name, end_index, dresistance, dpressure, Q_final,dinstantaneous):

    len_cycle = 30
    num_cycle = 2
    onlydat, data_indices, pathwd = get_list_files_dat(pinfo, case, num_cycle)

    P_0 = [dpressure.get("{}".format(data_indices[k])).get(
        "pressure{}".format(i_vessel))[1][1][1, 0]
        for k in range(len_cycle)]

    P_last = [dpressure.get("{}".format(data_indices[k])).get(
        "pressure{}".format(i_vessel))[1][1][1, end_index]
        for k in range(len_cycle)]
    
    deltaP_flip_direction_flag = [dpressure.get("{}".format(data_indices[k])).get(
        "pressure{}".format(i_vessel))[1][0] for k in range(len_cycle)]

    deltaP_instantaneous = np.abs(np.subtract(P_0, P_last)) # positive pressure
    Q_instantaneous = np.abs(Q_final['Q_{}'.format(vessel_name)]) # positive flow rate
    R_instantaneous = np.divide(deltaP_instantaneous, Q_instantaneous)
    
    dinstantaneous_vessel = {}
    dinstantaneous_vessel[vessel_name] = {"pressure_drop": deltaP_instantaneous, "flow_rate": Q_instantaneous, "resistance": R_instantaneous, "flip_direction_flag": deltaP_flip_direction_flag}
    dinstantaneous.update(dinstantaneous_vessel)
    
    #% Exclude entries that blow up
    R_median = np.median(R_instantaneous)
    threshold = 5*R_median
    R_instantaneous_new = []
    
    for r_local in R_instantaneous:
        if r_local < threshold:
            R_instantaneous_new.append(r_local)

    dresistance["resistance{}".format(i_vessel)] = vessel_name, R_instantaneous_new

    # Find mean pressure drop
    deltaP_mean = np.mean(deltaP_instantaneous)
    
    # Convert flow rate from m^3/s to mL/min
    Q_mean = np.mean(Q_instantaneous)*60*1e6
    
    # Resistance [Pa/(mL/min)]
    R_mean = np.mean(R_instantaneous_new)/(60*1e6)

    return dresistance, deltaP_mean, Q_mean, R_mean, dinstantaneous



# %%

# def main(pinfo, num_cycle):

pinfo = input('Patient number -- ')
num_cycle = 2

# LOAD BASELINE DATA

pathused = "L:/vasospasm/" + pinfo + "/baseline/4-results/pressure_resistance/"

case = "baseline"

dpoints_bas = load_dict(pathused + "points_" + pinfo + "_" + case)
dvectors_bas = load_dict(pathused + "vectors_" + pinfo + "_" + case)
dpressure_bas = load_dict(pathused + "pressure_" + pinfo + "_" + case)
ddist_bas = load_dict(pathused + "dist_" + pinfo + "_" + case)
dCS_bas = load_dict(pathused + "cross_section_" + pinfo + "_" + case)
dradii_bas = load_dict(pathused + "radii_" + pinfo + "_" + case)

#  LOAD VASOSPASM DATA

pathused = "L:/vasospasm/" + pinfo + "/vasospasm/4-results/pressure_resistance/"

case = "vasospasm"

dpoints_vas = load_dict(pathused + "points_" + pinfo + "_" + case)
dvectors_vas = load_dict(pathused + "vectors_" + pinfo + "_" + case)
dpressure_vas = load_dict(pathused + "pressure_" + pinfo + "_" + case)
ddist_vas = load_dict(pathused + "dist_" + pinfo + "_" + case)
dCS_vas = load_dict(pathused + "cross_section_" + pinfo + "_" + case)
dradii_vas = load_dict(pathused + "radii_" + pinfo + "_" + case)

#%% Remove existing directories and create new ones for figures

# Save figures in both 4-results baseline and vasospasm directories
figure_path_baseline = (
    "L:/vasospasm/"
    + pinfo
    + "/baseline/4-results/pressure_resistance/figures"
)
figure_path_vasospasm = (
    "L:/vasospasm/"
    + pinfo
    + "/vasospasm/4-results/pressure_resistance/figures"
)

if os.path.exists(figure_path_baseline):
    shutil.rmtree(figure_path_baseline)
if os.path.exists(figure_path_vasospasm):
    shutil.rmtree(figure_path_vasospasm)

if not os.path.exists(figure_path_baseline):
    os.makedirs(figure_path_baseline)
if not os.path.exists(figure_path_vasospasm):
    os.makedirs(figure_path_vasospasm)


#%% Plot resistance, pressure drop, and cross-sectional area for each vessel
# Store data for each vessel in data frame for heat map and color scale of percent change in resistance

# Define a color map for percent difference in resistance
cmap = cm.get_cmap('Reds')
N_colors = 10
percent_diff_min = -100
percent_diff_max = 1000
color_range = cmap(np.linspace(0,1,N_colors))
resistance_percent_difference_range = np.linspace(percent_diff_min,percent_diff_max,N_colors)

# Instantiate variables
df_all_vessels = pd.DataFrame()
df_colors_all_vessels = pd.DataFrame()
dresistance_bas, dresistance_vas = {}, {}
dinstantaneous_bas, dinstantaneous_vas = {}, {}

num_vessels = len(dpoints_bas)

for i_vessel in range(num_vessels):

    len_vessel_bas = dpoints_bas.get("points{}".format(i_vessel))[1].shape[0]
    len_vessel_vas = dpoints_vas.get("points{}".format(i_vessel))[1].shape[0]

    if len_vessel_bas > 2 and len_vessel_vas > 2:

        # Create figure
        fig, (ax1, ax4, ax2, ax3) = plt.subplots(4, 1, figsize=(10, 15))
        plt.rcParams["axes.grid"] = True

        # Plot local and global resistance. Averaged over time
        vessel_name = dpoints_bas.get("points{}".format(i_vessel))[0]
        plt.suptitle("Plots in the " + vessel_name)

        plot_bas, Q_final_bas = plot_R(
            dpressure_bas,
            ddist_bas,
            dpoints_bas,
            i_vessel,
            pinfo,
            "baseline",
            num_cycle,
            ax1,
            ax4,
        )
        plot_vas, Q_final_vas = plot_R(
            dpressure_vas,
            ddist_vas,
            dpoints_vas,
            i_vessel,
            pinfo,
            "vasospasm",
            num_cycle,
            ax1,
            ax4,
        )

        # # Find which vessel is longer to set the x limits
        dist_bas = ddist_bas.get("distance{}".format(i_vessel))[1]
        dist_bas_max = np.max(dist_bas)
        dist_vas = ddist_vas.get("distance{}".format(i_vessel))[1]
        dist_vas_max = np.max(dist_vas)
        max_vessel_length = np.max([dist_bas_max, dist_vas_max])

        ax1.set_ylabel("resistance")
        ax1.set_title("local resistance along the vessel", fontsize="small")
        ax1.legend(fontsize="small")

        ax1.set_xlim([0, max_vessel_length])
        ax4.set_xlabel("distance along the vessel (m)", fontsize="small")
        ax4.set_ylabel("resistance")
        ax4.set_title("global resistance along the vessel", fontsize="small")
        ax4.set_xlim([0, max_vessel_length])
        ax4.legend(fontsize="small")

        # Plot pressure drop averaged over time

        plot_bas = plot_pressure_vs_distance(
            dpressure_bas, ddist_bas, i_vessel, pinfo, "baseline", num_cycle, ax2
        )
        ax2.set_xlim([0, max_vessel_length])

        plot_vas = plot_pressure_vs_distance(
            dpressure_vas, ddist_vas, i_vessel, pinfo, "vasospasm", num_cycle, ax2
        )

        ax2.set_ylabel("pressure")
        ax2.set_xlabel("distance along the vessel (m)", fontsize="small")
        ax2.set_title("pressure along the vessel", fontsize="small")
        ax2.legend(fontsize="small")

        # # Plot cross sectional area

        plot_bas = plot_area(ddist_bas, dradii_bas, i_vessel, 'baseline',ax3)
        plot_vas = plot_area(ddist_vas, dradii_vas, i_vessel, 'vasospasm',ax3)
        ax3.set_ylabel("Area")
        ax3.set_xlabel("Distance along the vessel")
        ax3.set_title("Cross-sectional area along the vessel",
                      fontsize="small")
        ax3.legend(fontsize="small")
        ax3.set_ylim([0, None])
        ax3.set_xlim([0, max_vessel_length])

        fig.tight_layout()
            
        plt.savefig(figure_path_baseline + "/final_plot_" + vessel_name+ ".png")
        print(figure_path_baseline + "/final_plot_" + vessel_name+ ".png")

        plt.savefig(figure_path_vasospasm + "/final_plot_" + vessel_name + ".png")
        print(figure_path_vasospasm + "/final_plot_" + vessel_name + ".png")
        plt.show()
        
        # Calculate resistance and store in data frame
        
        bas_end_index, vas_end_index = find_matching_distance(
            ddist_bas, ddist_vas, i_vessel)

        dresistance_bas, deltaP_mean_bas, Q_mean_bas, R_mean_bas, dinstantaneous_bas = calculate_resistance(
            pinfo, 'baseline', i_vessel, vessel_name, bas_end_index, dresistance_bas, dpressure_bas, Q_final_bas, dinstantaneous_bas)
        dresistance_vas, deltaP_mean_vas, Q_mean_vas, R_mean_vas, dinstantaneous_vas = calculate_resistance(
            pinfo, 'vasospasm', i_vessel, vessel_name, vas_end_index, dresistance_vas, dpressure_vas, Q_final_vas, dinstantaneous_vas)

        percent_diff_deltaP = (deltaP_mean_vas-deltaP_mean_bas)/deltaP_mean_bas*100
        percent_diff_Q = (Q_mean_vas-Q_mean_bas)/Q_mean_bas*100
        percent_diff_R = (R_mean_vas-R_mean_bas)/R_mean_bas*100
        
        # Determine which color is associated with that percent change
        percent_change_difference = np.abs(
            np.subtract(resistance_percent_difference_range, percent_diff_R))
        closest_color_index = np.argmin(percent_change_difference)
        
        # Convet it to the RGB value from [0,255]
        color_vessel = np.round(np.multiply(color_range[closest_color_index,0:3],255))
        
        vessel_data = {
            'pressure drop baseline': deltaP_mean_bas,
            'pressure drop vasospasm': deltaP_mean_vas,
            'pressure drop percent difference': percent_diff_deltaP,
            'flow rate baseline': Q_mean_bas,
            'flow rate vasospasm': Q_mean_vas,
            'flow rate percent difference': percent_diff_Q,
            'resistance baseline': R_mean_bas,
            'resistance vasospasm': R_mean_vas,
            'resistance percent difference': percent_diff_R,
        }

        df_vessel = pd.DataFrame(vessel_data, index=[vessel_name])
        #df_vessel.loc()
        df_all_vessels = pd.concat([df_all_vessels,df_vessel])
        
        
        # Save color in data frame
        color_data = {vessel_name: color_vessel}
        
        df_color_vessel = pd.DataFrame(color_data, index=['red','green','blue'])
        #df_color_vessel.loc()
        df_colors_all_vessels = pd.concat([df_colors_all_vessels, df_color_vessel],axis=1)
        
        

#%% Plot heat map

f,(ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize = (25,17))

plt.suptitle("Resistance heatmap for "+ pinfo, fontsize = 40)

sns.set(font_scale=1.8)

xlabels = ['baseline','vasospasm']

sns.heatmap(df_all_vessels.loc[:,['pressure drop baseline','pressure drop vasospasm']],xticklabels=xlabels, annot = True,cmap =plt.cm.Reds,fmt = '.0f',linewidth=0.5,ax=ax1)
ax1.set_title('Pressure drop [Pa]',fontsize=30)   
sns.heatmap(df_all_vessels.loc[:,['flow rate baseline','flow rate vasospasm']],xticklabels=xlabels,annot = True,cmap =plt.cm.Reds,fmt = '.0f',linewidth=0.5,ax=ax2)

ax2.set_yticks([])
sns.heatmap(df_all_vessels.loc[:,['resistance baseline','resistance vasospasm']],xticklabels=xlabels,annot = True,cmap =plt.cm.Reds,fmt = '.2f',linewidth=0.5,ax=ax3)
ax2.set_title('Flow rate [mL/min]',fontsize=30)  


ax3.set_yticks([])

sns.heatmap(df_all_vessels.loc[:,['pressure drop percent difference','flow rate percent difference','resistance percent difference']],xticklabels=['Pressure','Flow rate','Resistance'],annot = True,fmt ='.0f',cmap =plt.cm.Reds,linewidth=0.5,ax=ax4,vmin=percent_diff_min,vmax=percent_diff_max)
ax3.set_title('Resistance [Pa/mL/min]',fontsize=30)
for t in ax4.texts: t.set_text(t.get_text() + " %")
ax4.set_yticks([])
ax4.set_title('Percent difference',fontsize=30)

plt.savefig("L:/vasospasm/" + pinfo + "/baseline/4-results/pressure_resistance/figures/plot_heatmap_threshold_" + str(percent_diff_max) + "_" + pinfo + "_new.png")
plt.savefig("L:/vasospasm/" + pinfo + "/vasospasm/4-results/pressure_resistance/figures/plot_heatmap_threshold_" + str(percent_diff_max) + "_" + pinfo + "_new.png")

plt.tight_layout()

## Save resistance
save_dict(dresistance_bas, "L:/vasospasm/" + pinfo + "/" + "baseline/4-results/pressure_resistance/resistance_" + pinfo + "_baseline")
save_dict(dresistance_vas, "L:/vasospasm/" + pinfo + "/" + "vasospasm/4-results/pressure_resistance/resistance_" + pinfo + "_vasospasm")

## Save instantaneous values
save_dict(dinstantaneous_bas, "L:/vasospasm/" + pinfo + "/" + "baseline/4-results/pressure_resistance/instantaneous_" + pinfo + "_baseline")
save_dict(dinstantaneous_vas, "L:/vasospasm/" + pinfo + "/" + "vasospasm/4-results/pressure_resistance/instantaneous_"+ pinfo +"_vasospasm")

# Export color data for percent change in resistance to CSV file

df_colors_all_vessels.to_csv("L:/vasospasm/" + pinfo + "/baseline/4-results/pressure_resistance/" + pinfo + "_colors_resistance_threshold_" + str(percent_diff_max) + "_new.csv")
df_colors_all_vessels.to_csv("L:/vasospasm/" + pinfo + "/vasospasm/4-results/pressure_resistance/" + pinfo + "_colors_resistance_threshold_" + str(percent_diff_max) + "_new.csv")




#%%

vessel_list = ["L_MCA","R_MCA","L_A2","R_A2","L_P2","R_P2","L_TICA","R_TICA","BAS",
                "L_A1","R_A1","L_PCOM","R_PCOM","L_P1","R_P1"]

df_percent_diff_all_vessels = pd.DataFrame()

for vessel_of_interest in vessel_list:

    # Create data frame with all vessels and percent change
    
    if vessel_of_interest in df_all_vessels.index:
        
        vessel_percent_diff_data = {vessel_of_interest: df_all_vessels.loc[vessel_of_interest,"resistance percent difference"]}
        
        print(df_all_vessels.loc[vessel_of_interest,"resistance percent difference"])
         
    else:
        vessel_percent_diff_data = {vessel_of_interest: 'nan'}
        print('missing')
        
    df_percent_diff_vessel =  pd.DataFrame(vessel_percent_diff_data, index=[pinfo])
    
    df_percent_diff_all_vessels = pd.concat([df_percent_diff_all_vessels, df_percent_diff_vessel],axis=1)

resistance_path_vasospasm = "L:/vasospasm/" + pinfo + "/vasospasm/4-results/pressure_resistance/"

df_percent_diff_all_vessels.to_csv(resistance_path_vasospasm + pinfo + "_resistance_percent_difference.csv")






