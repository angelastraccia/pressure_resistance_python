# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 11:18:34 2022

@author: GALADRIEL_GUEST
"""

import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import os
import pickle
import scipy.io
import scipy

# Function to find the indices associated with equally spaced points


def equally_spaced_points(centerline_points, num_points):
    """
    Parameters
    ----------
    centerline_points : array with complete set of points for a specific vessel
    num_points : integer for the final number of downsampled points

    Returns
    -------
    indices_downsampled_points : array of indices to downsample the original 
    arrays of points, normals, radii, and distance

    """

    # Assign x, y, z coordinates for original centerline points
    x = centerline_points[:, 0]
    y = centerline_points[:, 1]
    z = centerline_points[:, 2]

    # Create spline representation
    spline_coeff, u_along_spline = splprep([x, y, z], s=0)

    # Downsample to num_points while giving clearance for bifurcations and the end of the vessel
    u_downsampled = np.linspace(0.05, 0.95, num_points)
    downsampled_points = splev(u_downsampled, spline_coeff)

    indices_downsampled_points = []
    # Search for indices of points in original centerlines array closest to downsampled points
    i = 0
    while i < num_points:
        # Assign x, y, z coordinates for downsampled point of interest
        x_i = downsampled_points[0][i]
        y_i = downsampled_points[1][i]
        z_i = downsampled_points[2][i]

        # Find the difference between the point of interest and the centerline points
        difference = np.power(np.subtract(
            x, x_i), 2) + np.power(np.subtract(y, y_i), 2
                                   ) + np.power(np.subtract(z, z_i), 2)

        # Find the index associated with the centerline point closest to the point of interest
        index_minimum = np.where(difference == difference.min())[0][0]
        indices_downsampled_points.append(index_minimum)
        i += 1

    return indices_downsampled_points


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
        
def downsample_points(vessel_label,P1_indices,which_branch_input,stl_surf_m,centerline_points,centerline_normal,centerline_radii,centerline_distance,dpoints,dvectors,dradii,ddist):
        """
        Parameters
        ----------
        vessel label: str. vessel name
        P1_indices: list. indices associated with the left P1 and right P1 segments
        which_branch_input: int. index for storing data in dictionaries
        centerline_points: array. centerline coordinates
        centerline_normal: array. normal vectors
        centerline_radii: list. vessel radii
        centerline_distance: list. distance along vessel
        
    
        Returns
        -------
        dpoints,dvectors,dradii,ddist: dictionaries with centerlines, normals, radii, and distances with new vessel data added
    
        """
        if which_branch_input in P1_indices:
            # Assign fewer points for the left and right P1 segments
            num_points = 5
        else:
            # Otherwise use more points
            num_points = 10
        
        # Determine the indices for equally spaced points
        indices_downsampled_points = equally_spaced_points(
            centerline_points, num_points)

        
        downsampled_points = np.array(
            [centerline_points[j, :] for j in indices_downsampled_points])
        downsampled_normal = np.array(
            [centerline_normal[j, :] for j in indices_downsampled_points])
        downsampled_radii = np.array(
            [centerline_radii[j] for j in indices_downsampled_points])
        downsampled_distance_offset = np.array(
            [centerline_distance[j] for j in indices_downsampled_points])
        
        # Start distance at zero
        downsampled_distance = np.subtract(
            downsampled_distance_offset, downsampled_distance_offset[0])

        # Plot STL and points
        branch_spaced_points_plot = pv.Plotter()
        branch_spaced_points_plot.add_mesh(stl_surf_m, opacity=0.3)

        plot_spaced_points = pv.PolyData(downsampled_points)
        branch_spaced_points_plot.add_mesh(plot_spaced_points,label=vessel_label)

        # Plot normal vectors
        plot_spaced_points["vectors"] = downsampled_normal*0.001
        plot_spaced_points.set_active_vectors("vectors")
        branch_spaced_points_plot.add_mesh(
            plot_spaced_points.arrows, lighting=False)
        
        branch_spaced_points_plot.add_legend(size=(.2, .2), loc='upper right')
        branch_spaced_points_plot.show()

        dpoints["points{}".format(
            which_branch_input)] = vessel_label, downsampled_points
        dvectors["vectors{}".format(
            which_branch_input)] = vessel_label, downsampled_normal
        dradii["radii{}".format(
            which_branch_input)] = vessel_label, downsampled_radii
        ddist["distance{}".format(
            which_branch_input)] = vessel_label, downsampled_distance
        
        return dpoints,dvectors,dradii,ddist

# %% pseudo code

# input pinfo, case
# return dpoints, dvectors, ddist, dradii
# cycle through all .vtp available --> assign vessels


def main(pinfo, case):
    """
    Parameters
    ----------
    pinfo : str. patient identifier
    case : str. baseline or vasospasm

    Returns
    -------
    dpoints: dictionary of vessel names and downsampled points 
    dvectors: dictionary of normal vectors corresponding to the downsampled points
    ddist: dictionary of the distances from the first point to the ith point
    dradii: dictionary of maximum inscribed radii for downsampled points
    """

    main_directory = 'L:/vasospasm/' + pinfo + '/' + case + '/1-geometry/' + pinfo + \
        '_' + case + '_centerlines_final_ext/ROMSimulations/'

    # Read in STLs in mm and m
    fname_stl_mm = 'L:/vasospasm/' + pinfo + '/' + case + \
        '/1-geometry/' + pinfo + '_' + case + '_final_mm.stl'
    stl_surf_mm = pv.read(fname_stl_mm)

    fname_stl_m = 'L:/vasospasm/' + pinfo + '/' + case + \
        '/1-geometry/' + pinfo + '_' + case + '_final.stl'
    stl_surf_m = pv.read(fname_stl_m)

    # Identifies the directory with the case_info.mat file
    dinfo = scipy.io.loadmat(
        "L:/vasospasm/" + pinfo + "/" + case + "/3-computational/case_info.mat"
    )

    # Extract the variable Q_final from the case_info.mat
    variation_input_case_info = dinfo.get("variation_input")
    variation_input = variation_input_case_info[0][0]
    print(variation_input)

    if variation_input == 1 or variation_input == 2 :
        # Complete or missing Acom
    
        # List of vessels of interest
        vessel_names = ['L_MCA', 'R_MCA', 'L_A1', 'L_A2', 'R_A1', 'R_A2',
                        'L_PCOM','L_P1', 'L_P2', 'R_PCOM','R_P1', 'R_P2', 
                        'BAS', 'L_TICA', 'R_TICA']
    
        vessel_options = "Vessel: 0-LMCA, 1-RMCA, 2-LA1, 3-LA2, 4-RA1, 5-RA2, 6-LPCOM, 7-LP1, 8-LP2, 9-RPCOM, 10-RP1, 11-RP2, 12-BAS, 13-LTICA, 14=RTICA, 22-skip, 33-split --   "
        P1_indices = [7,10]
    
    elif variation_input == 3: 
        # Missing left P1
        
        # List of vessels of interest
        vessel_names = ['L_MCA', 'R_MCA', 'L_A1', 'L_A2', 'R_A1', 'R_A2',
                        'L_PCOM','L_P2', 'R_PCOM','R_P1', 'R_P2', 
                        'BAS', 'L_TICA', 'R_TICA']
        
        vessel_options = "Vessel: 0-LMCA, 1-RMCA, 2-LA1, 3-LA2, 4-RA1, 5-RA2, 6-LPCOM, 7-LP2, 8-RPCOM, 9-RP1, 10-RP2, 11-BAS, 12-LTICA, 13=RTICA, 22-skip, 33-split --   "
        P1_indices = [9]
    
    elif variation_input == 4:
        # Missing right P1
        
        # List of vessels of interest
        vessel_names = ['L_MCA', 'R_MCA', 'L_A1', 'L_A2', 'R_A1', 'R_A2',
                        'L_PCOM','L_P1', 'L_P2','R_PCOM', 'R_P2', 
                        'BAS', 'L_TICA', 'R_TICA']
    
        vessel_options = "Vessel: 0-LMCA, 1-RMCA, 2-LA1, 3-LA2, 4-RA1, 5-RA2, 6-LPCOM, 7-LP1, 8-LP2, 9-RPCOM, 10-RP2, 11-BAS, 12-LTICA, 13=RTICA, 22-skip, 33-split --   "
        P1_indices = [7]
            
    elif variation_input == 5:
        # Missing left A1
        
       # List of vessels of interest
       vessel_names = ['L_MCA', 'R_MCA','L_A2', 'R_A1', 'R_A2',
                      'L_PCOM','L_P1', 'L_P2', 'R_PCOM','R_P1', 'R_P2', 
                      'BAS', 'L_TICA', 'R_TICA']
  
       vessel_options = "Vessel: 0-LMCA, 1-RMCA, 2-LA2, 3-RA1, 4-RA2, 5-LPCOM, 6-LP1, 7-LP2, 8-RPCOM, 9-RP1, 10-RP2, 11-BAS, 12-LTICA, 13=RTICA, 22-skip, 33-split --   "
       P1_indices = [6,9]
    
    elif variation_input == 6:
        # Missing right A1
        
        # List of vessels of interest
        vessel_names = ['L_MCA', 'R_MCA', 'L_A1', 'L_A2', 'R_A2',
                        'L_PCOM','L_P1', 'L_P2', 'R_PCOM','R_P1', 'R_P2', 
                        'BAS', 'L_TICA', 'R_TICA']
    
        vessel_options = "Vessel: 0-LMCA, 1-RMCA, 2-LA1, 3-LA2, 4-RA2, 5-LPCOM, 6-LP1, 7-LP2, 8-RPCOM, 9-RP1, 10-RP2, 11-BAS, 12-LTICA, 13=RTICA, 22-skip, 33-split --   "
        P1_indices = [6,9]

    elif variation_input == 7:
        # Missing left Pcom
        
        # List of vessels of interest
        vessel_names = ['L_MCA', 'R_MCA', 'L_A1', 'L_A2', 'R_A1', 'R_A2',
                        'L_P1', 'L_P2', 'R_PCOM','R_P1', 'R_P2', 
                        'BAS', 'L_TICA', 'R_TICA']
    
        vessel_options = "Vessel: 0-LMCA, 1-RMCA, 2-LA1, 3-LA2, 4-RA1, 5-RA2, 6-LP1, 7-LP2, 8-RPCOM, 9-RP1, 10-RP2, 11-BAS, 12-LTICA, 13=RTICA, 22-skip, 33-split --   "
        P1_indices = [6,9]      
    
    elif variation_input == 8:
        # Missing right Pcom
        
        # List of vessels of interest
        vessel_names = ['L_MCA', 'R_MCA', 'L_A1', 'L_A2', 'R_A1', 'R_A2',
                        'L_PCOM','L_P1', 'L_P2', 'R_P1', 'R_P2', 
                        'BAS', 'L_TICA', 'R_TICA']
    
        vessel_options = "Vessel: 0-LMCA, 1-RMCA, 2-LA1, 3-LA2, 4-RA1, 5-RA2, 6-LPCOM, 7-LP1, 8-LP2, 9-RP1, 10-RP2, 11-BAS, 12-LTICA, 13=RTICA, 22-skip, 33-split --   "
        P1_indices = [7,9]
        

    # Identify the subdirectories for the ROM Simulations
    sub_directories = [directory[0] for directory in os.walk(main_directory)]
    # Remove the parent directory from the list of directories
    sub_directories = sub_directories[1:]
    print(sub_directories)

    # Instatiate empty vector for assigned vessels and dictionaries for data outputs
    dpoints, dvectors, dradii, ddist = {},{},{},{}
    assigned_vessels, assigned_vessel_indices = [],[]

    # View centerlines

    # Cycle through different centerline.vtp files
    for sub_directory in sub_directories:
        print(sub_directory)

        fname_centerlines = sub_directory + '/centerlines.vtp'
        centerlines = pv.read(fname_centerlines)

        # Identify how many unique branches are present
        branches = np.unique(centerlines['BranchId'])

        # Define a discretized color map
        cmap = plt.get_cmap('jet')
        colors = cmap(np.linspace(0, 1, len(branches)))

        # Add the STL to the plot
        cow_plot = pv.Plotter()
        cow_plot.add_mesh(stl_surf_mm, opacity=0.3)

        # Cycle through each branch
        for count, branchid in enumerate(branches):

            # Identify which indices are associated with a branch ID ("where" returns a tuple, want first element)
            branch_indices = np.where(centerlines['BranchId'] == branchid)[0]

            # Extract the points associated with those indices
            branch = centerlines.extract_points(
                branch_indices, adjacent_cells=False)

            # Plot the branch colored by the branch ID
            cow_plot.add_mesh(branch, color=colors[count], label=str(branchid))

        cow_plot.add_legend(size=(.2, .2), loc='upper right')
        cow_plot.show()

    # Assign vessel segments

    # Cycle through different centerline.vtp files
    for sub_directory in sub_directories:
        print(sub_directory)

        # Read in centerlines.vtp
        fname_centerlines = sub_directory + '/centerlines.vtp'
        centerlines = pv.read(fname_centerlines)

        # Identify how many unique branches are present
        branches = np.unique(centerlines['BranchId'])

        # Define a discretized color map
        cmap = plt.get_cmap('jet')
        colors = cmap(np.linspace(0, 1, len(branches)))

        # Add the STL to the plot
        cow_plot = pv.Plotter()
        cow_plot.add_mesh(stl_surf_mm, opacity=0.3)

        # Cycle through each branch
        for count, branchid in enumerate(branches):

            # Identify which indices are associated with a branch ID ("where" returns a tuple, want first element)
            branch_indices = np.where(centerlines['BranchId'] == branchid)[0]

            # Extract the points associated with those indices
            branch = centerlines.extract_points(
                branch_indices, adjacent_cells=False)

            # Plot the branch colored by the branch ID
            cow_plot.add_mesh(branch, color=colors[count], label=str(branchid))

        cow_plot.add_legend(size=(.2, .2), loc='upper right')
        cow_plot.show()

        # Cycle through each branch
        for count, branchid in enumerate(branches):
            # print(branchid)

            # Identify which indices are associated with a branch ID ("where" returns a tuple, want first element)
            branch_indices = np.where(centerlines['BranchId'] == branchid)[0]

            # Identify data associated with a branch
            branch = centerlines.extract_points(
                branch_indices, adjacent_cells=False)

            # Extract the centerline points, normal, radii
            centerline_points_mm = branch.points
            centerline_radii_mm = branch['MaximumInscribedSphereRadius']
            centerline_normal = branch['CenterlineSectionNormal']

            # Convert to meters!
            centerline_points = np.divide(centerline_points_mm, 1000)
            centerline_radii = np.divide(centerline_radii_mm, 1000)

            # Calculate distance between all centerline points
            centerline_vectors = np.subtract(
                centerline_points[1:, :], centerline_points[:-1, :])
            centerline_vector_lengths = [np.linalg.norm(
                centerline_vectors[i, :]) for i in range(centerline_vectors.shape[0])]
            centerline_distance = [np.sum(centerline_vector_lengths[0:i]) for i in range(
                len(centerline_vector_lengths)+1)]

            # Add original centerline points to plot
            branch_plot = pv.Plotter()
            branch_plot.add_mesh(stl_surf_m, opacity=0.3)
            plot_original_points = pv.PolyData(centerline_points)
            branch_plot.add_mesh(plot_original_points)
            branch_plot.show()

            # Assign branch to label
            which_branch_input = int(input(vessel_options))
            
            # If it is a vessel of interest
            if which_branch_input < len(vessel_names):

                # Check to see if the vessel has already been assigned
                if which_branch_input in assigned_vessel_indices:
                    print('Already assigned')            
            
                else:
                    # Identify the vessel
                    vessel_label = vessel_names[int(which_branch_input)]
                    print(vessel_label)
            
                    dpoints,dvectors,dradii,ddist = downsample_points(vessel_label,P1_indices,which_branch_input,stl_surf_m,centerline_points,centerline_normal,centerline_radii,centerline_distance,dpoints,dvectors,dradii,ddist)
            
                    # Store which vessels have been assigned
                    assigned_vessel_indices.append(which_branch_input)
                    assigned_vessels.append(vessel_names[which_branch_input])
        
            elif which_branch_input == 33:
                print('Split vessel')
                
                split_segment_plot = pv.Plotter()
                
                # Add original centerline points to plot
                split_segment_plot = pv.Plotter()
                split_segment_plot.add_mesh(stl_surf_m, opacity=0.3,pickable=False)
                split_segment_plot.add_mesh(plot_original_points)
                
                # Enable picking
                split_segment_plot.enable_point_picking(show_message='Pick the dividing point', left_clicking=True,use_mesh=True)
                split_segment_plot.show()
                
                # Identify ID for picked point
                pointID = np.where(centerline_points == split_segment_plot.picked_point)[0]
                
                
                # Define centerlines for 1st segment and plot them
                segment1_plot = pv.Plotter()
                segment1_plot.add_mesh(stl_surf_m, opacity=0.3)
                plot_centerlines_segment1 = pv.PolyData(centerline_points[0:pointID[0],:])
                segment1_plot.add_mesh(plot_centerlines_segment1)
                segment1_plot.show()
                
                # Identify which vessel for 1st segment
                which_branch_input = int(input(vessel_options))
                
                # If it is a vessel of interest
                if which_branch_input < len(vessel_names):

                    # Check to see if the vessel has already been assigned
                    if which_branch_input in assigned_vessel_indices:
                        print('Already assigned')            

                    else:
                        # Split the centerlines, normal, radii and distance
                        centerline_points_segment1 = centerline_points[0:pointID[0],:]
                        centerline_normal_segment1 = centerline_normal[0:pointID[0],:]
                        centerline_radii_segment1 = centerline_radii[0:pointID[0]]
                        centerline_distance_segment1 = centerline_distance[0:pointID[0]]
                
                        # Identify the vessel
                        vessel_label = vessel_names[int(which_branch_input)]
                        print(vessel_label)
                    
                    
                        dpoints,dvectors,dradii,ddist = downsample_points(vessel_label,P1_indices,which_branch_input,stl_surf_m,centerline_points_segment1,centerline_normal_segment1,centerline_radii_segment1,centerline_distance_segment1,dpoints,dvectors,dradii,ddist)
                        
                        # Store which vessels have been assigned
                        assigned_vessel_indices.append(which_branch_input)
                        assigned_vessels.append(vessel_names[which_branch_input])
                
                
                # Define centerlines for 2nd segment and plot them
                segment2_plot = pv.Plotter()
                segment2_plot.add_mesh(stl_surf_m, opacity=0.3)
                plot_centerlines_segment2 = pv.PolyData(centerline_points[pointID[0]+1:,:])
                segment2_plot.add_mesh(plot_centerlines_segment2)
                segment2_plot.show()
                
                # Identify which vessel for 2nd segment
                which_branch_input = int(input(vessel_options))
                
                # If it is a vessel of interest
                if which_branch_input < len(vessel_names):

                    # Check to see if the vessel has already been assigned
                    if which_branch_input in assigned_vessel_indices:
                        print('Already assigned')            

                    else:
                        # Split the centerlines, normal, radii and distance
                        centerline_points_segment2 = centerline_points[pointID[0]+1:,:]
                        centerline_normal_segment2 = centerline_normal[pointID[0]+1:,:]
                        centerline_radii_segment2 = centerline_radii[pointID[0]+1:]
                        centerline_distance_segment2 = centerline_distance[pointID[0]+1:]
                
                        # Identify the vessel
                        vessel_label = vessel_names[int(which_branch_input)]
                        print(vessel_label)

                    
                        dpoints,dvectors,dradii,ddist = downsample_points(vessel_label,P1_indices,which_branch_input,stl_surf_m,centerline_points_segment2,centerline_normal_segment2,centerline_radii_segment2,centerline_distance_segment2,dpoints,dvectors,dradii,ddist)
                        
                        # Store which vessels have been assigned
                        assigned_vessel_indices.append(which_branch_input)
                        assigned_vessels.append(vessel_names[which_branch_input])


    # Check to see if all vessels have been assigned
    assigned_vessel_indices.sort()
    all_vessel_indices = np.arange(0,len(vessel_names),1)
    if np.array_equal(assigned_vessel_indices,all_vessel_indices):
        print('All vessels assigned')
    else:
        print('Not all vessels are assigned!')
        print('Assigned vessels : ')
        print(assigned_vessels)

    return dpoints, dvectors, ddist, dradii


# %%
pinfo = input('Patient number -- ') 
case = input ('Condition -- ')

save_dict_directory = 'L:/vasospasm/' + pinfo + \
    '/' + case + '/4-results/pressure_resistance/'



# if __name__ == "__main__":

dpoints_original, dvectors_original, ddist_original, dradii_original = main(
    pinfo, case)

#%% Check that the final points are assigned correctly

fname_stl_m = 'L:/vasospasm/' + pinfo + '/' + case + \
    '/1-geometry/' + pinfo + '_' + case + '_final.stl'
stl_surf_m = pv.read(fname_stl_m)

for i_vessel in range(len(dpoints_original)):

    vessel_name = dpoints_original["points{}".format(i_vessel)][0]
    # Plot STL and points
    check_vessels_plot = pv.Plotter()
    check_vessels_plot.add_mesh(stl_surf_m, opacity=0.3)

    plot_vessel_points = pv.PolyData(
        dpoints_original["points{}".format(i_vessel)][1])
    check_vessels_plot.add_mesh(plot_vessel_points,label=vessel_name)
    check_vessels_plot.add_legend(size=(.2, .2), loc='upper right')
    check_vessels_plot.show()


# %%

if not os.path.exists(save_dict_directory):
    os.makedirs(save_dict_directory)

save_dict(dpoints_original, save_dict_directory +
          "points_original_" + pinfo + "_" + case)
save_dict(dvectors_original, save_dict_directory +
          "vectors_original_" + pinfo + "_" + case)
save_dict(ddist_original, save_dict_directory +
          "dist_original_" + pinfo + "_" + case)
save_dict(dradii_original, save_dict_directory +
          "radii_original_" + pinfo + "_" + case)
print('Dictionaries of original data saved')
