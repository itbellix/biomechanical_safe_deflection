"""
This file stores the parameters for the experimental setups
It contains information that should be available both to TO_main.py 
and to robot_control.py. By defining them here and then importing
this file where needed, we avoid duplicating things.
"""

import os
import numpy as np
from scipy.spatial.transform import Rotation as R

#-------------------------------------------------------------------------
# choose which experiment to perform
experiment = 2      # 1: Passive human subject (strain map is only position-dependent)
                    # 2: Active human subject (strain maps change with muscle activation)
                    # 3: tuning of the cost function weights, in simulation, for experiment #1
#-------------------------------------------------------------------------

# define the required paths
code_path = os.path.dirname(os.path.realpath(__file__))     # getting path to where this script resides
path_to_repo = os.path.join(code_path, '..', '..')          # getting path to the repository

# set the frequency to check for new message and execute loops
loop_frequency = 20    # [Hz]

# select here with which setup/subject we are working
subject = 'subject1'       # list of available subjects: subject1

setup = 'newLab_facingRobot'        # list of setups: 'OldLab' (configuration that the robot had before 12/12/2023)

# physical parameters related to the experimental setup (they could be different every time)
#   * l_arm:  (subject-dependent) length of the segment between the glenohumeral joint center 
#             and the elbow tip, when the elbow of the subject is bent at 90 degrees 
#   * l_brace: (fixed) length of the segment between the elbow tip and the robot end-effector, when
#              the subject is wearing the brace - it is the thickness of the brace along the arm direction 
#   * base_R_shoulder: rotation that expresses the orientation of the human shoulder frame in the 
#                      base frame of the robot  
#   * position_gh_in_base: position of the center of the shoulder frame (GH joint) in the base frame [m]


if subject=='subject1':
    l_arm = 0.32            # length of the subject's right arm, from the center of the glenohumeral (GH) joint to the elbow [m]
    l_brace = 0.02          # thickness of the brace [m]


if setup=='newLab_facingRobot':                  
    # This is the setup used now that the KUKA7 is mounted on the table
    rot_sh_in_base = np.array([np.pi/2, 0, 0])      # intrinsic series of rotations around the x-y-z axis of the robot base 
                                                    # to align it with shoulder frame [rad]
    # if the person is facing the robot, looking along the +X direction of the base: np.array([np.pi/2, 0, 0]) 

    base_R_shoulder = R.from_euler('x', rot_sh_in_base[0]) * R.from_euler('y', rot_sh_in_base[1]) * R.from_euler('z', rot_sh_in_base[2])

    if subject == 'subject1':
        position_gh_in_base = np.array([-0.9, 0, 0.62]) # position of the center of the shoulder frame (GH joint) in the base frame [m]

    ar_offset = - np.pi/4

if setup=='OldLab':                  
    # This is the setup used before the lab was moved
    rot_sh_in_base = np.pi/2    # rotation around the x axis of the robot base to align it with shoulder frame [rad]
    base_R_shoulder = R.from_euler('x', rot_sh_in_base, degrees=False)

    position_gh_in_base = np.array([-0.2, 0.8, 0.6]) # position of the center of the shoulder frame (GH joint) in the base frame [m]

# depending on the experiment, set the parameters used only by the trajectory optimization module
if experiment == 1:
    # determine the time horizon and control intervals for the NLP problem, on the basis of the experiment
    N = 50  # control intervals used (control will be constant during each interval)
    T = 5.  # time horizon for the optimization

    # file from which to read the precomputed parameters that define the strainmaps
    file_strainmaps = os.path.join(path_to_repo, 'Musculoskeletal Models','Strain Maps', 'Passive', 'differentiable_strainmaps_allTendons.pkl')

    # set the cost weights
    gamma_strain = 1        # weight for increase in strain
    gamma_goal = 0          # weight for distance to goal
    gamma_velocities = 0     # weight on the coordinates' velocities
    gamma_acceleration = 10  # weight on the coordinates' acceleration

    # initial state (referred explicitly to the position of the patient's GH joint) 
    # Therapy will start in this position - used to build the NLP structure, and to command first position of the robot
    # x = [pe, pe_dot, se, se_dot, ar, ar_dot], but note that ar and ar_dot are not tracked
    x_0 = np.deg2rad(np.array([50, 0, 100, 0, 0, 0]))

    # goal states (if more than one, the next one is used once the previous is reached)
    # here we command multiple ones in sequence to achieve a rather large range of motion
    x_goal_1 = np.deg2rad(np.array([100, 0, 100, 0, 0, 0]))
    x_goal_2 = np.deg2rad(np.array([60, 0, 60, 0, 0, 0]))
    x_goal_3 = np.deg2rad(np.array([60, 0, 110, 0, 0, 0]))

    x_goal = np.vstack((x_0, x_goal_1, x_goal_2, x_goal_3))     # the first goal is the current position
                                                                # in this way gravity compensation is activated
                                                                # before starting the actual experiment
    
    # enforce constraints on the final state
    constrain_final_position = True
    constrain_final_velocity = True

    speed_estimate = True

    # specify the stiffness and damping for the cartesian impedance controller implemented on the robot
    # we do this both for the nullspace controlling the elbow (ns_elb) and for the end effector
    ns_elb_stiffness = np.array([10, 10, 10])
    ns_elb_damping = 2*np.sqrt(ns_elb_stiffness)

    ee_stiffness = np.array([550, 550, 550, 15, 15, 4])
    ee_damping = 2*np.sqrt(ee_stiffness) 

    # option to change these only in simulation
    ns_elb_stiffness_sim = ns_elb_stiffness
    ns_elb_damping_sim = ns_elb_damping

    ee_stiffness_sim = ee_stiffness
    ee_damping_sim = ee_damping

if experiment == 2:
    # determine the time horizon and control intervals for the NLP problem, on the basis of the experiment
    N = 10      # control intervals used (control will be constant during each interval)
    T = 1.      # time horizon for the optimization

    target_tendon = "SSPA_sim" # available :  "SSPA"      (supraspinatus anterior - used in the paper for real robot experiment)
                               #              "SSPA_sim"  (supraspinatus anterior - used in the paper for simulated experiment with different activation ramps)
                               #              "custom_1"
                               #              "ISI" (infraspinatus inferior) 

    # file from which to read the precomputed parameters that define the strainmaps
    if target_tendon == "ISI":
        file_strainmaps = os.path.join(path_to_repo, 'Musculoskeletal Models','Strain Maps', 'Active', 'differentiable_strainmaps_ISI.pkl')
    elif target_tendon[0:4] == "SSPA":
        file_strainmaps = os.path.join(path_to_repo, 'Musculoskeletal Models','Strain Maps', 'Active', 'differentiable_strainmaps_SSPA.pkl')
    elif target_tendon == "custom_1":
        file_strainmaps = None

    # set the cost weights
    if target_tendon[0:4] == "SSPA" or target_tendon == "custom_1":
        gamma_strain = 2        # weight for increase in strain
        gamma_goal = 1          # weight for distance to goal (real robot)
        # gamma_goal = 2          # weight for distance to goal (simulation)
        gamma_velocities = 0        # weight for use of generalized torques
        gamma_acceleration = 0.3  # weight on the coordinates' acceleration (real robot)
        # gamma_acceleration = 0.3  # weight on the coordinates' acceleration (simulation)

    if target_tendon == "ISI":
        gamma_strain = 1        # weight for increase in strain
        gamma_goal = 2          # weight for distance to goal
        gamma_velocities = 0        # weight for use of generalized torques
        gamma_acceleration = 0.3  # weight on the coordinates' acceleration

    if target_tendon == "ISI":
        # initial state (referred explicitly to the position of the patient's GH joint) 
        # Therapy will start in this position - used to build the NLP structure, and to command first position of the robot
        # x = [pe, pe_dot, se, se_dot, ar, ar_dot], but note that ar and ar_dot are not tracked
        x_0 = np.deg2rad(np.array([60, 0, 70, 0, 0, 0]))

        # goal states (if more than one, the next one is used once the previous is reached)
        x_goal_1 = np.deg2rad(np.array([100, 0, 100, 0, 0, 0]))

        x_goal = np.vstack((x_goal_1, x_0, x_goal_1, x_0))

        speed_estimate = True

    if target_tendon[0:4] == "SSPA":
        x_0 = np.deg2rad(np.array([45, 0, 95, 0, 0, 0]))

        # goal states (if more than one, the next one is used once the previous is reached)
        x_goal_1 = np.deg2rad(np.array([60, 0, 60, 0, 0, 0]))

        if target_tendon[-3:]== 'sim':
            # used in simulation, to test effects of different activation ramps
            # here we perform only one motion
            x_goal = x_0                
            x_0 = x_goal_1

            speed_estimate = False

        else:
            x_goal = np.vstack((x_goal_1, x_0, x_goal_1, x_0))  # used on the real robot
            speed_estimate = True

    if target_tendon == "custom_1":
        x_0 = np.deg2rad(np.array([45, 0, 95, 0, 0, 0]))

        # goal states (if more than one, the next one is used once the previous is reached)
        x_goal_1 = np.deg2rad(np.array([60, 0, 60, 0, 0, 0]))

        x_goal = x_goal_1

        speed_estimate = False

    # enforce constraints on the final state
    constrain_final_position = False
    constrain_final_velocity = True

    # parameters for considering the varying activation
    min_activation = 0
    max_activation = 0.5
    delta_activation = 0.01

    # specify the stiffness and damping for the cartesian impedance controller implemented on the robot
    # we do this both for the nullspace controlling the elbow (ns_elb) and for the end effector
    ns_elb_stiffness = np.array([10, 10, 10])
    ns_elb_damping = 2*np.sqrt(ns_elb_stiffness)

    ee_stiffness = np.array([300, 300, 300, 5, 5, 1])
    ee_damping = 2*np.sqrt(ee_stiffness) 

    # option to change these only in simulation
    ns_elb_stiffness_sim = ns_elb_stiffness
    ns_elb_damping_sim = ns_elb_damping

    ee_stiffness_sim = ee_stiffness
    ee_damping_sim = ee_damping

if experiment == 3: # TUNING OF OCP COST FUNCTION WEIGHTS (Fig. 4 in the paper)
    # determine the time horizon and control intervals for the NLP problem, on the basis of the experiment
    N = 50  # control intervals used (control will be constant during each interval)
    T = 5.  # time horizon for the optimization

    # file from which to read the precomputed parameters that define the strainmaps
    file_strainmaps = os.path.join(path_to_repo, 'Musculoskeletal Models','Strain Maps', 'Passive', 'differentiable_strainmaps_allTendons.pkl')

    # set the cost weights
    gamma_strain = 0        # weight for increase in strain
    gamma_goal = 1          # weight for distance to goal
    gamma_velocities = 0     # weight on the coordinates' velocities
    gamma_acceleration = 10  # weight on the coordinates' acceleration

    # initial state (referred explicitly to the position of the patient's GH joint) 
    # Therapy will start in this position - used to build the NLP structure, and to command first position of the robot
    # x = [pe, pe_dot, se, se_dot, ar, ar_dot], but note that ar and ar_dot are not tracked
    x_0 = np.deg2rad(np.array([120, 0, 100, 0, 0, 0]))

    # goal states (if more than one, the next one is used once the previous is reached)
    x_goal_1 = np.deg2rad(np.array([50, 0, 40, 0, 0, 0]))

    x_goal = x_goal_1
    
    # enforce constraints on the final state (in the paper, we keep both the flags to True to generate the results for this experiment)
    constrain_final_position = True
    constrain_final_velocity = True

    speed_estimate = False

    # specify the stiffness and damping for the cartesian impedance controller implemented on the robot
    # we do this both for the nullspace controlling the elbow (ns_elb) and for the end effector
    ns_elb_stiffness = np.array([10, 10, 10])
    ns_elb_damping = 2*np.sqrt(ns_elb_stiffness)

    ee_stiffness = np.array([500, 450, 600, 15, 15, 2])
    ee_damping = 2*np.sqrt(ee_stiffness) 

    # option to change these only in simulation
    ns_elb_stiffness_sim = ns_elb_stiffness
    ns_elb_damping_sim = ns_elb_damping

    ee_stiffness_sim = ee_stiffness
    ee_damping_sim = ee_damping


# -------------------------------------------------------------------------------
# experimental parameters used by both the TO and the robot control modules
dist_shoulder_ee = np.array([0, -(l_arm+l_brace), 0])   # evaluate the distance between GH center and robot ee, in shoulder frame 
                                                        # (once the subject is wearing the brace)
elb_R_ee = R.from_euler('x', -np.pi/2, degrees=False)   # rotation matrix expressing the orientation of the ee in the elbow frame

experimental_params = {}
experimental_params['p_gh_in_base'] = position_gh_in_base
experimental_params['base_R_shoulder'] = base_R_shoulder
experimental_params['L_tot'] = l_arm + l_brace
experimental_params['d_gh_ee_in_shoulder'] = dist_shoulder_ee
experimental_params['elb_R_ee'] = elb_R_ee
# -------------------------------------------------------------------------------
# names of the ROS topics on which the shared communication between biomechanical-based optimization and robot control will happen
shared_ros_topics = {}
shared_ros_topics['estimated_shoulder_pose'] = 'estimated_shoulder_pose'
shared_ros_topics['cartesian_init_pose'] = 'cartesian_init_pose'
shared_ros_topics['cartesian_ref_ee'] = 'cartesian_ref_ee'
shared_ros_topics['request_reference'] = 'request_reference'
shared_ros_topics['optimization_output'] = 'optimization_output'
shared_ros_topics['z_level'] = 'uncompensated_z_ref'
                                                         