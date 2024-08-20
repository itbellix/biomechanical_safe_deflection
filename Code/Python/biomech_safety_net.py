import os
import casadi as ca
import numpy as np
import rospy
import time
import matplotlib.pyplot as plt
import utilities_TO as utils_TO
import utilities_casadi as utils_ca
from scipy.spatial.transform import Rotation as R
import pickle
from std_msgs.msg import Float64MultiArray, Bool
import threading

# import parser
import argparse

# import the strain visualizer
import realTime_strainMap_visualizer as sv

class BS_net:
    """
    Biomechanics Safety Net module.
    """
    def __init__(self, shared_ros_topics, debug_mode, rate=200, simulation = 'true', speed_estimate:Bool=False):
        """"
        Initialization of the module, given inputs and default values
        """
        # set debug level
        self.debug_mode = debug_mode
        self.simulation = simulation

        # variables related to the definition of the biomechanical (shoulder) model
        self.pe_boundaries = [-20, 160] # defines the interval of physiologically plausible values for the plane of elevation [deg]
        self.se_boundaries = [0, 144]   # as above, for the shoulder elevation [deg]
        self.ar_boundaries = [-90, 100] # as above, for the axial rotation [deg]

        self.strainmap_step = 4         # discretization step used along the model's coordinate [in degrees]
                                        # By default we set it to 4, as the strainmaps are generated from the biomechanical model
                                        # with this grid accuracy
        
        self.ar_values = np.arange(self.ar_boundaries[0], self.ar_boundaries[1], self.strainmap_step)
        
        # Strainmap parameters
        self.pe_datapoints = np.array(np.arange(self.pe_boundaries[0], self.pe_boundaries[1], self.strainmap_step))
        self.se_datapoints = np.array(np.arange(self.se_boundaries[0], self.se_boundaries[1], self.strainmap_step))

        self.X,self.Y = np.meshgrid(self.pe_datapoints, self.se_datapoints, indexing='ij')
        self.X_norm = self.X/np.max(np.abs(self.pe_boundaries))
        self.Y_norm = self.Y/np.max(np.abs(self.se_boundaries))

        self.num_gaussians = 0          # the number of 2D Gaussians used to approximate the current strainmap
        self.num_params_gaussian = 6    # number of parameters that each Gaussian is defined of (for a 2D Gaussian, we need 6)
        self.all_params_gaussians = []  # list containing the values of all the parameters defining all the Gaussians
                                        # For each Gaussian, we have: amplitude, x0, y0, sigma_x, sigma_y, offset
                                        # (if more Gaussians are present, the parameters of all of them are concatenated)

        self.strainmaps_params_dict = None          # Dictionary containing of all the precomputed (and interpolated) strainmaps. 
                                                    # They are stored in memory as parameters of the corresponding gaussians and 
                                                    # the correct set of parameters is selected at every time instant (and set to strainmap_current)

        self.strainmap_current = None               # strainmap corresponding to the current model state

        # Definition of the unsafe zones
        self.num_params_ellipses = 4    # each unsafe zone will be defined by a 2D ellipse. Its parameters are x0, y0, a^2 and b^2
                                        # the expression for the ellipse is (x-x0)^2/a^2+(y-y0)^2/b^2=1

        self.max_safe_strain = 2.7      # value of strain [%] that is considered unsafe

        # definition of the state variables
        self.state_space_names = None               # The names of the coordinates in the model that describe the state-space in which we move
                                                    # For the case of our shoulder rehabilitation, it will be ["plane_elev", "shoulder_elev", "axial_rot"]

        self.state_values_current = None            # The current value of the variables defining the state
        self.speed_estimate = speed_estimate        # Are estimated velocities of the model computed?
                                                    # False: quasi_static assumption, True: updated through measurements
        
        # parameters regarding the muscle activation
        self.varying_activation = False             # initialize the module assuming that the activation will not change
        self.activation_level = 0                   # level of activation of the muscles whose strain we are considering
                                                    # this is an index 
        self.max_activation = 1                     # maximum activation considered (defaulted to 1)
        self.min_activation = 0                     # minimum activation considered (defaulted to 0)
        self.delta_activation = 0.005               # resolution on the activation value that the strainmap captures

        # Robot-related variables
        self.current_ee_pose = None

        # filter the reference that is generated, to avoid noise injection from the human-pose estimator
        self.last_cart_ee_cmd = np.zeros(3)         # store the last command sent to the robot (position)
        self.last_rotation_ee_cmd = np.zeros(3)     # store the last command sent to the robot (rotation)
        self.alpha_p = 0.9                          # set the weight used in the exponential moving average filter
        self.alpha_r = 0.9                          # set the weight used in the exponential moving average filter
        self.filter_initialized = False             # has the filter been initialized already?

        # initialize the visualization module for the strain maps
        self.strain_visualizer = sv.RealTimeStrainMapVisualizer(self.X_norm, self.Y_norm, self.num_params_gaussian, self.pe_boundaries, self.se_boundaries)

        # ROS part
        # initialize ROS node and set required frequency
        rospy.init_node('BS_net_node')
        self.ros_rate = rospy.Rate(rate)

        # Create publisher for the cartesian trajectory for the KUKA end-effector
        self.topic_opt_traj = shared_ros_topics['cartesian_ref_ee']
        self.pub_trajectory = rospy.Publisher(self.topic_opt_traj, Float64MultiArray, queue_size=1)
        self.flag_pub_trajectory = False    # flag to check if trajectory is being published (default: False = no publishing)
        
        # Create the publisher for the unused z_reference
        # It will publish the uncompensated z reference when running a real experiment,
        # and the gravity compensated reference when running in simulation.
        self.topic_z_level = shared_ros_topics['z_level']
        self.pub_z_level = rospy.Publisher(self.topic_z_level, Float64MultiArray, queue_size=1)

        # Create the publisher dedicated to stream the optimal trajectories and controls
        self.topic_optimization_output = shared_ros_topics['optimization_output']
        self.pub_optimization_output = rospy.Publisher(self.topic_optimization_output, Float64MultiArray, queue_size=1)

        # Create a subscriber to listen to the current value of the shoulder pose
        self.topic_shoulder_pose = shared_ros_topics['estimated_shoulder_pose']
        self.sub_curr_shoulder_pose = rospy.Subscriber(self.topic_shoulder_pose, Float64MultiArray, self._shoulder_pose_cb, queue_size=1)
        self.flag_receiving_shoulder_pose = False       # flag indicating whether the shoulder pose is being received

        # Set up the structure to deal with the new thread, to allow continuous publication of the optimal trajectory and torques
        # The thread itself is created later, with parameters known at run time
        self.x_opt_lock = threading.Lock()          # Lock for synchronizing access to self.x_opt
        self.publish_thread = None                  # creating the variable that will host the thread

        # create a subscriber to catch when the trajectory optimization should be running
        self.flag_run = False
        self.sub_run= rospy.Subscriber(shared_ros_topics['request_reference'], Bool, self._flag_run_cb, queue_size=1)



    def setCurrentStrainMapParams(self, all_params_gaussians):
        """"
        This function is used to manually update the parameters fo the strainmap that is being considered, 
        based on user input.
        The strainmap parameters are [amplitude, x0, y0, sigma_x, sigma_y, offset] for each of the 2D-Gaussian
        functions used to approximate the original discrete strainmap.

        Out of these parameters, we can already obtain the analytical expression of the high-strain zones, that is
        computed and saved below. 
        
        It is useful for debugging, then the update of the strainmaps should be done based on the 
        state_values_current returned by sensor input (like the robotic encoder, or visual input).
        """
        self.all_params_gaussians = all_params_gaussians        # store the parameters defining the strainmap

        num_gaussians = len(self.all_params_gaussians)//self.num_params_gaussian    # find the number of gaussians employed

        all_params_ellipses = np.zeros(num_gaussians*self.num_params_ellipse)       # corresponding number of parameters for the
                                                                                    # ellipses defining the unsafe zones

        for i in range(num_gaussians):
            amplitude = self.all_params_gaussians[i*self.num_params_ellipse + 0]
            x0 = self.all_params_gaussians[i*self.num_params_ellipse + 1]
            y0 = self.all_params_gaussians[i*self.num_params_ellipse + 2]
            sigma_x = self.all_params_gaussians[i*self.num_params_ellipse + 3]
            sigma_y = self.all_params_gaussians[i*self.num_params_ellipse + 4]
            offset = self.all_params_gaussians[i*self.num_params_ellipse + 5]

            a_squared = - 2 * sigma_x^2 / np.log10((self.max_safe_strain-offset)/amplitude)
            if a_squared <= 0:
                a_squared = np.nan
            b_squared = - 2 * sigma_y^2 / np.log10((self.max_safe_strain-offset)/amplitude)
            if b_squared <= 0:
                b_squared = np.nan

            all_params_ellipses[i*self.num_params_ellipse:i*self.num_params_ellipse+self.num_params_ellipses] = np.array([x0, y0, a_squared, b_squared])

        self.all_params_ellipses = all_params_ellipses


    def publishCartRef(self, shoulder_pose_ref, torque_ref, position_gh_in_base, base_R_sh, dist_gh_elbow):
        """"
        This function publishes a given reference shoulder state as the equivalent 6D cartesian pose corresponding
        to the position of the elbow tip, expressed the world frame. The center of the shoulder in this frame needs 
        to be given by the user, so that they need to specify the position of the GH joint center in the world frame 
        (px, py, pz), and the orientation of the shoulder reference frame (i.e., the scapula reference frame) as well.
        The underlying assumption is that the scapula/shoulder frame remains fixed over time wrt the world frame.

        The inputs are:
            - shoulder_pose_ref: 3x1 numpy array, storing the values of plane of elevation, shoulder 
              elevation and axial rotation at a given time instant
            - torque_ref: 2x1 numpy array, storing the torques to be applied to the plane of elevation and 
              shoulder elevation (output of the trajectory optimization step)
            - position_gh_in_base: the coordinates (px, py, pz) as a numpy array
            - base_R_sh: rotation matrix defining the orientation of the shoulder frame wrt the world frame
                         (as a scipy.spatial.transform.Rotation object)
            - dist_gh_elbow: the vector expressing the distance of the elbow tip from the GH center, expressed in the 
                             shoulder frame when plane of elevation = shoulder elevation =  axial rotation= 0

        The positions and distances must be expressed in meters, and the rotations in radians.
        """
        # distinguish between the various coordinates, so that debugging is simpler
        pe = shoulder_pose_ref[0]
        se = shoulder_pose_ref[1]
        ar = shoulder_pose_ref[2]

        # define the required rotations
        base_R_elb = base_R_sh*R.from_euler('y', pe)*R.from_euler('x', -se)*R.from_euler('y', ar-ar_offset)

        base_R_ee = base_R_elb * R.from_euler('x', -np.pi/2)

        euler_angles_cmd = base_R_ee.as_euler('xyz') # store also equivalent Euler angles

        # find position for the end-effector origin
        ref_cart_point = np.matmul(base_R_elb.as_matrix(), dist_gh_elbow) + position_gh_in_base

        # modify the reference along the Z direction, to account for the increased interaction force
        # due to the human arm resting on the robot. We do this only if we are not in simulation.
        if torque_ref is not None:
            k_z = ee_stiffness[2]
            se_estimated = self.state_values_current[2]
            torque_se = torque_ref[1]
            z_current = self.current_ee_pose[2]

            new_z_ref = z_current + torque_se/(k_z * experimental_params['L_tot'] * np.sin(se_estimated))
            
            # append new z reference (in this way, we can filter both the new and the old)
            ref_cart_point = np.hstack((ref_cart_point, new_z_ref))

        # check if filter has been initialized already
        if not self.filter_initialized:
            self.last_cart_ee_cmd = ref_cart_point
            self.last_rotation_ee_cmd = euler_angles_cmd
            self.filter_initialized = True

        # when we start computing the adjusted z reference, update the state of the filter too
        if self.last_cart_ee_cmd.shape[0] < ref_cart_point.shape[0]:
            self.last_cart_ee_cmd = np.hstack((self.last_cart_ee_cmd, self.last_cart_ee_cmd[2]))
        
        # filter position to command to the robot
        ref_cart_point = self.alpha_p * ref_cart_point + (1-self.alpha_p)*self.last_cart_ee_cmd

        # substitute the old (uncorrected) z reference with the new one
        if torque_ref is not None:
            if self.simulation == 'false':
                alternative_z_ref = np.atleast_2d(ref_cart_point[2])    # save the filtered old (uncorrected) ref
                ref_cart_point = np.delete(ref_cart_point, [2])         # now ref_cart point contains [x,y, gravity_compensated_z] filtered
            else:
                alternative_z_ref = np.atleast_2d(ref_cart_point[3])    # save the gravity compensated reference (unused)
                ref_cart_point = np.delete(ref_cart_point, [3])         # now ref_cart point contains [x,y, gravity_uncompensated_z] filtered
        else:
            alternative_z_ref = np.atleast_2d(np.array([np.nan]))

        # filter orientation to command to the robot (through equivalent Euler angles)
        euler_angles_cmd = self.alpha_r * euler_angles_cmd + (1-self.alpha_r)*self.last_rotation_ee_cmd

        # update the filter state
        self.last_cart_ee_cmd = ref_cart_point
        self.last_rotation_ee_cmd = euler_angles_cmd

        # retrieve rotation matrix from filtered angles
        base_R_ee = R.from_euler('xyz', euler_angles_cmd)

        # instantiate the homogenous matrix corresponding to the pose reference.
        homogeneous_matrix = np.eye(4)
        homogeneous_matrix[0:3, 3] = np.transpose(ref_cart_point)       # store the 3D cartesian position
        homogeneous_matrix[0:3, 0:3] = base_R_ee.as_matrix()            # store the rotation information

        # build the message
        message_ref = Float64MultiArray()
        message_ref.layout.data_offset = 0
        message_ref.data = np.reshape(homogeneous_matrix, (16,1))

        # publish the message
        self.pub_trajectory.publish(message_ref)

        # publish also the alternative_z_ref
        message_z = Float64MultiArray()
        message_z.layout.data_offset = 0
        message_z.data = alternative_z_ref
        self.pub_z_level.publish(message_z)
        

    def publishInitialPoseAsCartRef(self, shoulder_pose_ref, position_gh_in_base, base_R_sh, dist_gh_elbow):
        """"
        This function publishes a given reference shoulder state as the equivalent 6D cartesian pose corresponding
        to the position of the elbow tip, expressed the world frame. The center of the shoulder in this frame needs 
        to be given by the user, so that they need to specify the position of the GH joint center in the world frame 
        (px, py, pz), and the orientation of the shoulder reference frame (i.e., the scapula reference frame) as well.
        The underlying assumption is that the scapula/shoulder frame remains fixed over time wrt the world frame.

        The inputs are:
            - shoulder_pose_ref: 3x1 numpy array, storing the values of plane of elevation, shoulder 
              elevation and axial rotation at a given time instant
            - position_gh_in_base: the coordinates (px, py, pz) as a numpy array
            - base_R_sh: rotation matrix defining the orientation of the shoulder frame wrt the world frame
                         (as a scipy.spatial.transform.Rotation object)
            - dist_gh_elbow: the vector expressing the distance of the elbow tip from the GH center, expressed in the 
                             shoulder frame when plane of elevation = shoulder elevation =  axial rotation= 0

        The positions and distances must be expressed in meters, and the rotations in radians.

        This function is conceptually equivalent to publishCartRef, but it uses another ROS publisher to publish the
        required reference to the topic associated to the initial pose for the robot. It will keep publishing the initial pose
        until the robot controller receives it, and then it stops.

        Once called, it will also return the position that the robot has been commanded to reach.
        """
        shoulder_state = np.zeros((6, 1))
        shoulder_state[0::2] = shoulder_pose_ref.reshape(-1, 1)
        self.x_opt = shoulder_state

        # perform extra things if this is the first time we execute this
        if not self.flag_pub_trajectory:
            # We need to set up the structure to deal with the new thread, to allow 
            # continuous publication of the optimal trajectory
            self.publish_thread = threading.Thread(target=self.publish_continuous_trajectory, 
                                                    args = (position_gh_in_base, base_R_sh, dist_gh_elbow))   # creating the thread
            
            self.publish_thread.daemon = True   # this allows to terminate the thread when the main program ends
            self.flag_pub_trajectory = True     # update flag
            self.publish_thread.start()         # start the publishing thread


    def _shoulder_pose_cb(self, data):
        """
        Callback receiving and processing the current shoulder pose.
        """
        if not self.flag_receiving_shoulder_pose:       # if this is the first time we receive something, update flag
            self.flag_receiving_shoulder_pose = True

        # retrieve the current state as estimated on the robot's side
        self.state_values_current = np.array(data.data[0:6])        # update current pose

        if not self.speed_estimate:                                 # choose whether we use the velocity estimate or not
            self.state_values_current[1::2] = 0

        time_now = time.time()

        self.current_ee_pose = np.array(data.data[-3:])   # update estimate of where the robot is now (last 3 elements)

        # set up a logic that allows to start and update the visualization of the current strainmap at a given frequency
        # We set this fixed frequency to be 10 Hz
        if np.round(time_now-np.fix(time_now), 2) == np.round(time_now-np.fix(time_now),1):
            self.strain_visualizer.updateStrainMap(self.all_params_gaussians, 
                                                   self.state_values_current[[0,2]], 
                                                   self.x_opt[[0,2],:],
                                                   None, 
                                                   self.state_values_current[[1,3]])


    def _flag_run_cb(self, data):
        """
        This callback catches the boolean message that is published on a default topic, informing whether the robot wants
        to receive new, optimized references or not.
        """
        self.flag_run = data.data

    
    def keepRunning(self):
        """
        This function retrieves the value stored in self.flag_run, allowing to continue or stop the reference 
        trajectory computations.
        """
        return self.flag_run


    def waitForShoulderState(self):
        """"
        Utility that stops the execution of the code until the current shoulder pose becomes available.
        This ensures that correct communication is established between the biomechanics simulation and the robot.
        """
        while not rospy.is_shutdown() and not self.flag_receiving_shoulder_pose:
            self.ros_rate.sleep()

        print("Receiving current shoulder pose.")


    def setReferenceToCurrentPose(self):
        """
        This function allows to overwrite the reference state/control values. They are
        substituted by the current state of the subject/patient sensed by the robot.
        """
        with self.x_opt_lock:
            self.x_opt = self.state_values_current.reshape(self.nlps_module.dim_x, 1)
            self.u_opt = np.zeros((2,1))


    def setReferenceToCurrentRefPose(self):
        """
        This function allows to overwrite the reference state/control values, by keeping
        only the one that is currently being tracked.
        """
        with self.x_opt_lock:
            self.x_opt = self.x_opt[:,0].reshape((6,1))
            self.u_opt = np.zeros((2,1))


    def reachedPose(self, pose, tolerance = 0):
        """
        This function checks whether the current position of the human model
        is the same as the one we are willing to reach ('pose'). Optionally, 
        a 'tolerance' on such pose can be considered too, in radians.
        """
        if np.linalg.norm(self.state_values_current[[0,2]] - pose[[0,2]]) <= tolerance:
            return True
        else:
            return False
        

    def monitor_unsafe_zones(self):
        """
        This function monitors whether we are in any unsafe zones, based on the ellipse parameters
        obtained by the analytical expression of the strainmaps. In particular:
            - if we are outside of all the ellipses, the reference point (in the human frame)
              is set to be the current pose
            - if we are inside one (or more) ellipses, the reference point is the closest point
              on the ellipses borders.
        The corresponding human pose is commanded to the robot, transforming it into EE pose.
        """
        num_ellipses = len(self.all_params_ellipses)/self.num_params_ellipses   # retrieve number of ellipses currently present in the map

        in_zone_i = np.zeros(num_ellipses)
        for i in range(num_ellipses):       # loop through all the ellipses, to check if the current model's state is inside any of them
            a_squared = self.all_params_ellipses[self.num_params_ellipses*i+2]
            b_squared = self.all_params_ellipses[self.num_params_ellipses*i+3]
            if a_squared is not None and b_squared is not None:     # if the ellipse really exists
                x0 = self.all_params_ellipses[self.num_params_ellipses*i]
                y0 = self.all_params_ellipses[self.num_params_ellipses*i+1]

                pe = self.state_values_current[0]
                se = self.state_values_current[2]

                in_zone_i[i] = ((pe-x0)^2/a_squared + (se-y0)^2/b_squared) < 1

            TODO: implement strategy to find closest point and send that to the robot


# ----------------------------------------------------------------------------------------------

if __name__ == '__main__':
    try:
        # check if we are running in simulation or not
        parser = argparse.ArgumentParser(description="Script that runs the Bio-aware safety net")
        parser.add_argument("--simulation", required=True, type=str)
        args = parser.parse_args()
        simulation = args.simulation

        # define the required paths
        code_path = os.path.dirname(os.path.realpath(__file__))     # getting path to where this script resides
        path_to_repo = os.path.join(code_path, '..', '..')          # getting path to the repository
        path_to_model = os.path.join(path_to_repo, 'Musculoskeletal Models')    # getting path to the OpenSim models

        ## PARAMETERS -----------------------------------------------------------------------------------------------
        # import the parameters for the experiment as defined in experiment_parameters.py
        from experiment_parameters import *     # this contains the experimental_params and the shared_ros_topics

        # are we debugging or not?
        debug_mode = False

        # mode
        mode = 1    # quite rough way of selecting the operating mode 
                    # (do this better, so that it can be changed at runtime by the user)

        # initialize the biomechanics-safety net module
        bsn_module = BS_net(shared_ros_topics, debug_mode, rate=200, simulation = simulation, speed_estimate=True)

        # Publish the initial position of the KUKA end-effector, according to the initial shoulder state
        # This code is blocking until an acknowledgement is received, indicating that the initial pose has been successfully
        # received by the RobotControlModule
        bsn_module.publishInitialPoseAsCartRef(shoulder_pose_ref = x_0[0::2], 
                                            position_gh_in_base = experimental_params['p_gh_in_base'], 
                                            base_R_sh = experimental_params['base_R_shoulder'], 
                                            dist_gh_elbow = experimental_params['d_gh_ee_in_shoulder'])
        
        params_strainmap_test = np.array([0,0,0,0,0,0])
        
        bsn_module.setCurrentStrainMapParams(params_strainmap_test)

        # Wait until the robot has reached the required position, and proceed only when the current shoulder pose is published
        bsn_module.waitForShoulderState()

        # wait until the robot is requesting the optimized reference
        print("Waiting to start therapy")
        while not bsn_module.keepRunning():
            rospy.sleep(0.1)

        # countdown for the user to be ready
        print("Starting in...")
        print("3")
        time.sleep(1)
        print("2")
        time.sleep(1)
        print("1")
        time.sleep(1)

        # start to provide the safe reference as long as the robot is requesting it
        # we do so in a way that allows temporarily pausing the therapy
        while not rospy.is_shutdown():
            if bsn_module.keepRunning():
                if mode == 1:
                    bsn_module.monitor_unsafe_zones()
            
            # if the user wants to interrupt the therapy, we stop the optimization and freeze 
            # the robot to its current reference position.
            if not bsn_module.keepRunning():
                # overwrite the future references with the current one
                    bsn_module.setReferenceToCurrentRefPose()   # to decrease bumps (as trajectory will be very smooth anyway)

            bsn_module.ros_rate.sleep()

    except rospy.ROSInterruptException:
        pass
