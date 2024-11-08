import os
import casadi as ca
import numpy as np
import rospy
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize_scalar
from std_msgs.msg import Float64MultiArray, Bool
import threading

# import parser
import argparse

# import the strain visualizer
import realTime_strainMap_visualizer as sv

# import the nlps_module
import nlps_module as nlps

class BS_net:
    """
    Biomechanics Safety Net module.
    """
    def __init__(self, debug_mode, rate=200, simulation = 'true', speed_estimate:Bool=False):
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

        self.pe_normalizer = 160        # normalization of the variables used to compute the interpolated strain maps
        self.se_normalizer = 144        # normalization of the variables used to compute the interpolated strain maps

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

        # Definition of the unsafe zones (related to INTERACTION MODE 1)
        self.in_zone_i = np.array([0])  # are we inside any unsafe zone? 0 = no, 1 = yes
        self.num_params_ellipses = 4    # each unsafe zone will be defined by a 2D ellipse. Its parameters are x0, y0, a^2 and b^2
                                        # the expression for the ellipse is (x-x0)^2/a^2+(y-y0)^2/b^2=1

        # Definition of the "risky strain" (related to INTERACTION MODE 2)
        self.risky_strain = 2.0      # value of strain [%] that is considered risky

        # definition of the state variables
        self.state_space_names = None               # The names of the coordinates in the model that describe the state-space in which we move
                                                    # For the case of our shoulder rehabilitation, it will be ["plane_elev", "shoulder_elev", "axial_rot"]

        self.state_values_current = None            # The current value of the variables defining the state
        self.speed_estimate = speed_estimate        # Are estimated velocities of the model computed?
                                                    # False: quasi_static assumption, True: updated through measurements
        
        self.future_trajectory = None               # future trajectory of the human model

        # define the references
        self.x_opt = None
        self.u_opt = None

        # parameters regarding the muscle activation
        self.varying_activation = False             # initialize the module assuming that the activation will not change
        self.activation_level = 0                   # level of activation of the muscles whose strain we are considering
                                                    # this is an index 
        self.max_activation = 1                     # maximum activation considered (defaulted to 1)
        self.min_activation = 0                     # minimum activation considered (defaulted to 0)
        self.delta_activation = 0.005               # resolution on the activation value that the strainmap captures

        # Robot-related variables
        self.current_ee_pose = None

        self.ee_cart_stiffness_cmd = None       # stiffness value sent to the Cartesian impedance controller
        self.ee_cart_damping_cmd = None         # damping value sent to the Cartesian impedance controller

        self.ee_cart_stiffness_default = np.array([400, 400, 400, 5, 5, 1]).reshape((6,1))
        self.ee_cart_damping_default = 2 * np.sqrt(self.ee_cart_stiffness_default)

        self.ee_cart_stiffness_low = np.array([20, 20, 20, 5, 5, 1]).reshape((6,1))
        self.ee_cart_damping_low = 2 * np.sqrt(self.ee_cart_stiffness_low)

        self.ee_cart_damping_dampVel_baseline = np.array([35, 35, 35, 10, 10, 1]).reshape((6,1))

        self.ee_cart_stiffness_dampVel_fixed = np.array([100, 100, 100, 5, 5, 1]).reshape((6,1))
        self.ee_cart_damping_dampVel_fixed = np.array([80, 80, 80, 10, 10, 1]).reshape((6,1))

        self.max_cart_damp = 70                # max value of linear damping allowed


        # filter the reference that is generated, to avoid noise injection from the human-pose estimator
        self.last_cart_ee_cmd = np.zeros(3)         # store the last command sent to the robot (position)
        self.last_rotation_ee_cmd = np.zeros(3)     # store the last command sent to the robot (rotation)
        self.alpha_p = 0.9                          # set the weight used in the exponential moving average filter
        self.alpha_r = 0.9                          # set the weight used in the exponential moving average filter
        self.filter_initialized = False             # has the filter been initialized already?

        # initialize the visualization module for the strain maps
        self.strain_visualizer = sv.RealTimeStrainMapVisualizer(self.X_norm, self.Y_norm, self.num_params_gaussian, self.pe_boundaries, self.se_boundaries)

        # initialize the NLP on strain maps, together with the equivalent CasADi function and the input it requires
        self.nlps = None
        self.mpc_iter = None
        self.input_mpc_call = None

        self.nlp_count = 0                          # keeps track of the amount of times the NLP is solved
        self.avg_nlp_time = 0                       # average time required for each NLP iteration
        self.failed_count = 0                       # number of failures encountered

        # ROS part
        # initialize ROS node and set required frequency
        rospy.init_node('BS_net_node')
        self.ros_rate = rospy.Rate(rate)

        # Create publisher for the cartesian trajectory for the KUKA end-effector
        self.topic_opt_traj = rospy.get_param('/rostopic/cartesian_ref_ee')
        self.pub_trajectory = rospy.Publisher(self.topic_opt_traj, Float64MultiArray, queue_size=1)
        self.flag_pub_trajectory = False    # flag to check if trajectory is being published (default: False = no publishing)
        
        # Create the publisher for the unused z_reference
        # It will publish the uncompensated z reference when running a real experiment,
        # and the gravity compensated reference when running in simulation.
        self.topic_z_level = rospy.get_param('/rostopic/z_level')
        self.pub_z_level = rospy.Publisher(self.topic_z_level, Float64MultiArray, queue_size=1)

        # Create the publisher dedicated to stream the optimal trajectories and controls
        self.topic_optimization_output = rospy.get_param('/rostopic/optimization_output')
        self.pub_optimization_output = rospy.Publisher(self.topic_optimization_output, Float64MultiArray, queue_size=1)

        # Create a subscriber to listen to the current value of the shoulder pose
        self.topic_shoulder_pose = rospy.get_param('/rostopic/estimated_shoulder_pose')
        self.sub_curr_shoulder_pose = rospy.Subscriber(self.topic_shoulder_pose, Float64MultiArray, self._shoulder_pose_cb, queue_size=1)
        self.flag_receiving_shoulder_pose = False       # flag indicating whether the shoulder pose is being received

        # Set up the structure to deal with the new thread, to allow continuous publication of the optimal trajectory and torques
        # The thread itself is created later, with parameters known at run time
        self.x_opt_lock = threading.Lock()          # Lock for synchronizing access to self.x_opt
        self.publish_thread = None                  # creating the variable that will host the thread

        # create a subscriber to catch when the trajectory optimization should be running
        self.flag_run = False
        self.sub_run= rospy.Subscriber(rospy.get_param('/rostopic/request_reference'), Bool, self._flag_run_cb, queue_size=1)

        # create a thread to catch the input from teh user, who will select the robot's interaction mode
        self.interaction_mode = 0
        self.input_thread = threading.Thread(target=self.input_thread_fnc, daemon=True)


    def setCurrentEllipseParams(self, all_params_ellipse):
        self.all_params_ellipses = all_params_ellipse


    def getCurrentEllipseParams(self):
        return self.all_params_ellipses, self.num_params_ellipses


    def setCurrentStrainMapParams(self, all_params_gaussians):
        """"
        This function is used to manually update the parameters fo the strainmap that is being considered, 
        based on user input.
        The strainmap parameters are [amplitude, x0, y0, sigma_x, sigma_y, offset] for each of the 2D-Gaussian
        functions used to approximate the original discrete strainmap.

        Out of these parameters, we can already obtain the analytical expression of the high-strain zones, that is
        computed and saved below. (TODO: wrong!)
        
        It is useful for debugging, then the update of the strainmaps should be done based on the 
        state_values_current returned by sensor input (like the robotic encoder, or visual input).
        """
        self.all_params_gaussians = all_params_gaussians        # store the parameters defining the strainmap

        self.num_gaussians = len(self.all_params_gaussians)//self.num_params_gaussian    # find the number of gaussians employed


    def publishCartRef(self, shoulder_pose_ref, torque_ref, base_R_sh):
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
            - base_R_sh: rotation matrix defining the orientation of the shoulder frame wrt the world frame
                         (as a scipy.spatial.transform.Rotation object)

        The positions and distances must be expressed in meters, and the rotations in radians.
        """
        # distinguish between the various coordinates, so that debugging is simpler
        pe = shoulder_pose_ref[0]
        se = shoulder_pose_ref[1]
        ar = shoulder_pose_ref[2]

        # define the required rotations
        base_R_elb = base_R_sh*R.from_euler('y', pe)*R.from_euler('x', -se)*R.from_euler('y', ar - rospy.get_param('/pu/ar_offset'))

        base_R_ee = base_R_elb * R.from_euler('x', -np.pi/2)

        euler_angles_cmd = base_R_ee.as_euler('xyz') # store also equivalent Euler angles

        # find position for the end-effector origin
        dist_gh_elbow = np.array([0, -(rospy.get_param('/pu/l_arm')+rospy.get_param('/pu/l_brace')), 0])
        if rospy.get_param('/pu/estimate_gh_position') and self.flag_receiving_shoulder_pose:
            ref_cart_point = np.matmul(base_R_elb.as_matrix(), dist_gh_elbow) + self.position_gh_in_base
        else:
            position_gh_in_base_fixed = np.array([rospy.get_param('/pu/p_gh_in_base_x'), 
                                                  rospy.get_param('/pu/p_gh_in_base_y'), 
                                                  rospy.get_param('/pu/p_gh_in_base_z')])
            ref_cart_point = np.matmul(base_R_elb.as_matrix(), dist_gh_elbow) + position_gh_in_base_fixed

        # modify the reference along the Z direction, to account for the increased interaction force
        # due to the human arm resting on the robot. We do this only if we are not in simulation.
        if torque_ref is not None:
            k_z = np.array(rospy.get_param('/pu/ee_stiffness'))[2]
            se_estimated = self.state_values_current[2]
            torque_se = torque_ref[1]
            z_current = self.current_ee_pose[2]

            L_tot = rospy.get_param('/pu/l_arm') + rospy.get_param('/pu/l_brace')   # extract total length between GH joint center and
                                                                                    # elbow tip
            new_z_ref = z_current + torque_se/(k_z * L_tot * np.sin(se_estimated))
            
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
        ee_pose_d = np.reshape(homogeneous_matrix, (16,1))              # desired end effector pose

        # build the message
        # if desired Cartesian stiffness and/or damping has been specified, we account for that too
        message_ref = Float64MultiArray()
        message_ref.layout.data_offset = 0
        if self.ee_cart_stiffness_cmd is None and self.ee_cart_damping_cmd is None:
            # if no stiffness/damping have been specified, we just send the pose
            # (controller on the robot's side will use default values)
            message_ref.data = ee_pose_d
        elif self.ee_cart_damping_cmd is None:
            # if no damping has been specified, we send only what we have
            message_ref.data = np.concatenate((ee_pose_d, self.ee_cart_stiffness_cmd))

        else:
            # if we have all of the parameters, then we just send everything
            message_ref.data = np.concatenate((ee_pose_d, self.ee_cart_stiffness_cmd, self.ee_cart_damping_cmd))


        # publish the message
        self.pub_trajectory.publish(message_ref)

        # publish also the alternative_z_ref (for logging it)
        message_z = Float64MultiArray()
        message_z.layout.data_offset = 0
        message_z.data = alternative_z_ref
        self.pub_z_level.publish(message_z)
        

    def publishInitialPoseAsCartRef(self, shoulder_pose_ref, base_R_sh):
        """"
        This function publishes a given reference shoulder state as the equivalent 6D cartesian pose corresponding
        to the position of the elbow tip, expressed the world frame. The center of the shoulder in this frame needs 
        to be given by the user, so that they need to specify the position of the GH joint center in the world frame 
        (px, py, pz), and the orientation of the shoulder reference frame (i.e., the scapula reference frame) as well.
        The underlying assumption is that the orientation of the scapula/shoulder frame remains fixed over time 
        wrt the world frame.

        The inputs are:
            - shoulder_pose_ref: 3x1 numpy array, storing the values of plane of elevation, shoulder 
              elevation and axial rotation at a given time instant
            - base_R_sh: rotation matrix defining the orientation of the shoulder frame wrt the world frame
                         (as a scipy.spatial.transform.Rotation object)

        The positions and distances must be expressed in meters, and the rotations in radians.

        This function is conceptually equivalent to publishCartRef, but it uses another ROS publisher to publish the
        required reference to the topic associated to the initial pose for the robot. It will keep publishing the initial pose
        until the robot controller receives it, and then it stops.

        Once called, it will also return the position that the robot has been commanded to reach.
        """
        shoulder_state = np.zeros((6, 1))
        shoulder_state[0::2] = shoulder_pose_ref.reshape(-1, 1)
        self.x_opt = shoulder_state

        # fix the position of the center of the shoulder/glenohumeral joint
        self.position_gh_in_base = np.array([rospy.get_param('/pu/p_gh_in_base_x'), 
                                             rospy.get_param('/pu/p_gh_in_base_y'), 
                                             rospy.get_param('/pu/p_gh_in_base_z')])

        # perform extra things if this is the first time we execute this
        if not self.flag_pub_trajectory:
            # We need to set up the structure to deal with the new thread, to allow 
            # continuous publication of the optimal trajectory
            self.publish_thread = threading.Thread(target=self.publish_continuous_trajectory, 
                                                    args = (base_R_sh,))   # creating the thread
            
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
        self.state_values_current = np.array(data.data[0:9])        # update current pose
        # this is pe, pe_dot, se, se_dot, ar, ar_dot, pe_ddot, se_ddot, ar_ddot
        
        if rospy.get_param('/pu/estimate_gh_position'):
            # retrieve the current pose of the shoulder/glenohumeral center in the base frame
            self.position_gh_in_base = np.array(data.data[9:12])

        if not self.speed_estimate:                                 # choose whether we use the velocity estimate or not
            self.state_values_current[1::2] = 0

        time_now = time.time()

        self.current_ee_pose = np.array(data.data[-3:])   # update estimate of where the robot is now (last 3 elements)

        # set up a logic that allows to start and update the visualization of the current strainmap at a given frequency
        # We set this fixed frequency to be 10 Hz
        if np.round(time_now-np.fix(time_now), 2) == np.round(time_now-np.fix(time_now),1):
            if self.future_trajectory is None:
                self.strain_visualizer.updateStrainMap(list_params = self.all_params_gaussians, 
                                                    pose_current = self.state_values_current[[0,2]], 
                                                    reference_current = self.x_opt[[0,2],:],
                                                    future_trajectory = None,
                                                    goal_current = None, 
                                                    vel_current = self.state_values_current[[1,3]],
                                                    ar_current = None)
            else:
                    self.strain_visualizer.updateStrainMap(list_params = self.all_params_gaussians, 
                                                    pose_current = self.state_values_current[[0,2]], 
                                                    reference_current = self.x_opt[[0,2],:],
                                                    future_trajectory = self.future_trajectory[[0,2],:],
                                                    goal_current = None, 
                                                    vel_current = self.state_values_current[[1,3]],
                                                    ar_current = None)


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


    def publish_continuous_trajectory(self, base_R_sh):
        """
        This function picks the most recent information regarding the optimal shoulder trajectory,
        converts it to end effector space and publishes the robot reference continuously. A flag enables/disables
        the computations/publishing to be performed, such that this happens only if the robot controller needs it.
        """
        # frequency of discretization of the future human movement is used
        if self.interaction_mode == 3:
            # if we are using the solution from the NLPS, then be coherent with the discretization
            rate = rospy.Rate(1/self.nlps.h)
        else:
            # otherwise, run at the expected frequency
            rate = self.ros_rate

        while not rospy.is_shutdown():
            if self.flag_pub_trajectory:    # perform the computations only if needed
                with self.x_opt_lock:       # get the lock to read and modify references
                    if self.x_opt is not None:
                        # We will pick always the first element of the trajectory, and then delete it (unless it is the last element)
                        # This is done in blocking mode to avoid conflicts, then we move on to processing the shoulder pose and convert
                        # it to cartesian trajectory
                        if np.shape(self.x_opt)[1]>1:
                            # If there are many elements left, get the first and then delete it
                            cmd_shoulder_pose = self.x_opt[0::2, 0]
                            self.x_opt = np.delete(self.x_opt, obj=0, axis=1)   # delete the first column

                            if self.u_opt is not None:
                                cmd_torques = self.u_opt[:,0]
                                self.u_opt = np.delete(self.u_opt, obj=0, axis=1)
                            else:
                                cmd_torques = None

                        else:
                            # If there is only one element left, keep picking it
                            cmd_shoulder_pose = self.x_opt[0::2, 0]
                            if self.u_opt is not None:
                                cmd_torques = self.u_opt[:,0]
                            else:
                                cmd_torques = None

                        self.publishCartRef(cmd_shoulder_pose, cmd_torques, base_R_sh)

            rate.sleep()


    def input_thread_fnc(self):
        while True:
            try:
                # Read user input (this happens asynchronously as executed in a thread)
                interaction_mode = input("Select interaction mode (0, 1, 2 or 3):\n")
                print(": interaction mode selected")
                self.interaction_mode = int(interaction_mode)
            except ValueError:
                print("Invalid input, please enter a number between 0 and 3")


    def setReferenceToCurrentPose(self):
        """
        This function allows to overwrite the reference state/control values. They are
        substituted by the current state of the subject/patient sensed by the robot.
        """
        with self.x_opt_lock:
            self.x_opt = self.state_values_current.reshape((9,1))[0:6]
            self.u_opt = np.zeros((self.nlps.dim_u, 1))

        # set the default stiffness and damping for the controller
        self.ee_cart_stiffness_cmd = self.ee_cart_stiffness_default
        self.ee_cart_damping_cmd = self.ee_cart_damping_default


    def setReferenceToCurrentRefPose(self):
        """
        This function allows to overwrite the reference state/control values, by keeping
        only the one that is currently being tracked.
        """
        with self.x_opt_lock:
            self.x_opt = self.x_opt[:,0].reshape((6,1))
            self.u_opt = np.zeros((self.nlps.dim_u,1))

        # set the default stiffness and damping for the controller
        self.ee_cart_stiffness_cmd = self.ee_cart_stiffness_default
        self.ee_cart_damping_cmd = self.ee_cart_damping_default


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
        

    def visualizeCurrentStrainMap(self, threeD = False, block = False):
        """
        Call this function to display the strainmap currently considered by safety net.
        Select whether you want to visualize it in 3D or not.
        """
        # inizialize empty strainmap
        current_strainmap = np.zeros(self.X_norm.shape)

        # loop through all of the Gaussians, and obtain the strainmap values
        for function in range(len(self.all_params_gaussians)//self.num_params_gaussian):

            #first, retrieve explicitly the parameters of the function considered in this iteration
            amplitude = self.all_params_gaussians[function*self.num_params_gaussian]
            x0 = self.all_params_gaussians[function*self.num_params_gaussian+1]
            y0 = self.all_params_gaussians[function*self.num_params_gaussian+2]
            sigma_x = self.all_params_gaussians[function*self.num_params_gaussian+3]
            sigma_y = self.all_params_gaussians[function*self.num_params_gaussian+4]
            offset = self.all_params_gaussians[function*self.num_params_gaussian+5]
            
            # then, compute the contribution of this particular Gaussian to the final strainmap
            current_strainmap += amplitude * np.exp(-((self.X_norm-x0)**2/(2*sigma_x**2)+(self.Y_norm-y0)**2/(2*sigma_y**2)))+offset
            
        # finally, plot the resulting current strainmap
        if threeD:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.plot_surface(self.X, self.Y, current_strainmap, cmap='plasma')
            ax.set_xlabel('Plane Elev [deg]')
            ax.set_ylabel('Shoulder Elev [deg]')
            ax.set_zlabel('Strain level [%]')
            ax.set_zlim([0, current_strainmap.max()])
        else:
            fig = plt.figure()
            ax = fig.add_subplot()
            heatmap = ax.imshow(np.flip(current_strainmap.T, axis = 0), cmap='plasma', extent=[self.X.min(), self.X.max(), self.Y.min(), self.Y.max()])
            fig.colorbar(heatmap, ax = ax, ticks=np.arange(0, current_strainmap.max() + 1), label='Strain level [%]')
            ax.set_xlabel('Plane Elev [deg]')
            ax.set_ylabel('Shoulder Elev [deg]')

        plt.show(block=block)
        

    def ellipse_y_1quad(self, x, a_squared, b_squared):
        """
        Function returning the y value of the ellipse given the x, in the first quadrant:
        y = b * sqrt(1-x^2/a^2)
        
        The ellipse is assumed to be centered in 0, and its full expression is 
        x^2/a^2+ y^2/b^2 = 1
        """
        return np.sqrt(b_squared * (1-x**2/a_squared))


    def distance_function_x_1quad(self, x, px, py, a_squared, b_squared):
        """
        Function expressing the distance between a point (px, py) in the first quadrant (px, py>0)
        and the ellipse defined by x^2/a^2+ y^2/b^2 = 1

        The distance is a function of the variable x, obtained by the following two equations:
        1. distance in the plane: d(x,y) = (x-px)^2 + (y-py)^2
        2. ellipse in the first quadrant: y = b * sqrt(1-x^2/a^2)
        """

        return (x-px)**2 + (np.sqrt(b_squared* (1 - x**2/a_squared)) - py)**2
        

    def monitor_unsafe_zones(self):
        """
        This function monitors whether we are in any unsafe zones, based on the ellipse parameters
        obtained by the analytical expression of the strainmaps. In particular:
            - if we are outside of all the ellipses, the reference point (in the human frame)
              is set to be the current pose
            - if we are inside one (or more) ellipses, the reference point is the closest point
              on the ellipses borders.
        The corresponding human pose is commanded to the robot, transforming it into EE pose.


        The ellipses are defined with model's angles in degrees!
        """

        # enable robot's reference trajectory publishing
        self.flag_pub_trajectory = True

        num_ellipses = int(len(self.all_params_ellipses)/self.num_params_ellipses)   # retrieve number of ellipses currently present in the map

        # retrieve the current point on the strain map
        # remember that the ellipses are defined in degrees!
        pe = np.rad2deg(self.state_values_current[0])
        se = np.rad2deg(self.state_values_current[2])
        ar = np.rad2deg(self.state_values_current[4])

        # check if (pe, se) is inside any unsafe zone
        self.in_zone_i = np.zeros(num_ellipses)
        for i in range(num_ellipses):
            a_squared = self.all_params_ellipses[self.num_params_ellipses*i+2]
            b_squared = self.all_params_ellipses[self.num_params_ellipses*i+3]
            if a_squared is not None and b_squared is not None:     # if the ellipse really exists
                x0 = self.all_params_ellipses[self.num_params_ellipses*i]
                y0 = self.all_params_ellipses[self.num_params_ellipses*i+1]

                # note that we use normalized variables here
                self.in_zone_i[i] = int(((pe-x0)**2/a_squared + (se-y0)**2/b_squared) < 1)

                # update strain visualization information for plotting the ellipse
                # this needs to be x0, y0, 2a, 2b
                params_ellipse_viz = np.array([x0, y0, 2*np.sqrt(a_squared), 2*np.sqrt(b_squared)]).reshape((1, self.num_params_ellipses))

                # update the ellipses for plotting
                self.strain_visualizer.update_ellipse_params(params_ellipse_viz, force = True)

        # now we know if we are inside one (or more ellipses). Given the current shoulder pose (pe, se), we find 
        # the closest point on the surface of the ellipses containing (pe, se).
        num_in_zone = int(self.in_zone_i.sum())
        if num_in_zone == 0:
            # if we are not inside any ellipse, then we are safe and the current position is tracked
            # (we convert back to radians for compatibility with the rest of the code)
            self.x_opt = np.deg2rad(np.array([pe, 0, se, 0, ar, 0]).reshape((6,1)))

            # choose Cartesian stiffness and damping for the robot's impedance controller
            # we set low values so that subject can move (almost) freely
            self.ee_cart_stiffness_cmd = self.ee_cart_stiffness_low
            self.ee_cart_damping_cmd = self.ee_cart_damping_low

        else:
            # otherwise check all of the zones in which we are in, and find the closest points
            closest_point = np.nan * np.ones((num_ellipses, 2))

            # loop through the ellipses
            for i in range(num_ellipses):
                if self.in_zone_i[i] == 1:           # are we in this ellipse?

                    # retrieve ellipse parameters
                    x0 = self.all_params_ellipses[self.num_params_ellipses*i]
                    y0 = self.all_params_ellipses[self.num_params_ellipses*i+1]
                    a_squared = self.all_params_ellipses[self.num_params_ellipses*i+2]
                    b_squared = self.all_params_ellipses[self.num_params_ellipses*i+3]

                    # express (se, pe) in the reference frame of the center of the ellipse
                    # we use normalized variables since the ellipse is calculated in that space
                    pe_ellipse = pe - x0
                    se_ellipse = se - y0

                    # record in which quadrant we are, and transform (pe_ellipse, se_ellipse) to
                    # the first quadrant of the ellipse
                    pe_ellipse_sgn = np.sign(pe_ellipse)
                    se_ellipse_sgn = np.sign(se_ellipse)

                    pe_1quad = np.abs(pe_ellipse)
                    se_1quad = np.abs(se_ellipse)

                    # find the closest point in the first quadrant
                    opt = minimize_scalar(self.distance_function_x_1quad, args=(pe_1quad, se_1quad, a_squared, b_squared), bounds = (0, np.sqrt(a_squared)))
                    if opt.success == False:
                        # if optimization does not converge, throw an error and freeze to current pose
                        self.setReferenceToCurrentPose
                        RuntimeError('Optimization did not converge: closest point on the ellipse could not be found!')

                    x_cp_1quad = opt.x
                    y_cp_1quad = self.ellipse_y_1quad(x_cp_1quad, a_squared, b_squared)

                    # re-map point in the correct quadrant 
                    pe_cp_ellipse = pe_ellipse_sgn * x_cp_1quad
                    se_cp_ellipse = se_ellipse_sgn * y_cp_1quad

                    # move point back to original coordinate and store them
                    # here, we take care of the de-normalization too
                    pe_cp = pe_cp_ellipse + x0
                    se_cp = se_cp_ellipse + y0

                    closest_point[i,:] = np.array([pe_cp, se_cp])

            # extract only the relevant rows from closest_point, retaining the point(s)
            # on each ellipse that we can consider as a reference
            closest_point = closest_point[~np.isnan(closest_point).all(axis=1)]

            # check if we are inside multiple ellipses. If so, check which point is really the closest
            num_points = closest_point.shape[1]>1
            if num_points:
                curr_closest_point = closest_point[0, :]
                curr_d = (curr_closest_point[0]-pe)**2 + (curr_closest_point[1]-se)**2

                for i in range(1, num_points):
                    new_d = (closest_point[i,0]-pe)**2 + (closest_point[i,1]-se)**2
                    if new_d < curr_d:
                        curr_d = new_d
                        curr_closest_point = closest_point[i, :]

            # send the closest point on the ellipse border as a reference for the robot
            # we need to reconvert back to radians for compatibility with the rest of the code
            self.x_opt = np.deg2rad(np.array([curr_closest_point[0], 0, curr_closest_point[1], 0, ar, 0]).reshape((6,1)))

            # choose Cartesian stiffness and damping for the robot's impedance controller
            # we set high values so that alternative reference is tracked
            # (the subject will be pulled out of the unsafe zone)
            self.ee_cart_stiffness_cmd = self.ee_cart_stiffness_default
            self.ee_cart_damping_cmd = self.ee_cart_damping_default


    def damp_unsafe_velocities(self):
        """
        This function takes care of increasing the damping of the cartesian impedance controller
        when the subject is traversing high-strain areas.
        Ideally, damping should be increased if the movement will result in increased strain, and
        kept low when the movement is "escaping" unsafe areas.
        """
        # enable robot's reference trajectory publishing
        self.flag_pub_trajectory = True

        # retrieve model's current pose and velocity
        # remember that the strain map is defined in degrees
        pe = np.rad2deg(self.state_values_current[0])
        pe_dot = np.rad2deg(self.state_values_current[1])
        se = np.rad2deg(self.state_values_current[2])
        se_dot = np.rad2deg(self.state_values_current[3])
        ar = np.rad2deg(self.state_values_current[4])
        ar_dot = np.rad2deg(self.state_values_current[5])

        current_strain = 0

        # loop through all of the Gaussians, to obtain the contribution of each of them
        # to the overall strain corresponding to the current position 
        for function in range(len(self.all_params_gaussians)//self.num_params_gaussian):

            #first, retrieve explicitly the parameters of the function considered in this iteration
            amplitude = self.all_params_gaussians[function*self.num_params_gaussian]
            x0 = self.all_params_gaussians[function*self.num_params_gaussian+1]
            y0 = self.all_params_gaussians[function*self.num_params_gaussian+2]
            sigma_x = self.all_params_gaussians[function*self.num_params_gaussian+3]
            sigma_y = self.all_params_gaussians[function*self.num_params_gaussian+4]
            offset = self.all_params_gaussians[function*self.num_params_gaussian+5]
            
            # then, compute the contribution of this particular Gaussian to the final strainmap
            # (remember that the strain maps are computed with normalized values!)
            current_strain += amplitude * np.exp(-((pe/self.pe_normalizer-x0)**2/(2*sigma_x**2)+(se/self.se_normalizer-y0)**2/(2*sigma_y**2)))+offset
        
        # check if the current strain is safe or risky
        if current_strain <= self.risky_strain:
            # if the strain is sufficiently low, then we are safe: we set the parameters
            # for the cartesian impedance controller to produce minimal interaction force with the subject
            self.ee_cart_stiffness_cmd = self.ee_cart_stiffness_low
            self.ee_cart_damping_cmd = self.ee_cart_damping_low

        else:
            # if we are in a risky area, then rapid movement should be limited if it would lead to
            # increased strain. For this, we calculate the directional derivative of the strain map
            # at the current location, along the direction given by the current (estimated) velocity.
            # If the directional derivative is negative, then the movement is still safe. Otherwise,
            # damping from the robot should be added.

            # calculate the directional vector from the estimated velocities
            vel_on_map = np.array([pe_dot, se_dot]) 
            norm_vel = vel_on_map / np.linalg.norm(vel_on_map + 1e-6)

            # calculate the directional derivative on the strain map
            dStrain_dx = 0
            dStrain_dy = 0
            for function in range(len(self.all_params_gaussians)//self.num_params_gaussian):
                #first, retrieve explicitly the parameters of the function considered in this iteration
                amplitude = self.all_params_gaussians[function*self.num_params_gaussian]
                x0 = self.all_params_gaussians[function*self.num_params_gaussian+1]
                y0 = self.all_params_gaussians[function*self.num_params_gaussian+2]
                sigma_x = self.all_params_gaussians[function*self.num_params_gaussian+3]
                sigma_y = self.all_params_gaussians[function*self.num_params_gaussian+4]
                offset = self.all_params_gaussians[function*self.num_params_gaussian+5]

                # then, compute the analytical (partial) derivatives of the strain map and add them
                dStrain_dx += - amplitude * (pe/self.pe_normalizer - x0) / (sigma_x**2) * \
                              np.exp(-((pe/self.pe_normalizer-x0)**2/(2*sigma_x**2)+(se/self.se_normalizer-y0)**2/(2*sigma_y**2)))
                
                dStrain_dy += - amplitude * (se/self.se_normalizer - y0) / (sigma_y**2) * \
                              np.exp(-((pe/self.pe_normalizer-x0)**2/(2*sigma_x**2)+(se/self.se_normalizer-y0)**2/(2*sigma_y**2)))
                
            
            dStrain_along_direction = np.dot(np.array([dStrain_dx, dStrain_dy]), norm_vel)

            # if the directional derivative is negative, then movement is safe (we set low damping)
            if dStrain_along_direction < 0:
                self.ee_cart_stiffness_cmd = self.ee_cart_stiffness_low
                self.ee_cart_damping_cmd = self.ee_cart_damping_low

            else:
                # # here we will change the damping of the interaction. However, we cannot do this without 
                # # stability problems if we do not change the stiffness too.

                # # I came up with a simple heuristic, in which stiff - stiff_low =  3 x (damp - damp_low)

                # ratio = (current_strain - self.risky_strain)**2 + 1         # we add + 1 to avoid having damping close to 0
                # damping  = ratio * self.ee_cart_damping_dampVel_baseline

                # if damping[0] > self.max_cart_damp:
                #     self.ee_cart_damping_cmd = self.max_cart_damp / damping[0] * damping
                # else:
                #     self.ee_cart_damping_cmd  = ratio * self.ee_cart_damping_dampVel_baseline

                # self.ee_cart_stiffness_cmd = 3 * (self.ee_cart_damping_cmd[0] - self.ee_cart_damping_low[0]) * self.ee_cart_stiffness_low
                # print(self.ee_cart_damping_cmd[0])
                
                self.ee_cart_stiffness_cmd = self.ee_cart_stiffness_dampVel_fixed
                self.ee_cart_damping_cmd = self.ee_cart_damping_dampVel_fixed
            

        # the current position is set as a reference
        # (we convert back to radians for compatibility with the rest of the code)
        self.x_opt = np.deg2rad(np.array([pe, 0, se, 0, ar, 0]).reshape((6,1)))


    def assign_nlps(self, nlps_object):
        """
        This function takes care of embedding the NLP problem for planning on strain maps into the BSN.
        """
        self.nlps = nlps_object

        # the overall NLP problem is formulated, meaning that its structure is determined based on the
        # previous inputs
        self.nlps.formulateNLP_simpleMass()

        # embed the whole NLP (solver included) into a CasADi function that can be called
        # both the function and the inputs it needs are initialized here
        self.mpc_iter, self.input_mpc_call = self.nlps.createOptimalMapWithoutInitialGuesses()


    def predict_future_state_kinematic(self, N, T):
        """
        This function is used to predict the future state of the human body, under the assumption that the current
        velocities will be maintained in the future along all the DoFs. The current velocities are used to propagate
        forward the state, by checking that it remains always safe with respect to (elliptical) unsafe zones.

        Inputs are the number N of time-steps to be considered in the prediction, and the total duration T of the
        prediction horizon.
        """
        assert self.state_values_current is not None, "The current state of the human model is unknown"

        # disable robot's reference trajectory publishing
        self.flag_pub_trajectory = True

        # retrieve number of (elliptical) unsafe zones present on current strain map
        num_ellipses = int(len(self.all_params_ellipses)/self.num_params_ellipses)

        self.in_zone_i = np.zeros((N+1, num_ellipses))   # initialize counter for unsafe states

        # initial state for the human model
        initial_state = self.state_values_current[0:6]        # TODO this is the correct thing

        # uncomment below for debugging
        # rng = np.random.default_rng()

        # pe_init = np.deg2rad(rng.uniform(low = 55, high = 67))
        # pe_dot_init = np.deg2rad(rng.uniform(low = -20, high = -5))

        # se_init = np.deg2rad(rng.uniform(low = 90, high = 110))
        # se_dot_init = np.deg2rad(rng.uniform(low = -20, high = 20))

        # ar_init = np.deg2rad(rng.uniform(low = -60, high = 60))
        # ar_dot_init = np.deg2rad(0)
        
        # initial_state = np.array([pe_init, pe_dot_init, se_init, se_dot_init, ar_init, ar_dot_init])

        # for the next N time-steps, predict the future states of the human model
        future_states = np.zeros((6, N+1))
        future_states[::2, 0] = initial_state[::2]      # initialize the first point of the state trajectory
        future_states[1::2, :] = initial_state[1::2][:, np.newaxis]    # velocities are assumed to be constant
        
        for timestep in range(1, N+1):
            # retrieve estimation for future human state at current time step (assuming constant velocity)
            future_states[::2, timestep] = future_states[::2, timestep-1] + T/N * future_states[1::2, timestep-1]

            # check that the estimated point of the trajectory will be safe
            # (for now, this is done in 2D for PE and SE only)
            for i in range(num_ellipses):
                a_squared = self.all_params_ellipses[self.num_params_ellipses*i+2]
                b_squared = self.all_params_ellipses[self.num_params_ellipses*i+3]
                if a_squared is not None and b_squared is not None:     # if the ellipse really exists
                    x0 = self.all_params_ellipses[self.num_params_ellipses*i]
                    y0 = self.all_params_ellipses[self.num_params_ellipses*i+1]

                    # note that we use normalized variables here (the ellipses are defined in degrees)
                    self.in_zone_i[timestep, i] = int(((np.rad2deg(future_states[0, timestep])-x0)**2/a_squared + (np.rad2deg(future_states[2, timestep])-y0)**2/b_squared) < 1)

                    # update strain visualization information for plotting the ellipse
                    # this needs to be x0, y0, 2a, 2b
                    params_ellipse_viz = np.array([x0, y0, 2*np.sqrt(a_squared), 2*np.sqrt(b_squared)]).reshape((1, self.num_params_ellipses))

                    # update the ellipses for plotting
                    self.strain_visualizer.update_ellipse_params(params_ellipse_viz, force = True)

        # the trajectory is saved as a parameter in the BSN module.
        # In this way, we can also display it in real time on the strain map
        self.future_trajectory = future_states[:, 1:]   # the first column is discarded, since it corresponds to x_0

        # let's sum up the flags, and see if any of the future states will be unsafe
        num_unsafe = int(self.in_zone_i.sum())

        if num_unsafe == 0:
            # if there is no future state which is unsafe, the current position is tracked
            # choose Cartesian stiffness and damping for the robot's impedance controller
            # we set low values so that subject can move (almost) freely
            self.ee_cart_stiffness_cmd = self.ee_cart_stiffness_low
            self.ee_cart_damping_cmd = self.ee_cart_damping_low

            self.x_opt = initial_state.reshape((6,1))
        else:
            # if not, then we have the NLP running to find an alternative path
            x_opt, u_opt, _,  j_opt, xddot_opt = self.mpc_iter(initial_state, self.future_trajectory, self.all_params_ellipses)

            # note the order for reshape!
            traj_opt = x_opt.full().reshape((6, N+1), order='F')[:, 1::]        # we discard the first state as it is the current one
            self.x_opt = traj_opt
            self.u_opt = None       # TODO: we are ignoring u_opt for now


            # fig = plt.figure()
            # ax = fig.add_subplot()
            # ax.scatter(np.rad2deg(traj_opt[0,:]), np.rad2deg(traj_opt[2,:]), c = 'blue', label = 'reference traj')
            # ax.scatter(np.rad2deg(traj_opt[0,0]), np.rad2deg(traj_opt[2,0]), c = 'cyan')
            # ax.scatter(np.rad2deg(self.future_trajectory[0,:]), np.rad2deg(self.future_trajectory[2,:]), c = 'red', label = 'future traj')
            # ax.add_patch(Ellipse((self.all_params_ellipses[0], self.all_params_ellipses[1]), width = 2*np.sqrt(self.all_params_ellipses[2]), height = 2*np.sqrt(self.all_params_ellipses[3]), alpha=0.2))
            # ax.legend()
            # plt.show()

            # increase stiffness to actually track the optimal deflected trajectory
            self.ee_cart_stiffness_cmd = self.ee_cart_stiffness_default
            self.ee_cart_damping_cmd = self.ee_cart_damping_default

            # sleep for the duration of the optimized trajectory
            rospy.sleep(self.nlps.T)

            # decrease stiffness again so that subject can continue their movement
            self.ee_cart_stiffness_cmd = self.ee_cart_stiffness_low
            self.ee_cart_damping_cmd = self.ee_cart_damping_low


    def predict_future_state_simple(self, N, T):
        """
        A variation with respect to predict_future_state_kinematic, where no optimization is run but only
        geometric rules are applied to find some sort of deflection.
        """
        assert self.state_values_current is not None, "The current state of the human model is unknown"

        # disable robot's reference trajectory publishing
        self.flag_pub_trajectory = True

        # retrieve number of (elliptical) unsafe zones present on current strain map
        num_ellipses = int(len(self.all_params_ellipses)/self.num_params_ellipses)

        self.in_zone_i = np.zeros((N+1, num_ellipses))   # initialize counter for unsafe states

        # initial state for the human model
        initial_state = self.state_values_current[0:6]

        # for the next N time-steps, predict the future states of the human model
        future_states = np.zeros((6, N+1))
        future_states[::2, 0] = initial_state[::2]      # initialize the first point of the state trajectory
        future_states[1::2, :] = initial_state[1::2][:, np.newaxis]    # velocities are assumed to be constant
        
        for timestep in range(1, N+1):
            # retrieve estimation for future human state at current time step (assuming constant velocity)
            future_states[::2, timestep] = future_states[::2, timestep-1] + T/N * future_states[1::2, timestep-1]

            # check that the estimated point of the trajectory will be safe
            # (for now, this is done in 2D for PE and SE only)
            for i in range(num_ellipses):
                a_squared = self.all_params_ellipses[self.num_params_ellipses*i+2]
                b_squared = self.all_params_ellipses[self.num_params_ellipses*i+3]
                if a_squared is not None and b_squared is not None:     # if the ellipse really exists
                    x0 = self.all_params_ellipses[self.num_params_ellipses*i]
                    y0 = self.all_params_ellipses[self.num_params_ellipses*i+1]

                    # note that we use normalized variables here (the ellipses are defined in degrees)
                    self.in_zone_i[timestep, i] = int(((np.rad2deg(future_states[0, timestep])-x0)**2/a_squared + (np.rad2deg(future_states[2, timestep])-y0)**2/b_squared) < 1)

                    # update strain visualization information for plotting the ellipse
                    # this needs to be x0, y0, 2a, 2b
                    params_ellipse_viz = np.array([x0, y0, 2*np.sqrt(a_squared), 2*np.sqrt(b_squared)]).reshape((1, self.num_params_ellipses))

                    # update the ellipses for plotting
                    self.strain_visualizer.update_ellipse_params(params_ellipse_viz, force = True)

        # the trajectory is saved as a parameter in the BSN module.
        # In this way, we can also display it in real time on the strain map
        self.future_trajectory = future_states[:, 1:]   # the first column is discarded, since it corresponds to x_0

        # let's sum up the flags, and see if any of the future states will be unsafe
        num_unsafe = int(self.in_zone_i.sum())

        if num_unsafe == 0:
            # if there is no future state which is unsafe, the current position is tracked
            # choose Cartesian stiffness and damping for the robot's impedance controller
            # we set low values so that subject can move (almost) freely
            self.ee_cart_stiffness_cmd = self.ee_cart_stiffness_low
            self.ee_cart_damping_cmd = self.ee_cart_damping_low

            self.x_opt = initial_state.reshape((6,1))
        else:
            # if not, then we run a simple optimization problem to find the optimal trajectory to avoid the unsafe areas.
            x_opt, u_opt, _,  j_opt, xddot_opt = self.mpc_iter(initial_state, self.future_trajectory, self.all_params_ellipses)

            # note the order for reshape!
            traj_opt = x_opt.full().reshape((6, N+1), order='F')[:, 1::]        # we discard the first state as it is the current one
            self.x_opt = traj_opt
            self.u_opt = None       # TODO: we are ignoring u_opt for now

            # increase stiffness to actually track the optimal deflected trajectory
            self.ee_cart_stiffness_cmd = self.ee_cart_stiffness_default
            self.ee_cart_damping_cmd = self.ee_cart_damping_default

            # sleep for the duration of the optimized trajectory
            rospy.sleep(self.nlps.T)

            # decrease stiffness again so that subject can continue their movement
            self.ee_cart_stiffness_cmd = self.ee_cart_stiffness_low
            self.ee_cart_damping_cmd = self.ee_cart_damping_low


    def predict_future_state_old(self):
        """
        This function is used to predict the future state of the human body, which is found assuming that the human
        will continue exerting a constant torque along their DoFs. The future trajectory for the human model is saved
        in the variable x_opt, that can be then visualized for debugging.
        """
        assert self.nlps is not None, "The NLP has not been defined yet, cannot continue"

        # disable robot's reference trajectory publishing
        self.flag_pub_trajectory = False

        # initialize flag to monitor if the solver found an optimal solution
        failed = 0

        # first, we estimate the current human torques given the current position, velocity and acceleration of the model
        u_hat_hum = self.nlps.opensimAD_ID(self.state_values_current)[0:2]

        # initial state for the human model
        initial_state = self.state_values_current[0:6]

        # if we are considering strain in our formulation, add the parameters of the strainmap to the numerical input
        if self.nlps.num_gaussians>0:
            params_g1 = self.nlps.all_params_gaussians[0:6]
            params_g2 = self.nlps.all_params_gaussians[6:12]
            params_g3 = self.nlps.all_params_gaussians[12:18]

            # solve the NLP problem given the current state of the system (with strain information)
            # we solve the NLP, and catch if there was an error. If so, notify the user and retry
            try:
                time_start = time.time()
                x_opt, u_opt, j_opt, _, strain_opt, xddot_opt = self.mpc_iter(initial_state, u_hat_hum, self.state_values_current[4], params_g1, params_g2, params_g3)
                time_execution = time.time()-time_start
                strain_opt = strain_opt.full().reshape(1, self.nlps.N+1, order='F')

            except Exception as e:
                print('Solver failed:', e)
                print('retrying ...')
                failed = 1

        else:
            # solve the NLP problem given the current state of the system (without strain information)
            # we solve the NLP, and catch if there was an error. If so, notify the user and retry
            try:
                time_start = time.time()
                x_opt, u_opt, _,  j_opt, xddot_opt = self.mpc_iter(initial_state, u_hat_hum, self.state_values_current[4])
                time_execution = time.time() - time_start

                strain_opt = np.nan * np.ones((1, self.nlps.N+1))  # still fill the optimal strain values with NaNs

            except Exception as e:
                print('Solver failed:', e)
                print('retrying ...')
                failed = 1

        self.nlp_count += 1         # update the number of iterations until now
        self.failed_count += failed # update number of failed iterations

        # only do the remaining steps if we have a new solution
        if not failed:
            # convert the solution to numpy arrays, and store them to be processed
            x_opt = x_opt.full().reshape(self.nlps.dim_x, self.nlps.N+1, order='F')
            u_opt = u_opt.full().reshape(self.nlps.dim_u, self.nlps.N, order='F')

            # save the strain value
            self.strain_opt = strain_opt
            
            # update the optimal values that are stored in the BSN module. 
            # They can be accessed only if there is no other process that is modifying them
            with self.x_opt_lock:
                # self.x_opt = x_opt[:, 1::]      # the first point is discarded, as it is the current one
                self.x_opt = x_opt[:, 2::]      # the first points are discarded, to compensate for relatively low stiffness of the controller
                self.u_opt = u_opt[:,1::]       # same as above, to guarantee consistency

            # update average running time
            self.avg_nlp_time = ((self.avg_nlp_time * self.nlp_count) + time_execution) / (self.nlp_count+1)

            # update the optimal values that are stored in the NLPS module as well
            self.nlps.x_opt = x_opt      
            self.nlps.u_opt = u_opt

            # publish the optimal values to a ROS topic, so that they can be recorded during experiments
            message = Float64MultiArray()
            u_opt = np.concatenate((u_opt, np.atleast_2d(np.nan*np.ones((self.nlps.dim_u,1)))), axis = 1)  # adding one NaN to match dimensions of other arrays
            activation = self.activation_level*self.delta_activation + self.min_activation
            message.data = np.hstack((np.vstack((x_opt, u_opt, strain_opt)).flatten(), activation))    # stack the three outputs in a single message (plus activation), and flatten it for publishing
            self.pub_optimization_output.publish(message)

            return u_opt, x_opt, j_opt, strain_opt

            # TODO is now to test that the current acceleration is received and properly used by f_ID,AD


    def debug_sysDynamics(self):
        # simulated initial state for the human model
        # the state is pe, pe_dot, se, se_dot, ar, ar_dot, pe_ddot, se_ddot, ar_ddot
        num_reps = 100
        sim_initial_state_perturbed = np.zeros((num_reps, 9))
        sim_initial_state_perturbed[0, :] = np.concatenate((self.nlps.x_0, np.zeros((3,))))

        ## SENSITIVITY ANALYSIS FOR THE ID
        # get the corresponding human torques using OpenSimAD
        # we do this num_reps times, with small perturbations of the initial state (the magnitude of the perturbation can be adjusted)
        delta_perturbation_torque = 1e-3
        delta_vec = np.zeros((num_reps, 9))

        hum_torques_hat = np.zeros((num_reps, 3))
        for count in range(0, num_reps):
            if count > 0:
                # the torque outputs will tell us how much a change in the input modifies the output
                delta_vec[count, :] = delta_perturbation_torque * np.random.random(np.shape(sim_initial_state_perturbed[0,:])).reshape((9,))

            sim_initial_state_perturbed [count, :] = sim_initial_state_perturbed[0, :] + delta_vec[count, :]

            hum_torques_hat[count, :] = self.nlps.opensimAD_ID(sim_initial_state_perturbed[count, :]).full().reshape((3,))

        # let's find the maximum variation for each component of the torque output
        max_diff_tauPe = np.abs(np.max(hum_torques_hat[:,0]) - np.min(hum_torques_hat[:,0]))
        max_diff_tauSe = np.abs(np.max(hum_torques_hat[:,1]) - np.min(hum_torques_hat[:,1]))
        max_diff_tauAr = np.abs(np.max(hum_torques_hat[:,2]) - np.min(hum_torques_hat[:,2]))

        # let's plot the results
        # plot PE torques
        fig = plt.figure()
        ax = fig.add_subplot(131)
        ax.scatter(np.ones((num_reps-1, 1)), hum_torques_hat[1:, 0], color = 'black')
        ax.scatter(1, hum_torques_hat[0,0], color = 'red', s = 35)
        ax.set_title("PE (max diff " + str(np.round(max_diff_tauPe, 6)) + ")")

        # plot SE torques
        ax = fig.add_subplot(132)
        ax.scatter(np.ones((num_reps-1, 1)), hum_torques_hat[1:, 1], color = 'black')
        ax.scatter(1, hum_torques_hat[0,1], color = 'red', s = 35)
        ax.set_title("SE (max diff " + str(np.round(max_diff_tauSe, 6)) + ")")

        # plot AR torques
        ax = fig.add_subplot(133)
        ax.scatter(np.ones((num_reps-1, 1)), hum_torques_hat[1:, 2], color = 'black')
        ax.scatter(1, hum_torques_hat[0,2], color = 'red', s = 35)
        ax.set_title("AR (max diff " + str(np.round(max_diff_tauAr, 6)) + ")")
        fig.suptitle("Tau sensitivities (delta = " + str(delta_perturbation_torque) + ")")

        ## SENSITIVITY ANALYSIS FOR THE FD  
        # now we do the same sensitivity analysis for the system dynamics (as OpenSimAD function)

        # let's start from the nominal input torque (corresponding to the unperturbed initial state)
        hum_torque_inputSysDin = hum_torques_hat[0, :]

        # get the corresponding derivative of the state using OpenSimAD
        # we do this num_reps times, with small perturbations of the initial state (the magnitude of the perturbation can be adjusted)
        num_reps = 100
        delta_perturbation_acc = 1e-3
        delta_vec = np.zeros((num_reps, 3))

        acc_output = np.zeros((num_reps, 3))

        for count in range(0, num_reps):
            if count > 0:
                # the torque outputs will tell us how much a change in the input modifies the output
                delta_vec[count, :] = delta_perturbation_acc * np.random.random(np.shape(hum_torque_inputSysDin)).reshape((3,))

            acc_output[count, :] = self.nlps.sys_fd_dynamics(np.concatenate((sim_initial_state_perturbed[count, :6], hum_torque_inputSysDin + delta_vec[count, :]))).full().reshape((6,))[1::2]

        # let's find the maximum output variation in the various DoF
        max_diff_peDdot = np.abs(np.max(acc_output[:,0]) - np.min(acc_output[:,0]))
        max_diff_seDdot = np.abs(np.max(acc_output[:,1]) - np.min(acc_output[:,1]))
        max_diff_arDdot = np.abs(np.max(acc_output[:,2]) - np.min(acc_output[:,2]))

        # let's plot the results
        # plot PE accelerations
        fig = plt.figure()
        ax = fig.add_subplot(131)
        ax.scatter(np.ones((num_reps-1, 1)), acc_output[1:, 0], color = 'black')
        ax.scatter(1, acc_output[0,0], color = 'red', s = 35)
        ax.set_title("PE (max diff " + str(np.round(max_diff_peDdot, 6)) + ")")

        # plot SE accelerations
        ax = fig.add_subplot(132)
        ax.scatter(np.ones((num_reps-1, 1)), acc_output[1:, 1], color = 'black')
        ax.scatter(1, acc_output[0,1], color = 'red', s = 35)
        ax.set_title("SE (max diff " + str(np.round(max_diff_seDdot, 6)) + ")")

        # plot AR accelerations
        ax = fig.add_subplot(133)
        ax.scatter(np.ones((num_reps-1, 1)), acc_output[1:, 2], color = 'black')
        ax.scatter(1, acc_output[0,2], color = 'red', s = 35)
        ax.set_title("AR (max diff " + str(np.round(max_diff_arDdot, 6)) + ")")
        fig.suptitle("Acc. sensitivities (delta = " + str(delta_perturbation_acc) + ")")

        plt.show()
        

    def debug_NLPS_formulation(self):
        # first solve the nlp problem
        time_start = time.time()
        x_opt, u_opt, sol, x_opt_coll = self.nlps.solveNLPOnce()
        time_execution_0 = time.time() - time_start

        print ("execution with Opti: ", np.round(time_execution_0,3))

        # time_vec_knots = self.nlps.h * np.arange(0,self.nlps.N+1)
        # time_vec_colPoints = np.array([])
        # for index in range(0, self.nlps.N):
        #     time_vec_colPoints = np.concatenate((time_vec_colPoints, time_vec_knots[index] + self.nlps.h * np.asarray(ca.collocation_points(self.nlps.pol_order, 'legendre'))))

        # index_pe = self.nlps.dim_x * np.arange(0,self.nlps.N+1)

        # fig = plt.figure()
        # ax = fig.add_subplot(211)
        # ax.scatter(time_vec_knots, x_opt[index_pe])         # plot state
        # ax.plot(time_vec_knots, x_opt[index_pe])
        # for interval in np.arange(0, self.nlps.N):          # plot state collocation points inside each interval
        #     ax.scatter(time_vec_colPoints[0 + self.nlps.pol_order * interval], x_opt_coll[int(interval*self.nlps.dim_x) , 0], color = 'orange')
        #     ax.scatter(time_vec_colPoints[1 + self.nlps.pol_order * interval], x_opt_coll[int(interval*self.nlps.dim_x) , 1], color = 'blue')
        #     ax.scatter(time_vec_colPoints[2 + self.nlps.pol_order * interval], x_opt_coll[int(interval*self.nlps.dim_x) , 2], color = 'red')
        # ax.set_title("Plane of elevation")
        # ax = fig.add_subplot(212)
        # ax.scatter(time_vec_knots, x_opt[index_pe+2])
        # ax.plot(time_vec_knots, x_opt[index_pe+2])
        # for interval in np.arange(0, self.nlps.N):          # plot state collocation points inside each interval
        #     ax.scatter(time_vec_colPoints[0 + self.nlps.pol_order * interval], x_opt_coll[int(interval*self.nlps.dim_x) +2, 0], color = 'orange')
        #     ax.scatter(time_vec_colPoints[1 + self.nlps.pol_order * interval], x_opt_coll[int(interval*self.nlps.dim_x) +2, 1], color = 'blue')
        #     ax.scatter(time_vec_colPoints[2 + self.nlps.pol_order * interval], x_opt_coll[int(interval*self.nlps.dim_x) +2,2], color = 'red')
        # # ax.set_title("Shoulder elevation")

        # fig = plt.figure()
        # ax = fig.add_subplot(211)
        # ax.stairs(u_opt[0::3])
        # ax.set_title("Torque PE")
        # ax = fig.add_subplot(212)
        # ax.stairs(u_opt[1::3])
        # ax.set_title("Torque SE")

        # traj_opt = x_opt.reshape((6, 11), order='F')[:, 1::]

        # fig = plt.figure()
        # ax = fig.add_subplot()
        # ax.scatter(traj_opt[0, :], traj_opt[2, :])

        # plt.show()

        # simulated initial state for the human model
        # sim_initial_state =self.nlps.x_0

        # Create a random number generator instance
        rng = np.random.default_rng()
        instances = 100

        time_duration = np.zeros((instances, 1))

        for instance in range(instances):
            pe_init = np.deg2rad(rng.uniform(low = 55, high = 60))
            pe_dot_init = np.deg2rad(rng.uniform(low = -15, high = -5))

            se_init = np.deg2rad(rng.uniform(low = 95, high = 105))
            se_dot_init = np.deg2rad(rng.uniform(low = -15, high = 5))

            ar_init = np.deg2rad(rng.uniform(low = -60, high = 60))
            ar_dot_init = np.deg2rad(0)
            
            # TODO: use this for robustness test with many initial conditions
            sim_initial_state = np.array([pe_init, pe_dot_init, se_init, se_dot_init, ar_init, ar_dot_init])

            # TODO: use this for benchmarking and repeatable trials
            # sim_initial_state =self.nlps.x_0

            # first, we estimate the future states given the initial one
            fut_traj_value = np.zeros((self.nlps.dim_x, self.nlps.N+1))
            fut_traj_value[:,0] = sim_initial_state
            fut_traj_value[1::2, :] = sim_initial_state[1::2][:, np.newaxis]    # velocities are assumed to be constant
            for timestep in range(1, self.nlps.N+1):
                fut_traj_value[::2, timestep] = fut_traj_value[::2, timestep-1] + self.nlps.h * fut_traj_value[1::2, timestep-1]

            # solve the NLP once
            time_start = time.time()
            try:
                x_opt, u_opt, _,  j_opt, xddot_opt = self.mpc_iter(sim_initial_state, fut_traj_value[:, :-1], self.all_params_ellipses)
            except:
                RuntimeError("Optimization not converged")

            # print the solution
            x_opt = x_opt.full().reshape((6, 11), order='F')
            # x_opt = x_opt.reshape((6, 11), order='F')

            fig = plt.figure()
            ax = fig.add_subplot(311)
            ax.scatter(range(self.nlps.N+1), np.rad2deg(x_opt[0,:]), c='blue', label='optimized')
            ax.scatter(range(self.nlps.N+1), np.rad2deg(fut_traj_value[0,:]), c='red', label='free')

            ax = fig.add_subplot(312)
            ax.scatter(range(self.nlps.N+1), np.rad2deg(x_opt[2,:]), c='blue')
            ax.scatter(range(self.nlps.N+1), np.rad2deg(fut_traj_value[2,:]), c='red')

            ax = fig.add_subplot(313)
            ax.scatter(range(self.nlps.N+1), np.rad2deg(x_opt[4,:]), c='blue')
            ax.scatter(range(self.nlps.N+1), np.rad2deg(fut_traj_value[4,:]), c='red')

            fig.legend()

            # title
            fig.suptitle("x0 = [" + 
                         str(np.round(np.rad2deg(pe_init), 1)) + ", " +
                         str(np.round(np.rad2deg(pe_dot_init), 1)) +  ", " +
                         str(np.round(np.rad2deg(se_init), 1)) +  ", " +
                         str(np.round(np.rad2deg(se_dot_init), 1)) +  ", " +
                         str(np.round(np.rad2deg(ar_init), 1)) +  ", " +
                         str(np.round(np.rad2deg(ar_dot_init), 1)) +
                           "]")

            fig = plt.figure()
            ax = fig.add_subplot()
            ax.scatter(np.rad2deg(x_opt[0,:]), np.rad2deg(x_opt[2,:]), c = 'blue')
            ax.scatter(np.rad2deg(x_opt[0,0]), np.rad2deg(x_opt[2,0]), c = 'cyan')
            ax.scatter(np.rad2deg(fut_traj_value[0,:]), np.rad2deg(fut_traj_value[2,:]), c = 'red')
            ax.add_patch(Ellipse((self.all_params_ellipses[0], self.all_params_ellipses[1]), width = 2*np.sqrt(self.all_params_ellipses[2]), height = 2*np.sqrt(self.all_params_ellipses[3]), alpha=0.1))

            plt.show()
            
            time_duration[instance] = time.time() - time_start

        
        print ("avg time: ", np.round(time_duration.sum()/instances, 3))

        aux = 0


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
        path_to_repo = os.path.join(code_path, '..')          # getting path to the repository
        path_to_model = os.path.join(path_to_repo, 'Musculoskeletal Models')    # getting path to the OpenSim models

        ## PARAMETERS -----------------------------------------------------------------------------------------------
        # are we debugging or not?
        debug_mode = True

        # initialize the biomechanics-safety net module
        bsn_module = BS_net(debug_mode, rate=200, simulation = simulation, speed_estimate=rospy.get_param('/pu/speed_estimate'))

        params_strainmap_test = np.array([4, 20/160, 90/144, 35/160, 25/144, 0])
        params_ellipse_test = np.array([20, 90, 35**2, 25**2])
        
        bsn_module.setCurrentEllipseParams(params_ellipse_test)
        bsn_module.setCurrentStrainMapParams(params_strainmap_test)

        # initialize the underlying nonlinear programming problem
        opensimAD_ID = ca.Function.load(os.path.join(path_to_model, 'right_arm_GH_full_scaled_preservingMass_ID.casadi'))
        opensimAD_FD = ca.Function.load(os.path.join(path_to_model, 'right_arm_GH_full_scaled_preservingMass_FD.casadi'))

        nlps_instance = nlps.nlps_module(opensimAD_FD, opensimAD_ID, bsn_module.getCurrentEllipseParams()[0], bsn_module.getCurrentEllipseParams()[1])

        nlps_instance.setTimeHorizonAndDiscretization(N = 10, T = 1)

        x = ca.MX.sym('x', 6)   # state vector: [theta, theta_dot, psi, psi_dot, phi, phi_dot], in rad or rad/s
        nlps_instance.initializeStateVariables(x)

        u = ca.MX.sym('u', 3)   # control vector: [tau_theta, tau_psi], in Nm (along the DoFs of the GH joint)  
        nlps_instance.initializeControlVariables(u)

        # define the constraints
        u_max = 1e-5
        u_min = -1e-5
        constraint_list = {'u_max':u_max, 'u_min':u_min}
        nlps_instance.setConstraints(constraint_list)

        # define order of polynomials and collocation points for direct collocation 
        d = 3
        coll_type = 'legendre'
        nlps_instance.populateCollocationMatrices(d, coll_type)

        # for now, there is no cost function
        nlps_instance.setCostFunction(0)

        # choose solver and set its options
        solver = 'ipopt'        # available solvers depend on CasADi interfaces

        opts = {
                # 'ipopt.print_level': 5,         # options for the solver (check CasADi/solver docs for changing these)
                # 'print_time': 0,
                # 'ipopt.mu_strategy': 'adaptive',
                # 'ipopt.nlp_scaling_method': 'gradient-based',
                'ipopt.tol': 1e-3,
                'error_on_fail':1,              # to guarantee transparency if solver fails
                'expand':1,                     # to leverage analytical expression of the Hessian
                'ipopt.linear_solver':'ma27'
                # 'ipopt.linear_solver':'mumps'
                # "jit": True, 
                # "compiler": "shell", 
                # "jit_options": 
                #     {
                #         "flags": ["-O3"], 
                #         "verbose": True
                #     } 
                }

        # solver = 'fatrop'
        # opts = {}
        # opts["expand"] = True
        # opts["fatrop"] = {"mu_init": 0.1}
        # opts["structure_detection"] = "auto"
        # opts["debug"] = True

        
        nlps_instance.setSolverOptions(solver, opts)

        nlps_instance.setInitialState(x_0 = np.array(rospy.get_param('/pu/x_0')))

        # after the NLPS is completely built, we assign it to the Biomechanics Safety Net
        bsn_module.assign_nlps(nlps_instance)

        # debug OpenSimAD functions
        # bsn_module.debug_sysDynamics()

        # debug the NLP formulation
        # bsn_module.debug_NLPS_formulation()

        # Publish the initial position of the KUKA end-effector, according to the initial shoulder state
        # This code is blocking until an acknowledgement is received, indicating that the initial pose has been successfully
        # received by the RobotControlModule
        bsn_module.publishInitialPoseAsCartRef(shoulder_pose_ref = np.array(rospy.get_param('/pu/x_0'))[0::2], 
                                            base_R_sh = R.from_matrix(np.array(rospy.get_param('/pu/base_R_shoulder'))))

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

        # start the loop processing user input
        bsn_module.input_thread.start()

        # start to provide the safe reference as long as the robot is requesting it
        # we do so in a way that allows temporarily pausing the therapy
        while not rospy.is_shutdown():
            if bsn_module.keepRunning():

                if bsn_module.interaction_mode == 0:        # mode = 0: keep current pose and wait
                    bsn_module.setReferenceToCurrentPose()

                elif bsn_module.interaction_mode == 1:        # mode = 1: monitor unsafe zones
                    bsn_module.monitor_unsafe_zones()
                    
                elif bsn_module.interaction_mode == 2:        # mode = 2: damp velocities in unsafe areas
                    bsn_module.damp_unsafe_velocities()

                elif bsn_module.interaction_mode == 3:
                    bsn_module.predict_future_state_kinematic(N = 10, T = 1)

                elif bsn_module.interaction_mode == 4:
                    bsn_module.predict_future_state_simple(N = 10, T = 1)
                    
            # if the user wants to interrupt the therapy, we stop the optimization and freeze 
            # the robot to its current reference position.
            if not bsn_module.keepRunning():
                # overwrite the future references with the current one
                    bsn_module.setReferenceToCurrentRefPose()   # to decrease bumps (as trajectory will be very smooth anyway)

            bsn_module.ros_rate.sleep()

    except rospy.ROSInterruptException:
        pass
