import os
import casadi as ca
import numpy as np
import rospy
import time
import matplotlib.pyplot as plt
import utilities_TO as utils_TO
import utilities_casadi as utils_ca
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize_scalar
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

        # create a thread to catch the input from teh user, who will select the robot's interaction mode
        self.interaction_mode = 0
        self.input_thread = threading.Thread(target=self.input_thread_fnc, daemon=True)


    def setCurrentEllipseParams(self, all_params_ellipse):
        self.all_params_ellipses = all_params_ellipse


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


    def publishCartRef(self, shoulder_pose_ref, torque_ref, base_R_sh, dist_gh_elbow):
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
        if experimental_params['estimate_gh_position'] and self.flag_receiving_shoulder_pose:
            ref_cart_point = np.matmul(base_R_elb.as_matrix(), dist_gh_elbow) + self.position_gh_in_base
        else:
            ref_cart_point = np.matmul(base_R_elb.as_matrix(), dist_gh_elbow) + experimental_params['p_gh_in_base']

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

        # fix the position of the center of the shoulder/glenohumeral joint
        self.position_gh_in_base = experimental_params['p_gh_in_base']

        # perform extra things if this is the first time we execute this
        if not self.flag_pub_trajectory:
            # We need to set up the structure to deal with the new thread, to allow 
            # continuous publication of the optimal trajectory
            self.publish_thread = threading.Thread(target=self.publish_continuous_trajectory, 
                                                    args = (base_R_sh, dist_gh_elbow))   # creating the thread
            
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
        
        if experimental_params['estimate_gh_position']:
            # retrieve the current pose of the shoulder/glenohumeral center in the base frame
            self.position_gh_in_base = np.array(data.data[6:9])

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


    def publish_continuous_trajectory(self, rot_ee_in_base_0, dist_shoulder_ee):
        """
        This function picks the most recent information regarding the optimal shoulder trajectory,
        converts it to end effector space and publishes the robot reference continuously. A flag enables/disables
        the computations/publishing to be performed, such that this happens only if the robot controller needs it.
        """
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

                        self.publishCartRef(cmd_shoulder_pose, cmd_torques, rot_ee_in_base_0, dist_shoulder_ee)

            rate.sleep()


    def input_thread_fnc(self):
        while True:
            try:
                # Read user input (this happens asynchronously as executed in a thread)
                interaction_mode = input("Select interaction mode (0, 1, or 2):\n")
                print(": interaction mode selected")
                self.interaction_mode = int(interaction_mode)
            except ValueError:
                print("Invalid input, please enter a number between 0 and 2")


    def setReferenceToCurrentPose(self):
        """
        This function allows to overwrite the reference state/control values. They are
        substituted by the current state of the subject/patient sensed by the robot.
        """
        with self.x_opt_lock:
            self.x_opt = self.state_values_current.reshape((6,1))
            self.u_opt = np.zeros((2,1))

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
            self.u_opt = np.zeros((2,1))

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
                    # TODO: check what is happening here
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

        # initialize the biomechanics-safety net module
        bsn_module = BS_net(shared_ros_topics, debug_mode, rate=200, simulation = simulation, speed_estimate=True)

        # Publish the initial position of the KUKA end-effector, according to the initial shoulder state
        # This code is blocking until an acknowledgement is received, indicating that the initial pose has been successfully
        # received by the RobotControlModule
        bsn_module.publishInitialPoseAsCartRef(shoulder_pose_ref = x_0[0::2], 
                                            position_gh_in_base = experimental_params['p_gh_in_base'], 
                                            base_R_sh = experimental_params['base_R_shoulder'], 
                                            dist_gh_elbow = experimental_params['d_gh_ee_in_shoulder'])

        params_strainmap_test = np.array([4, 20/160, 90/144, 35/160, 25/144, 0])
        params_ellipse_test = np.array([20, 90, 35**2, 25**2])
        
        bsn_module.setCurrentEllipseParams(params_ellipse_test)
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
                    
                elif bsn_module.interaction_mode == 2:        # 
                    bsn_module.damp_unsafe_velocities()
            # if the user wants to interrupt the therapy, we stop the optimization and freeze 
            # the robot to its current reference position.
            if not bsn_module.keepRunning():
                # overwrite the future references with the current one
                    bsn_module.setReferenceToCurrentRefPose()   # to decrease bumps (as trajectory will be very smooth anyway)

            bsn_module.ros_rate.sleep()

    except rospy.ROSInterruptException:
        pass
