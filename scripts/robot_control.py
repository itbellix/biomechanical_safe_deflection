"""
Script to control the KUKA LBR iiwa both in simulation and in the lab.
It builds on top of the iiwa-ros repository, available at https://gitlab.tudelft.nl/kuka-iiwa-7-cor-lab/iiwa_ros

To run this code, first source your ROS distribution (tested with Noetic), then source the iiwa-ros package,
in two different terminals. On the first terminal, launch ROS and the Gazebo environment with controller.launch
(located in this repository, under Code/launch/). Then, run this script in the second terminal. You will be prompted
with an interface to interact with the robot. For some tasks (like approaching the position to begin therapy),
TO_main.py should also be started in another terminal.

Example usage: $ python robot_control --simulation=true
"""

# Used in the main computations
import numpy as np

# ROS libraries
import rospy # CC
import actionlib

# Messages to and from KUKA robot
from cor_tud_msgs.msg import ControllerAction, ControllerGoal, CartesianState
from std_msgs.msg import Float64MultiArray, Bool, Float32MultiArray
from spatialmath import SE3

# import scipy to deal with rotation matrices
from scipy.spatial.transform import Rotation as R
import time

# import tkinter to be used as an interface for executing code in a state-machine fashion
import tkinter as tk

# import threading for testing
import threading

# import parser
import argparse

# import the parameters for the experiment as defined in experiment_parameters.py
from experiment_parameters import *

class RobotControlModule:
    """
    This class implements the robot control
    """
    def __init__(self):
        """
        Initializes a RobotControlModule object.
        """
        # get the name of the robot that we are working with
        self.ns = rospy.get_param('/namespaces')

        # define the ros client - "/cor_tud/torque_controller" is defined on the "bringup" file
        self.client = actionlib.SimpleActionClient(self.ns+'/torque_controller', ControllerAction)

        # instantiate the controller goal that will be used to send commands to the robot
        self.reference_tracker = ControllerGoal()

        # instantiate the controller goal that will stream the end-effector (EE) cartesian pose to the given topic
        self.cartesian_pose_publisher = ControllerGoal()
        self.cartesian_pose_publisher.mode = 'pub_cartesian_state'  # mode for the goal, to publish the cartesian state of the EE
        self.cartesian_pose_publisher.topic = 'ee_cartesian_pose'   # name of the topic on which to publish the cartesian state
        self.cartesian_pose_publisher.flag = False                  # setting this initializes the publisher to be inactive

        # send immediately the first ControllerGoal, so that the robot will keep its current position until commanded otherwise
        goal = ControllerGoal()
        goal.mode = 'ee_cartesian'
        goal.reference = []             # no reference, so that the current one will be used
        goal.stiffness = goal.stiffness = np.array([150, 150, 150, 25, 25, 25])
        goal.damping = 2*np.sqrt(goal.stiffness)
        goal.time = 0.1
        goal.rate = 20

        self.client.wait_for_server()       # necessary the first time to establish communication properly
        self.client.send_goal(goal)         # this will allow us to set what we want in the rest of the code
        self.client.wait_for_result()

        # define a ROS subscriber to have access to the cartesian pose of the EE if needed
        self.sub_to_cartesian_pose = rospy.Subscriber(self.ns+'/'+self.cartesian_pose_publisher.topic,  # name of the topic to subscribe to
                                                      CartesianState,                                   # type of ROS message to receive
                                                      self._callback_ee_pose,                           # callback
                                                      queue_size=1)
        
        # define a ROS publisher to convert current cartesian pose into shoulder pose
        self.topic_shoulder_pose= rospy.get_param('/rostopic/estimated_shoulder_pose')
        self.pub_shoulder_pose = rospy.Publisher(self.topic_shoulder_pose, Float64MultiArray, queue_size = 1)

        # define a ROS subscriber to receive the commanded (optimal) trajectory for the EE, from the
        # biomechanical-based optimization
        self.topic_optimal_pose_ee = rospy.get_param('/rostopic/cartesian_ref_ee')
        self.sub_to_optimal_pose_ee = rospy.Subscriber(self.topic_optimal_pose_ee, 
                                                       Float64MultiArray,
                                                       self._callback_ee_ref_pose,
                                                       queue_size=1)
        
        # define the parameters that will be used to store information about the robot status and state estimation
        self.ee_pose_curr = None            # EE current pose
        self.ee_desired_pose = None         # EE desired pose
        self.ee_twist_curr = None           # EE current twist (linear and angular velocity in robot base frame)
        self.desired_pose_reached = None    # the end effector has effectively reached the desired point
        self.initial_pose_reached = False   # one-time flag to be adjusted when the robot reaches the required
                                            # starting point. When set to true, the estimated shoulder state are meaningful


        # define parameters for the filtering of the human state estimation
        self.alpha_p = 0.9                  # weight of the exponential moving average filter (position part)
        self.alpha_v = 0.9                  # weight of the exponential moving average filter (velocity part)
        self.human_pose_estimated = np.zeros(12)    # contains q, q_dot, q_ddot, and position_gh_center_in_base (all are 3D vecs)
        self.last_timestamp = 0
        self.filter_initialized = False     # whether the filter applied on the human pose estimation has
                                            # already been initialized
        
        # set up a publisher and its thread for publishing continuously whether the robot is tracking the 
        # optimal trajectory or not
        self.topic_request_reference = rospy.get_param('/rostopic/request_reference')
        self.pub_request_reference = rospy.Publisher(self.topic_request_reference, Bool, queue_size = 1)
        self.requesting_reference = False         # flag indicating whether the robot is requesting a reference
        self.thread_therapy_status = threading.Thread(target=self.requestUpdatedReference)   # creating the thread
        self.thread_therapy_status.daemon = True    # this allows to terminate the thread when the main program ends

        # store information about the physical robot
        self.joint_limits = np.array([[-170, 170],      # joint limits for each joint of the robot, in degrees
                                      [-120, 120],
                                      [-170, 170],
                                      [-120, 120],
                                      [-170, 170],
                                      [-120, 120],
                                      [-175, 175]])
        
        self.base_nullspace_gains_j = 1/np.deg2rad(self.joint_limits[:,1] - self.joint_limits[:, 0])    # base nullspace gains (for joints)

        # create a publisher to interface with the RMR solver
        self.topic_rmr = '/kukadat'
        self.pub_rmr_data = rospy.Publisher(self.topic_rmr, Float32MultiArray, queue_size = 1)

    def _callback_ee_pose(self,data):
        """
        This callback is linked to the ROS subscriber that listens to the topic where the cartesian pose of the EE is published.
        It processes the data received and updates the internal parameters of the RobotControlModule accordingly.
        If the desired position for the therapy/experiment has been reached, it also converts the current EE pose into shoulder pose 
        (under the assumption that the glenohumeral joint center is fixed in space), and publishes this information on a topic.
        """
        self.ee_pose_curr = SE3(np.array(data.pose).reshape(4,4))   # value for the homogenous matrix describing ee pose
        self.ee_twist_curr = np.array(data.velocity)                # value for the twist (v and omega, in base frame)
        # check if the robot has already reached its desired pose (if so, publish shoulder poses too)
        if self.initial_pose_reached:
            R_ee = self.ee_pose_curr.R                              # retrieve the rotation matrix defining orientation of EE frame
            cart_pose_ee = self.ee_pose_curr.t                      # retrieve the vector defining 3D position of the EE (in robot base frame)
            sh_R_elb = np.transpose(R.from_matrix(np.array(rospy.get_param('/pu/base_R_shoulder'))).as_matrix())@R_ee@np.transpose(R.from_matrix(np.array(rospy.get_param('/pu/elb_R_ee'))).as_matrix())

            # calculate the instantaneous position of the center of the shoulder/glenohumeral joint
            if rospy.get_param('/pu/estimate_gh_position'):
                position_gh_in_ee = np.array([0, 0, rospy.get_param('/pu/l_arm')+rospy.get_param('/pu/l_brace')])
                position_gh_in_base = cart_pose_ee + R_ee@position_gh_in_ee
            else:
                position_gh_in_base = np.array([rospy.get_param('/pu/p_gh_in_base_x'), 
                                                rospy.get_param('/pu/p_gh_in_base_y'), 
                                                rospy.get_param('/pu/p_gh_in_base_z')])
                
            direction_vector = cart_pose_ee - position_gh_in_base
            direction_vector_norm = direction_vector / np.linalg.norm(direction_vector)

            direction_vector_norm_in_shoulder = np.transpose(R.from_matrix(np.array(rospy.get_param('/pu/base_R_shoulder'))).as_matrix())@direction_vector_norm

            # 1. we estimate the coordinate values
            # The rotation matrix expressing the elbow frame in shoulder frame is approximated as:
            # sh_R_elb = R_y(pe) * R_x(-se) * R_y (ar) = 
            # 
            #   | c_pe*c_ar-s_pe*c_se*s_ar    -s_pe*s_se    c_pe*s_ar+s_pe*c_se*c_ar  |
            # = |          -s_ar                 c_se               s_se*c_ar         |
            #   | -s_pe*c_ar-c_pe*c_se*s_ar   -c_pe*s_se    -s_pe*s_ar+c_pe*c_se*c_ar |
            #
            # Thus, the following formulas retrieve the shoulder state:
            
            # estimation of {pe,se} based on the cartesian position of the EE
            pe = np.arctan2(direction_vector_norm_in_shoulder[0], direction_vector_norm_in_shoulder[2])
            se = np.arccos(np.dot(direction_vector_norm_in_shoulder, np.array([0, -1, 0])))
            
            # estimation of {pe,se} based on the orientation of the EE
            # se = np.arctan2(np.sqrt(sh_R_elb[0,1]**2+sh_R_elb[2,1]**2), sh_R_elb[1,1])
            # pe = np.arctan2(-sh_R_elb[0,1]/np.sin(se), -sh_R_elb[2,1]/np.sin(se))

            # once the previous angles have been retrieved, use the orientation of the EE to find ar
            # (we assume that the EE orientation points towards the center of the shoulder for this)
            ar = np.arctan2(-sh_R_elb[1,0], sh_R_elb[1,2]/np.sin(se)) + rospy.get_param('/pu/ar_offset')

            # 2. we estimate the coordinate velocities (here the robot and the human are interacting as a geared mechanism)
            # 2.1 estimate the velocity along the plane of elevation
            sh_twist = R.from_matrix(np.array(rospy.get_param('/pu/base_R_shoulder'))).as_matrix().T @ self.ee_twist_curr.reshape((2,3)).T
            L_tot = rospy.get_param('/pu/l_arm') + rospy.get_param('/pu/l_brace')   # total distance between GH joint and elbow tip
            
            r = np.array([L_tot * np.cos(pe), L_tot * np.sin(pe)])

            # calculating the angular velocity around the Y axis of the shoulder frame (pointing upwards)
            # formula : omega = radius_vec X velocity_vec / (||radius_vec||^2)
            # velocities and radius are considered on the plane perpendicular to Y (so, the Z-X plane)
            pe_dot = np.cross(r, np.array([sh_twist[2,0], sh_twist[0,0]]))/(L_tot**2)

            # 2.2. estimate the velocity along the shoulder elevation
            # First transform the twist in the local frame where this DoF is defined, then apply the same
            # formula as above to obtain angular velocity given the linear speed and distance from the rotation axis
            # Note the minus sign in front of the cross-product, for consistency with the model definition.
            local_twist = R.from_euler('y', pe).as_matrix().T @ sh_twist
            r = np.array([L_tot * np.sin(se), -L_tot * np.cos(se)])
            se_dot = np.cross(r, np.array([local_twist[2,0], local_twist[1,0]]))/(L_tot**2)

            # 2.3 estimate the velocity along the axial rotation
            elb_twist = R.from_euler('x', -se).as_matrix().T @ local_twist
            ar_dot = elb_twist[1,1]

            # filter the state values (we use an exponential moving average filter)
            if not self.filter_initialized:
                # if the filter state has not been initialized yet, do it now
                self.human_pose_estimated[0] = pe
                self.human_pose_estimated[1] = pe_dot
                self.human_pose_estimated[2] = se
                self.human_pose_estimated[3] = se_dot
                self.human_pose_estimated[4] = ar
                self.human_pose_estimated[5] = ar_dot
                self.human_pose_estimated[6] = 0            # we initialize the accelerations to be 0
                self.human_pose_estimated[7] = 0            # we initialize the accelerations to be 0
                self.human_pose_estimated[8] = 0            # we initialize the accelerations to be 0
                self.human_pose_estimated[9:] = position_gh_in_base     # also the position of the GH joint center is 
                                                                        # part of the human pose


                self.last_timestamp = rospy.Time.now().to_time()

                self.filter_initialized = True

            # filter the estimates for the human pose (glenohumeral angles and velocities)    
            pe = self.alpha_p * pe + (1-self.alpha_p) * self.human_pose_estimated[0]
            pe_dot = self.alpha_v * pe_dot + (1-self.alpha_v) * self.human_pose_estimated[1]
            se = self.alpha_p * se + (1-self.alpha_p) * self.human_pose_estimated[2]
            se_dot = self.alpha_v * se_dot + (1-self.alpha_v) * self.human_pose_estimated[3]
            ar = self.alpha_p * ar + (1-self.alpha_p) * self.human_pose_estimated[4]
            ar_dot = self.alpha_v * ar_dot + (1-self.alpha_v) * self.human_pose_estimated[5]
            position_gh_in_base = self.alpha_p * position_gh_in_base + (1-self.alpha_p) * self.human_pose_estimated[9:]

            # retrieve accelerations differentiating numerically the velocities
            time_now = rospy.Time.now().to_time()
            delta_t = time_now - self.last_timestamp
            pe_ddot = (pe_dot - self.human_pose_estimated[1])/delta_t
            se_ddot = (se_dot - self.human_pose_estimated[3])/delta_t
            ar_ddot = (ar_dot - self.human_pose_estimated[5])/delta_t

            # update last estimated pose (used as state of the filter)
            self.human_pose_estimated[0] = pe
            self.human_pose_estimated[1] = pe_dot
            self.human_pose_estimated[2] = se
            self.human_pose_estimated[3] = se_dot
            self.human_pose_estimated[4] = ar
            self.human_pose_estimated[5] = ar_dot
            self.human_pose_estimated[6] = pe_ddot
            self.human_pose_estimated[7] = se_ddot
            self.human_pose_estimated[8] = ar_ddot
            self.human_pose_estimated[9:] = position_gh_in_base

            self.last_timestamp = time_now      # used as the timestamp of the previous information

            # retrieve also the current XYZ cartesian position of the robot
            xyz_ee = self.ee_pose_curr.t

            # build the message and fill it with information
            message = Float64MultiArray()
            message.data = np.round(np.concatenate((np.array([pe, pe_dot, se, se_dot, ar, ar_dot, pe_ddot, se_ddot, ar_ddot]), 
                                                    position_gh_in_base, xyz_ee)), 3)

            message_rmr = Float32MultiArray()
            message_rmr.data = np.round(np.array([pe, se, ar, pe_dot, se_dot, ar_dot, pe_ddot, se_ddot, ar_ddot, 0, 0, 0, time_now]), 3)

            # publish only if ROSCORE is running
            if not rospy.is_shutdown():
                self.pub_shoulder_pose.publish(message)
                self.pub_rmr_data.publish(message_rmr)


    def _callback_ee_ref_pose(self,data):
        """
        This callback is linked to the ROS subscriber that listens to the topic where the desired cartesian pose of the EE 
        is published. It processes the data received and updates the internal parameters of the RobotControlModule accordingly.
        """
        # the message that we receive contains the required cartesian pose for the end effector in terms of an homogenous matrix.
        homogeneous_matrix = np.array(data.data[0:16]).reshape((4,4))
        self.ee_desired_pose = homogeneous_matrix

        # calculate corresponding 3D position and euler angles just in case
        xyz_position = homogeneous_matrix[0:3, 3]
        euler_angles = R.from_matrix(homogeneous_matrix[0:3, 0:3]).as_euler('zxy', degrees=False)

    
    def togglePublishingEEPose(self, flag, last_controller_mode=None):
        """
        This function allows to enable/disable the stream of data related to the position of the end-effector in cartesian
        space (expressed in the frame centered at the robot base).
        The inputs are:
            - flag: can be set to True/False if we want to start/stop publishing the ee pose
            - last_controller_mode: it is a ControllerGoal object that represents the one that is being used to 
                                    control the robot when this function is called. The same goal is sent again 
                                    at the end of the function call, so that the robot will maintain the position,
                                    stiffness and damping that it had before. If it is None, a default cartesian
                                    impedance controller will be used instead to keep the current position.

        Note: the execution of this function might be a bit slow, since two interactions with the action server
        are necessary. 
        This should not be a problem as it is meant to be used at the beginning of the experiment 
        (otherwise, make sure that it does not cause delays).
        """

        # now we can actually send what we wanted without worrying 
        self.cartesian_pose_publisher.flag = flag
        self.client.wait_for_server()
        self.client.send_goal(self.cartesian_pose_publisher)
        self.client.wait_for_result()

        if last_controller_mode is not None:
            self.client.wait_for_server()
            self.client.send_goal(last_controller_mode)
            self.client.wait_for_result()


    def goHoming(self, time):
        """
        This function is a utility to send the robot in its "home" configuration (i.e. all links are aligned, straight up).
        The only parameter that can be used is "time", to specify the duration of the movement to reach the home configuration.
        """
        # set the parameters of the ControllerGoal
        self.reference_tracker.mode = 'joint_ds'
        self.reference_tracker.reference = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.reference_tracker.time = time
        self.reference_tracker.precision = 1e-1
        self.reference_tracker.rate = 20
        self.reference_tracker.stiffness = 30* np.ones(7)
        self.reference_tracker.damping = np.sqrt(self.reference_tracker.stiffness)

        # send the ControllerGoal to start the motion towards the required reference
        self.client.wait_for_server()
        self.client.send_goal(self.reference_tracker)
        self.client.wait_for_result()


    def getEEDesiredPose(self):
        """
        Utility function to retrieve the value of the end effector desired pose stored in the RobotControlModule
        """
        return self.ee_desired_pose

    
    def moveToEEDesiredPose(self, time, rate, precision, cart_stiffness, cart_damping=None):
        """
        This function allows to move the KUKA EE towards the desired pose.
        The value of the EE pose is stored inside the RobotControlModule object, so a check
        is implemented to discriminate whether the pose is indeed valid.

        To guarantee that the robot joint configuration varies as smoothly as possible, we will use the
        ee_cartesian_jds mode: this means that the reference is a 6D Cartesian pose, but stiffness and damping
        must be expressed in joint space (7x1).
        """
        assert self.ee_desired_pose is not None, "Desired pose cannot be empty! Make sure of this before calling this function."

        # use ee_cartesian_ds mode to actually reach the cartesian pose we want
        self.reference_tracker.mode = 'ee_cartesian_jds'
        # retrieve the reference in a 6D format
        cartesian_position = self.ee_desired_pose[0:3, 3]
        euler_angles = np.flip(R.from_matrix(self.ee_desired_pose[0:3, 0:3]).as_euler('XYZ', degrees=False))    # the controller requires this angle sequence
        self.reference_tracker.reference = np.concatenate((cartesian_position, euler_angles))
        self.reference_tracker.time = time
        self.reference_tracker.rate = rate
        self.reference_tracker.precision = precision
        self.reference_tracker.stiffness = np.array([30, 30, 30, 30, 30, 30, 30])
        self.reference_tracker.damping = 2*np.sqrt(self.reference_tracker.stiffness)

        self.client.send_goal(self.reference_tracker)
        self.desired_pose_reached = self.client.wait_for_result()

        print("Switch to Cartesian Mode")
        # use ee_cartesian_ds mode to actually reach the cartesian pose we want
        self.reference_tracker.mode = 'ee_cartesian_ds'

        # retrieve the reference in a 6D format
        cartesian_position = self.ee_desired_pose[0:3, 3]
        euler_angles = np.flip(R.from_matrix(self.ee_desired_pose[0:3, 0:3]).as_euler('XYZ', degrees=False))
        self.reference_tracker.reference = np.concatenate((cartesian_position, euler_angles))
        self.reference_tracker.time = 5
        self.reference_tracker.rate = rate
        self.reference_tracker.precision = precision
        self.reference_tracker.stiffness = cart_stiffness

        # set the damping
        if cart_damping is not None:
            self.reference_tracker.damping = cart_damping
        else:
            self.reference_tracker.damping = 2*np.sqrt(self.reference_tracker.stiffness)
        
        self.client.send_goal(self.reference_tracker)
        self.desired_pose_reached = self.client.wait_for_result()
        

    def trackReferenceOnTopic(self, topic, flag):
        """
        This function takes as inputs:
        * the topic on which the reference to track will be published
        * a flag to toggle the tracking of the reference (True or False)
        """

        control_module.toggleListenCartRef(topic, flag) # listening to a reference on the given topic is toggled
        self.requesting_reference = flag

        if flag and not self.thread_therapy_status.is_alive():
            # start the thread the first time (only if it is not running already)
            self.thread_therapy_status.start()
        

    def requestUpdatedReference(self):
        """
        This function publishes on a default topic whether or not the robot is requesting an updated reference to track.
        This allows to perform all the start up procedures, and the optimal trajectory for the robot EE will be published
        only when the subject/experimenter are ready to start.
        """
        rate = rospy.Rate(5)        # setting the rate quite low so that the publishing happens not too often
        while not rospy.is_shutdown():
            self.pub_request_reference.publish(self.requesting_reference)
            rate.sleep()
    
    
    def stop(self):
        """
        This function allows to stop the execution. 
        If the robot is being operated in CARTESIAN mode, the current EE pose is set as a reference and kept.
        If the robot is in JOINT mode, the last commanded joint pose is kep, and the robot is driven to that.
        """
        current_mode = self.reference_tracker.mode
        if current_mode[0:12] == 'ee_cartesian':
            # retrieve the current position of the end effector in cartesian coordinates
            current_position = self.ee_pose_curr.t
            current_orientation = R.from_matrix(self.ee_pose_curr.R).as_euler('zyx', degrees=False)
            current_pose = np.concatenate((current_position, current_orientation))

            # set the parameters to the ControllerGoal to freeze the robot where it is
            self.reference_tracker.mode = 'ee_cartesian_ds'
            self.reference_tracker.time = 0.1
            self.reference_tracker.reference = current_pose
            self.reference_tracker.stiffness = np.array([150, 150, 150, 10, 10, 10])           # the rotational DoF are left almost free for safety
            self.reference_tracker.damping = 2*np.sqrt(self.reference_tracker.stiffness)
        
        elif current_mode[0:5] == 'joint':
            # retrieve the last reference commanded
            last_joint_command = self.reference_tracker.reference

            # set the parameters to the ControllerGoal to freeze the robot where it is
            self.reference_tracker.mode = 'joint_ds'
            self.reference_tracker.time = 5
            self.reference_tracker.reference = last_joint_command
            self.reference_tracker.stiffness = 30* np.ones(7)
            self.reference_tracker.damping = 2*np.sqrt(self.reference_tracker.stiffness)
            
        # send the goal for termination to take effect
        self.client.send_goal(self.reference_tracker)
        self.client.wait_for_result()

        # set correct flags to let TO_module know we are done
        self.requesting_reference = False
        self.pub_request_reference.publish(self.requesting_reference)


    def test_publish_cartesian(self, reference, time_movement, stiffness, damping = None, nullspace_gain = np.zeros(7), nullspace_reference = np.zeros(7)):
        """
        This is a testing utility to publish a given reference to the robot.
        It requires a reference (6x1 pose of the ee), and a time over which
        the movement will be executed.
        """
        self.reference_tracker.mode = 'ee_cartesian_ds'
        self.reference_tracker.time = time_movement
        self.reference_tracker.reference = reference
        self.reference_tracker.precision = 1e-2

        self.reference_tracker.stiffness = stiffness
        if damping is None:
            self.reference_tracker.damping = 2*np.sqrt(stiffness)
        else:
            self.reference_tracker.damping = damping

        # add nullspace gains for the control at the joint level
        self.reference_tracker.nullspace_gain = nullspace_gain
        self.reference_tracker.nullspace_reference = nullspace_reference
        
        # send the goal for the robot to move towards the reference
        self.client.send_goal(self.reference_tracker)
        self.client.wait_for_result()


    def test_publish_joint(self, reference, time_movement):
        """
        This is a testing utility to publish a given reference to the robot.
        It requires a reference (7x1 joint positions), and a time over which
        the movement will be executed
        """
        self.reference_tracker.mode = 'ee_cartesian_jds'
        self.reference_tracker.time = time_movement
        self.reference_tracker.reference = reference
        self.reference_tracker.precision = 1e-2

        self.reference_tracker.stiffness = np.array([30, 30, 30, 30, 30, 30, 30])
        self.reference_tracker.damping = 1/2*np.sqrt(self.reference_tracker.stiffness)  # lower damping for stability
        
        # send the goal for the robot to move towards the reference
        self.client.send_goal(self.reference_tracker)
        self.client.wait_for_result()


    def toggleListenCartRef(self, topic, flag):
        """
        This function provides a utility to start/stop listening to the cartesian
        reference for the ee, published on a given topic. Publishing on the topic should
        happen soon after (~1s) the function is called. If flag=False, the robot controller
        will stop tracking the reference on the topic.
        """
        self.reference_tracker.mode = "listen_cartesian_ref"
        self.reference_tracker.topic = topic
        self.reference_tracker.flag = flag

        # send the goal
        self.client.send_goal(self.reference_tracker)
        self.client.wait_for_result()


    def test_publish_ref_on_topic(self, publisher, ref, frequency=100):
        """
        This function will be executed in another thread. It will publish the given reference (6x1 ee pose),
        as long as the "pub_from_thread" variable is true.
        It can be useful to start publishing a fixed reference to ensure that the mode 'listen_cartesian_ref'
        will succeed and will start reading from where the publisher is publishing.
        Optionally, a frequency for publishing the reference message can be given
        """
        rate = rospy.Rate(frequency)  # Set the publishing rate

        while not rospy.is_shutdown() and self.pub_from_thread:
            homogeneous_matrix = np.eye(4)
            homogeneous_matrix[0:3, 3] = np.transpose(ref[0:3])
            rot = R.from_euler('xyz', ref[3::], degrees=False).as_matrix()
            homogeneous_matrix[0:3, 0:3] = rot

            # create the message
            message_ref = Float64MultiArray()
            message_ref.layout.data_offset = 0
            message_ref.data = np.reshape(homogeneous_matrix, (16,1))

            # send the message and sleep
            publisher.publish(message_ref)
            rate.sleep()


if __name__ == "__main__":
    try:
        # check if we are running in simulation or not
        parser = argparse.ArgumentParser(description="Script that controls our Kuka robot")
        parser.add_argument("--simulation", required=True, type=str)
        args = parser.parse_args()
        simulation = args.simulation

        # define real-time factor if we are in simulation or not
        if simulation == 'true':
            rt_factor = 5
        else:
            rt_factor = 1

        print("Running with simulation settings: ", simulation)
        print("real time factor: ", rt_factor)

        # initialize ros node and set desired rate (imported above as a parameter)
        rospy.init_node("robot_control_module")
        ros_rate = rospy.Rate(rospy.get_param('/pu/loop_frequency'))

        # flag that determines if the robotic therapy should keep going
        ongoing_therapy = True

        # initialize tkinter to set up a state machine in the code execution logic
        root = tk.Tk()
        root.title("Robot Interface (keep it selected)")

        # create the window that the user will have to keep selected to give their inputs
        window = tk.Canvas(root, width=500, height=200)
        window.pack()

        # Static caption
        caption_text = """Use the following keys to control the robot:
                        - 'a' to approach the starting pose for the therapy
                        - 's' to (re)start the therapy
                        - 'p' to pause the therapy
                        - 't' to run do some testing
                        - 'z' to set cartesian stiffness and damping to 0
                        - 'q' to quit the therapy"""
        
        window.create_text(250, 100, text=caption_text, font=("Helvetica", 12), fill="black")


        # StringVar to store the pressed key
        pressed_key = tk.StringVar()

        def on_key_press(event):
            # update the StringVar with the pressed key
            pressed_key.set(event.char)

        # bind the key press event to the callback function
        root.bind("<Key>", on_key_press)

        # instantiate a RobotControlModule with the correct shared ros topics (coming from experiment_parameters.py)
        control_module = RobotControlModule()

        print("control module instantiated")
        # set the robot in the homing pose, to always start from there
        homing_time = 5/rt_factor             # [s] time required to complete homing
        print("Initiating homing")
        control_module.goHoming(homing_time)
        print("Completed homing")

        print("start publishing current EE pose")
        
        # the last ControlGoal to be sent is the one inside goHoming() -> it is the reference_tracker
        # here we change the execution time, and then send it to the togglePublishingEEPose so that the
        # same control mode, stiffness, damping and reference are maintained
        control_module.reference_tracker.time = 1/rt_factor
        control_module.togglePublishingEEPose(True, control_module.reference_tracker)
        print("Current pose is being published")

        while not rospy.is_shutdown() and ongoing_therapy:
            # state machine to allow for easier interface
            # it checks if there is a user input, and set parameters accordingly
            try:
                # wait for an event to occur
                event = root.wait_variable(pressed_key)

                # handle the event

                # 'a': approach initial pose
                if pressed_key.get() == "a":
                    if control_module.getEEDesiredPose() is None:         # be sure that we know where to go
                        print("waiting for initial pose to be known")
                        while control_module.getEEDesiredPose() is None:
                            ros_rate.sleep()
                    
                    duration_movement = 20/rt_factor           # duration of the approach to the initial pose [s]
                    rate = 20                                   # num. of via-points per second 
                                                                # (from interpolation between current state and desired pose)
                    precision = 1e-2                            # precision required for reaching the goal [m]
                    stiffness = np.array([300, 300, 300, 10, 10, 2])
                    # stiffness = np.array([200, 200, 200, 15, 15, 2])    # low stiffness (issue #162)
                    # stiffness = np.array([350, 350, 350, 45, 45, 10])   # high stiffness (issue #162)
                    damping = np.sqrt(stiffness)             # decrease damping to increase stability

                    print("moving to initial position")
                    control_module.moveToEEDesiredPose(duration_movement, rate, precision, stiffness, damping)

                    if control_module.desired_pose_reached:
                        # add nullspace control on robot's elbow
                        control_module.reference_tracker.mode = 'elbow'
                        control_module.reference_tracker.reference = np.array([0.0, 0.0, 0.55])
                        control_module.reference_tracker.flag = True
                        control_module.reference_tracker.joints = [3]
                        if simulation == True:
                            control_module.reference_tracker.stiffness = np.array(rospy.get_param('/pu/ns_elb_stiffness_sim'))
                            control_module.reference_tracker.damping = np.array(rospy.get_param('/pu/ns_elb_damping_sim'))
                        else:
                            control_module.reference_tracker.stiffness = np.array(rospy.get_param('/pu/ns_elb_stiffness'))
                            control_module.reference_tracker.damping = np.array(rospy.get_param('/pu/ns_elb_damping'))
                        
                        control_module.client.send_goal(control_module.reference_tracker)
                        result = control_module.client.wait_for_result()

                        # add nullspace control on robot's last links
                        control_module.reference_tracker.mode = 'ee_cartesian_ds'
                        control_module.reference_tracker.reference = []
                        control_module.reference_tracker.time = 1
                        control_module.reference_tracker.rate = 20
                        control_module.reference_tracker.stiffness = stiffness
                        control_module.reference_tracker.damping = damping
                        control_module.reference_tracker.nullspace_gain = np.array([0, 0, 0, 0, 0.01, 0.01, 0.01])
                        control_module.reference_tracker.nullspace_reference = np.array([0, 0, 0, 0, 0, 0, 0])
                        control_module.client.send_goal(control_module.reference_tracker)
                        result = control_module.client.wait_for_result()

                        # set the flag for indicating completion, and inform the user
                        control_module.initial_pose_reached = True
                        print("Reached initial position. Starting to publish on topic %s (order data is [pe, se, ar])" % control_module.topic_shoulder_pose)
                        print("-----------------------------------------------")
                        print("Therapy can start")
                        print("-----------------------------------------------")

                    else:
                        print("Initial position could not be reached... Try again!")

                # 's' : start the therapy
                if pressed_key.get() == "s":
                    if control_module.initial_pose_reached:
                        # switch to pure cartesian mode
                        # further increase stiffness
                        if simulation == True:
                            stiffness_higher = np.array(rospy.get_param('/pu/ee_stiffness_sim'))
                            damping_higher = np.array(rospy.get_param('/pu/ee_damping_sim'))
                        else:
                            stiffness_higher = np.array(rospy.get_param('/pu/ee_stiffness'))
                            damping_higher = np.array(rospy.get_param('/pu/ee_damping'))

                        control_module.reference_tracker.mode = 'ee_cartesian'
                        control_module.reference_tracker.reference = []
                        control_module.reference_tracker.stiffness = stiffness_higher
                        control_module.reference_tracker.damping = damping_higher

                        # wait a couple of seconds
                        time.sleep(3)

                        control_module.client.send_goal(control_module.reference_tracker)
                        control_module.client.wait_for_result()
                        print("Start to follow optimal reference")
                        topic = rospy.get_param('/rostopic/cartesian_ref_ee')
                        control_module.trackReferenceOnTopic('/'+topic, True)
                    else:
                        print("Robot is not in the initial pose yet. Doing nothing.")

                # 'p' : pause the therapy
                if pressed_key.get() == "p":
                    print("Pause trajectory generation")
                    topic = rospy.get_param('/rostopic/cartesian_ref_ee')
                    control_module.trackReferenceOnTopic('/'+topic, False)

                # 't' : test something
                if pressed_key.get() == "t":
                    print("Moving to initial pose for testing")
                    # TEST: send a fixed goal to move to a known position, then start sending a trajectory reference 
                    # to see if we can do it properly
                    cartesian_0 = np.array([-0.4, 0, 0.6])

                    euler_0 = np.array([0, 0, 0])

                    ref = np.concatenate((cartesian_0, euler_0))
                    time_movement = 20

                    # go to ref
                    control_module.test_publish_cartesian(ref, time_movement)

                    # now, enable elbow 
                    input("\nPress any key to turn elbow on")
                    control_module.reference_tracker.mode = 'elbow'
                    control_module.reference_tracker.reference = np.array([0.0, 0.0, 0.55])
                    control_module.reference_tracker.flag = True
                    control_module.reference_tracker.joints = [3]
                    control_module.client.send_goal(control_module.reference_tracker)
                    result = control_module.client.wait_for_result()

                    
                    input("Press any key to move ee to new position")
                    cartesian_0 = np.array([-0.6, 0, 0.7])

                    euler_0 = np.array([0, 0, 0])

                    ref = np.concatenate((cartesian_0, euler_0))
                    time_movement = 10

                    control_module.test_publish_cartesian(ref, time_movement)

                    print("Done")

                # 'z' set cartesian stiffness and damping to 0
                if pressed_key.get() == 'z':
                    print("Zero stiffness/damping in 5 seconds")
                    time.sleep(2)
                    print("3")
                    time.sleep(1)
                    print("2")
                    time.sleep(1)
                    print("1")
                    time.sleep(1)

                    # switch to zero stiffness and damping
                    ref = []
                    time_movement = 3
                    stiffness = np.array([0, 0, 0, 0, 0, 0])
                    damping = np.array([0, 0, 0, 0, 0, 0])
                    nullspace_gain = np.array([0, 0, 0, 0, 1, 1, 0.5])
                    nullspace_reference = np.array([0, 0, 0, 0, 0, 0, 0])
                    control_module.test_publish_cartesian(ref, time_movement, stiffness, damping, nullspace_gain, nullspace_reference)
                    
                    #confirm that everything went smoothly
                    print("Free movement possible")

                # 'q': quit the execution, and freeze the robot to the current pose
                if pressed_key.get() == "q":
                    print("shutting down - freezing to current position")
                    control_module.stop()

                    # adjust flag for termination
                    ongoing_therapy = False

            except tk.TclError:
                # root is destroyed (e.g., window closed)
                break
            # to allow ROS to execute other tasks if needed
            ros_rate.sleep()

    except rospy.ROSInterruptException:
        pass
