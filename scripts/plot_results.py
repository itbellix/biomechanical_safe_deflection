import rosbag
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import os
from spatialmath import SO3
from roboticstoolbox.robot.ERobot import ERobot, ERobot2
import roboticstoolbox as rtb



class strainMap:

    def __init__(self):
        # Definition of the biomechanical model parameters
        self.pe_boundaries = [-20, 160] # defines the interval of physiologically plausible values for the plane of elevation [deg]
        self.se_boundaries = [0, 144]   # as above, for the shoulder elevation [deg]
        self.ar_boundaries = [-90, 100] # as above, for the axial rotation [deg]

        self.pe_normalizer = 160        # normalization of the variables used to compute the interpolated strain maps
        self.se_normalizer = 144        # normalization of the variables used to compute the interpolated strain maps

        self.strainmap_step = 0.1    # discretization step used along the model's coordinate [in degrees]
                                # By default we set it to 4, as the strainmaps are generated from the biomechanical model
                                # with this grid accuracy

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

        self.num_params_ellipse = 4
        self.all_params_ellipse = []

        self.landscape_strain = None


    def setGaussianParams(self, all_params_gaussians):
        self.all_params_gaussians = all_params_gaussians


    def setEllipseParams(self, all_params_ellipse):
        self.all_params_ellipse = all_params_ellipse


    def calcLandscape(self, all_params_gaussians):
        # inizialize empty strainmap
        self.landscape_strain = np.zeros(self.X_norm.shape)

        for function in range(len(self.all_params_gaussians)//self.num_params_gaussian):

            amplitude, x0, y0, sigma_x, sigma_y, strain_offset = all_params_gaussians[function*self.num_params_gaussian:function*self.num_params_gaussian+self.num_params_gaussian]

            # then, compute the contribution of this particular Gaussian to the final strainmap
            self.landscape_strain += amplitude * np.exp(-((self.X_norm-x0/self.pe_normalizer)**2/(2*(sigma_x/self.pe_normalizer)**2)+(self.Y_norm-y0/self.se_normalizer)**2/(2*(sigma_y/self.se_normalizer)**2))) + strain_offset

    def plotLandscape(self):
        assert self.landscape_strain is not None, "There is no landscape to plot!"

        fig = plt.figure()
        ax = fig.add_subplot()
        heatmap = ax.imshow(np.flip(self.landscape_strain.T, axis = 0), cmap='hot', extent=[self.X.min(), self.X.max(), self.Y.min(), self.Y.max()])
        fig.colorbar(heatmap, ax = ax, ticks=np.arange(0, self.landscape_strain.max() + 1), label='Strain level [%]')
        ax.set_xlabel('Plane Elev [deg]')
        ax.set_ylabel('Shoulder Elev [deg]')
        
        return fig, ax
    

    def plotUnsafeZones(self, fig=None, ax=None):
        assert self.all_params_ellipse is not None, "There are no ellipses to plot!"

        if fig==None or ax==None:
            fig=plt.figure()
            ax = fig.add_subplot()
        else:
            x0, y0, sigma_x, sigma_y = self.all_params_ellipse
            ellipse = Ellipse(xy=(x0, y0), width=sigma_x, height=sigma_y, edgecolor='black', fc='None', lw=2)
            ax.add_patch(ellipse)
        
        return fig, ax

    
if __name__=='__main__':
    # user inputs
    experiment = 1      # 1: experiment with reactive robot behaviour
                        # 2: experiment with predictive robot behaviour

    time_start_perc = 0.3   # percentage of the experiment to be considered as "start"
    time_end_perc = 0.5     # percentage of the experiment to be considered as "end"

    # instantiate robot model
    kuka = ERobot.URDF('/home/itbellix/Desktop//GitHub/iiwa_ros/src/iiwa_description/urdf/iiwa7.urdf.xacro')

    # STEP 1: define the strain map used in the experiments
    # definition of parameters for artificial strain map and ellipse
    x0 = 40                 # PE center in degrees
    y0 = 90                 # SE center in degrees
    sigma_x = 35            # standard deviation along PE in degrees
    sigma_y = 25            # standard deviation along SE in degrees
    amplitude = 5           # max strain at the top
    strain_offset = 0       # constant offset to elevate strain-map
    strain_threshold = 1.5  # strain threshold to define risky zones

    all_params_gaussians = np.array([amplitude, x0, y0, sigma_x, sigma_y, strain_offset])
    all_params_ellipse = np.array([x0, y0, 2*sigma_x, 2*sigma_y])
    sm = strainMap()
    sm.setGaussianParams(all_params_gaussians)
    sm.setEllipseParams(all_params_ellipse)
    sm.calcLandscape(sm.all_params_gaussians)

    # fig_sm, ax_sm = sm.plotLandscape()
    # fig_sm, ax_sm = sm.plotUnsafeZones(fig_sm, ax_sm)

    # STEP 2: extract values from the ROS bags
    bag_file_name = 'reactUnsafe_0.bag'
    bag_file_name = 'expetiment.bag'

    # list of variables we are interested in
    estimated_shoulder_state = None
    xyz_ref = None
    stiffness_ref = None
    damping_ref = None
    xyz_curr = None
    z_uncompensated = None
    angvec_ref = None
    angvec_curr = None
    optimal_trajectory = None
    joint_pos = None
    joint_vol = None
    joint_eff = None
    external_force = None

    # Analyze what the bag containss
    with rosbag.Bag(os.path.join('bags', bag_file_name), 'r') as bag:
        # Get info about topics
        topic_info = bag.get_type_and_topic_info()

        # Print out details about each topic
        for topic, info in topic_info.topics.items():
            print(f"Topic: {topic}")
            print(f"  Message Type: {info.msg_type}")
            print(f"  Message Count: {info.message_count}")
            if info.frequency is not None:
                print(f"  Frequency: {info.frequency:.2f} Hz")
            else:
                print(f"  Frequency: NaN Hz")


        # extracting estimated shoulder poses
        print('\n\nExtracting estimated shoulder poses')
        for _, msg, time_msg in bag.read_messages(topics=['/estimated_shoulder_pose']):
            timestamp_est = time_msg.to_time()
            if estimated_shoulder_state is None:
                estimated_shoulder_state = [msg.data + (timestamp_est,)]
            else:
                estimated_shoulder_state = np.vstack((estimated_shoulder_state, [msg.data + (timestamp_est,)]))

        # extracting reference ee cartesian poses
        # Note that this message can carry also the stiffness and damping for the cartesian impedance controller
        # We need to check its dimensions to extract all the info
        print('Extracting reference ee cartesian poses')
        for _, msg, time_msg in bag.read_messages(topics=['/cartesian_ref_ee']):
            if xyz_ref is None:
                timestamps_ref = time_msg.to_time()
                data_msg = np.reshape(msg.data[0:16], (4,4))
                xyz_ref = np.hstack((data_msg[0:3, 3], timestamps_ref))
                angle, vector = SO3(data_msg[0:3, 0:3]).angvec(unit='deg')
                angvec_ref = np.hstack((angle, vector, timestamps_ref))

            else:
                timestamps_ref = time_msg.to_time()
                data_msg = np.reshape(msg.data[0:16], (4,4))
                xyz_ref = np.vstack((xyz_ref, np.hstack((data_msg[0:3, 3], timestamps_ref))))
                angle, vector = SO3(data_msg[0:3, 0:3]).angvec(unit='deg')
                angvec_ref = np.vstack((angvec_ref, np.hstack((angle, vector, timestamps_ref))))

            if stiffness_ref is None:
                if len(msg.data) == 22:     # cartesian stiffness has been specified, with critical damping
                    stiffness_ref = np.hstack((np.reshape(msg.data[16:22], (6, 1)), timestamps_ref))
                    damping_ref = np.hstack((2* np.sqrt(np.reshape(msg.data[16:22]), (6, 1)), timestamps_ref))

                if len(msg.data) == 28:     # cartesian stiffness and damping have been specified
                    stiffness_ref = np.hstack((msg.data[16:22], timestamps_ref))
                    damping_ref = np.hstack((msg.data[22:28], timestamps_ref))
            else:

                if len(msg.data) == 22:     # cartesian stiffness has been specified, with critical damping
                    stiffness_ref = np.vstack((stiffness_ref, np.hstack((msg.data[16:22], timestamps_ref))))
                    damping_ref = np.vstack((damping_ref, np.hstack((2* np.sqrt(msg.data[16:22]), timestamps_ref))))

                if len(msg.data) == 28:     # cartesian stiffness and damping have been specified
                    stiffness_ref = np.vstack((stiffness_ref, np.hstack((msg.data[16:22], timestamps_ref))))
                    damping_ref = np.vstack((damping_ref, np.hstack((msg.data[22:28], timestamps_ref))))

        # extracting actual ee cartesian poses
        print('Extracting actual ee cartesian poses')
        for _, msg, time_msg in bag.read_messages(topics=['/iiwa7/ee_cartesian_pose']):
            time_curr = time_msg.to_time()
            if time_curr >= xyz_ref[0, -1]:     # only interested in the poses after there is a reference to follow
                if xyz_curr is None:
                    timestamps_curr = time_msg.to_time()
                    data_msg = np.reshape(msg.pose, (4,4))
                    xyz_curr = np.hstack((data_msg[0:3, 3], timestamps_curr))
                    angle, vector = SO3(data_msg[0:3, 0:3]).angvec(unit='deg')
                    angvec_curr = np.hstack((angle, vector, timestamps_curr))
                else:
                    timestamps_curr = time_msg.to_time()
                    data_msg = np.reshape(msg.pose, (4,4))
                    xyz_curr = np.vstack((xyz_curr, np.hstack((data_msg[0:3, 3], timestamps_curr))))
                    angle, vector = SO3(data_msg[0:3, 0:3]).angvec(unit='deg')
                    angvec_curr = np.vstack((angvec_curr, np.hstack((angle, vector, timestamps_curr))))

        # extracting output of the trajectory optimization
        print('Extracting optimization outputs')
        for _, msg, time_msg in bag.read_messages(topics=['/optimization_output']):
            data = np.reshape(msg.data[0:-1], (6, 11))
            time_opt = time_msg.to_time()
            if optimal_trajectory is None:
                optimal_trajectory = np.hstack((data[0:6,:], time_opt * np.ones((6,1))))
            else:
                optimal_trajectory = np.vstack((optimal_trajectory, np.hstack((data[0:6,:], time_opt * np.ones((6,1))))))


        # extracting the joint state values
        print("Extracting joint values")
        for _, msg, time_msg in bag.read_messages(topics=['/iiwa7/joint_states']):
            time_joint = time_msg.to_time()


            # here we also EXTRACT CONTACT FORCE MAGNITUDE
            # considering the full dynamic model of the robot:
            # tau = M(q) * q_ddot + C(q, q_dot) * q_dot + G(q) + J^T * F_ext  -> we want the F_ext!
            # 
            # However, the Kuka already runs its own gravity compensation, so effectively the term G(q) should be cancelled,
            # as tau includes it already. Moreover, if we assume minimal acceleration, we can disregard M(q):
            # F_ext = (J^T)^(-1) * (tau - C(q, q_dot) * q_dot)
            # 
            # Let's use the robotics toolbox functionalities to calculate it!
            J = kuka.jacob0(msg.position)
            J_inv = np.linalg.pinv(J)
            # C_qq_dot = kuka.coriolis(np.array(msg.position), np.array(msg.velocity))      # TODO: coriolis method does not work
            # wrench_ee = J_inv.transpose() @ (np.array(msg.effort) - C_qq_dot @ np.array(msg.velocity))
            wrench_ee = J_inv.transpose() @ np.array(msg.effort)

            if time_joint >= xyz_ref[0, -1]:
                if joint_pos is None:
                    joint_pos = np.hstack((msg.position, time_joint))
                    joint_vel = np.hstack((msg.velocity, time_joint))
                    joint_eff = np.hstack((msg.effort, time_joint))
                    external_force = np.hstack((wrench_ee, time_joint))
                else:
                    joint_pos = np.vstack((joint_pos, np.hstack((msg.position, time_joint))))
                    joint_vel = np.vstack((joint_vel, np.hstack((msg.position, time_joint))))
                    joint_eff = np.vstack((joint_eff, np.hstack((msg.position, time_joint))))
                    external_force = np.vstack((external_force, np.hstack((wrench_ee, time_joint))))


    # Now, let's filter the data to retain only the interesting part of the experiment
    # (i.e., when the subject is wearing the brace properly and the robot is moving)
    init_time = xyz_curr[int(xyz_curr.shape[0]*time_start_perc), -1]            # identify initial time-step
    end_time = xyz_curr[int(xyz_curr.shape[0]*time_end_perc), -1] - init_time   # identify final time-step
    
    # variables used for plotting (strictly speaking, this is useless, but speeds up debugging!)
    estimated_shoulder_state_plot = estimated_shoulder_state
    xyz_curr_plot = xyz_curr
    xyz_ref_plot = xyz_ref
    stiffness_ref_plot = stiffness_ref
    damping_ref_plot = damping_ref
    angvec_curr_plot = angvec_curr
    angvec_ref_plot = angvec_ref
    z_uncompensated_plot = z_uncompensated
    optimal_trajectory_plot = optimal_trajectory
    joint_pos_plot = joint_pos
    joint_vel_plot = joint_vel
    joint_eff_plot = joint_eff

    estimated_shoulder_state_plot[:,-1] = estimated_shoulder_state_plot[:,-1] - init_time   # center time values starting at initial time
    xyz_curr_plot[:,-1] = xyz_curr_plot[:,-1] - init_time
    xyz_ref_plot[:,-1] = xyz_ref_plot[:,-1] - init_time
    stiffness_ref_plot[:,-1] = stiffness_ref_plot[:,-1] - init_time
    damping_ref_plot[:,-1] = damping_ref_plot[:,-1] - init_time
    angvec_curr_plot[:,-1] = angvec_curr_plot[:,-1] - init_time
    angvec_ref_plot[:,-1] = angvec_ref_plot[:,-1] - init_time
    joint_pos_plot[:,-1] = joint_pos_plot[:,-1] - init_time
    joint_vel_plot[:,-1] = joint_vel_plot[:,-1] - init_time
    joint_eff_plot[:,-1] = joint_eff_plot[:,-1] - init_time


    if z_uncompensated_plot is not None:
        z_uncompensated_plot[:,-1] = z_uncompensated_plot[:,-1] - init_time

    if optimal_trajectory_plot is not None:
        optimal_trajectory_plot[:,:,-1] = optimal_trajectory_plot[:,:,-1] - init_time

    estimated_shoulder_state_plot = estimated_shoulder_state_plot[(estimated_shoulder_state_plot[:,-1]>0) & (estimated_shoulder_state_plot[:,-1]<end_time)]
    xyz_curr_plot = xyz_curr_plot[(xyz_curr_plot[:,-1]>0) & (xyz_curr_plot[:,-1]<end_time)]    # retain data after initial time
    xyz_ref_plot = xyz_ref_plot[(xyz_ref_plot[:,-1]>0) & (xyz_ref_plot[:,-1]<end_time)]
    stiffness_ref_plot = stiffness_ref_plot[(stiffness_ref_plot[:,-1]>0) & (stiffness_ref_plot[:,-1]<end_time)]
    damping_ref_plot = damping_ref_plot[(damping_ref_plot[:,-1]>0) & (damping_ref_plot[:,-1]<end_time)]
    angvec_curr_plot = angvec_curr_plot[(angvec_curr_plot[:,-1]>0) & (angvec_curr_plot[:,-1]<end_time)]
    angvec_ref_plot = angvec_ref_plot[(angvec_ref_plot[:,-1]>0) & (angvec_ref_plot[:,-1]<end_time)]
    joint_pos_plot = joint_pos_plot[(joint_pos_plot[:,-1]>0) & (joint_pos_plot[:,-1]<end_time)]
    joint_vel_plot = joint_vel_plot[(joint_vel_plot[:,-1]>0) & (joint_vel_plot[:,-1]<end_time)]
    joint_eff_plot = joint_eff_plot[(joint_eff_plot[:,-1]>0) & (joint_eff_plot[:,-1]<end_time)]
    
    if z_uncompensated_plot is not None:
        z_uncompensated_plot = z_uncompensated_plot[(z_uncompensated_plot[:,-1]>0) & (z_uncompensated_plot[:,-1]<end_time)]

    # now let's plot the figures
    # TRAJECTORY ON THE STRAIN MAP
    fig_sm, ax_sm = sm.plotLandscape()
    fig_sm, ax_sm = sm.plotUnsafeZones(fig_sm, ax_sm)
    ax_sm.plot(np.rad2deg(estimated_shoulder_state_plot[:, 0]), np.rad2deg(estimated_shoulder_state_plot[:, 2]), c='blue')

    # EVOLUTION OF STIFFNESS AND DAMPING
    fig_sd = plt.figure()
    ax_sd1 = fig_sd.add_subplot(411)
    ax_sd1.plot(stiffness_ref_plot[:,-1], stiffness_ref_plot[:,0], label='trans stiff')
    ax_sd2 = fig_sd.add_subplot(412)
    ax_sd2.plot(stiffness_ref_plot[:,-1], stiffness_ref_plot[:,4], label='rot stiff')
    ax_sd3 = fig_sd.add_subplot(413)
    ax_sd3.plot(damping_ref_plot[:,-1], damping_ref_plot[:,0], label='trans damp')
    ax_sd4 = fig_sd.add_subplot(414)
    ax_sd4.plot(damping_ref_plot[:,-1], damping_ref_plot[:,4], label='rot damp')
    fig_sd.legend()
    fig_sd.suptitle("Stiffness and Damping")

    # EVOLUTION OF THE INTERACTION FORCE
    force_magnitude = np.sqrt(np.sum(external_force[:, :3]**2, axis=1))
    fig_f = plt.figure()
    ax_f = fig_f.add_subplot()
    ax_f.plot(force_magnitude)

