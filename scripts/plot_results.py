import rosbag
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import os
from spatialmath import SO3
from biomechanical_safe_deflection.lbr_iiwa_robot_model import LBR7_iiwa_ros_DH

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
            ellipse = Ellipse(xy=(x0, y0), width=sigma_x, height=sigma_y, edgecolor='black', fc='black', lw=2, alpha=0.1, fill = True)
            ax.add_patch(ellipse)
        
        return fig, ax

    
if __name__=='__main__':
    # user inputs
    experiment = 1      # 1: experiment with reactive robot behaviour
                        # 2: experiment with predictive robot behaviour

    time_start_perc = 0.1   # percentage of the experiment to be considered as "start"
    time_end_perc = 0.9

    # instantiate robot model
    kuka = LBR7_iiwa_ros_DH()

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

    # STEP 2: extract values from the ROS bags
    bag_file_name = 'deflection_interpTraj.bag'

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
    optimal_trajectory_pe = None
    optimal_trajectory_se = None
    joint_pos = None
    joint_vol = None
    joint_eff = None
    external_force_minGq = None
    external_force = None
    cmd_torques = None

    timestamps_init = 0
    

    # Analyze what the bag contains
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

        # extracting the reference request (only first time stamp, can be used later to disregard useless stuff)
        for _, msg, time_msg in bag.read_messages(topics=['/request_reference']):
            timestamps_init = time_msg.to_time()
            break   # we need to get the first one only

        # extracting reference ee cartesian poses
        # Note that this message can carry also the stiffness and damping for the cartesian impedance controller
        # We need to check its dimensions to extract all the info
        print('\n\nExtracting reference ee cartesian poses')
        for _, msg, time_msg in bag.read_messages(topics=['/cartesian_ref_ee']):
            timestamps_ref = time_msg.to_time()

            if timestamps_ref>=timestamps_init:
                if xyz_ref is None:
                    data_msg = np.reshape(msg.data[0:16], (4,4))
                    xyz_ref = np.hstack((data_msg[0:3, 3], timestamps_ref))
                    angle, vector = SO3(data_msg[0:3, 0:3]).angvec(unit='deg')
                    angvec_ref = np.hstack((angle, vector, timestamps_ref))

                else:
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

        # extracting estimated shoulder poses
        print('Extracting estimated shoulder poses')
        for _, msg, time_msg in bag.read_messages(topics=['/estimated_shoulder_pose']):
            timestamp_est = time_msg.to_time()
            if timestamp_est >= xyz_ref[0, -1]:
                if estimated_shoulder_state is None:
                    estimated_shoulder_state = [msg.data + (timestamp_est,)]
                else:
                    estimated_shoulder_state = np.vstack((estimated_shoulder_state, [msg.data + (timestamp_est,)]))

        # extracting actual ee cartesian poses
        print('Extracting actual ee cartesian poses')
        for _, msg, time_msg in bag.read_messages(topics=['/iiwa7/ee_cartesian_pose']):
            time_curr = time_msg.to_time()
            if time_curr >= xyz_ref[0, -1] and time_curr >=timestamps_init:     # only interested in the poses after there is a reference to follow
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
            data = np.reshape(msg.data, (6, 11))
            time_opt = time_msg.to_time()
            if time_opt>=xyz_ref[0,-1] and time_opt>=timestamps_init:
                if optimal_trajectory is None:
                    optimal_trajectory = np.hstack((data, time_opt * np.ones((6,1))))
                    optimal_trajectory_pe = np.hstack((data[0, :], time_opt))
                    optimal_trajectory_se = np.hstack((data[2, :], time_opt))
                else:
                    optimal_trajectory = np.vstack((optimal_trajectory, np.hstack((data, time_opt * np.ones((6,1))))))
                    optimal_trajectory_pe = np.vstack((optimal_trajectory_pe, np.hstack((data[0,:], time_opt))))
                    optimal_trajectory_se = np.vstack((optimal_trajectory_se, np.hstack((data[2,:], time_opt))))


        # extracting the commanded forces to the CIC
        print("Extracting commanded torques")
        for _, msg, time_msg in bag.read_messages(topics=['/iiwa7/TorqueController/command']):
            time_cf = time_msg.to_time()
            if time_cf>=xyz_ref[0,-1] and time_cf>=timestamps_init:
                if cmd_torques is None:
                    cmd_torques = np.hstack((msg.data, time_cf))
                else:
                    cmd_torques = np.vstack((cmd_torques, np.hstack((msg.data, time_cf))))

        # extracting the joint state values
        print("Extracting joint values")
        for _, msg, time_msg in bag.read_messages(topics=['/iiwa7/joint_states']):
            time_joint = time_msg.to_time()

            if time_joint >= xyz_ref[0, -1] and time_joint>=timestamps_init:
                # We extract the commanded EE force from the commanded torques.
                if joint_pos is None:
                    joint_pos = np.hstack((msg.position, time_joint))
                else:
                    joint_pos = np.vstack((joint_pos, np.hstack((msg.position, time_joint))))

    # Now let's compute the commanded interaction force
    # Extract timestamps (last column)
    timestamps_torques = cmd_torques[:, -1]
    timestamps_joints = joint_pos[:, -1]

    # Initialize lists for aligned rows
    aligned_cmd_torques = []
    aligned_joint_pos = []

    # Iterate through torque timestamps to find closest joint timestamps
    for idx_torque, time_torque in enumerate(timestamps_torques):
        # Find the index of the closest joint timestamp
        closest_idx_joint = np.argmin(np.abs(timestamps_joints - time_torque))
        time_joint = timestamps_joints[closest_idx_joint]

        # Add rows where the timestamp difference is minimal
        if np.abs(time_torque - time_joint) <= 0.001:  # Optional: Add a tolerance if needed
            aligned_cmd_torques.append(cmd_torques[idx_torque])
            aligned_joint_pos.append(joint_pos[closest_idx_joint])

    # Convert lists back to numpy arrays
    aligned_cmd_torques = np.array(aligned_cmd_torques)[::2, :]
    aligned_joint_pos = np.array(aligned_joint_pos)[::2, :]

    external_force = np.zeros((aligned_cmd_torques.shape[0], 7))
    # Loop through them to extract the commanded contact force
    for instant in range(aligned_cmd_torques.shape[0]):
        J = kuka.jacob0(aligned_joint_pos[instant, 0:-1])
        J_inv = np.linalg.pinv(J)
        external_force[instant,:] =np.concatenate((np.matmul(J_inv.transpose(), aligned_cmd_torques[instant, 0:-1]), np.array([aligned_cmd_torques[instant, -1]]))) 

    external_force = np.array(external_force)
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
    optimal_trajectory_pe_plot = optimal_trajectory_pe
    optimal_trajectory_se_plot = optimal_trajectory_se
    joint_pos_plot = joint_pos
    external_force_plot = external_force
    external_force_minGq_plot = external_force_minGq

    estimated_shoulder_state_plot[:,-1] = estimated_shoulder_state_plot[:,-1] - init_time   # center time values starting at initial time
    xyz_curr_plot[:,-1] = xyz_curr_plot[:,-1] - init_time
    xyz_ref_plot[:,-1] = xyz_ref_plot[:,-1] - init_time
    angvec_curr_plot[:,-1] = angvec_curr_plot[:,-1] - init_time
    angvec_ref_plot[:,-1] = angvec_ref_plot[:,-1] - init_time
    joint_pos_plot[:,-1] = joint_pos_plot[:,-1] - init_time
    external_force_plot[:,-1] = external_force_plot[:,-1] - init_time

    if stiffness_ref_plot is not None:
        stiffness_ref_plot[:,-1] = stiffness_ref_plot[:,-1] - init_time
        damping_ref_plot[:,-1] = damping_ref_plot[:,-1] - init_time

    if z_uncompensated_plot is not None:
        z_uncompensated_plot[:,-1] = z_uncompensated_plot[:,-1] - init_time

    if optimal_trajectory_plot is not None:
        optimal_trajectory_plot[:,-1] = optimal_trajectory_plot[:,-1] - init_time
        optimal_trajectory_pe_plot[:,-1] = optimal_trajectory_pe_plot[:,-1] - init_time
        optimal_trajectory_se_plot[:,-1] = optimal_trajectory_se_plot[:,-1] - init_time

    estimated_shoulder_state_plot = estimated_shoulder_state_plot[(estimated_shoulder_state_plot[:,-1]>0) & (estimated_shoulder_state_plot[:,-1]<end_time)]
    xyz_curr_plot = xyz_curr_plot[(xyz_curr_plot[:,-1]>0) & (xyz_curr_plot[:,-1]<end_time)]    # retain data after initial time
    xyz_ref_plot = xyz_ref_plot[(xyz_ref_plot[:,-1]>0) & (xyz_ref_plot[:,-1]<end_time)]
    angvec_curr_plot = angvec_curr_plot[(angvec_curr_plot[:,-1]>0) & (angvec_curr_plot[:,-1]<end_time)]
    angvec_ref_plot = angvec_ref_plot[(angvec_ref_plot[:,-1]>0) & (angvec_ref_plot[:,-1]<end_time)]
    joint_pos_plot = joint_pos_plot[(joint_pos_plot[:,-1]>0) & (joint_pos_plot[:,-1]<end_time)]
    external_force_plot = external_force_plot[(external_force_plot[:,-1]>0) & (external_force_plot[:,-1]<end_time)]
    
    if optimal_trajectory_pe_plot is not None:
        optimal_trajectory_pe_plot = optimal_trajectory_pe_plot[(optimal_trajectory_pe_plot[:,-1]>0) & (optimal_trajectory_pe_plot[:,-1]<end_time)]
        optimal_trajectory_se_plot = optimal_trajectory_se_plot[(optimal_trajectory_se_plot[:,-1]>0) & (optimal_trajectory_se_plot[:,-1]<end_time)]

    if z_uncompensated_plot is not None:
        z_uncompensated_plot = z_uncompensated_plot[(z_uncompensated_plot[:,-1]>0) & (z_uncompensated_plot[:,-1]<end_time)]

    if stiffness_ref_plot is not None:
        stiffness_ref_plot = stiffness_ref_plot[(stiffness_ref_plot[:,-1]>0) & (stiffness_ref_plot[:,-1]<end_time)]
        damping_ref_plot = damping_ref_plot[(damping_ref_plot[:,-1]>0) & (damping_ref_plot[:,-1]<end_time)]

    # now let's plot the figures
    # TRAJECTORY ON THE STRAIN MAP
    init_state = estimated_shoulder_state_plot[np.abs(estimated_shoulder_state_plot[:,-1] - optimal_trajectory_pe_plot[0, -1]) < 0.0025]
    initial_state = init_state[:,0:4]
    initial_state = initial_state.transpose()
    future_states = np.zeros((4, 10+1))
    future_states[0, 0] = initial_state[0]
    future_states[2, 0] = initial_state[2]
    future_states[1::2, :] = initial_state[1::2]
    for timestep in range(1, 11):
            # retrieve estimation for future human state at current time step (assuming constant velocity)
            future_states[::2, timestep] = future_states[::2, timestep-1] + 1/10 * future_states[1::2, timestep-1]


    fig_sm, ax_sm = sm.plotLandscape()
    fig_sm, ax_sm = sm.plotUnsafeZones(fig_sm, ax_sm)
    ax_sm.plot(np.rad2deg(estimated_shoulder_state_plot[:, 0]), np.rad2deg(estimated_shoulder_state_plot[:, 2]), c='blue', linewidth=2)
    ax_sm.plot(np.rad2deg(optimal_trajectory_pe_plot[0, 0:-1]), np.rad2deg(optimal_trajectory_se_plot[0, 0:-1]), c='green', linewidth=2)
    ax_sm.scatter(np.rad2deg(future_states[0,:]), np.rad2deg(future_states[2,:]), c='blue')

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
    force_magnitude = np.sqrt(np.sum(external_force_plot[:, :3]**2, axis=1))
    fig_f = plt.figure()
    ax_f = fig_f.add_subplot()
    ax_f.plot(external_force_plot[:,-1], force_magnitude, c = "red", label="Commanded")
    ax_f.set_ylabel('Force [N]')
    ax_f.set_xlabel('time [s]')
    ax_f.legend()
    fig_f.suptitle('Commanded EE force')

    # EVOLUTION OF THE ERRORS
    # POSITION ERROR
    min_length = np.min((xyz_ref_plot.shape[0], xyz_curr_plot.shape[0]))

    fig_err_pos = plt.figure()
    ax = fig_err_pos.add_subplot(321)
    ax.plot(xyz_curr_plot[:,-1], xyz_curr_plot[:,0], label = 'x_curr', color='black')
    ax.plot(xyz_ref_plot[:,-1], xyz_ref_plot[:,0], label = 'x_ref', color='black', linestyle='dashed')
    ax.set_ylabel('[m]')
    ax.legend()
    ax = fig_err_pos.add_subplot(322)
    ax.plot(xyz_curr_plot[0:min_length,-1], xyz_ref_plot[0:min_length,0] - xyz_curr_plot[0:min_length,0], label = 'x_err')
    ax.set_ylabel('[m]')
    ax.legend()
    ax = fig_err_pos.add_subplot(323)
    ax.plot(xyz_curr_plot[:,-1], xyz_curr_plot[:,1], label = 'y_curr', color='black')
    ax.plot(xyz_ref_plot[:,-1], xyz_ref_plot[:,1], label = 'y_ref', color='black', linestyle='dashed')
    ax.set_ylabel('[m]')
    ax.legend()
    ax = fig_err_pos.add_subplot(324)
    ax.plot(xyz_curr_plot[0:min_length,-1], xyz_ref_plot[0:min_length,1] - xyz_curr_plot[0:min_length,1], label = 'y_err')
    ax.set_ylabel('[m]')
    ax.legend()
    ax = fig_err_pos.add_subplot(325)
    ax.plot(xyz_curr_plot[:,-1], xyz_curr_plot[:,2], label = 'z_curr', color='black')
    ax.plot(xyz_ref_plot[:,-1], xyz_ref_plot[:,2], label = 'z_ref', color='black', linestyle='dashed')
    ax.set_ylabel('[m]')
    ax.legend()
    ax = fig_err_pos.add_subplot(326)
    ax.plot(xyz_curr_plot[0:min_length,-1], xyz_ref_plot[0:min_length,2] - xyz_curr_plot[0:min_length,2], label = 'z_err')
    ax.set_ylabel('[m]')
    ax.legend()

    fig_err_pos.suptitle("EE cartesian position")

    # ORIENTATION ERROR (AS AXIS-ANGLE)
    abs_time_diff = np.abs(angvec_ref_plot[0:min_length,-1] - angvec_curr_plot[0:min_length,-1])

    orientation_mismatch = np.rad2deg(np.arccos(np.sum(angvec_ref_plot[0:min_length, 1:4] * angvec_curr_plot[0:min_length, 1:4], axis=1) /
                            (np.linalg.norm(angvec_ref_plot[0:min_length, 1:4], axis=1) * np.linalg.norm(angvec_curr_plot[0:min_length, 1:4], axis=1))))

    angle_mismatch = angvec_ref_plot[0:min_length, 0] - angvec_curr_plot[0:min_length, 0]

    fig_err_ori = plt.figure()
    ax = fig_err_ori.add_subplot(211)
    ax.plot(angvec_ref_plot[0:min_length, -1], orientation_mismatch, label='axis mismatch')
    ax.set_ylabel('[deg]')
    ax.legend()
    ax = fig_err_ori.add_subplot(212)
    ax.plot(angvec_ref_plot[0:min_length, -1], angle_mismatch, label='angle mismatch')
    ax.set_ylabel('[deg]')
    ax.set_xlabel('time [s]')
    ax.legend()
    fig_err_ori.suptitle("EE orientation error")

    plt.show(block=True)
