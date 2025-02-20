This folder contains the launch file that starts the ROS master, together with the Cartesian impedance controller contained in the `cor_tud_controllers` folder contained in the `iiwa_ros` package. Please follow the instructions in the main README of this repository to obtain it.

Example usage:
1. open 2 terminals
2. source your ROS distribution in both terminals (tested with ROS 1 Noetic)
3. source the `iiwa_ros` package (available at https://gitlab.tudelft.nl/kuka-iiwa-7-cor-lab/iiwa_ros/)
4. on one terminal, launch the impedance controller implemented in `iiwa_ros` by running:   
    - `roslaunch controller.launch simulation:=true` (for starting the Gazebo simulation environment)
    - `roslaunch controller.launch simulation:=false` (for working directly with the robot, once a connection has been established)
5. on the other terminal, launch all the scripts to send actual references to the robot based on the estimated human pose, underlying musculoskeletal computations, and for visualization and behavior selection by running `roslaunch start_all.launch`. There are several options available:
    - you can specify whether to run in simulation or not with the boolean argument `simulation:=true/false`;
    - you can specify where to record rosbags with the arguments `record:=true/false` and `bag_name:=$(your_bag_name)`. Note that the bag is created in the `bags` folder of this repository;
    - you can specify the virtual environment to be used with `venv:=$(path_to_your_venv)`. Dependencies can be install there from the `requirements.txt` provided in the repo.

When running in simulation, you should have Gazebo Classic 11 firing up with the robot model (Kuka 7) in the scene, and a dynamic reconfigure window allowing you to specify the behavior of the robot. Note that by default the position of the human shoulder (i.e., the center of the glenohumeral joint) is estimated based on the pose of the end-effector. This works well for the real experiments, but causes visible drifts in the simulation environment.


For examples on how to work and connect with the real Kuka LBR iiwa 7, please follow the instructions at https://gitlab.tudelft.nl/kuka-iiwa-7-cor-lab/how-to/-/blob/726083cc72a70ca5879e9f7a6fb42bdcd67bc2f1/kuka/README.md