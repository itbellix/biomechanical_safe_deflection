This folder contains the launch file that starts the ROS master, together with the Cartesian impedance controller contained in the `cor_tud_controllers` folder contained in the `iiwa_ros` package. Please follow the instructions in the main README of this repository to obtain it.

Example usage:
- `roslaunch controller.launch simulation:=true` (for starting the Gazebo simulation environment)
- `roslaunch controller.launch simulation:=false` (for working directly with the robot, once a connection has been established)


For examples on how to work and connect with the real Kuka LBR iiwa 7, please follow the instructions at https://gitlab.tudelft.nl/kuka-iiwa-7-cor-lab/how-to/-/blob/726083cc72a70ca5879e9f7a6fb42bdcd67bc2f1/kuka/README.md