# Biomechanic-aware shared control for robotic-assisted physiotherapy

In this repository, we collect the code associated to our publication "A Shared Control Approach to Safely Limiting Patient Motion Based on Tendon Strain During Robotic-Assisted Shoulder Rehabilitation", to appear at the IEEE International Consortium on Rehabilitation Robotics 2025 (ICORR).

We provide the scripts we used to control our robotic physiotherapist (KUKA iiwa 7), as well as the basic plotting scripts employed to generate the images in the paper. Moreover, we provide a link to the rosbags containing our experimental data.

### Paper
For an explanation of our approach, we refer to our paper:


If you find our work useful, :star: and citations are appreciated!

<table align="center">
  <tr>
    <td colspan="2" align="center">Funding Institutions</td>
  </tr>
  <tr>
    <td align="center">
      <a>
        <img src="https://user-images.githubusercontent.com/50029203/226883398-97b28065-e144-493b-8a6c-5cbbd9000411.png" alt="TUD logo" height="128">
        <br />
        <a href="https://www.tudelft.nl/3me/over/afdelingen/biomechanical-engineering">Biomechanical Engineering</a> and <br />
        <a href="https://www.tudelft.nl/3me/over/afdelingen/cognitive-robotics-cor">Cognitive Robotics</a> at TU Delft</p>
      </a>
    </td>
    <td align="center">
      <a href="https://chanzuckerberg.com/">
        <img src="https://user-images.githubusercontent.com/50029203/226883506-fbb59348-38a4-43f9-93c9-2c7b8ba63619.png" alt="CZI logo" width="128" height="128">
        <br />
        Chan Zuckerberg Initiative
      </a>
    </td>
  </tr>
</table>

### Dependencies
Our code depends on:
- the Robot Operating System (ROS1)
- the impedance controller implemented at https://gitlab.tudelft.nl/kuka-iiwa-7-cor-lab/iiwa_ros/
- several other (python) packages, documented in the `requirements.txt`
- (we also depend indirectly from the OpenSim project and OpenSimAD, which we use to generate our differentiable models of the human shoulder joint)

### Code
The code is structured as a ROS package to simplify its execution. The most important scripts are:
- `biomech_safety_net.py`, implementing the bulk of the work (estimating the most suitable robotic intervention to guarantee low-strain rehabilitation exercises for the shoulder complex, on the basis of trajectory optimization);
- `robot_control.py`, which takes care of initializing the communication with the robot, sending it to the initial position and estimating the current human pose;
- `plot_results.py`, which can be run to analyze one of the rosbags we provide below (some parameters need to be adjusted, such as the interval of interest for the analysis).

The `launch` folder contains the launch files and a readme on how to use them.

### Experimental data
The experimental data to reproduce our figures is available at the following DOI: 10.4121/2016260b-d4bb-4b4e-94d5-e2e177913e11

### Contributing and trouble-shooting
If you would like to contribute to this project or have troubles getting things to work, just reach out or open an issue!

https://github.com/user-attachments/assets/fb8b568c-e494-41ca-8cb9-3e3e3a0cb629
