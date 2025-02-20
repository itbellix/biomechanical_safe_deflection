from roboticstoolbox import DHRobot, RevoluteMDH
import numpy as np

# Define the robot model as in the Robotic Toolbox
class LBR7_iiwa_ros_DH(DHRobot):
    """
    Class that imports a LBR URDF model
    ``LBR()`` is a class which imports a Kuka LBR robot definition
    from a URDF file.  The model describes its kinematic and graphical
    characteristics.
    .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.LBR()
        >>> print(robot)
    Defined joint configurations are:
    - qz, zero joint angle configuration, 'L' shaped configuration
    - qr, vertical 'READY' configuration
    - qs, arm is stretched out in the x-direction
    - qn, arm is at a nominal non-singular configuration
    .. codeauthor:: Jesse Haviland
    .. sectionauthor:: Peter Corke
    """

    def __init__(self):

        # deg = np.pi/180
        
        # This LBR model is defined using modified
        # Denavit-Hartenberg parameters
        L = [
            RevoluteMDH(
                a=0.,
                d=0.340,
                alpha=0.,
                I=[0.02183, 0.007703, 0.02083, 0., -0.003887, 0.],
                r=[0., -0.03, -0.07],
                m=3.4525,
                G=1,
                qlim=[-2.96706, 2.96706]
            ),

            RevoluteMDH(
                a=0.,
                d=0.,
                alpha=-np.pi/2.,
                I=[0.02076, 0.02179, 0.00779, 0., 0., 0.003626],
                r=[-0.0003, -0.059, 0.042],
                m=3.4821,
                G=1,
                qlim=[-2.094395, 2.094395]
            ),

            RevoluteMDH(
                a=0.,
                d=0.400,
                alpha=np.pi/2.,
                I=[0.03204, 0.00972, 0.03042, 0., 0.006227, 0.],
                r=[0., 0.03, -0.06],
                m=4.05623,
                G=1,
                qlim=[-2.96706, 2.96706]
            ),

            RevoluteMDH(
                a=0.,
                d=0.,
                alpha=np.pi/2.,
                I=[0.02178, 0.02075, 0.007785, 0., -0.003625, 0.],
                r=[0., 0.067, 0.034],
                m=3.4822,
                G=1,
                qlim=[-2.094395, 2.094395]
            ),

            RevoluteMDH(
                a=0.,
                d=0.400,
                alpha=-np.pi/2.,
                I=[0.01287, 0.005708, 0.01112, 0., 0.003946, 0.],
                r=[-0.0001, -0.021, -0.114],
                m=2.1633,
                G=1,
                qlim=[-2.96706, 2.96706]
            ),

            RevoluteMDH(
                a=0.,
                d=0.,
                alpha=-np.pi/2.,
                I=[0.006509, 0.006259, 0.004527, 0., -0.000319, -0.],
                r=[0., -0.0006, -0.0603],
                m=2.3466,
                G=1,
                qlim=[-2.094395, 2.094395]
            ),

            RevoluteMDH(
                a=0.,
                d=0.126,
                alpha=np.pi/2.,
                I=[0.01464, 0.01465, 0.002872, 0.000591, 0., 0.],
                r=[0., 0., -0.025],
                m=3.129,
                G=1,
                qlim=[-3.054326, 3.054326]
            )
        ]

        super().__init__(
            L,
            name='LBR7_iiwa_ros_DH',
            manufacturer="Kuka")

        self.addconfiguration("qz", np.array([0, 0, 0, 0, 0, 0, 0]))
        self.addconfiguration("qr", np.array([0, -0.3, 0, -1.9, 0, 1.5, np.pi / 4]))
