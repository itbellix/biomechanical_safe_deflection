#!/usr/bin/env python

# This is a server that allows to update parameters of the ROS parameter serve as the code is being executed

import rospy

from dynamic_reconfigure.server import Server
from biomechanical_safe_deflection.cfg import BSDConfig
# Note, in the line above, that the name of the .cfg file is Tutorials.cfg, but we need to import TutorialsConfig
# otherwise it will not work!

def callback(config, level):
    """
    This callback sets parameters of the ROS Parameter Server, so that
    the modifications are accessible to other nodes too. This is done specifically
    with the return command. Before, we can print something to make the user happy!
    """
    rospy.loginfo("""Reconfigure Request: {int_param}, {double_param},\ 
          {str_param}, {bool_param}, {size}""".format(**config))
    return config

if __name__ == "__main__":
    rospy.init_node("param_updater", anonymous = False)

    srv = Server(BSDConfig, callback)
    rospy.spin()