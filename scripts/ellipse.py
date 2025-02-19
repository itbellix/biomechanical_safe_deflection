"""
This class implements an ellipse
"""

class Ellipse():
    def __init__(self, xc, yc, a, b) -> None:
        """
        Initializes the ellipse parameters.
        The formula of the generic point belonging to the ellipse is:
        (x-xc)^2/a^2+(y-yc)^2/b^2 = 1

        TODO: take into account orientations too?
        """
        self.xc = xc            # x coordinate of the center of the ellipse
        self.yc = yc            # y coordinate of the center of the ellipse
        self.a = a              # semi-diameter
        self.b = b              # semi-diameter
        self.theta = 0          # orientation of the ellipse

    def closest_point_1quad(self, xp, yp):
        """
        This function finds the point on the ellipse which is closer to (xp, yp).
        It computes the distance from the point (xp, yp) to the ellipse, 
        by re-projecting the point in the first quadrant, and then minimizes it 
        to find the corresponding desired point.
        """
        pass