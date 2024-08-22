import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import pygame


class RealTimeStrainMapVisualizer:
    """
    This class is built to have a way to visualize the 2D strainmaps and update them in real-time
    """
    def __init__(self, X_norm, Y_norm, num_params_gaussian, pe_boundaries, se_boundaries):
        """
        Initialize the visualizer with the parameters that define
        """
        self.X_norm = X_norm
        self.Y_norm = Y_norm
        self.num_params_gaussian = num_params_gaussian
        self.ar_current = None          # current (rounded) value of the axial rotation, associated to the strainmap
        self.ar_prev = None             # previous (rounded) value of the axial rotation, associated to the strainmap
        self.pe_boundaries = pe_boundaries
        self.se_boundaries = se_boundaries

        self.act_current = None         # current (rounded) value of the activation
        self.act_prev = None            # previous (rounded) value of the activation

        # create a surface to hold the image of the strainmap
        self.image_surface = pygame.Surface(np.shape(X_norm))

        # define the bound values for the strain
        self.min_strain = 0
        self.max_strain = 8

        self.ellipse_params = None  # the parameters of the ellipses representing unsafe zones
                                    # (stored as: x0, y0, diameter_x, diameter_y)


        # dimensions of the resulting windows in pixels
        self.widow_dimensions = (800, 600)
        self.tick_length = 5                # length of axis ticks, in pixels
        self.width_lines = 3                # width of the lines, in pixels
        self.color_lines = (255, 255, 255)

        self.font_title_pygame = None
        self.font_title = 24
        self.font_ticks_pygame = None
        self.font_ticks = 18
        self.ticks_x = None

        self.debug = 0

        if self.debug:
            # Define a custom color gradient based on the data range
            num_colors = 256  # Adjust as needed
            self.custom_palette = np.zeros((num_colors, 3), dtype=np.uint8)

            # Adjust the color gradient to match the data distribution
            data_min = 0
            data_max = 7
            for i in range(num_colors):
                # Interpolate color components based on the data range
                value = (i / (num_colors - 1))  # Normalize value to range [0, 1]
                self.custom_palette[i, 0] = 255 * (value - data_min) / (data_max - data_min)  # Red component
                self.custom_palette[i, 1] = 0  # Green component (adjust as needed)
                self.custom_palette[i, 2] = 255 * (1 - (value - data_min) / (data_max - data_min))  # Blue component

            # Convert the custom palette to a list of tuples
            self.custom_palette = [tuple(color) for color in self.custom_palette.tolist()]

        self.is_running = False


    def map_to_color(self, values):
        """
        Function to map values to colors. We want to reproduce the "hot" colormap of matplotlib.
        """ 
        # Normalize the values to the range [0, 1]
        # normalized_values = (values - self.min_strain) / (self.max_strain - self.min_strain)
        normalized_values = (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-5)

        # Apply colormap transformation
        red = np.clip(2 * normalized_values, 0, 1)
        green = np.clip(2 * normalized_values - 1, 0, 1)
        blue = np.clip(4 * normalized_values - 3, 0, 1)
        
        # Stack color channels and convert to uint8
        colors = np.stack((red, green, blue), axis=-1) * 255
        colors = colors.astype(np.uint8)
        return colors


    def remapPointOnScreen(self, point):
        """
        Utility to remap a given point (pe,se) in the correct position on the screen.
        """
        # first, we find the position of our point as a scale factor for both the coordinates
        # essentially, if the point is very close to the upper bound of its two coordinates, this value is ~1
        # while if it is very close to the lower bound, it is ~0
        pe_scale_factor = (point[0]-self.pe_boundaries[0])/(self.pe_boundaries[1]-self.pe_boundaries[0])
        se_scale_factor = (point[1]-self.se_boundaries[0])/(self.se_boundaries[1]-self.se_boundaries[0])

        # we then use this information to map the point on the actual screen, as we know the screen size
        # mind that the origin of the screen is in the top left corner, with X from left to right, and Y downwards
        pe_on_screen = pe_scale_factor * self.widow_dimensions[0]
        se_on_screen = (1-se_scale_factor) * self.widow_dimensions[1]   # note the correction, as the strainmap has SE positive
                                                                        # upwards. So we flip the position wrt the Y axis
        
        # return the re-projected point
        return np.array([pe_on_screen, se_on_screen])
    

    def draw_x_axis(self, ticks):
        pygame.draw.line(self.screen, 
                         self.color_lines, 
                         (0, self.widow_dimensions[1]), 
                         (self.widow_dimensions[0], self.widow_dimensions[1]), 
                         self.width_lines)

        for i in range(0, self.widow_dimensions[0] + 1, int(self.widow_dimensions[0]/((self.pe_boundaries[1]-self.pe_boundaries[0])//10))):
            x = i
            y = self.widow_dimensions[1] - self.width_lines
            pygame.draw.line(self.screen, self.color_lines, (x, y - self.tick_length), (x, y + self.tick_length), self.width_lines)

        # Render and blit the ticks text
        num_ticks = np.shape(ticks)[0]

        for i in range(num_ticks):
            caption_text = self.font_ticks_pygame.render(str(ticks[i]), True, self.color_lines)
            self.screen.blit(caption_text, np.array([self.remapPointOnScreen(np.array([ticks[i], 0]))[0]-2*i, self.widow_dimensions[1] - 20]))
            # the -2*i above is just to compensate for some weird drift. Might be a mistake I made as well, not sure...

    def draw_y_axis(self, ticks):
        pygame.draw.line(self.screen, 
                         self.color_lines, 
                         (0, 0), 
                         (0, self.widow_dimensions[1]), 
                         self.width_lines)

        for i in range(0, self.widow_dimensions[1] + 1, int(self.widow_dimensions[1]/((self.se_boundaries[1]-self.se_boundaries[0])//10))):
            x = self.width_lines
            y = i
            pygame.draw.line(self.screen, self.color_lines, (x - self.tick_length, y), (x + self.tick_length, y), self.width_lines)

        # Render and blit the ticks text
        num_ticks = np.shape(ticks)[0]

        for i in range(num_ticks):
            caption_text = self.font_ticks_pygame.render(str(ticks[i]), True, self.color_lines)
            self.screen.blit(caption_text, np.array([10, self.remapPointOnScreen(np.array([0, ticks[i]]))[1]-4*(num_ticks -i)]))


    def updateStrainMap(self, list_params, pose_current = None, trajectory_current = None, goal_current = None, vel_current = None, ar_current = None):
        """
        This function allows to update the strainmap.
        Inputs:
        * list_params: contains the list of parameters that define the gaussians
                       that need to be plotted.
        * pose_current: the current estimated pose on the strainmap
        * trajectory current: the current optimal trajectory (as points)
        TODO: we could also add capability to plot what was considered optimal at the previous time
        step, and the position where the optimization started?
        """
        # check if this is the first time that the visualizer is used
        # if so, instantiate the window first
        if not self.is_running:
            pygame.init()
            self.clock = pygame.time.Clock()
            self.screen = pygame.display.set_mode(self.widow_dimensions)
            # font for the caption
            self.font_title_pygame = pygame.font.Font(None, self.font_title)
            pygame.display.set_caption('Real-Time Strain Map')

            # font for the axis
            self.font_ticks_pygame = pygame.font.Font(None, self.font_ticks)

            self.is_running = True

        current_strainmap = np.zeros(self.X_norm.shape)

        for function in range(len(list_params) // self.num_params_gaussian):
            amplitude = list_params[function * self.num_params_gaussian]
            x0 = list_params[function * self.num_params_gaussian + 1]
            y0 = list_params[function * self.num_params_gaussian + 2]
            sigma_x = list_params[function * self.num_params_gaussian + 3]
            sigma_y = list_params[function * self.num_params_gaussian + 4]
            offset = list_params[function * self.num_params_gaussian + 5]

            current_strainmap += amplitude * np.exp(
                -((self.X_norm - x0) ** 2 / (2 * sigma_x ** 2) + (self.Y_norm - y0) ** 2 / (2 * sigma_y ** 2))) + offset

        # Map values to colors directly using numpy array indexing
        colors = self.map_to_color(current_strainmap)

        # Set the entire surface with colors
        pygame.surfarray.blit_array(self.image_surface, np.flip(colors, axis=1))  # Transpose to match pygame surface format

        self.screen.blit(pygame.transform.scale(self.image_surface, self.widow_dimensions), (0, 0))

        # Render and blit the caption
        self.ar_current = ar_current
        ar_label = self.font_title_pygame.render(f'Axial rotation:{self.ar_current}', True, self.color_lines)
        self.screen.blit(ar_label, (self.widow_dimensions[0]-200, 10))

        act_label = self.font_title_pygame.render(f'Muscle activation:{self.act_current}', True, self.color_lines)
        self.screen.blit(act_label, (self.widow_dimensions[0]-200, 40))

        # if given, display the current 2D pose on the strainmap (plane of elevation, shoulder elevation)
        if pose_current is not None:
            marker_radius = 5      # define the radius of the marker for the current pose
            pygame.draw.circle(self.screen, (255, 0, 0), self.remapPointOnScreen(np.rad2deg(pose_current)), marker_radius)

        if vel_current is not None:
            pygame.draw.line(self.screen, 
                             (255, 0, 0), 
                             self.remapPointOnScreen(np.rad2deg(pose_current)), 
                             self.remapPointOnScreen(np.rad2deg(pose_current+vel_current)), 
                             self.width_lines)

        # if given, display the reference trajectory scattering its points
        if trajectory_current is not None:
            traj_point_radius = 3
            
            for index in range(np.shape(trajectory_current)[1]):
                pygame.draw.circle(self.screen, (0, 0, 255), self.remapPointOnScreen(np.rad2deg(trajectory_current[:,index])), traj_point_radius)

        # visualize also the goal on the current strainmap (if it has been set)
        if goal_current is not None:
            goal_radius = 3.5
            pygame.draw.circle(self.screen, (0, 255, 0), self.remapPointOnScreen(np.rad2deg(goal_current)), goal_radius)

        # visualize ellipses
        if self.ellipse_params is not None:
            for i in range(self.ellipse_params.shape[0]):
                width = self.ellipse_params[i, 2] * self.widow_dimensions[0]/(self.pe_boundaries[1] - self.pe_boundaries[0])
                heigth = self.ellipse_params[i, 3] * self.widow_dimensions[1]/(self.se_boundaries[1] - self.se_boundaries[0])
                rect = pygame.Rect(0, 0, width, heigth)
                rect.center = self.remapPointOnScreen(np.array([self.ellipse_params[i, 0], self.ellipse_params[i, 1]]))
                pygame.draw.ellipse(self.screen, (96, 245, 66), rect, width = 2)

        # draw the X and Y axis on the map
        self.draw_x_axis(np.array([0, 40, 80, 120, 140]))
        self.draw_y_axis(np.array([20, 40, 80, 140]))

        pygame.display.flip()


    def update_ellipse_params(self, ellipse_params, force = False):
        """
        function used to update the ellipse parameters representing the unsafe zones.
        Only the one relevant for the current strain-maps are considered.

        We can also force the update by setting force=True
        """
        if self.act_current != self.act_prev or self.ar_current != self.ar_prev:
            self.ellipse_params = ellipse_params

        if force:
            self.ellipse_params = ellipse_params


    def quit(self):
        pygame.quit()
