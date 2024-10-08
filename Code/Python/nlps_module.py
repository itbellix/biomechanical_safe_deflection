import opensim as osim
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

from experiment_parameters import *     # this contains the experimental_params and the shared_ros_topics


class nlps_module():
    """
    Class defining the nonlinear programming (NLP) problem defined to navigate the Strain maps.
    We refer to this as NLPS (where the "s" stands for "strain")

    CasADi and OpenSimAD are used to formalize the optimization problem to find the trajectory 
    that a given OpenSim model should follow in order to navigate the corresponding  strain-space 
    safely.

    """
    def __init__(self, sys_forw_dynamics, sys_inv_dynamics = None):
        """"
        Initialization
        """
        self.T = None                   # time horizon for the optimization
        self.N = None                   # number of control intervals
        self.h = None                   # duration of each control interval

        # CasADi symbolic variables
        self.x = None                   # CasADi variable indicating the state of the system (position and velocities of relevant coordinates)
                                        # In our case, the coordinates we care about are plane of elevation (pe) and shoulder elevation (se)
        
        self.u = None                   # CasADi variable representing the control vector to be applied to the system
                                        # In our case, they are forces to be applied to the human at their elbow

        # Initial condition expressed in state-space
        self.x_0 = None

        # Naming and ordering of states and controls
        self.state_names = None
        self.dim_x = 0
        self.control_names = None
        self.dim_u = 0

        # Systems dynamics as a CasADi function
        self.sys_forw_dynamics = sys_forw_dynamics

        # Define the differentiable inverse dynamics with a CasADi function
        self.sys_inv_dynamics = sys_inv_dynamics

        # Strainmap parameters
        self.num_gaussians = 0          # the number of 2D Gaussians used to approximate the current strainmap
        self.num_params_gaussian = 6    # number of parameters that each Gaussian is defined of (for a 2D Gaussian, we need 6)
        self.all_params_gaussians = []  # list containing the values of all the parameters defining all the Gaussians
                                        # For each Gaussian, we have: amplitude, x0, y0, sigma_x, sigma_y, offset
                                        # (if more Gaussians are present, the parameters of all of them are concatenated)
        
        self.pe_boundaries = [-20, 160] # defines the interval of physiologically plausible values for the plane of elevation [deg]
        self.se_boundaries = [0, 144]   # as above, for the shoulder elevation [deg]
        self.ar_boundaries = [-90, 100] # as above, for the axial rotation [deg]

        self.strainmap_step = 4         # discretization step used along the model's coordinate [in degrees]
                                        # By default we set it to 4, as the strainmaps are generated from the biomechanical model
                                        # with this grid accuracy
        
        self.ar_values = np.arange(self.ar_boundaries[0], self.ar_boundaries[1], self.strainmap_step)
        
        # Strainmap parameters (for visualization)
        self.pe_datapoints = np.array(np.arange(self.pe_boundaries[0], self.pe_boundaries[1], self.strainmap_step))
        self.se_datapoints = np.array(np.arange(self.se_boundaries[0], self.se_boundaries[1], self.strainmap_step))

        self.X,self.Y = np.meshgrid(self.pe_datapoints, self.se_datapoints, indexing='ij')
        self.X_norm = self.X/np.max(np.abs(self.pe_boundaries))
        self.Y_norm = self.Y/np.max(np.abs(self.se_boundaries))

        # Type of collocation used and corresponding matrices
        self.pol_order = None
        self.collocation_type = None
        self.B = None                   # quadrature matrix
        self.C = None                   # collocation matrix
        self.D = None                   # end of the intervaprobleml

        # cost function
        self.cost_function = None       # CasADi function expressing the cost to be minimized
        self.gamma_strain = 0           # weight of the strain value in the cost function
        self.gamma_velocities = 0       # weight for the coordinates' velocities
        self.gamma_acceleration = 0     # weight for the coordinates' accelerations

        # constraints
        self.constrain_final_vel = False    # constrain the final velocity
        self.eps_final_vel = 0              # tolerance for velocity

        # CasADi optimization problem
        self.opti = ca.Opti()
        self.nlp_is_formulated = False              # has the nlp being built with formulateNLP()?
        self.solver_options_set = False             # have the solver options being set by the user?
        self.default_solver = 'ipopt'
        self.default_solver_opts = {'ipopt.print_level': 0,
                                    'print_time': 0, 
                                    'error_on_fail': 1,
                                    'ipopt.tol': 1e-3,
                                    'expand': 0,
                                    'ipopt.hessian_approximation': 'limited-memory'}


        # parameters of the NLP
        self.params_list = []           # it collects all the parameters that are used in a single instance
                                        # of the NLP
        
        # solution of the NLP
        self.Xs = None                  # the symbolic optimal trajectory for the state x
        self.Xs_collocation = None      # symbolic states at collocation points
        self.Xddot_s = None             # the symbolic expression for the optimal accelerations of the model's DoFs
        self.Us = None                  # the symbolic optimal sequence of the controls u
        self.Js = None                  # the symbolic expression for the cost function
        self.strain_s = None            # the symbolic strain history, as a function of the optimal solution
        self.solution = None            # the numerical solution (where state and controls are mixed together)
        self.x_opt = None               # the numerical optimal trajectory for the state x
        self.u_opt = None               # the numerical optimal sequence of the controls u
        self.strain_opt = None          # the numerical values of the strain, along the optimal trajectory
        self.lam_g0 = None              # initial guess for the dual variables (populated running solveNLPonce())
        self.solution_is_updated = False


    def setTimeHorizonAndDiscretization(self, N, T):
        self.T = T      # time horizon for the optimal control
        self.N = N      # number of control intervals
        self.h = T/N


    def setSystemForwDynamics(self, sys_forw_dynamics):
        """
        Utility to set the system forward dynamics. It can receive as an input either a CasADi callback
        (if an OpenSim model is to be used) or a CasADi function directly (if the ODE are know analytically).
        """
        self.sys_forw_dynamics = sys_forw_dynamics


    def setSystemInvDynamics(self, sys_inv_dynamics):
        """
        Utility to set the system inverse dynamics, as a CasADi function obtained through OpenSimAD.
        """
        self.sys_inv_dynamics = sys_inv_dynamics
    

    def initializeStateVariables(self, x):
        self.x = x
        self.state_names = x.name()
        self.dim_x = x.shape[0]


    def initializeControlVariables(self, u):
        self.u = u
        self.control_names = u.name()
        self.dim_u = u.shape[0]


    def populateCollocationMatrices(self, order_polynomials, collocation_type):
        """
        Here, given an order of the collocation polynomials, the corresponding matrices are generated
        The collocation polynomials used are Lagrange polynomials.
        The collocation type determines the collocation points that are used ('legendre', 'radau').
        """
        # Degree of interpolating polynomial
        self.pol_order = order_polynomials

        # Get collocation points
        tau = ca.collocation_points(self.pol_order, collocation_type)

        # Get linear maps
        self.C, self.D, self.B = ca.collocation_coeff(tau)

    
    def setCostFunction(self, cost_function):
        """"
        The user-provided CasADi function is used as the cost function of the problem
        """
        self.cost_function = cost_function


    def enforceFinalConstraints(self, on_velocity, eps_vel = 0.01):
        """
        This function allows to choose whether to enforce constraints on the final velocity in the NLP. 
        The user can input the tolerances with which the constraints will be respected (in rad/s)
        """
        self.constrain_final_vel = on_velocity
        self.eps_final_vel = eps_vel


    def setInitialState(self, x_0):
        self.x_0 = x_0

    def setConstraints(self, constraint_list):
        """
        We set the constraints as a list of elements.
        """
        self.constraint_list = constraint_list
     

    def formulateNLP_functionDynamics(self, initial_guess_prim_vars = None, initial_guess_dual_vars = None):
        """"
        This takes care of formulating the specific problem that we are aiming to solve.
        """ 
        if self.sys_forw_dynamics is None:
            RuntimeError("Unable to continue. The system forward dynamics have not been specified. \
                         Do so with setSystemForwDynamics()!")
            
        if self.sys_inv_dynamics is None:
            RuntimeError("Unable to continue. The system inverse dynamics have not been specified. \
                         Do so with setSystemInvDynamics()!")
            
        if self.cost_function is None:
            RuntimeError("Unable to continue. The cost function have not been specified. \
                         Do so with setCostFunction()!")
            
        if len(self.all_params_gaussians) == self.num_gaussians * 6:
            if len(self.all_params_gaussians)==0:
                print("No (complete) information about strainmaps have been included \nThe NLP will not consider them")
        else:
            RuntimeError("Unable to continue. The specified strainmaps are not correct. \
                         Check if the parameters provided are complete wrt the number of Gaussians specified. \
                         Note: we assume that 6 parameters per (2D) Gaussian are given")

        # initialize the cost function value
        J = 0

        if self.constraint_list is not None:
            u_max = self.constraint_list['u_max']
            u_min = self.constraint_list['u_min']

        # initialize empty list for the parameters used in the problem
        # the parameters collected here can be changed at runtime   
        self.params_list = []

        # if this information is present, define the strainmap to navigate onto
        if self.num_gaussians>0:
            # for now, let's assume that there will always be 3 Gaussians (if this is not true, consider
            # if it is better to have a fixed higher number or a variable one)
            tmp_param_list = []     # this is an auxiliary list, to collect all the strain-related params and 
                                    # append them at the end.
            
            # parameters of the 1st Gaussian
            p_g1 = self.opti.parameter(self.num_params_gaussian)    # the order is [amplitude, x0, y0, sigma_x, sigma_y, offset]
            tmp_param_list.append(p_g1)

            # definition of the 1st Gaussian (note that the state variables are normalized!)
            g1 = p_g1[0] * np.exp(-((self.x[0]/np.max(np.abs(self.pe_boundaries))-p_g1[1])**2/(2*p_g1[3]**2) + (self.x[2]/np.max(np.abs(self.se_boundaries))-p_g1[2])**2/(2*p_g1[4]**2))) + p_g1[5]

            # parameters of the 2nd Gaussian
            p_g2 = self.opti.parameter(self.num_params_gaussian)    # the order is [amplitude, x0, y0, sigma_x, sigma_y, offset]
            tmp_param_list.append(p_g2)

            # definition of the 2nd Gaussian (note that the state variables are normalized!)
            g2 = p_g2[0] * np.exp(-((self.x[0]/np.max(np.abs(self.pe_boundaries))-p_g2[1])**2/(2*p_g2[3]**2) + (self.x[2]/np.max(np.abs(self.se_boundaries))-p_g2[2])**2/(2*p_g2[4]**2))) + p_g2[5]

            # parameters of the 3rd Gaussian
            p_g3 = self.opti.parameter(self.num_params_gaussian)    # the order is [amplitude, x0, y0, sigma_x, sigma_y, offset]
            tmp_param_list.append(p_g3)

            # definition of the 3rd Gaussian (note that the state variables are normalized!)
            g3 = p_g3[0] * np.exp(-((self.x[0]/np.max(np.abs(self.pe_boundaries))-p_g3[1])**2/(2*p_g3[3]**2) + (self.x[2]/np.max(np.abs(self.se_boundaries))-p_g3[2])**2/(2*p_g3[4]**2))) + p_g3[5]

            # definition of the symbolic cumulative strainmap
            strainmap = g1 + g2 + g3
            # strainmap_sym = ca.Function('strainmap_sym', [self.x], [strainmap], {"allow_free":True})
            strainmap_sym = ca.Function('strainmap_sym', [self.x, p_g1, p_g2, p_g3], [strainmap])

            # save the symbolic strainmap for some debugging
            self.strainmap_sym = strainmap_sym

        #  "Lift" initial conditions
        Xk = self.opti.variable(self.dim_x)
        init_state = self.opti.parameter(self.dim_x)     # parametrize initial condition
        self.params_list.append(init_state)              # add the parameter to the list of parameters for the NLP
        self.opti.subject_to(Xk==init_state)

        # the control torques are parameters that can be changed at execution
        estimated_human_torques = self.opti.parameter(self.dim_u)
        self.params_list.append(estimated_human_torques)

        # the current value of ar and ar_dot are also input parameters for the problem
        phi_prm = self.opti.parameter(1)
        self.params_list.append(phi_prm)
        phi_dot_prm = self.opti.parameter(1)             # this is an internal parameter, not modifiable from outside

        # Collect all states/controls, and strain along the trajectory
        Xs = [Xk]
        Xs_collocation = []
        Us = []
        Xddot_s = []
        strain_s = []

        # formulate the NLP
        for k in range(self.N):
            # New NLP variable for the control
            Uk = self.opti.variable(self.dim_u)
            Us.append(Uk)

            # limits to the additional controls that the robot interaction should generate
            if self.constraint_list is not None:
                self.opti.subject_to(self.opti.bounded(u_min, Uk, u_max))

            J = J + Uk[0]**2 + Uk[1]**2 + Uk[2]**2
            
            # add the robot torque and the human torque together
            Uk_tot = Uk + estimated_human_torques

            # optimization variable (state) at collocation points
            Xc = self.opti.variable(self.dim_x, self.pol_order)
            Xs_collocation.append(Xc)

            # we need to create a proper input to the casadi function
            # we want to evaluate x_dot at every point in the collocation mesh
            input_sys_forw_dynamics = ca.vertcat(Xc[0:4, :],                     # theta, theta dot, psi, psi_dot at collocation points
                                            ca.repmat(phi_prm, 1, 3),       # phi at collocation points
                                            ca.repmat(phi_dot_prm, 1, 3),   # phi_dot at collocation points
                                            ca.repmat(Uk_tot, 1, 3))        # controls at collocation points (constant)
                                            # ca.repmat(estimated_human_torques_Ar, 1, 3))    # robot torque around uncontrolled DoF

            # evaluate ODE right-hand-side at collocation points. This allows to simulate the trajectory forward
            ode = self.sys_forw_dynamics(input_sys_forw_dynamics)

            # check if we care about the strain
            if self.num_gaussians>0:
                # if so, add term related to current strain to the cost
                # the strain is evaluated only at the knots of the optimization mesh
                # (note that the strainmap is defined in degrees, so we convert our state to that)
                J = J + self.gamma_strain * strainmap_sym(Xk*180/ca.pi, p_g1, p_g2, p_g3)

                # record the strain level in the current state value
                # (note that the strainmap is defined in degrees, so we convert our state to that)
                strain_s.append(strainmap_sym(Xk*180/ca.pi, p_g1, p_g2, p_g3))

            # get interpolating points of collocation polynomial
            Z = ca.horzcat(Xk, Xc)

            # get slope of interpolating polynomial (normalized)
            Pidot = ca.mtimes(Z, self.C)

            # match with ODE right-hand-side
            self.opti.subject_to(Pidot[0:4,:]==self.h*ode[0:4, :])    # theta, theta_dot, psi, psi_dot (no constraint phi)

            # save coordinates' accelerations (only for the first collocation point)
            Xddot_s.append(ode[1::2, 0])

            # state at the end of collocation interval
            Xk_end = ca.mtimes(Z, self.D)

            # new decision variable for state at the end of interval
            Xk = self.opti.variable(self.dim_x)
            Xs.append(Xk)

            # continuity constraint
            self.opti.subject_to(Xk_end==Xk)

        if self.num_gaussians>0:
            # record the strain level at the final step
            # (note that the strainmap is defined in degrees, so we convert our state to that)
            strain_s.append(strainmap_sym(Xk*180/ca.pi, p_g1, p_g2, p_g3))
            self.strain_s = ca.vertcat(*strain_s)

        # adding constraint to reach the final desired state
        if self.constrain_final_vel:
            self.opti.subject_to((Xk[1])**2<self.eps_final_vel)   # bound on final velocity
            self.opti.subject_to((Xk[3])**2<self.eps_final_vel)   # bound on final velocity

        self.Us = ca.vertcat(*Us)
        self.Xs = ca.vertcat(*Xs)
        self.Xs_collocation = ca.vertcat(*Xs_collocation)
        self.Xddot_s = ca.vertcat(*Xddot_s)

        # explicitly provide an initial guess for the primal variables
        if initial_guess_prim_vars is not None:
            self.opti.set_initial(initial_guess_prim_vars)

        # explicitly provide an initial guess for the dual variables
        if initial_guess_dual_vars is not None:
            self.opti.set_initial(self.opti.lam_g, initial_guess_dual_vars)

        # set the values of the parameters (these will be changed at runtime)
        self.opti.set_value(init_state, self.x_0)
        self.opti.set_value(estimated_human_torques, self.sys_inv_dynamics(np.concatenate((self.x_0, np.zeros((3,))))))
        # self.opti.set_value(estimated_human_torques_Ar, self.sys_inv_dynamics(np.concatenate((self.x_0, np.zeros((3,)))))[2])
        self.opti.set_value(phi_prm, 0)
        self.opti.set_value(phi_dot_prm, 0)

        if self.num_gaussians>0:
            self.params_list.extend(tmp_param_list)     # append at the end the parameters for the strains

            self.opti.set_value(p_g1, self.all_params_gaussians[0:6])   # TODO: hardcoded!
            self.opti.set_value(p_g2, self.all_params_gaussians[6:12])
            self.opti.set_value(p_g3, self.all_params_gaussians[12:18])

        # define the cost function to be minimized, and store its symbolic expression
        self.opti.minimize(J) # + ca.sumsqr(ca.vertcat(*self.params_list)))
        self.Js = J

        # set flag indicating process was successful
        self.nlp_is_formulated = True


    def formulateNLP_functionDynamics_IDintheloop(self, initial_guess_prim_vars = None, initial_guess_dual_vars = None):
        """"
        This takes care of formulating the specific problem that we are aiming to solve, including the ID function call
        inside the optimization loop.
        """ 
        if self.sys_forw_dynamics is None:
            RuntimeError("Unable to continue. The system dynamics have not been specified. \
                         Do so with setSystemForwDynamics()!")
            
        if self.sys_inv_dynamics is None:
            RuntimeError("Unable to continue. The system dynamics have not been specified. \
                         Do so with setSystemInvDynamics()!")
            
        if self.cost_function is None:
            RuntimeError("Unable to continue. The cost function have not been specified. \
                         Do so with setCostFunction()!")
            
        if len(self.all_params_gaussians) == self.num_gaussians * 6:
            if len(self.all_params_gaussians)==0:
                print("No (complete) information about strainmaps have been included \nThe NLP will not consider them")
        else:
            RuntimeError("Unable to continue. The specified strainmaps are not correct. \
                         Check if the parameters provided are complete wrt the number of Gaussians specified. \
                         Note: we assume that 6 parameters per (2D) Gaussian are given")

        # initialize the cost function value
        J = 0

        if self.constraint_list is not None:
            u_max = self.constraint_list['u_max']
            u_min = self.constraint_list['u_min']

        # initialize empty list for the parameters used in the problem
        # the parameters collected here can be changed at runtime   
        self.params_list = []

        #  "Lift" initial conditions
        Xk = self.opti.variable(self.dim_x)
        init_state = self.opti.parameter(self.dim_x)     # parametrize initial condition
        self.params_list.append(init_state)              # add the parameter to the list of parameters for the NLP
        self.opti.subject_to(Xk==init_state)

        # the control torques are parameters that can be changed at execution
        estimated_human_torques_0 = self.opti.parameter(self.dim_u)
        self.params_list.append(estimated_human_torques_0)

        # the current value of ar and ar_dot are also input parameters for the problem
        phi_prm = self.opti.parameter(1)
        self.params_list.append(phi_prm)
        phi_dot_prm = self.opti.parameter(1)             # this is an internal parameter, not modifiable from outside

        # Collect all states/controls, and strain along the trajectory
        Xs = [Xk]
        Xs_collocation = []
        Us = []
        Xddot_s = []
        strain_s = []

        # formulate the NLP
        for k in range(self.N):
            # New NLP variable for the control
            Uk = self.opti.variable(self.dim_u)
            Us.append(Uk)

            if k == 0:
                # add the robot torque and the human torque together
                Uk_tot = Uk + estimated_human_torques_0
            else:
                Xk_dot = self.sys_forw_dynamics(ca.vertcat(Xk[0:4], phi_prm, phi_dot_prm, Uk_tot))

                estimated_human_torques_k = self.sys_inv_dynamics(ca.vertcat(Xk, Xk_dot[1::2]))

                Uk_tot = Uk + estimated_human_torques_k


            # limits to the additional controls that the robot interaction should generate
            if self.constraint_list is not None:
                self.opti.subject_to(self.opti.bounded(u_min, Uk, u_max))

            J = J + Uk[0]**2 + Uk[1]**2 + Uk[2]**2

            # optimization variable (state) at collocation points
            Xc = self.opti.variable(self.dim_x, self.pol_order)
            Xs_collocation.append(Xc)

            # we need to create a proper input to the casadi function
            # we want to evaluate x_dot at every point in the collocation mesh
            input_sys_forw_dynamics = ca.vertcat(Xc[0:4, :],                     # theta, theta dot, psi, psi_dot at collocation points
                                            ca.repmat(phi_prm, 1, 3),       # phi at collocation points
                                            ca.repmat(phi_dot_prm, 1, 3),   # phi_dot at collocation points
                                            ca.repmat(Uk_tot, 1, 3))        # controls at collocation points (constant)
                                            # ca.repmat(estimated_human_torques_Ar, 1, 3))    # robot torque around uncontrolled DoF

            # evaluate ODE right-hand-side at collocation points. This allows to simulate the trajectory forward
            ode = self.sys_forw_dynamics(input_sys_forw_dynamics)

            # get interpolating points of collocation polynomial
            Z = ca.horzcat(Xk, Xc)

            # get slope of interpolating polynomial (normalized)
            Pidot = ca.mtimes(Z, self.C)

            # match with ODE right-hand-side
            self.opti.subject_to(Pidot[0:4,:]==self.h*ode[0:4, :])    # theta, theta_dot, psi, psi_dot (no constraint phi)

            # save coordinates' accelerations (only for the first collocation point)
            Xddot_s.append(ode[1::2, 0])

            # state at the end of collocation interval
            Xk_end = ca.mtimes(Z, self.D)

            # new decision variable for state at the end of interval
            Xk = self.opti.variable(self.dim_x)
            Xs.append(Xk)

            # continuity constraint
            self.opti.subject_to(Xk_end==Xk)

        # adding constraint to reach the final desired state
        if self.constrain_final_vel:
            self.opti.subject_to((Xk[1])**2<self.eps_final_vel)   # bound on final velocity
            self.opti.subject_to((Xk[3])**2<self.eps_final_vel)   # bound on final velocity

        self.Us = ca.vertcat(*Us)
        self.Xs = ca.vertcat(*Xs)
        self.Xs_collocation = ca.vertcat(*Xs_collocation)
        self.Xddot_s = ca.vertcat(*Xddot_s)

        # explicitly provide an initial guess for the primal variables
        if initial_guess_prim_vars is not None:
            self.opti.set_initial(initial_guess_prim_vars)

        # explicitly provide an initial guess for the dual variables
        if initial_guess_dual_vars is not None:
            self.opti.set_initial(self.opti.lam_g, initial_guess_dual_vars)

        # set the values of the parameters (these will be changed at runtime)
        self.opti.set_value(init_state, self.x_0)
        self.opti.set_value(estimated_human_torques_0, self.sys_inv_dynamics(np.concatenate((self.x_0, np.zeros((3,))))))
        # self.opti.set_value(estimated_human_torques_Ar, self.sys_inv_dynamics(np.concatenate((self.x_0, np.zeros((3,)))))[2])
        self.opti.set_value(phi_prm, 0)
        self.opti.set_value(phi_dot_prm, 0)

        # define the cost function to be minimized, and store its symbolic expression
        self.opti.minimize(J + ca.sumsqr(ca.vertcat(*self.params_list)))
        self.Js = J

        # set flag indicating process was successful
        self.nlp_is_formulated = True


    def formulateNLP_newVersion(self):
        """
        New NLP relying on minimal deviations from given trajectories.
        """

        if self.sys_forw_dynamics is None:
            RuntimeError("Unable to continue. The system forward dynamics have not been specified. \
                         Do so with setSystemForwDynamics()!")
            
        if self.sys_inv_dynamics is None:
            RuntimeError("Unable to continue. The system inverse dynamics have not been specified. \
                         Do so with setSystemInvDynamics()!")
            
        if self.cost_function is None:
            RuntimeError("Unable to continue. The cost function have not been specified. \
                         Do so with setCostFunction()!")
            
        if len(self.all_params_gaussians) == self.num_gaussians * 6:
            if len(self.all_params_gaussians)==0:
                print("No (complete) information about strainmaps have been included \nThe NLP will not consider them")
        else:
            RuntimeError("Unable to continue. The specified strainmaps are not correct. \
                         Check if the parameters provided are complete wrt the number of Gaussians specified. \
                         Note: we assume that 6 parameters per (2D) Gaussian are given")

        # initialize the cost function value
        J = 0

        # weights of the cost function
        w_traj = 5
        w_torque = 0.1

        # tolerances 
        delta_vel = np.deg2rad(5)  # on the final velocity (in rad/s)
        delta_torque = 0.05         # on the final torque values (in N/m)

        # initialize empty list for the parameters used in the problem
        # the parameters collected here can be changed at runtime   
        self.params_list = []

        #  "Lift" initial conditions
        Xk = self.opti.variable(self.dim_x)
        init_state = self.opti.parameter(self.dim_x)     # parametrize initial condition
        self.params_list.append(init_state)              # add the parameter to the list of parameters for the NLP
        self.opti.subject_to(Xk==init_state)

        # parametrize the future human states (if no robot intervention is given)
        future_trajectory_0 = self.opti.parameter(self.dim_x, self.N)
        self.params_list.append(future_trajectory_0)

        # the current value of ar and ar_dot are also input parameters for the problem
        phi_prm = init_state[4]
        phi_dot_prm = self.opti.parameter(1)             # this is an internal parameter, not modifiable from outside

        # Collect all states/controls, and strain along the trajectory
        Xs = [Xk]
        Xs_collocation = []
        Us = []
        Xddot_s = []
        strain_s = []

        # formulate the NLP
        for k in range(self.N):
            # New NLP variable for the control
            Uk = self.opti.variable(self.dim_u)
            Us.append(Uk)

            # in the cost function, we weight:
            # - the control effort difference with respect to the torques necessary to maintain the current state
            # - the deviation from the estimated future states
            torque_stabil_k = self.sys_inv_dynamics(ca.vertcat(future_trajectory_0[:, k], np.zeros((3,))))
            J = J \
                + w_torque * ca.sumsqr(Uk - torque_stabil_k) \
                + w_traj * ca.sumsqr(Xk - future_trajectory_0[:, k])

            # optimization variable (state) at collocation points
            Xc = self.opti.variable(self.dim_x, self.pol_order)
            Xs_collocation.append(Xc)

            # we need to create a proper input to the casadi function
            # we want to evaluate x_dot at every point in the collocation mesh
            input_sys_forw_dynamics = ca.vertcat(Xc[0:4, :],                     # theta, theta dot, psi, psi_dot at collocation points
                                            ca.repmat(phi_prm, 1, 3),       # phi at collocation points
                                            ca.repmat(phi_dot_prm, 1, 3),   # phi_dot at collocation points
                                            ca.repmat(Uk, 1, 3))        # controls at collocation points (constant)
            
            # evaluate ODE right-hand-side at collocation points. This allows to simulate the trajectory forward
            ode = self.sys_forw_dynamics(input_sys_forw_dynamics)

            # get interpolating points of collocation polynomial
            Z = ca.horzcat(Xk, Xc)

            # get slope of interpolating polynomial (normalized)
            Pidot = ca.mtimes(Z, self.C)

            # match with ODE right-hand-side (only part of the state, to plan onto 2D strain maps)
            self.opti.subject_to(Pidot[0:4,:]==self.h*ode[0:4, :])    # theta, theta_dot, psi, psi_dot (no constraint phi)

            # save coordinates' accelerations (only for the first collocation point)
            Xddot_s.append(ode[1::2, 0])

            # state at the end of collocation interval
            Xk_end = ca.mtimes(Z, self.D)

            # new decision variable for state at the end of interval
            Xk = self.opti.variable(self.dim_x)
            Xs.append(Xk)

            # continuity constraint
            self.opti.subject_to(Xk_end==Xk)

        # bounding final velocities according to initial ones
        self.opti.subject_to((Xk[1] - future_trajectory_0[1, -1])**2 < delta_vel)
        self.opti.subject_to((Xk[3] - future_trajectory_0[3, -1])**2 < delta_vel)

        # bounding final velocities
        # self.opti.subject_to((Xk[1])**2 < delta_vel)
        # self.opti.subject_to((Xk[3])**2 < delta_vel)

        # bounding final torques 
        # (we need to resort to ID here, to find the torques that can stabilize the final state)
        # Note that the final velocity can also be non-zero (depending on the bounds imposed above)
        torque_stabil_end = self.sys_inv_dynamics(ca.vertcat(Xk, np.zeros((3,))))
        self.opti.subject_to(self.opti.bounded(torque_stabil_end - delta_torque, Uk, torque_stabil_end + delta_torque))

        # manipulate variables to retrieve their values after the NLP is solved
        self.Us = ca.vertcat(*Us)
        self.Xs = ca.vertcat(*Xs)
        self.Xs_collocation = ca.vertcat(*Xs_collocation)
        self.Xddot_s = ca.vertcat(*Xddot_s)

        # find the most likely evolution of the human state in the next N timesteps
        # (this is used to initialize the future_trajectory_0 parameter)
        fut_traj_value = np.zeros((self.dim_x, self.N))
        fut_traj_value[:,0] = self.x_0
        fut_traj_value[1::2, :] = self.x_0[1::2][:, np.newaxis]    # velocities are assumed to be constant
        for timestep in range(1, self.N):
            fut_traj_value[::2, timestep] = fut_traj_value[::2, timestep-1] + self.h * fut_traj_value[1::2, timestep-1]

        # set the values of the parameters (these will be changed at runtime)
        self.opti.set_value(init_state, self.x_0)
        self.opti.set_value(future_trajectory_0, fut_traj_value)
        self.opti.set_value(phi_dot_prm, 0)

        # define the cost function to be minimized, and store its symbolic expression
        self.opti.minimize(J)
        self.Js = J

        # set flag indicating process was successful
        self.nlp_is_formulated = True


    def setSolverOptions(self, solver, opts):
        """
        This function allows to set the solver and the solver options that will be used
        when solving the NLP. It sets to true the corresponding flag, so that other methods
        can operate safely.
        """
        self.opti.solver(solver, opts)
        self.solver_options_set = True


    def solveNLPOnce(self):
        """
        This function solves the NLP problem that has been formulated, assuming that the constraints,
        the initial position, the goal, the cost function and the solver have been specified already.
        It retrieves the optimal trajectory for the state and control variables using the symbolic 
        mappings computes in formulateNLP(), storing those variable and returning them to the caller 
        as well.
        """
        # change the flag so that others know that the current solution is not up-to-date
        self.solution_is_updated = False

        if self.nlp_is_formulated == False:
            RuntimeError("The NLP problem has not been formulated yet! \
                         Do so with the formulateNLP() function")
            
        if self.solver_options_set == False:
            print("No user-provided solver options. \
                  Default solver options will be used. You can provide yours with setSolverOptions()")
            self.setSolverOptions(self.default_solver, self.default_solver_opts)

        self.solution = self.opti.solve()

        self.x_opt = self.solution.value(self.Xs)
        self.x_opt_coll = self.solution.value(self.Xs_collocation)
        self.u_opt = self.solution.value(self.Us)
        self.J_opt = self.solution.value(self.Js)
        self.lam_g0 = self.solution.value(self.opti.lam_g)
        self.xddot_opt = self.solution.value(self.Xddot_s)

        # change the flag so that others know that the current solution is up to date
        self.solution_is_updated = True

        return self.x_opt, self.u_opt, self.solution, self.x_opt_coll
    
    def createOptimalMapWithoutInitialGuesses(self):
        """
        Provides a utility to retrieve a CasADi function out of an opti object, once the NLP stucture 
        has been formulated. It does formally not require solving the NLP problem beforehand.
        However, you should first run an instance of solveNLPonce() so that a good initial guess for
        primal and dual variables for the problem are used - this should speed up the solver massively.
        The function that will be generated can be used as follows (adapting it to your case):

        numerical_outputs_list = MPC_iter(numerical_values_for_parameters)

        The generated function does not allow warm-starting it.
        """

        if self.nlp_is_formulated == False:
            RuntimeError("Unable to continue. The NLP problem has not been formulated yet \
                         Do so with formulateNLP()!")
        
        if self.num_gaussians>0:
            symbolic_output_list = [self.Xs, self.Us, self.opti.lam_g, self.Js, self.strain_s, self.Xddot_s]  
        else:
            symbolic_output_list = [self.Xs, self.Us, self.opti.lam_g, self.Js, self.Xddot_s] 

        # inputs to the function
        input_list = self.params_list.copy()       # the parameters that are needed when building the NLP

        MPC_iter = self.opti.to_function('MPC_iter', input_list, symbolic_output_list)
        return MPC_iter, input_list


    def getSizePrimalVars(self):
        """
        This function allows to retrieve the dimension of the primal variables of the problem, after it 
        has been solved at least once
        """
        if self.Xs is None or self.Us is None:
            RuntimeError('No stored values for primal variables!\
                         Run solveNLPonce() first. \n')
        else:
            return (self.Xs.shape, self.Us.shape)
    

    def getSizeDualVars(self):
        """
        This function allows to retrieve the dimension of the dual variables of the problem, after it 
        has been solved at least once
        """

        if self.lam_g0 is None:
            RuntimeError('No stored values for dual variables! \
                         Run solveNLPonce() first \n')
        else:
            return np.shape(self.lam_g0)
