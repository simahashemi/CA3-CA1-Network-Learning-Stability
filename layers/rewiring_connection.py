# -*- coding: utf-8 -*-
"""TODO"""

import numpy as np
from scipy.special import expit
from scipy.stats import logistic

import core.core_global as core
from core.core_definitions import simulation_timestep
from core.connection import Connection
from core.euler_trace_2d import EulerTrace2D


class RewiringConnection(Connection):
    """TODO"""

    def __init__(self, source, destination, transmitter, params={},
                 name="RewiringConnection"):
        """TODO"""
        super().__init__(source, destination, transmitter, name)

        if destination.get_post_size() == 0:
            return

        core.kernel.register_connection(self)

        self.size = (destination.num_compartments, destination.soma.size, source.size)
        


################################################################
        self.AcD_num = destination.soma.AcD_num
        self.nonAcD_num = destination.soma.nonAcD_num
        
        self.n_syn_max = np.full((destination.num_compartments,destination.soma.size),params.get("n_syn_max_nonAcD", 10))
        self.n_syn_max[0,:self.AcD_num] = params.get("n_syn_max_AcD", 20)
        self.n_syn_start = np.copy(self.n_syn_max)

#         self.n_syn_start = np.full((destination.num_compartments,destination.soma.size),params.get("n_syn_start", 10))

        
        
        #################################################
#         self.n_syn_start = params.get("n_syn_start", 4)
        self.w_max = params.get("w_max", 8.0)
        self.w_ini_min = params.get("w_ini_min", 1.0)
        self.w_ini_max = params.get("w_ini_max", 8.0)
        self.theta_ini = params.get("theta_ini", -10.0)
        self.theta_min = params.get("theta_min", -100.0)
        

        self.learn = params.get("learn", True)
        self.T = params.get("T", 1.0)
        self.eta = params.get("eta", 0.005)
        self.lambd = params.get("lambd", 1.0)
        self.gamma = params.get("gamma", 0.5)
        self.grad_sigmoid_clip =self.w_max # params.get("grad_sigmoid_clip", 8.0)
        self.A = params.get("A", 0.0)
        
        self.c_BAP = params.get("c_BAP", 4.0)
        self.gamma_BAP = params.get("gamma_BAP", 0.2)
        self.BAP_active = params.get("BAP_active", True)
        self.d_spike_events = np.zeros((destination.num_compartments,destination.soma.size))
        self.d_spike_events_correct_dim = np.zeros(self.size)
        
        
        self.scale_theta = params.get("scale_theta", 1.0)
        self.theta_max = self.w_max/self.scale_theta 
        self.scale_w = params.get("scale_w", 0.01)
        self.scale_prior = params.get("scale_prior", 1.0)
        self.scale_likelihood = params.get("scale_likelihood", 4.0)
        self.scale_noise = np.sqrt(2 * self.T * self.eta)

        self.tau_pre = params.get("tau_pre", 20e-3)
        self.tau_post = params.get("tau_post", 20e-3)

        self.tr_pre = self.get_pre_trace(self.tau_pre)
        self.tr_post = self.get_post_trace(self.tau_post)

        self.prior = np.zeros(self.size)
        self.likelihood = np.zeros(self.size)
        self.noise = np.zeros(self.size)
        
        
        self.n_syn = np.zeros((destination.num_compartments,destination.soma.size))


        self._init_weights()

    def _init_weights(self):
        """TODO"""
        self.c = np.zeros(self.size)
        self.w = np.zeros(self.size)
        self.theta = np.full(self.size, self.theta_ini, dtype=np.float)
        
        for dendrite_ind in range(len(self.theta)):
            dendrite = self.theta[dendrite_ind]
            for cell_ind in range(len(dendrite)):
                cell = dendrite[cell_ind]
                cell[core.kernel.rng.choice(cell.size, self.n_syn_start[dendrite_ind][cell_ind],
                                       replace=False)] =\
                    core.kernel.rng.uniform(low=self.w_ini_min,
                                        high=self.w_ini_max,
                                        size=self.n_syn_start[dendrite_ind][cell_ind])
#                 print("conn",dendrite)

        np.maximum(0, self.scale_theta*self.theta, self.w)
        np.heaviside(self.theta, 0, self.c)

    def set_weights(self, w):
        """TODO"""
#         if theta.shape != self.size:
#             core.logger.warning("Warning: Shape of the weight matrix"
#                                 " does not match (post size, pre size)"
#                                 " -- ingoring.")
#             return
        if w.shape != self.size:
            core.logger.warning("Warning: Shape of the weight matrix"
                                " does not match (post size, pre size)"
                                " -- ingoring.")
            return
        self.w = w
        self.c = np.zeros(self.size)
        self.theta = np.zeros(self.size)
        
        np.maximum(0, self.w/self.scale_theta, self.theta)
        np.heaviside(self.theta, 0, self.c)
        
#         self.w = np.zeros(self.size)
#         self.theta = theta

#         np.maximum(0, self.scale_theta * self.theta, self.w)
#         np.heaviside(self.theta, 0, self.c)

    def evolve(self):
        """TODO"""
        if self.learn:
            
            # Compute the prior.
            np.add.reduce(2 * (expit(self.scale_w * self.w) - 0.5),
                          axis=2, out=self.n_syn)
            sigmoid = expit(self.lambd * (self.n_syn_max - self.n_syn))
            correct_dim_sigmoid = np.repeat(sigmoid[:, :, np.newaxis], repeats=self.size[2], axis=2)
            grad_sigmoid = logistic._pdf(self.scale_w * self.w)
            grad_sigmoid[self.w > self.grad_sigmoid_clip] = 0
            np.multiply(-self.lambd * self.scale_w * (1 - correct_dim_sigmoid),grad_sigmoid, self.prior)
            
            # Compute the Structural term.
            
#             Connection_count = np.heaviside(np.add.reduce(np.heaviside(self.w,0), axis=2)-self.n_syn_max ,0)
#             Connection_count_dim_correction = np.repeat(Connection_count[:, :, np.newaxis], repeats=self.size[2], axis=2)
#             grad_sigmoid = logistic._pdf(self.scale_w * self.w)
#             np.multiply(- self.lambd ,grad_sigmoid, self.prior)
#             np.multiply(self.prior,Connection_count_dim_correction, self.prior)

            
            # Compute the Functional term.
            if self.destination.branch.branch_dynamics:
                ###### x-gamma * (1-x)
#                 reshaped_tr_val = np.tile(self.tr_pre.val, (self.destination.num_compartments,self.destination.soma.size, 1))
#                 self.likelihood = np.multiply(np.heaviside(self.destination.branch.pla, 0)[:, :, np.newaxis],
#                             (reshaped_tr_val - self.gamma * (1 - reshaped_tr_val)))                
                ###### (x-gamma)
                reshaped_tr_val = np.tile(self.tr_pre.val, (self.destination.num_compartments,self.destination.soma.size, 1))
                self.likelihood = np.multiply(np.heaviside(self.destination.branch.pla, 0)[:, :, np.newaxis],
                            (reshaped_tr_val - self.gamma )) 
#                 print("time",core.kernel.get_time())
#                 print("assembly 2:","min",np.min(self.tr_pre.val[60:90]),"max",np.max(self.tr_pre.val[60:90]))
#                 print("assembly 2:","mean",np.mean(self.tr_pre.val[60:90]))
#                 print("expectation",np.min(self.tr_pre.val[60:90]) - self.gamma , np.max(self.tr_pre.val[60:90]) - self.gamma  )
#                 print("likelihood 2:",np.min(self.likelihood[:,:,60:90]),np.max(self.likelihood[:,:,60:90]))
                
                
# #                 print("assembly_1","min",np.min(self.tr_pre.val[30:60]),"max",np.max(self.tr_pre.val[30:60]))
#                 print("assembly_1","mean",np.mean(self.tr_pre.val[30:60]))

#                 print("expectation",np.min(self.tr_pre.val[30:60]) - self.gamma  , np.max(self.tr_pre.val[30:60]) - self.gamma )
#                 print("likelihood 1:",np.min(self.likelihood[:,:,30:60]),np.max(self.likelihood[:,:,30:60]),'\n')
#                 self.likelihood = np.multiply(np.heaviside(self.destination.branch.pla, 0),
#                             (self.tr_pre.val[:, np.newaxis, np.newaxis] - 1.0)).reshape(self.size)   
#                 print("branch likelihood",self.scale_likelihood *np.max(self.likelihood),self.scale_likelihood *np.mean(self.likelihood[self.likelihood>0]))
                   
            # Add contribution from prior and likelihood to active synapses.
            np.add(self.theta, np.multiply(
                self.eta, np.add(self.scale_prior * self.prior,
                                 self.scale_likelihood * self.likelihood)),self.theta, where=self.c == 1)



            # Add noise to all synapses.
            # W_t+dt - W_t is normally distributed with mean 0 and variance dt (N(0, dt)). So
            # rng.normal(loc=0, scale=np.sqrt(dt))
            np.add(self.theta, self.scale_noise * core.kernel.rng.normal(
                loc=0, scale=np.sqrt(simulation_timestep), size=self.size), self.theta)


            # Clip weights and parameters.
            np.clip(self.theta, self.theta_min, self.theta_max, self.theta)
            np.maximum(0, self.scale_theta * self.theta, self.w)
            np.heaviside(self.theta, 0, self.c)
            
            
                

    def propagate(self):
        """TODO"""
        self.propagate_forward()
        self.propagate_backward()

    def propagate_forward(self):
        """TODO"""
        for spike in self.source.get_spikes():
            self.target_state_vector += self.w[:, :, spike]


        
#             if self.BAP_active and self.learn:
#                 self.theta[((self.c[:, spike] == 1) &
#                             (self.destination.branch.mem >= self.stdp_th).flatten()), spike] += self._on_pre()

#             if self.stdp_active and self.learn:

#                 self.theta[(self.c[:, :,spike] == 1) , spike] += self._on_pre()

#                 # Clip weights and parameters.
#                 np.clip(self.theta, self.theta_min, self.theta_max, self.theta)
#                 np.maximum(0, self.theta, self.w)
#                 np.heaviside(self.theta, 0, self.c)

    def propagate_backward(self):
        """TODO"""
#         if len(self.destination.get_spikes_immediate()) > 0:
#             print("with delay",self.destination.get_spikes_immediate(),core.kernel.get_time())
        if self.BAP_active and self.learn:
            S_t = np.zeros(self.size)
            for spike in self.destination.get_spikes_immediate():
                S_t[:,spike,:] = 1
            np.add(self.theta, np.multiply(S_t,self._on_post()), self.theta,where=(self.c == 1))
            # Clip weights and parameters.
            np.clip(self.theta, self.theta_min, self.theta_max, self.theta)
            np.maximum(0, self.theta, self.w)
            np.heaviside(self.theta, 0, self.c)

                
#         if self.BAP_active and self.learn:
#             self.d_spike_events = (self.destination.branch.pla > 0) * 1
#             self.d_spike_events_correct_dim = np.repeat(self.d_spike_events[:, :, np.newaxis], repeats=self.size[2], axis=2)

#             for spike in self.destination.get_spikes_immediate():
#                 np.add(self.theta, self._on_post(), self.theta,
#                        where=(self.c == 1) & (self.d_spike_events_correct_dim > 0))
#             d_theta = self.theta - before_theta##########
#             if np.max(abs(d_theta)) > 0:#####################
#                 print("BAP effect", np.min(d_theta), np. max(d_theta)) #########
#                 print("likelihood effect", np.min(self.eta * self.scale_likelihood * self.likelihood[self.c == 1]), np.max(self.eta * self.scale_likelihood * self.likelihood[self.c == 1])) #########
#             # Clip weights and parameters.
#             np.clip(self.theta, self.theta_min, self.theta_max, self.theta)
#             np.maximum(0, self.theta, self.w)
#             np.heaviside(self.theta, 0, self.c)

    def _on_pre(self):
        """TODO"""
        return self.eta * self.A * self.tr_post.val
    
    def _on_post(self):
        """TODO"""
        reshaped_tr_val = np.tile(self.tr_pre.val, (self.destination.num_compartments,self.destination.soma.size, 1))
        return self.eta * self.c_BAP *(reshaped_tr_val - self.gamma_BAP )
#         return self.eta * self.c_BAP * (reshaped_tr_val - self.gamma_BAP * (1 - reshaped_tr_val))

#         return self.eta * self.c_BAP * (self.tr_pre.val[:, np.newaxis, np.newaxis] - 
#                              self.gamma_BAP * (1 - self.tr_pre.val[:, np.newaxis, np.newaxis])).reshape(-1)
