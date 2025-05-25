# -*- coding: utf-8 -*-
"""TODO"""

import numpy as np
import math
import copy

import core.core_global as core
from core.core_definitions import simulation_timestep
from core.mc_neuron_group import McNeuronGroup


class McLifGroup(McNeuronGroup):
    """TODO"""

    def __init__(self, size, num_branches, params={}):
        """TODO"""
        super().__init__(size, num_branches)

        core.kernel.register_spiking_group(self)

        if self.evolve_locally:
            self.branch = self.Branch((num_branches, self.rank_size), self,
                                      params.get("branch_parameters", {}))
            self.soma = self.Soma(self.rank_size, self.branch, self,
                                  params.get("soma_parameters", {}))

    def evolve(self):
        """TODO"""
        self.branch.evolve()
        self.soma.evolve()


    class Soma:
        """TODO"""

        def __init__(self, size, branch, nrn, params={}):
            """TODO"""
            self.size = size
            self.branch = branch
            self.nrn = nrn
            
            ############ reading parameters
            ### Network neuron number
            self.AcD_num = params.get("AcD_num", 60)
            self.nonAcD_num = params.get("nonAcD_num", 60)
            ### neuron parameters
            self.v_thr = np.full(self.size ,params.get("PC_v_thr", -50.0) )
            self.v_rest = np.full(self.size ,params.get("PC_v_rest", -67.0) )
            self.refractory_period = np.full(self.size ,params.get("PC_refractory_period", 2e-3) )
            self.r_mem = params.get("PC_r_mem", 40e6)
            self.c_mem = params.get("PC_c_mem", 275e-12)
            self.tau_mem = self.r_mem * self.c_mem
            self.tau_syn = params.get("tau_syn", 2e-3)
            self.r_l = params.get("r_l", 10.0e9)
            ####### inhibitory current
            self.inh_max_amp  = params.get("inh_amp", -1.5e-9)


            #######################################
            ############currents
            self._tmp_current = np.zeros(self.size)
            self.recurrent_current = np.zeros(self.size)
            self.br_current = np.zeros(self.size)
            self.I_inh  = np.zeros(self.size)

            ############membrane potentials
            ####AIS####
            self.mem = np.full(self.size , self.v_rest)
            self._tmp = np.zeros(self.size)
            self._last_mem = np.zeros(self.size)
            self._slope_mem = np.zeros(self.size)

            ###############STDP########
            self.cell_spike_list = [[] for cell_ind in range(self.size)]
            self.new_pairs = [[[] for post_cell in range(self.size)] for pre_cell in range(self.size)]
            self.new_spike = np.zeros(self.size)
            self.stdp_time_window = params.get("stdp_time_window", 50.0e-3)
            
            
            self.on_spike = np.zeros(self.size)
            
            self.ref = np.zeros(size, dtype=np.int)
            self.set_refractory_period(self.refractory_period)
            
            self.calculate_scale_constants()

        def rank2global(self, i):
            """TODO"""
            return self.nrn.rank2global(i)

        def get_post_size(self):
            """TODO"""
            return self.nrn.get_post_size()

        def calculate_scale_constants(self):
            """TODO"""
            self.scale_mem = simulation_timestep / self.tau_mem
            self.scale_syn = np.exp(-simulation_timestep / self.tau_syn)
            self.mul_syn = simulation_timestep / self.tau_syn


        def set_refractory_period(self, t):
            """TODO"""
            self.refractory_period = t / simulation_timestep

        def evolve(self):
            """TODO"""
            self.compute_inhibitory_current() ### calculate perisomatic inhibition
            self.compute_branch_current() ### calculate the I_inp to the AIS
            self.integrate_AIS_membrane() ### calculate the leaky integrator eq. for AIS
            self.check_thresholds() ### check for AP generation



        def compute_inhibitory_current(self):### Perisomatic inhibition
            """TODO""" 
            #### constant perisomatic inhibition
            self.I_inh = np.full(self.branch.size, self.inh_max_amp)### make an array of size(num_branches, num_cells) that indicates the value of inhibition on each branch
            self.I_inh[0,:self.AcD_num] = np.zeros(self.AcD_num) ### zero inhibition on the AcD branches

            
        def compute_branch_current(self): ### Somatic summation
            """TODO"""
            self.br_current = np.add.reduce(np.clip(self.branch.I_dend+self.I_inh,0,None))
            
            
        def integrate_AIS_membrane(self): # dV = dt/tau (V_rest - V(t)) + dt/c * I_branches + dt/c
            """TODO"""
            np.copyto(self._last_mem, self.mem)
            ### Leakage:  dt/tau (V_rest - V(t))
            np.subtract(self.v_rest, self.mem, self._tmp)
            np.multiply(self.scale_mem, self._tmp, self._tmp)
            ### current from branches to the AIS: dt/c * I_branches
            np.add(self._tmp, np.multiply(simulation_timestep*1.0e3/self.c_mem,self.br_current), self._tmp)### *1e3:converting V to mV
            ### calculate the new membrane potential: V(t+dt) = V(t) + dV
            np.add(self._tmp,self.mem, self.mem)
            ### calculate the slope of membrane potential
            np.subtract(self.mem, self._last_mem, self._slope_mem)


        def check_thresholds(self):
            """TODO"""
            self.new_spike = np.zeros(self.size)
            for i in range(self.size):
                if self.ref[i] == 0:
                    if self.mem[i] >= self.v_thr[i]:
                        self.nrn.push_spike(i)
                        self.new_spike[i] = core.kernel.get_time()
                        self.cell_spike_list[i].append(self.new_spike[i])
    
                        self.mem[i] = self.v_rest[i]
                        self.on_spike[i] = 1
                        self.ref[i] = self.refractory_period[i]
                else:
                    self.mem[i] = self.v_rest[i]
                    self.on_spike[i] = 0
                    self.ref[i] -= 1

                    
                    

    class Branch:
        """TODO"""

        def __init__(self, size, nrn, params={}):
            """TODO"""
            self.size = size
            self.nrn = nrn
            ############ reading parameters
            self.v_thr = params.get("v_thr", -60.0)
            self.v_rest = params.get("v_rest", -67.0)
            self.r_mem = params.get("r_mem", 40e6)
            self.c_mem = params.get("c_mem", 250e-12)
            self.tau_mem = self.r_mem * self.c_mem
            self.tau_syn = params.get("tau_syn", 3.0e-3)
            self.branch_dynamics = params.get("branch_dynamics" , True)
            self.AcD_num = params.get("AcD_num", 75)
            self.nonAcD_num = params.get("nonAcD_num", 75)
            self.refractory_time = params.get("dendrite_refractory_period" , 5.0e-3)
            self.dendritic_spike_max_amp = params.get("dendritic_spike_max_amp" , 500.0e-12)
            self.tau_decay = params.get("d_spike_decay_time" , 4.0e-3)
            self.tau_rise = params.get("d_spike_rise_time" , 1.0e-3)
            self.tau_DS = params.get("d_spike_delay" , 2.7e-3) 
            self.d_spike_memory_time_window = params.get("d_spike_memory_time_window" , 20.0e-3)
            self.pattern_duration = params.get("pattern_duration" , 60e-3)
            self.pattern_delay = params.get("pattern_delay" , 1.0)
            
            ############# membrane potentials
            self.v_thr = np.full(self.size, params.get("v_thr", -57.0))
            self.mem = np.full(self.size, self.v_rest)
            self._tmp = np.zeros(self.size)
            self._last_mem = np.zeros(self.size)
            self._slope_mem = np.zeros(self.size)
            self._tmp_current = np.zeros(self.size)
            
            ####### inhibitory current
            self.inh_max_amp  = params.get("inh_amp", 0.0)
            self.inh_freq = params.get("inh_freq", 0.0)
            self.inh_phase = params.get("inh_phase", 0.0)
            self.inh_current = np.zeros(self.size)
            self.I_inh  = np.zeros(self.size)
            
            ############## currents
            self._current = np.zeros(self.size)
            self.syn_current = np.zeros(self.size)
            self._a_sod = np.zeros(self.size)
            
            ############# plateaus
            self.dendritic_spike_list = np.zeros(self.size)
            self.I_dend = np.zeros(self.size)
            self.pla = np.zeros(self.size, dtype=np.int)
            self.pla_on = np.zeros(self.size, dtype=np.int)
            
            self.all_dendritic_spike_list  = np.empty(self.size, dtype=object)
            self.all_dendritic_spike_list[:] = [[[] for _ in range(self.size[1])] for _ in range(self.size[0])]

           

            self.calculate_scale_constants()

            if self.branch_dynamics:
                self.threshold_function = self.check_thresholds
            else:
                self.threshold_function = self.check_thresholds2

        def rank2global(self, i):
            """TODO"""
            return self.nrn.rank2global(i)

        def get_post_size(self):
            """TODO"""
            return self.nrn.get_post_size()

        def calculate_scale_constants(self):
            """TODO"""
            self.scale_mem = simulation_timestep / self.tau_mem
            self.mul_syn = simulation_timestep / self.tau_syn
            self.scale_syn = np.exp(-simulation_timestep / self.tau_syn)

        def evolve(self):
            """TODO"""
            self.integrate_synapses() ### calculate CA3 input current
            self.compute_inhibitory_current()
            self.integrate_membrane() ### calculate leaky integrator for each dendritic branch
            self.threshold_function() ### check for d-spike
            self.calculate_I_dend() ### calculate the I_dend for each dendritic spike

        def integrate_synapses(self): ### current from CA3 neurons
            """TODO"""
            np.multiply(self.syn_current, self.scale_syn, self.syn_current)
            np.add(np.multiply(
                np.subtract(self.syn_current, self._tmp_current),
                self.mul_syn), self._tmp_current, self._tmp_current)
            np.multiply(100e-12*np.e, self._tmp_current, self._current)### calculate current in Amper unit


        def compute_inhibitory_current(self):### inhibitory input
            """TODO"""
            self.inh_current = self.inh_max_amp*(1 + np.sin(2*np.pi*self.inh_freq*(core.kernel.get_time()% (self.pattern_duration + self.pattern_delay))))*(math.ceil( (core.kernel.get_time()% (self.pattern_duration + self.pattern_delay))-self.pattern_delay ))
            self.I_inh = np.full(self.size, self.inh_current)### make an array of size(num_branches, num_cells) that indicates 
        
        def integrate_membrane(self):
        ### dV = dt/tau (V_rest - V(t)) + dt/c * I_CA3 + dt/c * I_inh
            """TODO"""
            np.copyto(self._last_mem, self.mem)
            ### Leakage: dt/tau (V_rest - V(t))
            np.subtract(self.v_rest, self.mem, self._tmp)
            np.multiply(self.scale_mem, self._tmp, self._tmp)
            ### CA3 input current: dt/c * I_CA3
            np.add(self._tmp, np.multiply(simulation_timestep*1e3/self.c_mem,self._current), self._tmp) ### *1e3:converting V to mV
            ### calculate dendritic inhibition(during sleep)
            np.add(self._tmp, np.multiply(simulation_timestep*1e3/self.c_mem,self.I_inh), self._tmp)### *1e3:converting V to mV
            ### calculate the new membrane potential: V(t+dt) = V(t) + dV
            np.add(self._tmp,self.mem, self.mem)
            ### calculate the membrane potential slope
            np.subtract(self.mem, self._last_mem, self._slope_mem)

            
        def check_thresholds(self):
            """TODO"""
            for i in range(self.size[1]):
                for j in range(self.size[0]):
                    if self.pla[j, i] == 0:
                        if self.mem[j, i] >= self.v_thr[j,i] :
                            self.dendritic_spike_list[j,i] = core.kernel.get_time()
                            self.all_dendritic_spike_list[j,i].append(self.dendritic_spike_list[j,i])

                            self.pla[j, i] = self.refractory_time/simulation_timestep
                            self.pla_on[j, i] = 1
                            self._a_sod[j, i] = 30.0
                            self.mem[j, i] = self.v_rest + self._a_sod[j, i]
                            

                    else:
                        self.pla[j, i] -= 1
                        self.pla_on[j, i] = 0
                        if self.pla[j, i] == 0:
                            self.mem[j, i] = self.v_rest
                        else:
                            self._a_sod[j, i] *= np.exp(-simulation_timestep / 4e-3)#self.tau_decay
                            self.mem[j, i] = self.v_rest


        
        
        
        def calculate_I_dend(self):
            ###with delay: I_max(exp((Delta_t-delay)/Tau_rise) - exp((Delta_t-delay)/Tau_decay)        
            filtered_dendritic_spike_list = np.array([[ [val for val in sublist if val >= core.kernel.get_time() - self.d_spike_memory_time_window] 
                                                   for sublist in row] 
                                                  for row in self.all_dendritic_spike_list], dtype=object)

            if np.all([len(sublist) == 0 for row in filtered_dendritic_spike_list for sublist in row]): 
                self.I_dend = np.zeros(self.size)
            else: 
                self.I_dend = np.array([
                              [
                               [sum(self.dendritic_spike_max_amp * np.heaviside((core.kernel.get_time() - elem -self.tau_DS),0)*(np.exp(-(core.kernel.get_time() - elem - self.tau_DS)/self.tau_decay) - np.exp(-(core.kernel.get_time() - elem - self.tau_DS)/self.tau_rise)) 
                                 for elem in sublist)][0]
                                  for sublist in row] 
                                    for row in filtered_dendritic_spike_list])
                
