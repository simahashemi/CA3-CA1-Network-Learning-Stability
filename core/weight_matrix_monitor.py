# -*- coding: utf-8 -*-
"""TODO"""

import numpy as np

import core.core_global as core
from core.core_definitions import simulation_timestep
from core.monitor import Monitor


class WeightMatrixMonitor(Monitor):
    """TODO"""

    def __init__(self, source, filename, cell_id = 0, interval=1.0):
        """TODO"""
        super().__init__(filename)

        core.kernel.register_device(self)

        self.source = source
        self.stepsize = int(interval / simulation_timestep)
        self.cell_id = cell_id

        if self.stepsize < 1:
            self.stepsize = 1

        self.outfile.write("# Recording with a sampling interval of %.2fs at a"
                           " timestep of %.2es\n".encode()
                           % (interval, simulation_timestep))
        self.outfile.write("# The shape (post size, pre size) of the matrix is"
                           " {0}\n".format(self.source.w[:,self.cell_id,:].shape).encode())
        


    def execute(self):
        """TODO"""
        if not self.active:
            return

        if self.source.get_destination().evolve_locally:
            if core.kernel.get_clock() % self.stepsize == 0:
#                   np.savetxt(self.outfile, self.source.w.reshape(-1, self.source.w.shape[-1]), fmt="%.6f",
#                            header= "%.6f" % (core.kernel.get_time()))
                np.savetxt(self.outfile, self.source.w[:,self.cell_id,:], fmt="%.6f",
                           header= "%.6f" % (core.kernel.get_time()))
