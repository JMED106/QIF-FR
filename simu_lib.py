import numpy as np
from scipy import stats, special
from scipy.fftpack import dct
import logging
from libNeuroDyn import lorentz, gauss
from sconf import create_dir, now

logging.getLogger('simu_lib').addHandler(logging.NullHandler())

__author__ = 'Jose M. Esnaola Acebes'

""" This file contains classes and functions to be used in the QIF network simulation.

    Data: (to store parameters, variables, and some functions)
    *****
"""

pi = np.pi


class Data:
    """ The data structure will have a general structure but must be shaped to match the simulation
        parameters and variables.
    """

    def __init__(self, opts, external=(None,)):
        self.logger = logging.getLogger('simu_lib.Data')
        self.logger.debug("Creating data structure.")

        # Mutable parameters will be stored in a dictionary called prmts
        self.opts = opts
        # Non-mutable parameters will be stored as separated variables
        self.t0 = opts['t0']  # Initial time
        self.tfinal = opts['tfinal']  # Final time
        self.total_time = opts['tfinal'] - opts['t0']  # Time of simulation
        self.dt = opts['dt']  # Time step
        self.taue = opts['taue']
        self.taui = opts['taui']
        self.faketau = float(opts['faketau'])
        self.n = opts['n']
        self.ne = opts['ne']
        self.ni = opts['ni']

        if self.ne + self.ni != 1:
            self.logger.warning("The proportion e:i of neurons %d:%d is odd..." % (self.ne * 100, self.ni * 100))

        # 0.2) Define the temporal resolution and other time-related variables
        self.tpoints = np.arange(self.t0, self.tfinal, self.dt)  # Points for the plots and others
        self.nsteps = len(self.tpoints)  # Total time steps

        self.systems = opts['systems']
        self.logger.debug("Systems to be simulated: %s" % self.systems)
        if set(self.systems).intersection({'nf', 'wc-nf', 'qif-nf', 'if-nf', 'eif-nf'}):
            self.nf = True
            self.logger.info("Type of system: Neural Network.")
        else:
            self.nf = False
            self.logger.info("Type of system: Neural Population(s).")
            if self.n > 2:
                self.n = 2
                self.opts['n'] = 2
                self.opts['network']['n'] = 2

        opts.update({'nf': self.nf})

        # Conductance-based neurons
        if opts.get('cond', False):
            self.cond = True
        else:
            self.cond = False

        # Other simulation options here:
        self.all = opts['ap']  # In the GUI it will expand all the mutable parameters entered in the conf file.

        # Simulation control variables
        self.opts['controls'] = {'exit': False, 'pause': True, 'stop': False, 'x': None, 'y': None}
        # Simulation raster control flags
        self.opts['raster'] = {'start': False, 'update': False, 'rate': 100, 'dynamic': False, 'pop': None}

        # Extra objects, such as perturbations, measures, etc.
        # TODO
        external = list(external)
        for obj in external:
            if obj:
                pass

        # ######## Edit below (and "population" function) to adapt for different simulations #####################
        self.vars = {'t': self.tpoints, 'tstep': 0, 'temps': 0.0, 'dummy': np.ones(self.nsteps) * 0.8, 'cycle': 0}
        self.lims = {'t': [0.0, self.tfinal * self.faketau], 're': [0, 100], 'ri': [0, 100], 'rwe': [0, 100], 'rwi': [0, 100],
                     'Pe': [-np.pi, np.pi],
                     've': [-2, 2], 'vi': [-2, 2], 'se': [0, 2], 'si': [0, 2], 'Pi': [-np.pi, np.pi]}
        # In case we have a neural field:
        if self.n > 2:
            self.vars.update({'phi': np.linspace(-np.pi, np.pi, self.n)})
            self.lims.update({'phi': [-np.pi, np.pi]})
            self.nf = True
            it = np.ones((self.nsteps, self.n)) * 0.0
        else:
            if not self.nf:
                self.n = 1
                opts['n'] = 1
                opts['network']['n'] = 1
                self.nf = False
            it = np.ones(self.nsteps) * 0.0

        # External time-dependent input current
        self.vars.update({'it': it})

        # Output variables will be stored in dictionaries to make the Queue handling easy
        exc, inh = (None, None)
        if set(self.systems).intersection({'fr', 'wc'}):
            exc = self.single_population(self.nsteps, 2.0, -1.0, 0.0, name="e")
            inh = self.single_population(self.nsteps, 1.0, -0.5, 0.0, name="i")
        elif set(self.systems).intersection({'nf', 'wc-nf'}):
            exc = self.network_population(self.nsteps, self.n, 2.0, -1.0, 0.0, name="e")
            inh = self.network_population(self.nsteps, self.n, 2.0, -1.0, 0.0, name="i")

        if set(self.systems).intersection({'fr', 'wc', 'nf', 'wc-nf'}):
            self.vars.update(exc)
            self.vars.update(inh)
            del exc, inh
        else:
            self.vars.update({'re': 0, 'ri': 0, 've': 0, 'vi': 0, 'rwe': 0, 'rwi': 0, 'ser': 0, 'sir': 0, 'swer': 0, 'swir': 0})

        # Spiking neurons
        # General configuration (number of neurons in each population, etc.)
        if set(self.systems).intersection({'qif-fr', 'if-fr', 'eif-fr', 'qif-nf', 'if-nf', 'eif-nf'}):
            # Configuration of the spiking neurons:
            self.vpeak = 100.0
            self.vreset = -100.0
            self.vth = 0.0  # Threshold voltage (not implemented)
            self.rte = self.taue * (1.0 / self.vpeak - 1.0 / self.vreset)
            self.rti = self.taui * (1.0 / self.vpeak - 1.0 / self.vreset)
            self.tpeak_e = self.taue / self.vpeak
            self.tpeak_i = self.taui / self.vpeak
            self.opts['sp'] = 'qif'
            if set(self.systems).intersection({'if-fr', 'eif-fr', 'if-nf', 'eif-nf'}):
                self.vpeak = opts.get('vpeak', 0)
                self.vreset = opts.get('vreset', -60.0)
                self.reversal = opts.get('revers', -50.0)
                self.sharp = opts.get('sharp', 3)
                self.rheo = opts.get('rheo', -53)
                if set(self.systems).intersection({'eif-fr', 'eif-nf'}):
                    self.rte = opts.get('rperiod', 0.25)
                    self.rti = self.rte*1.0
                    # self.rte = 0.1  # 5 ms assuming tau = 20 ms and dt = 2 * 10^-5 s
                    # self.rti = 0.1
                self.tpeak_e = self.dt
                self.tpeak_i = self.dt
                if set(self.systems).intersection({'if-fr', 'if-nf'}):
                    self.opts['sp'] = 'if'
                else:
                    self.opts['sp'] = 'eif'
            # Configuration of the network
            self.N = opts['N']
            self.Ne = int(self.N * self.ne)
            self.Ni = int(self.N * self.ni)
            self.logger.debug("Total number of neurons: %d."
                              "\n\t\t\t\t\t\tExcitatory: %d.\n\t\t\t\t\t\tInhibitory: %d." % (self.N, self.Ne, self.Ni))
            opts.update({'Ne': self.Ne, 'Ni': self.Ni})
            # It is necessary to load the FR class to be able to measure something
            self.fr = FiringRate(opts)
            self.sye = np.ones(self.nsteps) * 0.0
            self.syi = np.ones(self.nsteps) * 0.0
            self.vars.update({'sp_re': self.fr.re, 'sp_ri': self.fr.ri, 'sp_r': self.fr.r,
                              'tfr': self.fr.t, 'frtstep': self.fr.tstep, 'frtstep2': self.fr.tstep2, 'sye': self.sye,
                              'syi': self.syi})
            self.lims.update({'tfr': [0.0, self.tfinal * self.faketau], 'sp_re': [0.0, 100.0], 'sp_ri': [0.0, 100.0]})

            # Synaptic activation computation requires a convolution with a weighting function (Heaviside, expo, alpha)
            self.t_syn = opts.get('tsyn', 10)
            self.t_syn = 1
            self.a_tau = self.synaptic_activation(self.t_syn, self.dt)
            # Distribution of external currents
            self.eta_e = self.external_currents(opts['etae'], opts['delta'], self.Ne, n=self.n, distribution=opts['D'])
            self.eta_i = self.external_currents(opts['etai'], opts['delta'], self.Ni, n=self.n, distribution=opts['D'])
            # Distribution of reversal potentials
            if self.cond:
                self.rev_e = self.external_currents(opts['reverse'], opts['gamma'], self.Ne, n=self.n,
                                                    distribution=opts['D'])
                self.rev_i = self.external_currents(opts['reversi'], opts['gamma'], self.Ni, n=self.n,
                                                    distribution=opts['G'])

            # Matrices containing voltages, spikes, and times of the spikes
            m_type = np.dtype([('i', np.int32), ('v', np.float64), ('t', np.float32), ('s', np.int8)])
            self.m_e = np.ndarray([self.Ne], dtype=m_type)
            self.m_e['i'] = range(self.Ne)
            # self.m_e['v'] = np.random.randn(self.Ne)
            self.m_e['v'] = np.ones(self.Ne) * (-0.1)
            self.m_e['t'] = 0.0
            self.m_e['s'] = 0
            self.m_i = np.ndarray([self.Ni], dtype=m_type)
            self.m_i['i'] = range(self.Ni)
            # self.m_i['v'] = np.random.randn(self.Ni)
            self.m_i['v'] = np.ones(self.Ni) * (-0.1)
            self.m_i['t'] = 0.0
            self.m_i['s'] = 0

            # Matrices containing spikes, to be able to compute the synaptic activation
            self.spk_e = np.ones(shape=(self.Ne, self.t_syn), dtype=np.int8) * 0
            self.spk_i = np.ones(shape=(self.Ni, self.t_syn), dtype=np.int8) * 0
            # Matrices registering the spikes in the appropriate time step (takes into account the refractory period)
            self.spk_time_e = int(self.tpeak_e / self.dt)
            self.spk_time_i = int(self.tpeak_i / self.dt)
            if self.spk_time_e == 0 or self.spk_time_i == 0:
                self.spk_e_mod = None
                self.spk_i_mod = None
            else:
                self.spk_e_mod = np.ones(shape=(self.Ne, self.spk_time_e), dtype=np.int8) * 0
                self.spk_i_mod = np.ones(shape=(self.Ni, self.spk_time_i), dtype=np.int8) * 0
        else:
            self.fr = None

        if set(self.systems).intersection({'qif-fr', 'if-fr', 'eif-fr'}):
            # Anything special for the spk-n fr simulations goes here
            pass
        elif set(self.systems).intersection({'qif-nf', 'if-nf', 'eif-nf'}):
            self.dN = self.N / self.n
            self.dNe = self.Ne / self.n
            self.dNi = self.Ni / self.n
            # Auxiliary matrices for the dot product
            self.auxMatE = np.zeros((self.n, self.Ne))
            self.auxMatI = np.zeros((self.n, self.Ni))
            for i in xrange(self.n):
                self.auxMatE[i, i * self.dNe:(i + 1) * self.dNe] = 1.0
                self.auxMatI[i, i * self.dNi:(i + 1) * self.dNi] = 1.0

            self.auxMat = np.zeros((self.n, self.N))
            for i in xrange(self.n):
                self.auxMat[i, i * self.dN:(i + 1) * self.dN] = 1.0
            self.aux = {'e': self.auxMatE, 'i': self.auxMatI, 'r': self.auxMat}

            # Arrays to select different populations in the raster plot
            self.pope = {}
            self.popi = {}
            for n in xrange(self.n):
                self.pope[n] = np.arange(n * self.dNe, (n+1) * self.dNe)
                self.popi[n] = np.arange(n * self.dNi, (n + 1) * self.dNi)

        # Create a registry of the dimensions of the variables
        for var in self.vars:
            shape = np.shape(self.vars[var])
            # self.logger.debug("Dim of %s: %s." % (var, str(shape)))
            if len(list(shape)) > 1:
                cols = len(self.vars[var][0])
                # self.logger.debug("\tIs a Matrix with %d colums." % cols)
                try:
                    self.lims[var] = list(np.concatenate((np.array(self.lims[var]), np.array([cols]))))
                    self.logger.debug("\tLim array is now %s" % self.lims[var])
                except KeyError:
                    pass

        self.Save = SaveResults

    @staticmethod
    def single_population(nsteps, r0=1.0, v0=-1.0, s0=0.0, name=""):
        r = np.ones(nsteps) * 0.1
        rw = np.ones(nsteps) * 0.1
        v = np.ones(nsteps) * (-0.01)
        r[len(r) - 1] = r0
        v[len(v) - 1] = -v0
        s = np.ones(nsteps) * 0.1
        s[len(s) - 1] = s0
        sw = np.ones(nsteps) * 0.1
        sw[len(sw) - 1] = s0
        # Kuramoto order parameter and phase
        R = np.ones(nsteps) * 0.1
        P = np.ones(nsteps) * 0.1
        return {'rw' + name: rw, 'r' + name: r, 'v' + name: v, 's' + name + 'r': s, 'sw' + name + 'r': sw, 'R' + name: R, 'P' + name: P}

    @staticmethod
    def network_population(nsteps, n, r0=1.0, v0=-1.0, s0=0.0, name=""):
        r = np.ones((nsteps, n)) * 0.0
        v = np.ones((nsteps, n)) * 0.0
        r[len(r) - 1, :] = r0
        v[len(v) - 1, :] = v0
        s = np.ones((2, n)) * 1.0
        s[len(s) - 1, :] = s0
        sw = np.ones((2, n)) * 0.1
        sw[len(sw) - 1, :] = s0
        # Kuramoto order parameter and phase
        R = np.ones(nsteps) * 0.1
        P = np.ones(nsteps) * 0.1
        # WC neural field
        rw = np.ones((nsteps, n)) * 0.1
        return {'rw' + name: rw, 'r' + name: r, 'v' + name: v, 's' + name + 'r': s, 'sw' + name + 'r': sw, 'R' + name: R, 'P' + name: P}

    @staticmethod
    def synaptic_activation(tsyn_steps, dt, funct='heaviside'):
        tau_syn = tsyn_steps * dt
        h_tau = 1.0 / tau_syn
        if funct == 'heaviside':
            a_tau0 = np.transpose(h_tau * np.ones(tsyn_steps))
        elif funct == 'exp_decay':
            a_tau0 = np.transpose(h_tau * np.array(np.exp(-dt * h_tau * np.arange(tsyn_steps))))
        else:
            return None
        a_tau = np.zeros((tsyn_steps, tsyn_steps))
        for i in xrange(tsyn_steps):
            a_tau[i] = np.roll(a_tau0, i, 0)
        return a_tau

    @staticmethod
    def external_currents(center, width, pop, n=1, distribution='lorentz'):
        # Implemented distributions:
        dist = {'lorentz': lorentz, 'gauss': gauss}
        if distribution == 'noise':
            logging.info("Setting an homogeneous population (identical neurons).")
            eta_i = np.ones(pop / n) * center
        else:
            logging.info("Setting an heterogeneous population.")
            logging.debug("Distribution of external currents: %s" % distribution)
            try:
                eta_i = dist[distribution](pop / n, center, width)
            except KeyError:
                logging.error("Distribution %s not implemented." % distribution)
                return -1
        # Set distributions for each node in case is a NF
        if n > 2:
            eta = np.zeros(pop)
            for i in xrange(n):
                eta[i * (pop / n):(i + 1) * (pop / n)] = 1.0 * eta_i
            del eta_i
            eta_i = eta * 1.0
        return eta_i


class FiringRate:
    """ Firing rate measurement of populations of spiking neurons."""

    def __init__(self, opts):
        self.logger = logging.getLogger('simu_lib.FiringRate')
        self.name = 'firingrate'

        # Parameters of the simulation:
        self.o = opts
        self.dt = self.o['dt']
        self.N = self.o['N']
        self.nsteps = int((opts['tfinal'] - opts['t0']) / self.dt)
        self.n = self.o['n']
        if self.n > 2:
            self.nf = True
            self.logger.debug("Firing rate measure for a neural field.")
        else:
            self.nf = False
            self.logger.debug("Firing rate measure for a single population.")

        try:
            self.Ne = self.o['Ne']
            self.Ni = self.o['Ni']
            if self.nf:
                self.dN = self.N / self.n
                self.dNe = self.Ne / self.n
                self.dNi = self.Ni / self.n
        except KeyError:
            self.logger.warning("Trying to create Firing Rate object without having spiking neurons!")
            raise KeyError

        # Fundamental parameters of FR measurement: sliding window, sampling rate,
        #                                           convolution function (heaviside, gauss).
        self.fo = opts['firingrate']
        self.sld_windw = None  # Sliding window in simulation time units (dt units)
        self.sld_steps = None  # Sliding window in simulation time steps
        self.wones = None  # Auxiliary vector of ones (for dot product)

        # Rate of sampling
        self.spl_time = None
        self.spl_steps = None

        # Time vector and firig rate vectors for FR measure
        self.t = None
        self.length = None
        self.tstep = 0
        self.tstep2 = 0
        self.re = None
        self.ri = None
        self.r = None

        self.spikes_e = None
        self.spikes_i = None

        self.update(self.fo)

    def update(self, opts, **kwargs):
        """ Function to prepare the firing rate observables """
        self.o.update(opts)
        self._init_prmts()
        self.t = self._t_vector()
        self.length = len(self.t)
        self.re = self._r_vector((self.length, self.o['n']))
        self.ri = self._r_vector((self.length, self.o['n']))
        self.r = self._r_vector((self.length, self.o['n']))
        try:
            a = kwargs['tstep'] * 1
            del a
            # TODO: be able to change the FR vector size in the shared memory or use the output queue (easier)
            self.logger.warning("Firing Rate measurement parameters can't "
                                "be changed (yet) without restarting the program.")
        except KeyError:
            pass

    def _r_vector(self, size):
        """ Returns a FR vector with a size that depends on the sampling rate."""
        if type(size) is not tuple:
            self.logger.error("Size must be tuple T x n")
            return -1
        tlen = size[0]
        n = size[1]
        if self.nf:  # For NF computing (n represents number of nodes, spatial dimension)
            r = 0.0 * np.zeros(shape=(tlen, n))
        else:
            r = 0.0 * np.zeros(tlen)
        return r

    def _t_vector(self):
        """ Returns a Time vector with a size that depends on the sampling rate."""
        # Measures of the firing rate are done every self.spl_steps
        vt_steps = np.arange(int(self.o['t0'] / self.dt) + self.sld_steps, self.nsteps, self.spl_steps)
        return (vt_steps - self.sld_steps / 2) * self.dt

    def _init_prmts(self):
        """ Function to set up the basic parameters for measurement."""
        # Fundamental parameters of FR measurement: sliding window, sampling rate,
        #                                           convolution function (heaviside, gauss).
        self.sld_windw = self.fo['sw']  # Sliding window in simulation time units (dt units)
        self.sld_steps = int(self.sld_windw / self.dt)  # Sliding window in simulation time steps
        self.wones = np.ones(int(self.sld_steps))  # Auxiliary vector of ones (for dot product)
        self.tstep = 0
        # Rate of sampling
        self.spl_time = self.fo['spr']
        self.spl_steps = int(self.spl_time / self.dt)

        # TODO: Compute how much memory is going to be use, and check availability

        # Create Spikes matrices. DOES NOT depend on the number of populations (n).
        #                         IT DOES depend on the type of populations (exc, inh, ...)
        # During the simulation the spikes are transferred to the following matrices
        self.spikes_e = 0 * np.zeros(shape=(self.Ne, self.sld_steps))
        self.spikes_i = 0 * np.zeros(shape=(self.Ni, self.sld_steps))

    def firing_rate(self, tstep, temps, var, aux=None):
        """ Computes the firing rate for a given matrix of spikes. Firing rate is computed
            every certain time (sampling). Therefore at some time steps the firing rate is not computed,
        :param tstep: time step of the simulation
        :param temps: time of the simulation
        :param var: variable dictionary (where re and ri and r are)
        :param aux: auxiliary matrix for neural field computation (defined in data)
        :return: nothing. modifies self.t, self.re, self.ri. self.r)
        """
        if tstep % self.spl_steps == 0 and (temps >= self.sld_windw):
            # We divide by cases ('nf', 'fr', 'exc', 'inh', ...)
            if self.nf and aux:
                if not aux:
                    self.logger.error("Auxiliary matrix is needed.")
                    return -1
                var['sp_re'][self.tstep] = (1.0 / self.sld_windw / self.dNe) * np.dot(aux['e'],
                                                                                      np.dot(self.spikes_e, self.wones))
                var['sp_ri'][self.tstep] = (1.0 / self.sld_windw / self.dNi) * np.dot(aux['i'],
                                                                                      np.dot(self.spikes_i, self.wones))
            else:
                var['sp_re'][self.tstep] = (1.0 / self.dt) * self.spikes_e.mean()
                var['sp_ri'][self.tstep] = (1.0 / self.dt) * self.spikes_i.mean()

            var['sp_r'][self.tstep] = (var['sp_re'][self.tstep] + var['sp_ri'][self.tstep]) / 2.0
            self.tstep += 1
            self.tstep2 += 1
            self.tstep = self.tstep % self.length
            var['frtstep'].value = self.tstep
            var['frtstep2'].value = self.tstep2


class Connectivity:
    def __init__(self, opts, kind='effective'):
        self.log = logging.getLogger('simu_lib.Connectivity')
        self.name = 'network'
        # Connectivity set up
        self.log.debug("Setting up connectivity, type: %s" % kind)
        # Mutable parameters of the connectivity must be merged with parameters['parameters']
        o = opts['network']  # Network parameters
        self.n = opts['n']  # Number of nodes in the network
        # Basic connectivity elements
        [i, j] = np.meshgrid(xrange(self.n), xrange(self.n))
        self.ij = (i - j) * (2.0 * pi / self.n)  # Grid
        del i, j
        self.eigenmodes = None
        self.eigenvectors = None
        self.kind = kind

        profile = o['c']
        self.log.debug("Connectivity profile: %s" % profile)
        # Types of connectivity: FS (Fourier Series), Mex-Hat, Random Laplace
        self.cntswitch = {'fs': self.cosine_series, 'cs': self.cosine_series, 'mex-hat': None, 'twopops': self.twopop}
        self.c, prmts = self.cntswitch[profile](o, kind)
        opts['parameters'].update(prmts)

    def update(self, opts, **kwargs):
        try:
            a = kwargs['tstep'] * 1
            del a
        except KeyError:
            pass

        self.c, dummy = self.cntswitch[opts['c']](opts, self.kind)

    def cosine_series(self, opts, kind='effective'):
        sign = {'exc': 1.0, 'inh': -1.0}
        # Default modes of effective (jk) connectivity are 0, 10, 7.5, -2.5
        if 'jk' in opts:
            modes = np.array(opts['jk'])
            self.log.debug("Loading custom modes of effective connectivity: %s" % modes)
        else:
            modes = 10.0 * np.array([0, 1, 0.75, -0.25])
            self.log.debug("Loading default modes of effective connectivity: %s" % modes)

        eimodes = {}
        if kind in ('exc', 'inh'):
            key = 'jk' + kind[0]
            if key in opts:
                modes = sign[kind] * np.array(opts[key])
                self.log.debug("Using custom modes of connectivity: %s" % modes)
                cnt = self.jcntvty(modes, coords=self.ij)
            else:
                self.log.debug("Using the effective connectivity to compute the %s cntvty. profile." % kind)

                # Separating connectivity modes to build a excitatory modulated connectivity and a flat inhibitory
                # connectivity. Create a dummy connectivity to see the minimum value:
                minvalue = np.min(self.jcntvty(modes, coords=self.ij)[0])
                self.log.debug('Minumum value of the cntvty.: %f. Projected mode 0 value: %f' %
                               (minvalue, -1.0 * np.floor(minvalue)))
                minvalue = np.floor(minvalue)
                eimodes = {'exc': list(modes), 'inh': [0.0, 0.0]}
                if minvalue < 0:
                    newmode0 = -1.0 * minvalue
                else:
                    newmode0 = modes[0]
                mode0 = modes[0]
                if mode0 < newmode0:
                    eimodes['exc'][0] = newmode0
                    eimodes['inh'][0] = minvalue
                    self.log.debug('Mode 0 set to %f' % eimodes['exc'][0])
                self.log.debug("%s modes: %s" % (kind, str(eimodes[kind])))
                cnt = self.jcntvty(eimodes[kind], coords=self.ij)
                modes = eimodes[kind]
            eimodes[key] = modes
        else:
            cnt = self.jcntvty(modes, coords=self.ij)
        self.eigenmodes = modes
        return cnt, eimodes

    def twopop(self, opts, kind='sym'):
        # Four parameters jee, jei, jii, jie
        ds = {'jee', 'jii', 'js'}.intersection(opts)
        dc = {'jei', 'jie', 'jc'}.intersection(opts)
        if len(ds) > 2 or len(dc) > 2:
            self.log.warning("Redundant information about connectivity, use either jc/js, or j{e}{i}. Expect the "
                             "unexpected.")
            # In case is symmetrical, jee=jii=js, jei=jie=jc
        j = {}
        if kind == 'sym':
            if ds and dc:
                for key in ds:
                    j['js'] = opts[key]
                for key in dc:
                    j['jc'] = opts[key]
            else:
                j['js'] = opts['jke'][0]
                j['jc'] = opts['jke'][1]
        elif ds and dc:
            for key in ds.union(dc):
                j[key] = opts[key]
        else:
            j['jee'], j['jei'], j['jii'], j['jie'] = (opts['jke'][0], opts['jke'][1], opts['jki'][0], opts['jki'][1])
        return j, j

    @staticmethod
    def gauss0_pdf(x, std):
        return stats.norm.pdf(x, 0, std)

    @staticmethod
    def mexhat0(a1, std1, a2, std2, length=500):
        x = np.linspace(-np.pi, np.pi, length)
        return x, a1 * Connectivity.gauss0_pdf(x, std1) + a2 * Connectivity.gauss0_pdf(x, std2)

    @staticmethod
    def vonmises(je, me, ji, mi, length=None, coords=None):
        if coords is None:
            if length is None:
                length = 500
            theta = (2.0 * np.pi / length) * np.arange(length)
        else:
            theta = 1.0 * coords
        return je / special.i0(me) * np.exp(me * np.cos(theta)) - ji / special.i0(mi) * np.exp(mi * np.cos(theta))

    @staticmethod
    def jcntvty(jk, coords=None):
        """ Fourier series generator.
        :param jk: array of eigenvalues. Odd ordered modes of Fourier series (only cos part)
        :param coords: matrix of coordinates
        :return: connectivity matrix J(|phi_i - phi_j|)
        """
        jphi = 0
        for i in xrange(len(jk)):
            if i == 0:
                jphi = jk[0]
            else:
                # Factor 2.0 is to be coherent with the computation of the mean-field S, where
                # we devide all connectivity profile by (2\pi) (which is the spatial normalization factor)
                jphi += 2.0 * jk[i] * np.cos(i * coords)
        return jphi

    @staticmethod
    def jmodes0(a1, std1, a2, std2, n=20):
        return 1.0 / (2.0 * np.pi) * (
            a1 * np.exp(-0.5 * (np.arange(n)) ** 2 * std1 ** 2) + a2 * np.exp(-0.5 * (np.arange(n)) ** 2 * std2 ** 2))

    @staticmethod
    def jmodesdct(jcnt):
        """ Extract fourier first 20 odd modes from jcnt function.
        :param jcnt: periodic odd function.
        :return: array of nmodes amplitudes corresponding to the FOurie modes
        """
        l = np.size(jcnt)
        jk = dct(jcnt, type=2, norm='ortho')
        for i in xrange(len(jk)):
            if i == 0:
                jk[i] *= np.sqrt(1.0 / (4.0 * l))
            else:
                jk[i] *= np.sqrt(1.0 / (2.0 * l))
        return jk

    @staticmethod
    def vonmises_modes(je, me, ji, mi, n=20):
        """ Computes Fourier modes of a given connectivity profile, built using
            Von Mises circular gaussian functions (see Marti, Rinzel, 2013)
        """
        modes = np.arange(n)
        return je * special.iv(modes, me) / special.i0(me) - ji * special.iv(modes, mi) / special.i0(mi)


class Perturbation:
    """ Tool to handle perturbations: time, duration, shape (attack, decay, sustain, release (ADSR), etc. """

    def __init__(self, opts, **kwargs):
        self.logger = logging.getLogger('tools.Perturbation')
        self.name = 'perturbation'

        # The class will create a external current "it" vector containing any external perturbation
        # Depending on the selected system, mainly "fr" or "nf" the possible properties of the perturbation vary
        # but many properties are common: such as amplitude, duration of the pulse (if pulse), rising and decay, etc.

        # Common fixed parameters:
        self.n = opts['n']
        self.nf = opts['nf']
        self.dt = opts['dt']
        self.nsteps = int(opts['tfinal'] / self.dt)
        # Spatial modulation (wavelengths)
        self.random = False
        self.pstart = None
        self.pend = None
        self.phi = np.linspace(-pi, pi, self.n)
        self.it = None
        self.active = True
        self.systems = opts['systems']
        opts.update(kwargs)
        self.update(opts[self.name], 0)

    def check(self, tstep):
        # self.logger.debug("Perturbation end: %d, current tstep: %d" % (self.pend, tstep))
        if tstep >= self.pend:
            self.logger.debug("Deactivating perturbation.")
            self.it = self.it * 0.0
            self.active = False

    def update(self, opts, tstep, tag=None):
        self.pstart = tstep * 1
        it = None
        try:
            it = self.it * 1.0
        except TypeError:
            pass
        if tstep != 0:
            opts['p0'] = self.pstart * self.dt + self.dt
        else:
            self.pstart = opts['p0'] / self.dt
        self.it = self._time_modulation(**opts)
        self.active = True
        if self.nf:
            self.it = opts['amp'] * np.dot(np.array((self.it,)).T, (self._spatial_modulation(**opts),))
        else:
            self.it = opts['amp'] * self.it
        try:
            self.it += it
        except TypeError:
            pass

        # opts.update({'it': self.it})
        if tag:
            opts[tag].update({'it': opts['it'], 'sym': opts['sym'], 'update': 'idle'})
        return 0

    def _spatial_modulation(self, modes=(1,), sprofile='fourier', cntmodes=None, phi=False, **kwargs):
        """ Gives the spatial profile of the perturbation: different wavelength and combinations
            of them can be produced.
        """
        self.logger.debug("Spatial profile of the perturbation: '%s'" % sprofile)
        self.logger.debug("Additional arguments are %s" % kwargs)
        # Check format of modes
        try:
            modes = list(modes)
        except TypeError:
            modes = [modes]
        # Typical modulations are gaussian or cosine (fourier) series
        if sprofile == 'gauss':
            return Connectivity.vonmises(1.0, 8.0, 0.0, 1.0, self.n, np.linspace(-pi, pi, self.n))
        elif sprofile == 'fourier':
            sp = 0.0
            for m in modes:
                if cntmodes is None:  # Use the connectivity eigenmodes to create the perturbation modes
                    if phi:  # Set a phase different of zero: the perturbation is centered at phi
                        if phi == 0.0:
                            phi = np.random.randn(1) * np.pi
                    else:
                        phi = 0.0
                    self.logger.debug("Perturbation of mode %d with phase %f" % (m, phi))
                    sp += np.cos(m * self.phi + phi)
                else:
                    sp += cntmodes[m] * 1.0
            return sp
        else:
            self.logger.error("Spatial profile '%s' not implemented." % sprofile)
            return 0.0

    def _time_modulation(self, p0, pd=0.5, rise=0.2, decay=0.0, tprofile='pulse', **kwargs):

        """ Function that produces a specific time modulated function to be applied to the perturbation vector.

        :param p0: initial time of the perturbation, given in dt units (not in tsteps). Redundant times should be hand
                      led previously.
        :param pd: duration of the pulse in dt units (not time steps)
        :param rise: time constant of the rising function
        :param decay: time constant of the decay function
        :param tprofile: type of perturbation: "pulse", "oscil", "chirp"
        """
        self.logger.debug("Additional arguments are %s" % kwargs)
        # Set up relevant times in time-step units
        # We create the vector from 0 to dts and then we roll the vector to much t0s
        t0s = int(p0 / self.dt) % self.nsteps
        rt, dt, ft = (np.array([0.0]), np.array([0.0]), np.array([0.0]))
        it = np.zeros(self.nsteps)
        if pd > 0:  # Finite perturbation
            dts = int(pd / self.dt)

            if dts > self.nsteps:  # Various cycles of the simulation # TODO
                self.cycles = (t0s + dts) // self.nsteps
                pd = self.nsteps * self.dt * self.cycles
            t = np.arange(0.0, pd, self.dt)

            if tprofile == 'pulse':
                self.pend = self.pstart + dts - 1
                self.logger.debug("Perturbation starting at %d (%f) and finishing at %d (%f)" % (
                self.pstart, self.pstart * self.dt, self.pend, self.pend * self.dt))
                if rise > 0.01:
                    rt = np.exp(t / rise) - 1.0
                    mask = (rt >= 1.0)
                    rt[mask] = rt[mask] * 0.0 + 1.0
                else:
                    rt = np.ones(len(t)) * 1.0
                if decay > 0:
                    dt = np.exp(-t / decay) - 1.0
                ft = np.concatenate((rt, dt))
            elif tprofile == 'oscil':
                pass
            elif tprofile == 'chirp':
                pass
            else:
                self.logger.error("Temporal profile '%s' not implemented." % tprofile)
                return 0.0
            it[0:len(ft)] = ft * 1.0
        # Roll the vector to start on t0
        it = np.roll(it, t0s)
        return it

    def _destroy(self):
        # Delete the perturbation
        pass


class SaveResults:
    """ Class to save and load results. Save pickled data, save csv data, select saving file. Load initial
        conditions (for spiking neuron simulaitons). """

    def __init__(self, opts, variables):
        self.logger = logging.getLogger('tools.SaveResults')
        self.o = opts
        self.v = variables
        self.nsteps = len(np.arange(opts['t0'], opts['tfinal'], opts['dt']))
        self.choice = None
        self.xtvars = {'t': None, 'tfr': None, 'phi': None}
        self.ic = False
        for xtvar in self.xtvars.copy():
            var = variables.get(xtvar, None)
            if var is not None:
                l = len(var)
                self.xtvars[xtvar] = l
            else:
                self.xtvars.pop(xtvar)
        self.logger.debug("XTvars: %s" % self.xtvars)

    def __call__(self, *args, **kwargs):
        self.choice = kwargs.get('choice', 'all')
        self.path = kwargs.get('path', '%s/saved_result-%s' % (self.o['dir'], self.choice))
        if kwargs.get('save_ic', False):
            self.save_ic(kwargs['data'])
            return 0
        elif kwargs.get('load_ic', False):
            self.logger.info("Loading initial conditions.")
            return self.load_ic()

        elif self.choice:
            self.logger.info("Saving %s data to %s." % (self.choice, self.path))
            self.save_vars()
            return 0

        return -1

    def save_vars(self):
        # Two options: save all variables, or save selected variable
        dvars = []
        if self.choice == 'all':
            for var in self.v:
                if var not in self.xtvars:
                    dvars.append(self.identify_domain(var))
                    if dvars[-1]:
                        dvars[-1].append(var)
                    else:
                        dvars.pop(-1)
        # Save selected variable.
        else:
            dvars.append(self.identify_domain(self.choice))
            dvars[-1].append(self.choice)

        # Save
        self.save(dvars, self.path)

    def save_ic(self, data):
        # Identify system.
        if self.o['nf']:
            self.o['ic_dir'] = self.o['dir'] + '/ic/nf'
        else:
            self.o['ic_dir'] = self.o['dir'] + '/ic/fr'
        d = {'ic_file': True, 'var': self.choice}
        d.update(data)
        # Variable is a spiking neuron network variable
        if self.choice[0:2] == 'sp':
            system = 'sn' + '_' + self.o['sp']
            # "data" is a dictionary
        else:
            system = 'fr'
            vdata = self.v[self.choice][::]
            d.update({self.choice: vdata})
        # Create dir tree.
        self.o['ic_dir'] = self.o['ic_dir'] + '/' + system
        create_dir(self.o['ic_dir'])
        # Save data necessary to be able to load the same system.
        path = self.o['ic_dir'] + '/' + '-'.join(now('_', '.'))
        np.save(path, d)
        self.logger.info("Initial conditions saved to %s." % path)

    def load_ic(self):
        # Load file
        try:
            self.logger.info("Loading %s..." % self.path)
            data = dict(np.load(self.path).item())
            # Check format
            if not data.get('ic_file', False):
                self.logger.error("This is not an initial condition container.")
                return None
        except (KeyError, IOError):
            self.logger.error("This is not an initial condition container.")
            return None
        # Load options and create new data object
        self.o = data['opts']
        self.logger.debug("Creating new 'data' instance...")
        new_data = Data(self.o, external=(None,))
        # Identify the variable which has to be loaded
        var = data['var']
        self.logger.debug("Loading '%s'..." % var)
        # Variable is a Spiking neuron variable
        if var[0:2] == 'sp':
            # Load voltages and spikes matrices
            new_data.m_e = data['m_e']
            new_data.m_i = data['m_i']
            new_data.spk_e = data['spk_e']
            new_data.spk_i = data['spk_i']
            try:
                new_data.spk_e_mod = data['spk_e_mod']
                new_data.spk_i_mod = data['spk_i_mod']
            except:
                pass
        else:
            new_data.vars[var][::] = data[var][::]
        return new_data

    def save(self, data, path):
        """ Save the data. Different possibilities: pickled, plot, ... """
        # TODO: different saving formats
        # Dictionary: to save a pickled object
        dictio = self.create_dict(data)
        dictio['opts'] = self.o
        dictio['last_step'] = self.v['tstep'].value * 1.0
        np.save(path, dictio)

    def identify_domain(self, var):
        """ Function that identifies the domain of the variable named var"""
        dim = np.shape(self.v[var])
        dvars = []
        # Identify the variable(s)
        for d in dim:
            for xtvar in self.xtvars:
                if d == self.xtvars[xtvar]:
                    dvars.append(xtvar)
        return dvars

    def create_dict(self, variables):
        """ Creates a dictionary with the variables to be saved in a pickled object"""
        dictio = {}
        for var in variables:
            dictio[var[-1]] = {}
            for v in var:
                if v[0:2] == 'sp':
                    tstep = self.v['frtstep'].value
                else:
                    tstep = self.v['tstep'].value % self.nsteps
                if v not in ('t', 'tfr', 'phi'):
                    if var[0] in ('t', 'tfr') and tstep != 0:
                        ydata = np.concatenate((self.v[v][tstep:], self.v[v][:tstep]))
                    else:
                        ydata = self.v[v][::]
                else:
                    ydata = self.v[v][::]

                dictio[var[-1]][v] = ydata
        return dictio
