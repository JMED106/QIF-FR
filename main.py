#!/usr/bin/python2.7

import sys
import numpy as np
import math
from sconf import parser_init, parser, log_conf, get_paths, LimitedSizeDict
from simu_lib import Data, Connectivity, Perturbation
from simu_gui import MainGui
import progressbar as pb
from timeit import default_timer as timer
from libNeuroDyn import qifint_fr, ifint_fr, qifint_nf, ifint_nf, eifint_fr, eifint_nf, \
    sigmoid_qif, noise, sigmoid_lif, qifint_cond
import Queue
import random

__author__ = 'jm'

pi = np.pi
pi2 = np.pi * np.pi

# -- Simulation configuration I: parsing, debugging.
conf_file, debug, args1, hlp = parser_init()
if not hlp:
    logger, mylog = log_conf(debug)
else:
    logger = None
# -- Simulation configuration II: data entry (second parser).
description = 'Conductance based QIF spiking neural network. All to all coupled with distributed external currents.'
groups = ('Parameters', 'Network', 'Perturbation', 'Firingrate')
opts, args = parser(conf_file, args1, description=description, groups=groups)  # opts is a dictionary, args is an object

opts.update(get_paths(__file__))


def simulation(dat, var, q_in=None, q_out=None):
    logger.info("New simulation launched.")
    # Set up perturbation class -> modifies parameters dictionary
    d = dat
    # General options dictionary (all the dictionaries below are connected, same memory)
    o = dat.opts

    # Dictionary for parameters
    p = o['parameters']
    # Dictionary for controlling the simulation
    c = o['controls']
    r = o['raster']
    # Dictionaries for the classes
    i = o['perturbation']
    n = o['network']

    # Assign the variables from the dictionary to easily named local variables
    rwe, rwi = (var['rwe'], var['rwi'])
    re, ve, ser, ri, vi, sir = (var['re'], var['ve'], var['ser'],
                                var['ri'], var['vi'], var['sir'])
    swer, swir = (var['swer'], var['swir'])
    if o.get('sp', False):
        sye, syi = (var['sye'], var['syi'])

    se = np.ones(2) * 0.0
    si = np.ones(2) * 0.0
    # Select transfer function for WC equations
    if set(d.systems).intersection({'if-fr', 'if-nf', 'all'}):
        transf = sigmoid_lif
    elif set(d.systems).intersection({'qif-fr', 'qif-nf', 'all'}):
        transf = sigmoid_qif
    else:
        transf = sigmoid_qif

    # Initial parameter values may be changed from now
    pause = None
    while c['pause']:
        try:
            incoming = q_in.get_nowait()
            for key in incoming.keys():
                o[key].update(incoming[key])

        except Queue.Empty:
            if not pause:
                logger.info("Data successfully updated. Simulation in hold.")
                pause = True

    # Set up Connectivity class(es) -> modifies parameters dictionary
    per = Perturbation(o)
    exc = Connectivity(o, 'exc')
    inh = Connectivity(o, 'inh')
    objects = [per, exc, inh, dat.fr]
    # Progress-bar configuration
    widgets = ['Progress: ', pb.Percentage(), ' ',
               pb.Bar(marker='='), ' ', pb.ETA(), ' ']

    # Raster configuration variables
    if d.opts.get('sp', False):
        r_maxlength = 1000
        raster = LimitedSizeDict(size_limit=r_maxlength)
        s_pop = None

    # Measuring frequencies of individual neurons
    spikes_e = []
    spikes_i = []
    tfreq = 0

    if o.get('sp', False):
        if d.nf:
            nmax = d.dNe
        else:
            nmax = d.Ne
        if nmax <= 1000:
            pop_max = nmax
        else:
            pop_max = 1000

    # ############################################################
    # 0) Prepare simulation environment
    if args.loop != 0:  # loop variable will force the time loop to iterate "loop" times more or endlessly if "loop" = 0
        barsteps = int((d.tfinal - d.t0) / d.dt)
        pbar = pb.ProgressBar(widgets=widgets, maxval=args.loop * (barsteps + 1)).start()
    else:
        args.loop = sys.maxint
        # noinspection PyTypeChecker
        pbar = pb.ProgressBar(max_value=pb.UnknownLength)

    time1 = timer()
    tstep = 0
    temps = 0.0
    tsk_e = 0
    tsk_i = 0

    noise_E = 0.0
    noise_I = 0.0

    np.seterr(all='raise')

    # Time loop: (if loop was 0 in the config step,
    #             we can break the time-loop by changing "loop"
    #             or explicitly with a break)
    while temps < d.tfinal * args.loop:
        pause = False
        while c['pause']:
            try:
                incoming = q_in.get_nowait()
                for key in incoming.keys():
                    o[key].update(incoming[key])
            except Queue.Empty:
                if not pause:
                    logger.info("Simulation in hold.")
                    pause = True
        # Time delay discretization
        delay = int(p['delay'] / d.dt)
        if len(se) != (2 + delay):
            se.resize(2 + delay)
            si.resize(2 + delay)
            print 'Delay! %d' % delay
        # Time step variables
        kp = tstep % d.nsteps
        k = (tstep + d.nsteps - 1) % d.nsteps
        kd = (tstep + d.nsteps - 1 - delay) % d.nsteps
        kd2 = (tstep + d.nsteps - delay) % d.nsteps
        # Time steps of the synaptic activation (just 2 positions, columns)
        k2p = tstep % (2 + delay)

        if set(d.systems).intersection({'fr', 'wc', 'all'}):  # Two population FR eqs simulations
            # Wilson Cowan
            if 'wc' in d.systems:
                input_e = p['etae'] + p['taue'] * n['jee'] * swer[k] - p['taue'] * n['jie'] * swir[k] + per.it[kp]
                rwe[kp] = rwe[k] + d.dt / p['taue'] * (-rwe[k] + transf(input_e, p['taue'], p['delta'],
                                                                        vr=p['vreset'], vth=p['vpeak'],
                                                                        vrevers=p['revers'], dt=p['rperiod']))

                input_i = p['etai'] - p['taui'] * n['jii'] * swir[k] + p['taui'] * n['jei'] * swer[k] + i['sym'] * \
                                                                                                        per.it[kp]
                rwi[kp] = rwi[k] + d.dt / p['taui'] * (-rwi[k] + transf(input_i, p['taui'], p['delta'],
                                                                        vr=p['vreset'], vth=p['vpeak'],
                                                                        vrevers=p['revers'], dt=p['rperiod']))
                if p.get('taude', False):
                    if p['taude'] > 0.001 and p['taudi'] > 0.001:
                        swer[kp] = swer[k] + d.dt / p['taude'] * (-swer[k] + rwe[kd])
                        swir[kp] = swir[k] + d.dt / p['taudi'] * (-swir[k] + rwi[kd])
                    else:
                        swer[kp] = rwe[kd2] * 1.0
                        swir[kp] = rwi[kd2] * 1.0
                else:
                    swer[kp] = rwe[kd2] * 1.0
                    swir[kp] = rwi[kd2] * 1.0

            if 'wc-eff' in d.systems:
                input_e = p['etae'] + p['taue'] * n['jee'] * swer[k] - p['taue'] * n['jei'] * swir[k] + per.it[
                    kp]
                rwe[kp] = rwe[k] + d.dt / p['taue'] * (-rwe[k] + transf(input_e, p['taue'], p['delta'],
                                                                        vr=p['vreset'], vth=p['vpeak'],
                                                                        vrevers=p['revers'], dt=p['rperiod']))

                input_i = p['etai'] + p['taui'] * n['jee'] * swir[k] - p['taui'] * n['jei'] * swer[k] + i[
                                                                                                            'sym'] * \
                                                                                                        per.it[
                                                                                                            kp]
                rwi[kp] = rwi[k] + d.dt / p['taui'] * (-rwi[k] + transf(input_i, p['taui'], p['delta'],
                                                                        vr=p['vreset'], vth=p['vpeak'],
                                                                        vrevers=p['revers'], dt=p['rperiod']))
                if p.get('taude', False):
                    if p['taude'] > 0.001 and p['taudi'] > 0.001:
                        swer[kp] = swer[k] + d.dt / p['taude'] * (-swer[k] + rwe[kd])
                        swir[kp] = swir[k] + d.dt / p['taudi'] * (-swir[k] + rwi[kd])
                    else:
                        swer[kp] = rwe[kd2] * 1.0
                        swir[kp] = rwi[kd2] * 1.0
                else:
                    swer[kp] = rwe[kd2] * 1.0
                    swir[kp] = rwi[kd2] * 1.0

            # QIF-FR
            if 'fr' in d.systems:
                if d.cond:
                    re[kp] = re[k] + d.dt / p['taue'] * (
                        p['delta'] / pi / p['taue'] + 2.0 * re[k] * ve[k] - p['taue'] * re[k] * (
                        1.0 / p['taue'] + p['ae'] * n['jee'] * ser[k] + p['ai'] * n['jie'] * sir[k]))
                    ve[kp] = ve[k] + d.dt / p['taue'] * (
                        ve[k] ** 2 + p['etae'] - pi2 * (re[k] * p['taue']) ** 2 - ve[k] - p['taue'] * (
                            p['ae'] * n['jee'] * ser[k] * (ve[k] - p['reverse']) + p['ai'] * n['jie'] * sir[k] * (
                                ve[k] - p['reversi'])) + per.it[kp])

                    ri[kp] = ri[k] + d.dt / p['taui'] * (
                        p['delta'] / pi / p['taui'] + 2.0 * ri[k] * vi[k] - p['taui'] * ri[k] * (
                            1.0 / p['taui'] + p['ae'] * n['jei'] * ser[k] + p['ai'] * n['jii'] * sir[k]))
                    vi[kp] = vi[k] + d.dt / p['taui'] * (
                        vi[k] ** 2 + p['etai'] - pi2 * (ri[k] * p['taui']) ** 2 - vi[k] - p['taui'] * (
                            p['ae'] * n['jei'] * ser[k] * (vi[k] - p['reverse']) + p['ai'] * n['jii'] * sir[k] * (
                                vi[k] - p['reversi'])) + i['sym'] * per.it[kp])
                else:
                    re[kp] = re[k] + d.dt / p['taue'] * (p['delta'] / pi / p['taue'] + 2.0 * re[k] * ve[k])
                    ve[kp] = ve[k] + d.dt / p['taue'] * (ve[k] ** 2 + p['etae'] - pi2 * (re[k] * p['taue']) ** 2
                                                         - p['taue'] * n['jie'] * sir[kd] + p['taue'] * n['jee'] * ser[
                                                             kd]
                                                         + per.it[kp])
                    ri[kp] = ri[k] + d.dt / p['taui'] * (p['delta'] / p['taui'] / pi + 2.0 * ri[k] * vi[k])
                    vi[kp] = vi[k] + d.dt / p['taui'] * (vi[k] ** 2 + p['etai'] - p['taui'] ** 2 * pi2 * ri[k] ** 2
                                                         + p['taui'] * n['jei'] * ser[kd] - p['taui'] * n['jii'] * sir[
                                                             kd]
                                                         + i['sym'] * per.it[kp])
                if p.get('taude', False):
                    if p['taude'] > 0.001 and p['taudi'] > 0.001:
                        ser[kp] = ser[k] + d.dt / p['taude'] * (-ser[k] + re[kd])
                        sir[kp] = sir[k] + d.dt / p['taudi'] * (-sir[k] + ri[kd])
                    else:
                        ser[kp] = re[kd2] * 1.0
                        sir[kp] = ri[kd2] * 1.0
                else:
                    ser[kp] = re[kd2] * 1.0
                    sir[kp] = ri[kd2] * 1.0

            if math.isnan(rwe[kp]) or math.isnan(rwi[kp]) or math.isnan(re[kp]) or math.isnan(ri[kp]):
                logger.error("Overflow encountered! Change parameters before running a new instance of the simulation.")
                break

        if set(d.systems).intersection({'nf', 'wc-nf', 'all'}):  # NF simulations
            # QIF-NF Equations
            if 'nf' in d.systems:
                # TODO: implement taue and taui !!!
                ser[k2p] = (2.0 * d.ne / d.n * np.dot(exc.c, re[k]) + 2.0 * d.ni / d.n * np.dot(inh.c, ri[k]))
                re[kp] = re[k] + d.dt * (p['delta'] / pi + 2.0 * re[k] * ve[k])
                ve[kp] = ve[k] + d.dt * (ve[k] ** 2 + p['etae'] + ser[k2p] - pi2 * re[k] ** 2 + per.it[kp])
                ri[kp] = ri[k] + d.dt * (p['delta'] / pi + 2.0 * ri[k] * vi[k])
                vi[kp] = vi[k] + d.dt * (vi[k] ** 2 + p['etai'] + ser[k2p] - pi2 * ri[k] ** 2 + i['sym'] * per.it[kp])
            # WC-NF Equations.
            if 'wc-nf' in d.systems:
                ser[k2p] = (2.0 * d.ne / d.n * np.dot(exc.c, rwe[k]) + 2.0 * d.ni / d.n * np.dot(inh.c, rwi[k]))
                rwe[kp] = rwe[k] + d.dt / p['taue'] * (
                    -rwe[k] + sigmoid_qif(p['etae'] + p['taue'] * ser[k2p] + per.it[kp], p['taue'], p['delta']))
                rwi[kp] = rwi[k] + d.dt / p['taui'] * (
                    -rwi[k] + sigmoid_qif(p['etai'] + p['taui'] * ser[k2p] + i['sym'] * per.it[kp], p['taui'],
                                          p['delta']))

            if math.isnan(rwe[kp, 0]) or math.isnan(rwi[kp, 0]) or math.isnan(re[kp, 0]) or math.isnan(ri[kp, 0]):
                logger.error("Overflow encountered! Change parameters before running a new instance of the simulation.")
                break

        # Spiking neurons
        if set(d.systems).intersection(
                {'qif-fr', 'if-fr', 'eif-fr', 'qif-nf', 'if-nf', 'eif-nf', 'all'}):  # Spiking neurons
            tsyp = tstep % d.t_syn
            if d.spk_time_e or d.spk_time_i:
                tskp_e = tstep % d.spk_time_e
                tskp_i = tstep % d.spk_time_i
                tsk_e = (tstep + d.spk_time_e - 1) % d.spk_time_e
                tsk_i = (tstep + d.spk_time_i - 1) % d.spk_time_i
            else:
                tsk_e = tsk_i = tskp_e = tskp_i = 0

            if o['D'] == 'noise':
                noise_E = p['taue'] / d.dt * p['delta'] * np.sqrt(d.dt / p['taue']) * noise(d.Ne)
                noise_I = p['taui'] / d.dt * p['delta'] * np.sqrt(d.dt / p['taui']) * noise(d.Ni)

            if set(d.systems).intersection(
                    {'qif-fr', 'if-fr', 'eif-fr', 'all'}):  # QIF or IF population simulation (FR)
                sep = np.dot(d.spk_e, d.a_tau[:, tsyp]).mean()
                sip = np.dot(d.spk_i, d.a_tau[:, tsyp]).mean()
                if p.get('taude', False):
                    if p['taude'] > 0.001 and p['taudi'] > 0.001:
                        sye[kp] = sye[k] + d.dt / p['taude'] * (-sye[k] + sep)
                        syi[kp] = syi[k] + d.dt / p['taudi'] * (-syi[k] + sip)
                    else:
                        sye[kp] = sep * 1.0
                        syi[kp] = sip * 1.0
                else:
                    sye[kp] = sep * 1.0
                    syi[kp] = sip * 1.0

                if set(d.systems).intersection({'qif-fr', 'all'}):  # QIF population simulation (FR)
                    if d.cond:
                        d.m_e = qifint_cond(d.m_e, d.m_e['v'], d.m_e['t'],
                                             d.eta_e + noise_E + per.it[kp] * np.ones(d.Ne),
                                             p['taue'] * p['ae'] * n['jee'] * sye[kp], p['taue'] * p['ai'] * n['jie'] * syi[kp], temps,
                                             d.Ne,
                                             d.dt,
                                             p['taue'], d.vpeak, d.rte, d.tpeak_e, p['reverse'], p['reversi'])
                        d.m_i = qifint_cond(d.m_i, d.m_i['v'], d.m_i['t'], d.eta_i + noise_I + i['sym'] * per.it[
                            kp] * np.ones(d.Ni),
                                             p['taui'] * p['ae'] * n['jei'] * sye[kp], p['taui'] * p['ai'] * n['jii'] * syi[kp],
                                             temps,
                                             d.Ni, d.dt, p['taui'], d.vpeak, d.rti, d.tpeak_i, p['reverse'],
                                             p['reversi'])
                    else:
                        d.m_e = qifint_fr(d.m_e, d.m_e['v'], d.m_e['t'], d.eta_e + noise_E,
                                          p['taue'] * n['jee'] * sye[kp] - p['taue'] * n['jie'] * syi[kp] + per.it[kp],
                                          temps, d.Ne,
                                          d.dt,
                                          p['taue'], d.vpeak, d.rte, d.tpeak_e)
                        d.m_i = qifint_fr(d.m_i, d.m_i['v'], d.m_i['t'], d.eta_i + noise_I,
                                          p['taui'] * n['jei'] * sye[kp] - p['taui'] * n['jii'] * syi[kp] + i['sym'] *
                                          per.it[kp],
                                          temps,
                                          d.Ni, d.dt, p['taui'], d.vpeak, d.rti, d.tpeak_i)

                if set(d.systems).intersection({'if-fr', 'all'}):  # LIF population simulation (FR)
                    d.m_e = ifint_fr(d.m_e, d.m_e['v'], d.m_e['t'], d.eta_e + noise_E,
                                     p['taue'] * n['jee'] * sye[kp] - p['taue'] * n['jie'] * syi[kp] + per.it[kp],
                                     temps, d.Ne,
                                     d.dt,
                                     p['taue'], p['vpeak'], p['vreset'], p['revers'], p['rperiod'], d.tpeak_e)
                    d.m_i = ifint_fr(d.m_i, d.m_i['v'], d.m_i['t'], d.eta_i + noise_I,
                                     p['taui'] * n['jei'] * sye[kp] - p['taui'] * n['jii'] * syi[kp] + i['sym'] *
                                     per.it[kp],
                                     temps,
                                     d.Ni, d.dt, p['taui'], p['vpeak'], p['vreset'], p['revers'], p['rperiod'],
                                     d.tpeak_i)

                if set(d.systems).intersection({'eif-fr', 'all'}):  # LIF population simulation (FR)
                    d.m_e = eifint_fr(d.m_e, d.m_e['v'], d.m_e['t'], d.eta_e + noise_E,
                                      p['taue'] * n['jee'] * sye[kp] - p['taue'] * n['jie'] * syi[kp] + per.it[kp],
                                      temps, d.Ne,
                                      d.dt, p['taue'], p['vpeak'], p['vreset'], p['revers'],
                                      p['rperiod'], d.tpeak_e, p['sharp'], p['rheo'])
                    d.m_i = eifint_fr(d.m_i, d.m_i['v'], d.m_i['t'], d.eta_i + noise_I,
                                      p['taui'] * n['jei'] * sye[kp] - p['taui'] * n['jii'] * syi[kp] + i['sym'] *
                                      per.it[kp],
                                      temps, d.Ni, d.dt, p['taui'], p['vpeak'], p['vreset'], p['revers'],
                                      p['rperiod'], d.tpeak_i, p['sharp'], p['rheo'])
                # Auxiliary matrices for FR computation
                if d.spk_time_e or d.spk_time_i:
                    d.spk_e_mod[:, tsk_e] = 1 * d.m_e['s']
                    d.spk_e[:, tsyp] = 1 * d.spk_e_mod[:, tskp_e]
                    d.spk_i_mod[:, tsk_i] = 1 * d.m_i['s']
                    d.spk_i[:, tsyp] = 1 * d.spk_i_mod[:, tskp_i]
                else:
                    d.spk_e[:, tsyp] = 1 * d.m_e['s']
                    d.spk_i[:, tsyp] = 1 * d.m_i['s']

                dat.fr.spikes_e[:, tstep % dat.fr.sld_steps] = 1 * d.spk_e[:, tsyp]
                dat.fr.spikes_i[:, tstep % dat.fr.sld_steps] = 1 * d.spk_i[:, tsyp]
                dat.fr.firing_rate(tstep, temps, var)

            if set(d.systems).intersection({'qif-nf', 'if-nf', 'eif-nf', 'all'}):  # QIF population simulation (NF)
                sep = (1.0 / d.Ne) * np.dot(exc.c, np.dot(d.aux['e'], np.dot(d.spk_e, d.a_tau[:, tsyp])))
                sip = (1.0 / d.Ni) * np.dot(inh.c, np.dot(d.aux['i'], np.dot(d.spk_i, d.a_tau[:, tsyp])))
                s = sep + sip

                if set(d.systems).intersection({'qif-nf', 'all'}):  # QIF population simulation (NF)
                    d.m_e = qifint_nf(d.m_e, d.m_e['v'], d.m_e['t'], d.eta_e + noise_E,
                                      p['taue'] * s + per.it[kp], temps, d.Ne, d.dNe, d.dt,
                                      p['taue'], d.vpeak, d.rte, d.tpeak_e)
                    d.m_i = qifint_nf(d.m_i, d.m_i['v'], d.m_i['t'], d.eta_i + noise_I,
                                      p['taue'] * s + i['sym'] * per.it[kp], temps, d.Ne, d.dNe, d.dt,
                                      p['taui'], d.vpeak, d.rti, d.tpeak_i)

                if set(d.systems).intersection({'if-nf', 'all'}):  # LIF population simulation (NF)
                    d.m_e = ifint_nf(d.m_e, d.m_e['v'], d.m_e['t'], d.eta_e + noise_E,
                                     p['taue'] * s + per.it[kp], temps, d.Ne, d.dNe, d.dt,
                                     p['taue'], p['vpeak'], p['vreset'], p['revers'], p['rperiod'], d.tpeak_e)
                    d.m_i = ifint_nf(d.m_i, d.m_i['v'], d.m_i['t'], d.eta_i + noise_I,
                                     p['taui'] * s + i['sym'] * per.it[kp], temps, d.Ne, d.dNe, d.dt,
                                     p['taui'], p['vpeak'], p['vreset'], p['revers'], p['rperiod'], d.tpeak_i)

                if set(d.systems).intersection({'eif-nf', 'all'}):  # EIF population simulation (NF)
                    d.m_e = eifint_nf(d.m_e, d.m_e['v'], d.m_e['t'], d.eta_e + noise_E, p['taue'] * s + per.it[kp],
                                      temps, d.Ne,
                                      d.dNe, d.dt, p['taue'], p['vpeak'], p['vreset'], p['revers'], p['rperiod'],
                                      d.tpeak_e,
                                      p['sharp'], p['rheo'])
                    d.m_i = eifint_nf(d.m_i, d.m_i['v'], d.m_i['t'], d.eta_i + noise_I,
                                      p['taui'] * s + i['sym'] * per.it[kp],
                                      temps, d.Ne, d.dNe, d.dt, p['taui'], p['vpeak'], p['vreset'], p['revers'],
                                      p['rperiod'],
                                      d.tpeak_i, p['sharp'], p['rheo'])

                # Auxiliary matrices for FR computation
                if d.spk_time_e or d.spk_time_i:
                    d.spk_e_mod[:, tsk_e] = 1 * d.m_e['s']
                    d.spk_e[:, tsyp] = 1 * d.spk_e_mod[:, tskp_e]
                    d.spk_i_mod[:, tsk_i] = 1 * d.m_i['s']
                    d.spk_i[:, tsyp] = 1 * d.spk_i_mod[:, tskp_i]
                else:
                    d.spk_e[:, tsyp] = 1 * d.m_e['s']
                    d.spk_i[:, tsyp] = 1 * d.m_i['s']

                dat.fr.spikes_e[:, tstep % dat.fr.sld_steps] = 1 * d.spk_e[:, tsyp]
                dat.fr.spikes_i[:, tstep % dat.fr.sld_steps] = 1 * d.spk_i[:, tsyp]
                dat.fr.firing_rate(tstep, temps, var, dat.aux)

            # Recording of spikie times for Raster Plotting (2 versions, snapshot and dynamic plotting)
            if r['start'] and tstep % r['rate'] == 0:
                # For the dynamic plot we can use a simple dictionary with a time as the key and an array of indexes.
                if r['dynamic']:
                    if q_out.qsize() < 10:
                        if r['pop']:
                            pop = r['pop']
                            q_out.put({'t': temps, 'sp': d.m_e[d.pope[pop]][d.m_e[d.pope[pop]]['s'] == 1]['i']})
                        else:
                            q_out.put({'t': temps, 'sp': d.m_e[d.m_e['s'] == 1]['i']})
                # For the snapshot we use a limited size dictionary with times as keys.
                else:
                    if not s_pop:
                        if d.nf and r['pop']:
                            s_pop = random.sample(range(d.dNe * r['pop'], d.dNe * (r['pop'] + 1)), pop_max)
                        else:
                            s_pop = random.sample(range(d.Ne), pop_max)
                    s_pop.sort()
                    raster[temps] = d.m_e[s_pop][d.m_e[s_pop]['s'] == 1]['i']

            # Measure firing rate of individual neurons
            if c.get('neuronf', False):
                if len(spikes_e) > 1:
                    spikes_e += d.m_e['s']
                    spikes_i += d.m_i['s']
                else:
                    tfreq = tstep
                    spikes_e = d.m_e['s'][:] * np.float64(1.0)
                    spikes_i = d.m_i['s'][:] * np.float64(1.0)

        pbar.update(tstep)

        # Perturbation management
        if per.active and tstep % int(p['upd']) == 0:
            per.check(tstep)

        if c['exit'] or c['stop']:
            break

        temps += d.dt
        tstep += 1

        if tstep % d.nsteps == 0:
            var['cycle'].value += 1
        var['tstep'].value = tstep
        var['temps'].value = temps
        if not d.nf:
            var['it'][kp] = per.it[kp]

        # We get the data from the GUI
        if q_in and tstep % int(p['upd']) == 0:
            try:
                incoming = q_in.get_nowait()
                for key in incoming.keys():
                    logger.debug("Updating data dictionary.")
                    o[key].update(incoming[key])
                    # Passing spiking neuron data to the Main GUI
                    if incoming[key].get('pause', False):
                        if o.get('sp', False):
                            sp_dict = {'opts': o, 'm_e': d.m_e.copy(), 'm_i': d.m_i.copy(),
                                       'spk_e': np.roll(d.spk_e, -(tsyp + 1)), 'spk_i': np.roll(d.spk_i, -(tsyp + 1))}
                            sp_dict['m_e']['t'] -= (temps - d.dt)
                            sp_dict['m_i']['t'] -= (temps - d.dt)
                            try:
                                sp_qif = {'spk_mod_e': np.roll(d.spk_e_mod, -(tsk_e + 1)),
                                          'spk_mod_i': np.roll(d.spk_i_mod, -(tsk_i + 1))}
                                sp_dict.update(sp_qif)
                            except:
                                pass
                            q_out.put(sp_dict)
                            del sp_dict
                        else:
                            q_out.put({'opts': o})
                    # Passing raster data to the main GUI
                    if incoming[key].get('update', False):
                        q_out.put(raster)
                    # Receiving order to save voltage data
                    if incoming[key].get('vsnapshot', False):
                        np.save('volt_distribution_%f' % (d.dt * kp * d.faketau), d.m_e['v'])
                    if incoming[key].get('neuronf', False):
                        if tfreq == 0:
                            logger.info('Measuring frequencies of individual neurons...')
                            spikes_e = []
                            spikes_i = []
                        else:
                            logger.info('Measure of frequencies done.')
                            mylog.info(0, True)
                            c['neuronf'] = False
                            dtfreq = tstep - 1 - tfreq
                            tfreq = 0
                            freqs_e = np.array(spikes_e) / dtfreq
                            freqs_i = np.array(spikes_i) / dtfreq
                            np.save('nfreqs.npy', {'time': dtfreq, 'freqse': freqs_e, 'freqsi': freqs_i})
                    if o.get('sp', False):
                        if isinstance(incoming[key].get('delta', False), float) \
                                or isinstance(incoming[key].get('etai', False), float) \
                                or isinstance(incoming[key].get('etae', False), float):
                            d.eta_e = Data.external_currents(p['etae'], p['delta'], d.Ne, n=d.n,
                                                             distribution=o['D'])
                            d.eta_i = Data.external_currents(p['etai'], p['delta'], d.Ni, n=d.n,
                                                             distribution=o['D'])
                        if incoming[key].get('gamma', False) \
                                or incoming[key].get('reversi', False) or incoming[key].get('reverse', False):
                            if d.cond:
                                d.rev_e = Data.external_currents(p['reverse'], p['gamma'], d.Ne, n=d.n,
                                                                 distribution=o['G'])
                                d.rev_i = Data.external_currents(p['reversi'], p['gamma'], d.Ni, n=d.n,
                                                                 distribution=o['G'])
                # Updating the objects' properties (perturbation, connectivity)
                for obj in objects:
                    if obj:
                        if obj.name in incoming.keys():
                            logger.debug("Updating %s" % obj.name)
                            obj.update(o[obj.name], tstep=tstep)
            except Queue.Empty:
                pass
            except KeyError:
                logger.error("KeyError when getting or sending objects through the queue.")

    # Finish pbar
    pbar.finish()
    temps -= d.dt
    tstep -= 1
    # Stop the timer
    logger.info('Total time: {}.'.format(timer() - time1))
    # Synchronize data object
    while not q_out.empty():
        q_out.get_nowait()
    q_out.put({'opts': o})
    if o.get('sp', False):
        sp_dict = {'opts': o, 'm_e': d.m_e.copy(), 'm_i': d.m_i.copy(),
                   'spk_e': np.roll(d.spk_e, -(tsyp + 1)), 'spk_i': np.roll(d.spk_i, -(tsyp + 1))}
        sp_dict['m_e']['t'] -= (temps - d.dt)
        sp_dict['m_i']['t'] -= (temps - d.dt)
        try:
            sp_qif = {'spk_mod_e': np.roll(d.spk_e_mod, -(tsk_e + 1)),
                      'spk_mod_i': np.roll(d.spk_i_mod, -(tsk_i + 1))}
            sp_dict.update(sp_qif)
        except:
            pass
        q_out.put(sp_dict)
        del sp_dict


# Set up data class (contains simulation data)
data = Data(opts, external=(None,))
logger.info("Simulation data successfully loaded.")

# GUI initializing
salir = False
while not salir:
    mg = MainGui(data, simulation=simulation, tab_sets=('perturbation', 'network', 'firingrate'))
    salir = mg()
    mg.window.destroy()
    # Load new initial conditions (and parameters)
    if not salir:
        if mg.data:
            data = mg.data
    del mg
