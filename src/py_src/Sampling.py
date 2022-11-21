#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    Mike Lindner (mike.lindner@kit.edu), 2018
"""
from obspy import UTCDateTime
import numpy as np
from scipy.signal import hilbert
from scipy.optimize import minimize
import multiprocessing
import itertools
from collections import Counter
import os
from os import listdir, path
import sys
from platform import python_version
from multiprocessing import Process
import inspect
import json
from scipy import signal, sparse
from operator import itemgetter
#import matplotlib.pyplot as plt
#from .mopad import MomentTensor
from obspy.imaging.beachball import beach, aux_plane
from obspy import Stream, Trace
#from Utils.Logger import *
from .util import update_Mw, Time_shift_xcorr_trunc_v2
from .Modeller import Full_MT_modeller
from .util import Tape2M, M2Tape
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import warnings
warnings.filterwarnings("ignore")

class Data_Modelling:

    def __init__(self,Container=None):
        self.Container = Container
        
        
        self.Mw_reference = Container['source']['ref_mag']
        self.magnitude_update = Container['inversion']['magnitude_update']
        
        self.perform_Tshift = Container['inversion']['time_shift']['perform']
        self.station_Tshift = Container['inversion']['time_shift']['station_Tshift']
        self.env_fac = Container['inversion']['envelope_portion']
        
        # waveform information
        self.dt = 1./Container['preprocessing']['resampling']['sampling_rate']
        
        # station information
        self.STAT_dict = Container['network']['STAT_dict']
        self.obs_trace_set = Container['general']['obs_trace_set']
        self.trace_selection = Container['network']['trace_selection']
        
        # reported source time (Event File)
        self.src_time = Container['source']['F1_loc'][0]
        
        # auto trace selection parameter
        self.E_max = Container['preprocessing']['equalizer']['e_max']
        self.t_unc = Container['preprocessing']['equalizer']['t_uncertainty']
        self.min_coverage = Container['preprocessing']['equalizer']['min_coverage']
        self.S2N_min = Container['preprocessing']['s2n']['s2n_min']
        self.H2V_min = Container['preprocessing']['H2V']
        
        # auto trace selection criteria
        self.N_criteria = Container['preprocessing']['selection_criteria']['N_criteria'] 
        self.stat_min = Container['preprocessing']['selection_criteria']['stat_min']
        self.trace_min = Container['preprocessing']['selection_criteria']['trace_min']
        self.env_shift = Container['preprocessing']['selection_criteria']['perform_envelope_shift']
        
        # simulation setting
        self.sim_doublet = Container['source']['simulate_doublet']
        self.bootstrap = Container['inversion']['bootstrap']['perform']
        self.bootstrap_modus = Container['inversion']['bootstrap']['modus']
        self.bootstrap_factor = Container['inversion']['bootstrap']['factor']
        self.bootstrap_N = Container['inversion']['bootstrap']['n']
        
        # output information
        self.path2fig = ''#Container['path']['figure']
        self.plot = Container['plotter']['equalizer']
        
        # error functions
        self.inv_function_final = Container['inversion']['function']
        self.normalize = Container['inversion']['normalize']
        self.L2_normalize = Container['inversion']['L2_normalize']
        self.rec_inv_procedure = Container['inversion']['rec_inv_procedure']
        if self.inv_function_final != 'L2':
            self.data_covariance = Container['inversion']['data_covariance']
        
        # bayesian
        self.dx_misloc = Container['source']['dx']
        self.daz = Container['source']['daz']
        self.t_del = Container['source']['t_del']
        self.exp_scaling_fac = Container['inversion']['exp_scaling_fac']

        # modus
        self.solve_for_misrot = Container['source']['solve_for_misrot']
        self.solve_for_delay = Container['source']['solve_for_delay']
        self.solve_for_misloc = Container['source']['solve_for_misloc']
        
        # get number of filter pertubations, is set to none only full band is used
        self.Filter = self.get_filter()
        if self.Filter['partition_type'] == None:
            self.FN = [0]
        else:
            self.FN = list(map(int,list(np.arange(0,len(self.Filter['fcut']),1))))
        
        
        # station selection
        #self.Station_Selection = 
        #create_Station_selection_list(self.Container,event,trace_set)
    
    def get_filter(self):
        if self.sim_doublet:
            FB1 = self.Container['preprocessing']['doublet_filter']['F1']
            FB2 = self.Container['preprocessing']['doublet_filter']['F2']
            Filter = {'ftype':'bandpass',
                      'freqmin':FB1['freqmin'],
                      'freqmax':np.max([FB1['freqmax'],FB2['freqmax']]),
                      'corners':FB1['corners'],
                      'zerophase':FB1['zerophase'],
                      'partition_type':'lfix',
                      'fcut':len(FB2['fcut'])*[np.max([FB1['freqmax'],FB2['freqmax']])]} 
            
        else:
            Filter = self.Container['preprocessing']['filter'] 
        return Filter
    
    def Time_Logger(self,func_name,file_name,run_time):
        debug_message = 'Function %s in file %s with runtime=%s' % (func_name,file_name.split('/')[-1],str(run_time))
        TLOG.debug(debug_message)

    def time_shift(self,obs_raw,synt_raw):
        synt,Tshift = Time_shift_xcorr_trunc_v2(obs_raw,synt_raw,self.station_Tshift)
        return synt,Tshift
    
    def get_magnitude_update(self,Magnitude_dict,fi):
        '''
        
        '''        
        if self.magnitude_update == 'network':
            Mnetw = Magnitude_dict['network']['Mfac']
            return {'Z':Mnetw,'R':Mnetw,'T':Mnetw}
        elif self.magnitude_update == 'vertical':
            Mvertical = Magnitude_dict['vertical']['Mfac']
            return {'Z':Mvertical,'R':Mvertical,'T':Mvertical}
        elif self.magnitude_update == 'horizontal':
            Mhorizontal = Magnitude_dict['horizontal']['Mfac']
            return {'Z':Mhorizontal,'R':Mhorizontal,'T':Mhorizontal}
        elif self.magnitude_update == 'HV':
            Mvertical = Magnitude_dict['vertical']['Mfac']
            Mhorizontal = Magnitude_dict['horizontal']['Mfac']
            return {'Z':Mvertical,'R':Mhorizontal,'T':Mhorizontal}
        elif self.magnitude_update == 'R':
            MR = Magnitude_dict['R']['Mfac']
            return {'Z':MR,'R':MR,'T':MR}
        elif self.magnitude_update == 'T':
            MT = Magnitude_dict['T']['Mfac']
            return {'Z':MT,'R':MT,'T':MT}
        else:
            print('magnitude update short cut is not set!')
            raise SystemExit
    
    def calc_Magnitude(self,obs,synt_raw):
        '''
        
        '''
        # get amplitude factor updated Magnitude information
        Mlog = {}
        Mfac_list ,Mw_list = {},{}   
        D,U = {'netw':[],'Z':[],'H':[],'R':[],'T':[]}, {'netw':[],'Z':[],'H':[],'R':[],'T':[]}
        for comp in ['Z','R','T']:
            Mfac_list[comp] = {} 
            Mw_list[comp] = {}
            for fi in self.FN:
                Mfac_list[comp][fi] = [] 
                Mw_list[comp][fi] = []
        
        # station wise magnitude update loop
        for sti, stat_id in enumerate(obs.keys()):
            Mlog[stat_id] = {}
            for ci,comp in enumerate(obs[stat_id].keys()):
                Mlog[stat_id][comp] = {}
                for fii,fi in enumerate(obs[stat_id][comp].keys()):
                    f = obs[stat_id][comp][fi].data
                    g = synt_raw['cF_X0'][stat_id][comp][fi].data
                    # full stream list
                    D['netw'] += list(f)
                    U['netw'] += list(g)
                    # z stream list
                    if comp == 'Z':
                        D['Z'] += list(f)
                        U['Z'] += list(g)
                    else:
                        if comp == 'R':
                            D['R'] += list(f)
                            U['R'] += list(g)
                        elif comp == 'T':
                            D['T'] += list(f)
                            U['T'] += list(g)
                        D['H'] += list(f)
                        U['H'] += list(g)
                    
                    
                    Mfac, Mw = update_Mw(self.Mw_reference,f,g)

                    Mfac_list[comp][fi].append(Mfac)
                    Mw_list[comp][fi].append(Mw)
                    Mlog[stat_id][comp][fi] = [Mfac,Mw]
        
        # full network update
        netw_mag = {}
        netw_mag['Mfac'], netw_mag['Mw'] = update_Mw(self.Mw_reference, np.asarray(D['netw']), np.asarray(U['netw']))
        
        
        # vertical update
        try:
            vertical_mag = {}
            vertical_mag['Mfac'], vertical_mag['Mw'] = update_Mw(self.Mw_reference ,np.asarray(D['Z']), np.asarray(U['Z']))
        except:
            pass
        
        # R update
        try:
            R_mag = {}
            R_mag['Mfac'], R_mag['Mw'] = update_Mw(self.Mw_reference ,np.asarray(D['R']), np.asarray(U['R']))
        except:
            pass
        
        # T update
        try:
            T_mag = {}
            T_mag['Mfac'], T_mag['Mw'] = update_Mw(self.Mw_reference ,np.asarray(D['T']), np.asarray(U['T']))
        except:
            pass
        
        # horizontal update
        try:
            horizontal_mag = {}
            horizontal_mag['Mfac'], horizontal_mag['Mw'] = update_Mw(self.Mw_reference ,np.asarray(D['H']), np.asarray(U['H']))
        except:
            pass
        
        # componential amplitude update
        comp_mag = {}
        for comp in Mw_list.keys():
            comp_mag[comp] = {}
            for fi in Mw_list[comp].keys():
                comp_mag[comp][fi] = [np.mean(Mfac_list[comp][fi]),np.mean(Mw_list[comp][fi])]   
                
        # create magnitude dictionary
        Magnitude = {'Mlog':Mlog,'comp_mag':comp_mag,'network':netw_mag,'vertical':vertical_mag,'horizontal':horizontal_mag,'R':R_mag,'T':T_mag}
        
        return Magnitude
        
        
    def data_preprocess(self,obs,synt_raw):
        '''
        
        '''        
        # time_shift  
        if self.perform_Tshift:
            synt,Tshift = self.time_shift(obs,synt_raw)
        else:
            synt = synt_raw
            Tshift = None
            
        # magnitude
        Magnitude = self.calc_Magnitude(obs,synt_raw)
        
        return obs, synt, Tshift, Magnitude 
    
    def get_freq_dependend_norm_L2(self,d,u):
        '''
        
        '''
        
        #plt.plot(d,'k-')
        #plt.plot(10*u,'r--')
        #plt.show()
        
        if self.env_fac > 0.:
            u_env = np.abs(signal.hilbert(u))
            d_env = np.abs(signal.hilbert(d))
            if self.L2_normalize:
                k = np.sum(d_env**2)
                #k = np.max([np.sum(d_env**2),np.sum((u_env)**2)])
            else:
                k = 1
            er_env = np.sum((d_env-u_env)**2)/k
            if np.isnan(er_env): er_env = 1.
        else:
            er_env = 0.
                                    
        if 1.-self.env_fac > 0.:
            if self.L2_normalize:
                #k = np.sum(d**2)
                k = np.max([np.sum(d**2),np.sum((u)**2)])
            else:
                k = 1
                
            er_fw = np.sum((d-u)**2)/k
            if np.isnan(er_fw): er_fw = 1.
        else:
            er_fw = 0.
        
        er_combi = er_env*self.env_fac + er_fw*(1.-self.env_fac)

        if np.isscalar(er_combi):
            return er_combi
        else:
            return None
    
    def error_bootstrap(self,error_dict):
        '''
            Input: 
                error list of combined (waveform + envelope) error for all traces
                --> here we do not care which component or station we are looking at 
                effect: speed update
            Perform:
                calc mean of random subset with fixed size (percentage from json)
            Output:
                mean and std of error for tested mechanism
        '''
        
        error = error_dict[self.bootstrap_modus]
        b_res = []
        
        if self.bootstrap_modus == 'trace':
            for ii in range(self.bootstrap_N):
                rand_sub = np.random.choice(error,int(len(error)*self.bootstrap_factor),replace=False)
                N_sub_mis = len(error)-len(rand_sub)
                rand_rep = np.append(rand_sub,np.random.choice(rand_sub,N_sub_mis,replace=True))
                b_res.append(np.mean(rand_rep))
        elif self.bootstrap_modus in  ['station','freqency']:
            for ii in range(self.bootstrap_N):
                rand_sub = np.random.choice(list(error.keys()),int(len(error.keys())*self.bootstrap_factor),replace=False)
                N_sub_mis = len(error)-len(rand_sub)
                rand_rep = np.append(rand_sub,np.random.choice(rand_sub,N_sub_mis,replace=True))
                temp = []
                for key in rand_rep: 
                    temp += error[key]
                b_res.append(np.mean(np.asarray(temp)))
        else:
            print('bootstrap_modus does not exist!')
            raise SystemExit
            
        b_res = np.asarray(b_res)
        b_mean = float(np.mean(b_res))
        b_std = float(np.std(b_res))
        
        return b_mean, b_std  
    
    def get_daz(self,stat=None,mean=False):
        '''
        
        '''
        if mean:
            mdaz_list = []
            for stat in self.daz:
                mdaz_list.append(self.daz[stat][1]*np.pi/180.)
            return np.mean(np.array(mdaz_list))
        else:
            if stat is not None:
                return self.daz[stat][1]*np.pi/180.
            else:
                return self.daz*np.pi/180.
    
    def get_pdf(self,obs_raw,synt_raw,inv_function):
        '''
        
        '''
        self.inv_function = inv_function
        if inv_function == 'L2':
            res = self.get_L2(obs_raw,synt_raw)
        elif inv_function.split('_')[0] == 'Bayesian':
            if self.rec_inv_procedure == 'per_station':
                res = self.get_bayesian_per_station(obs_raw,synt_raw)
            elif self.rec_inv_procedure == 'full_network':
                res = self.get_bayesian_full_network(obs_raw,synt_raw)
            else:
                print('Inversion procedure '+self.rec_inv_procedure+' does not exist!')
                raise SystemExit
        else:
            print('Cost function '+inv_function+' does not exist!')
            raise SystemExit
        
        return res
        
    
    def get_L2(self,obs_raw,synt_raw):
        '''
        
        '''        
        # preprocessing (shift, taper)
        obs, synt, Tshift, Magnitude = self.data_preprocess(obs_raw,synt_raw)

        # declare variables
        res = {'per_station':{},'network':{},
               'TShift':Tshift,'Magnitude':Magnitude}
        bs_stat = [] # bayesian
        ls_stat = [] # l2
        U_netw, D_netw = [], [] # obs and synt list for full network case
        
        for sti,stat_id in enumerate(obs):
            D,U = [], [] # obs and synt list overwrite for each new station
            for ci,comp in enumerate(obs[stat_id]):
                for fi in obs[stat_id][comp]:
                    # magnitude update
                    Mfac = self.get_magnitude_update(Magnitude,fi)
                    Tr_Net_Ampl_Fac = Magnitude['Mlog'][stat_id][comp][fi][0]/Mfac[comp]
                    
                    # set waveform reference
                    U += list(Mfac[comp]*synt['cF_X0'][stat_id][comp][fi])
                    D += list(obs[stat_id][comp][fi])
                    U_netw += list(Mfac[comp]*synt['cF_X0'][stat_id][comp][fi])
                    D_netw += list(obs[stat_id][comp][fi])
                    
            # get freq dependend norm L2
            D = np.asarray(D)
            U = np.asarray(U)
            nl2 = self.get_freq_dependend_norm_L2(D,U)  
                        
            # calc pdf
            k = 1
            po = np.exp(-0.5*nl2)/k
                        
            # save to dictionary
            res['per_station'][stat_id] = [float(po),nl2]
            bs_stat.append(po)
            ls_stat.append(nl2)

        # network error
        if self.bootstrap:
            self.bootstrap_modus = 'trace' # bayesian station pdf is treated as trace wise error
            b_error_dict = {'trace':bs_stat}
            b_mean, b_std = self.error_bootstrap(b_error_dict)
            l2_error_dict = {'trace':ls_stat}
            l2_mean, l2_std = self.error_bootstrap(l2_error_dict)
            res['network'] = [b_mean,b_std,l2_mean,l2_std]
        else:
            # get freq dependend norm L2
            D_netw = np.asarray(D_netw)
            U_netw = np.asarray(U_netw)
            nl2_netw = self.get_freq_dependend_norm_L2(D_netw,U_netw)
            # calc pdf
            k = 1
            po_netw = np.exp(-0.5*nl2_netw)/k
            # save to dictionary
            res['network'] = [po_netw,0,nl2_netw,0]
        
        return res

    
    def get_bayesian_per_station(self,obs_raw,synt_raw):
        '''
        
        '''
        
        # preprocessing (shift, taper)
        obs, synt, Tshift, Magnitude = self.data_preprocess(obs_raw,synt_raw)
        
        # get data_covariance matrix
        Data_o = self.data_covariance
        
        # set dictionaries
        res = {'per_station':{},'network':[],'TShift':Tshift,'Magnitude':Magnitude}
        bs_stat = []
        ls_stat = []
                
        # station loop
        for stat_id in obs:
            error, dU, D1, U1 = [], [], [], []
            for ci in range(len(Data_o['CfL'][stat_id])):
                comp = Data_o['CfL'][stat_id][ci].split('-')[1]
                fi = int(Data_o['CfL'][stat_id][ci].split('-')[2])
                
                # observed data
                d = obs[stat_id][comp][fi].copy()
                D1 += list(d.data)
                
                # synthetics (cF_X0)
                u0 = synt['cF_X0'][stat_id][comp][fi].copy()
                Mfac = self.get_magnitude_update(Magnitude,fi)
                U1 += list(Mfac[comp]*u0.data)
                
                # error 
                error += list(Mfac[comp]*u0.data-d.data)
                if self.inv_function == 'Bayesian_CD':
                    if self.solve_for_misrot:
                        u4 = synt['cF_X4'][stat_id][comp][fi].copy()
                        du = Mfac[comp]*u4.data                       
                        dU += list(np.abs(du))
                    if self.solve_for_delay:
                        u5 = synt['cF_X5'][stat_id][comp][fi].copy()
                        du = Mfac[comp]*u5.data                       
                        dU += list(np.abs(du))
                    if self.solve_for_misloc:
                        u1 = synt['cF_X1'][stat_id][comp][fi].copy()
                        u2 = synt['cF_X2'][stat_id][comp][fi].copy()
                        u3 = synt['cF_X3'][stat_id][comp][fi].copy()
                        dU1 += list(Mfac[comp]*(u1.data - u0.data) / self.dx_misloc[0])
                        dU2 += list(Mfac[comp]*(u2.data - u0.data) / self.dx_misloc[0])
                        dU3 += list(Mfac[comp]*(u3.data - u0.data) / self.dx_misloc[1])
                
            # get normalization factor
            #if self.normalize:
            #    norm = np.sum(np.asarray(D1)**2)
            #else:
            #    norm = 1.0
            
            # convert to numpy matrix
            L1m = np.asmatrix(np.asarray(error))

            # assemble and calc inverse covariance matrix C_T^-1
            if self.inv_function == 'Bayesian_CD':
                if self.solve_for_misrot:
                    dUm = np.asmatrix(np.asarray(dU))
                    daz = self.get_daz(stat=stat_id.split('_')[1],mean=False)
                    Cx_mr = np.identity(1)*daz**2
                    CTD = np.matmul(np.transpose(dUm),np.matmul(Cx_mr,dUm))
                    CTD_c = CTD + Data_o['C_d'][stat_id]
                    CTI = np.linalg.inv(CTD_c)
                if self.solve_for_delay:
                    dUm = np.asmatrix(np.asarray(dU))
                    Cx_md = np.identity(1)*self.dt # self.dt should the station dependent
                    CTD = np.matmul(np.transpose(dUm),np.matmul(Cx_md,dUm))
                    CTD_c = CTD + Data_o['C_d'][stat_id]
                    CTI = np.linalg.inv(CTD_c)
                if self.solve_for_misloc:
                    dUm = np.matrix([dU1,dU2,dU3])
                    Cx_ml = np.identity(3)
                    Cx_ml[0,0] = self.dx_misloc[0]**2
                    Cx_ml[1,1] = self.dx_misloc[0]**2
                    Cx_ml[2,2] = self.dx_misloc[1]**2
                    CTD = np.matmul(np.transpose(dUm),np.matmul(Cx_ml,dUm))
                    CTD += Data_o['C_d'][stat_id]
                    CTI = np.linalg.inv(CTD)        
            elif self.inv_function == 'Bayesian_Cd':
                CTI = Data_o['C_dI'][stat_id]

            # get freq dependend norm L2
            D1 = np.asarray(D1)
            U1 = np.asarray(U1)
            nl2 = self.get_freq_dependend_norm_L2(D1,U1)
            
            # calc pdf
            #k0 = np.sqrt(Data_o['det_C_d'][stat_id]*(2*np.pi)**1)
            k = 1
            exp = np.matmul(L1m,np.matmul(CTI.toarray(),np.transpose(L1m),dtype=np.float128),dtype=np.float128)
            pdf = np.exp(-0.5*exp)/k
            
            # save to dictionary
            res['per_station'][stat_id] = [float(pdf[0,0]),nl2]
            bs_stat.append(float(pdf[0,0]))
            ls_stat.append(nl2)

        # network error
        if self.bootstrap:
            self.bootstrap_modus = 'trace' # bayesian station pdf is treated as trace wise error
            b_error_dict = {'trace':bs_stat}
            b_mean, b_std = self.error_bootstrap(b_error_dict)
            l2_error_dict = {'trace':ls_stat}
            l2_mean, l2_std = self.error_bootstrap(l2_error_dict)
            res['network'] = [b_mean,b_std,l2_mean,l2_std]
        else:
            b_mean = float(np.mean(np.asarray(bs_stat)))
            b_std = float(np.std(np.asarray(bs_stat)))
            l2_mean = float(np.mean(np.asarray(ls_stat)))
            l2_std = float(np.std(np.asarray(ls_stat)))
            res['network'] = [b_mean,b_std,l2_mean,l2_std]

        return res
    
    
    def get_bayesian_full_network(self,obs_raw,synt_raw):
        '''
        
        '''
        
        # preprocessing (shift, taper)
        obs, synt, Tshift, Magnitude = self.data_preprocess(obs_raw,synt_raw)
        #print(Magnitude)
        #print()
        #print(Tshift)
        #print()
        
        # get data_covariance matrix
        Data_o = self.data_covariance
        
        # set dictionaries
        res = {'per_station':{},'network':[],'TShift':Tshift,'Magnitude':Magnitude}
        bs_stat = []
        
        # Cx: mean error in rad
        if self.inv_function != 'Bayesian_Cd':
            if self.solve_for_misrot:
                daz = self.get_daz(stat=None,mean=True)
                Cx_mr = np.identity(1)*daz**2
            if self.solve_for_delay:
                Cx_md = np.identity(1)*self.dt#*(self.t_del)**2
            if self.solve_for_misloc:
                Cx_ml = np.identity(3)
                Cx_ml[0,0] = self.dx_misloc[0]**2
                Cx_ml[1,1] = self.dx_misloc[0]**2
                Cx_ml[2,2] = self.dx_misloc[1]**2
        

        res_D, res_d = {}, {}
        res_l2 = {}
        error, D1, U1 = [], [], []
        dU1, dU2, dU3, dU4, dU5 = {}, {}, {}, {}, {}
        sub_du = []
        D1_stat, U1_stat, sIndex  = [], [], 0
        for cfli, CfL in enumerate(Data_o['CfL']):
            stat_id = CfL.split('-')[0]
            comp = CfL.split('-')[1]
            fi = int(CfL.split('-')[2])
                        
            # observed data
            d = obs[stat_id][comp][fi].copy()
            D1 += list(d.data)
            D1_stat += list(d.data)
                
            # synthetics (cF_X0)
            u0 = synt['cF_X0'][stat_id][comp][fi].copy()
            Mfac = self.get_magnitude_update(Magnitude,fi)
            U1 += list(Mfac[comp]*u0.data)
            U1_stat += list(Mfac[comp]*u0.data)
            
            # error 
            error += list(Mfac[comp]*u0.data-d.data)            
            
            # station error
            if cfli != sIndex:
                # get freq dependend norm L2
                D1_stat = np.asarray(D1_stat)
                U1_stat = np.asarray(U1_stat)
                nl2 = self.get_freq_dependend_norm_L2(D1_stat,U1_stat)
                
                # update
                res['per_station'][stat_id] = [None,nl2]
                D1_stat, U1_stat = [], []
                sIndex += 1
            
            # model covariance matrix
            if self.inv_function != 'Bayesian_Cd':
                if self.solve_for_misrot:
                    u4 = synt['cF_X4'][stat_id][comp][fi].copy()
                    dU4[CfL] = list(Mfac[comp]*u4.data)
                if self.solve_for_delay:
                    u5 = synt['cF_X5'][stat_id][comp][fi].copy()
                    dU5[CfL] = list(Mfac[comp]*u5.data)
                if self.solve_for_misloc:
                    u1 = synt['cF_X1'][stat_id][comp][fi].copy()
                    u2 = synt['cF_X2'][stat_id][comp][fi].copy()
                    u3 = synt['cF_X3'][stat_id][comp][fi].copy()
                    dU1[CfL] = list(Mfac[comp]*(u1.data - u0.data) / self.dx_misloc[0])
                    dU2[CfL] = list(Mfac[comp]*(u2.data - u0.data) / self.dx_misloc[0])
                    dU3[CfL] = list(Mfac[comp]*(u3.data - u0.data) / self.dx_misloc[1])
            
        # get normalization factor (outdated)
        #if self.normalize:
        #    norm = np.sum(np.asarray(D1)**2)
        #else:
        #    norm = 1.0

        # convert to numpy matrix
        L1m = np.asmatrix(np.asarray(error))
            
        # assemble and calc inverse covariance matrix C_T^-1
        if self.inv_function == 'Bayesian_Cd':
           CDI = Data_o['C_dI'] 
           #CD = Data_o['C_d']
           kpdf = Data_o['k']
        elif self.inv_function == 'Bayesian_CD':
            # data covariance matrix
            Cd = Data_o['C_d']
            kpdf = Data_o['k']
            # model covariance cases
            if self.solve_for_misrot:
                CT_tup = ()
                for cfli, CfL in enumerate(Data_o['CfL']):
                    #dU4_temp = np.abs(signal.hilbert(np.abs(np.asarray(dU4[CfL]))))
                    dU4_temp = np.abs(np.asarray(dU4[CfL]))
                    sub_du += dU4_temp.tolist() # append trace for three entries (1 Station)
                    if (cfli+1)%3 == 0: # modulo for station or fi change (3 components)
                        dUm = np.asmatrix(sub_du)
                        CTm = np.matmul(np.transpose(dUm),np.matmul(Cx_mr,dUm))
                        '''
                        fig = plt.figure(figsize=(20,10), facecolor='w', edgecolor='k')
                        plt.subplot(1,2,1)
                        plt.plot(sub_du)
                        plt.plot(error)
                        plt.plot(D1)
                        plt.plot(U1)
                        plt.xlim((0,len(sub_du)))
                        plt.title(CfL)
                        plt.subplot(1,2,2)
                        plt.imshow(CTm)
                        plt.show()
                        print(asdf)
                        '''
                        CT_tup += (CTm,)
                        sub_du = [] # empty sub_du list
            if self.solve_for_delay:
                CT_tup = ()
                for cfli, CfL in enumerate(Data_o['CfL']):
                    dUm = np.matrix([dU5[CfL]])
                    CTm = np.matmul(np.transpose(dUm),np.matmul(Cx_md,dUm))
                    CT_tup += (CTm,)
            if self.solve_for_misloc:
                CT_tup = ()
                for cfli, CfL in enumerate(Data_o['CfL']):
                    dUm = np.matrix([dU1[CfL],dU2[CfL],dU3[CfL]])
                    CTm = np.matmul(np.transpose(dUm),np.matmul(Cx_ml,dUm))
                    CT_tup += (CTm,)
                    
            # construct CT sparse matrix
            CT = sparse.block_diag(CT_tup)
            # combine cov. matrices
            CD = Cd + CT
            # get inverse CDI
            CDI = sparse.linalg.inv(CD)
        else:
            print('Cost function '+self.inv_function+' does not exist!')
        
        '''
        N0,N1 = 1000, 2000#0,len(D1)#
        fig = plt.figure(figsize=(20,10), facecolor='w', edgecolor='k')
        plt.subplot(1,3,1)
        cD = CD.toarray()
        cD[cD==0] = None
        plt.imshow(cD)
        plt.xlim((N0,N1))
        plt.ylim((N0,N1))
        plt.subplot(1,3,2)
        cd = Cd.toarray()
        cd[cd==0] = None
        im = plt.imshow(cd)
        plt.xlim((N0,N1))
        plt.ylim((N0,N1))
        plt.colorbar(im)
        plt.subplot(1,3,3)
        cT = CT.toarray()
        cT[cT==0] = None
        plt.imshow(cT)
        plt.xlim((N0,N1))
        plt.ylim((N0,N1))
        plt.show()
        print(asdf)
        '''
        
        # get freq dependend norm L2
        D1 = np.asarray(D1)
        U1 = np.asarray(U1)
        nl2 = self.get_freq_dependend_norm_L2(D1,U1)
                
        # calc pdf
        exp = np.matmul(L1m,np.matmul(CDI.toarray(),np.transpose(L1m),dtype=np.float128), dtype=np.float128)  
        #print(kpdf,exp,np.exp(-0.5*exp))
        #print(Data_o['det_C_d'],Data_o['mdet_C_d'])
        pdf = kpdf*np.exp(-0.5*exp*self.exp_scaling_fac, dtype=np.float128)*np.exp(1./(self.exp_scaling_fac), dtype=np.float128)
        #print(pdf,np.exp(-0.5*exp*10**-4, dtype=np.float128),np.exp(1./(10**-4), dtype=np.float128))
        #print('in network',pdf,[pdf[0,0],0,nl2,0])
        #print(asdf)
        # save to dictionary
        res['network'] = [pdf[0,0],0,nl2,0]
        #print(asdf)
        
        return res


##################################################################################################
##################################################################################################
### Linear Inversion (simple)
##################################################################################################
##################################################################################################

class Linear_Inversion:
    
    def __init__(self,Container=None,Observed=None,Fundamentals=None):
        '''
            __init__ of linear inversion
        '''
        self.Container = Container
        self.event_id = Container['source']['event_id']
        self.Mw_ref = Container['source']['ref_mag'][0]
        self.Observed = Observed
        self.Fundamentals = Fundamentals
        self.STAT_dict = Container['network']['STAT_dict']['F1_X0']
        self.src_loc = Container['source']['SRC_dict']['F1_X0']
        
    def calc_El2(self):
        if self.add_iso: 
            NE = 6
        else: 
            NE = 5
        El2a = np.zeros((0,NE))
        obs = np.zeros((0,1))
        for stat_id in sorted(self.Observed):
            for comp in self.Observed[stat_id]:
                for fi in self.Observed[stat_id][comp]:
                    e0 = np.zeros(len(self.Fundamentals['F1_X0']['F1'][stat_id][comp][fi].data))
                    if self.add_iso:
                        E = np.matrix([e0,e0,e0,e0,e0,e0])
                    else:
                        E = np.matrix([e0,e0,e0,e0,e0])
                    for jj in range(NE):
                        E[jj] = self.Fundamentals['F1_X0']['F'+str(jj+1)][stat_id][comp][fi].data
                    El2a = np.vstack((El2a,np.transpose(E)))
                    obs = np.append(obs,self.Observed[stat_id][comp][fi])
        El2 = np.matmul(np.linalg.inv(np.matmul( np.transpose(El2a),El2a)),np.transpose(El2a))
        return np.transpose(El2), obs


    def inversion(self,El2, obs):
        a = np.matmul(np.transpose(El2),np.transpose(np.asmatrix(obs)))
        a = np.array(np.transpose(a))[0].tolist()
        if self.add_iso:
            a_opt = [a[3]+a[4]+a[5],-a[3]+a[5],-a[4]+a[5],a[1],a[2],-a[0]]
        else:
            a_opt = [a[3]+a[4],-a[3],-a[4],a[1],a[2],-a[0]]
        return a_opt+(10**-20)*np.random.rand(len(a_opt)), a

    
    def vec2mat(self,v):
        m = [[v[0],v[3],v[4]],
            [v[3],v[1],v[5]],
            [v[4],v[5],v[2]]]
        return m

    def get_weights(self,M):
        '''
            Aki-Convention:
                    | -a4+a6    a1      a2    |
                M = |   a1   -a5+a6    -a3    |
                    |   a2     -a3   a4+a5+a6 |

            Dziewonski-Convention:
                    | a4+a5+a6    a2      a3  |
                M = |   a2   -a4+a6    -a1    |
                    |   a3     -a1   -a5+a6   |   

            CAP only samples for a pure DC mechanism, hence a6 is set to zero --> new sampling in full space (-1: implosion to 1: explosion)
        '''
        # rounding factor
        rf = 5

        # remove trace (isotropic part)
        MT = self.vec2mat(M)
        trM = np.trace(MT) # get trace tr(M)
        MT_dev = MT - (1./3.)*trM*np.eye(3) # remove isotropic influence to get deviatoric part
        # write weight vector "a" based on the given moment tensor
        #a = [MT_dev[0][1],MT_dev[0][2],-MT_dev[1][2],-MT_dev[0][0],-MT_dev[1][1],trM/3.] # Aki
        a = [round(-MT_dev[1][2],rf),
                round(MT_dev[0][1],rf),
                round(MT_dev[0][2],rf),
                round(-MT_dev[1][1],rf),
                round(-MT_dev[2][2],rf),
                round(trM/3.,rf)] # Dziewonski


        return a 
    
    def SixElementModeller(self,a):
        synt = {}
        for stat_id in self.Fundamentals['F1_X0']['F1']:
            synt[stat_id] = {}
            for comp in self.Fundamentals['F1_X0']['F1'][stat_id]:
                synt[stat_id][comp] = {}
                for fi in self.Fundamentals['F1_X0']['F1'][stat_id][comp]:
                    trace = self.Fundamentals['F1_X0']['F1'][stat_id][comp][fi].copy()
                    trace.data *= 0 # create zero trace trace-object
                    synt[stat_id][comp][fi] = trace
                    for jj in range(len(a)):
                        synt[stat_id][comp][fi].data += self.Fundamentals['F1_X0']['F'+str(jj+1)][stat_id][comp][fi].data * a[jj]                                     
        return synt
    
    def get_error(self,synt):
        '''

        '''
        d_full, u_full = [], [] # list for network error
        d_stat, u_stat = {}, {} # dictionary for station error
        for stat_id in self.Observed:
            d_stat[stat_id], u_stat[stat_id] = [], [] 
            for comp in self.Observed[stat_id]:
                for fi in self.Observed[stat_id][comp]:
                    d_full += list(self.Observed[stat_id][comp][fi].data)
                    u_full += list(synt[stat_id][comp][fi].data)
                    d_stat[stat_id] += list(self.Observed[stat_id][comp][fi].data) 
                    u_stat[stat_id] += list(synt[stat_id][comp][fi].data)
                    
        # network error
        d, u = np.asarray(d_full), np.asarray(u_full)
        nl2_netw = np.sum((d-u)**2)/np.sum(d**2)
        VR_netw = 1.-nl2_netw
        pdf_netw = np.exp(-0.5*nl2_netw)
        
        # station error
        nl2_stat = {}
        VR_stat = {}
        pdf_stat = {}
        for stat_id in d_stat:
            d, u = np.asarray(d_stat[stat_id]), np.asarray(u_stat[stat_id])
            nl2_stat[stat_id] = np.sum((d-u)**2)/np.sum(d**2)
            VR_stat[stat_id] = 1.-nl2_stat[stat_id]
            pdf_stat[stat_id] = np.exp(-0.5*nl2_stat[stat_id])
            
        # construct error dictionary
        error = {'Network':{'NL2':nl2_netw,'VR':VR_netw,'pdf':pdf_netw},
               'Station':{'NL2':nl2_stat,'VR':VR_stat,'pdf':pdf_stat}}
            
            
        return error    
    
    def display_station_error(self,filename=None):
        '''
        
        '''
        error_stat = self.res['Error']['Station']['NL2']
        
        stat_List = sorted(list(error_stat.keys()))
        x,NL2,VR = [], [], []
        for si,stat_id in enumerate(stat_List):
            x.append(si)
            NL2.append(self.res['Error']['Station']['NL2'][stat_id])
            VR.append(self.res['Error']['Station']['VR'][stat_id])
        
        fig = plt.figure(figsize=(20, 5), facecolor='w', edgecolor='k')
        ax = plt.subplot(1,2,1)
        plt.stem(x, NL2, use_line_collection=True)
        plt.xticks(x, stat_List,rotation='vertical',fontsize=18)
        plt.ylabel('NL2',fontsize=18)        
        plt.title('Normalized Least Square Error',fontsize=18)
        
        ax = plt.subplot(1,2,2)
        plt.stem(x, VR, use_line_collection=True)
        plt.xticks(x, stat_List,rotation='vertical',fontsize=18)
        plt.ylabel('VR',fontsize=18)
        plt.title('Variance Reduction',fontsize=18)
        if filename == None:
            plt.show()
        else:
            plt.savefig(filename+self.event_id+'_LinInv_station_error.png', 
                    bbox_inches='tight', 
                    transparent=False,
                    pad_inches=0)
    
    def display_beachball(self,filename=None):
        '''
        
        '''
        MT = self.res['MT']
        DC = self.res['Src_Param']
        
        fig = plt.figure(figsize=(5, 5), facecolor='w', edgecolor='k')
        ax = plt.subplot(111)
        b = beach(MT, size=200, xy=(0,0), linewidth=1, facecolor='b', alpha=0.9, nofill=False,zorder=10)
        ax.add_collection(b) 
        b = beach(DC[:3], size=200, xy=(0, 0), linewidth=4, facecolor='b', alpha=0.9, nofill=True,zorder=10)
        ax.add_collection(b) 
        ax.set_aspect('equal')
        plt.xlim((-103,103))
        plt.ylim((-103,103))
        plt.axis('off')
        if filename == None:
            plt.show()
        else:
            plt.savefig(filename+self.event_id+'_LinInv_beachball.png', 
                    bbox_inches='tight', 
                    transparent=False,
                    pad_inches=0)
                    
    def display_waveforms(self,xlim=[],filename=None,scaleY=False):
        '''
        
        '''
        from .Modeller import Full_MT_modeller
        TP = self.res['Src_Param']
        fac = self.res['Afac']
        
        Modeller = Full_MT_modeller(Container=self.Container,Fundamentals=self.Fundamentals)
        
        c_coord = np.array([TP[0],TP[1],TP[2],TP[3],TP[4],0,0,0,0,0,0.0,0.0])
        Synt_Full = Modeller.simulate(source_mechanism=c_coord)
        c_coord = np.array([TP[0],TP[1],TP[2],TP[3],0*TP[4],0,0,0,0,0,0.0,0.0])
        Synt_Dev = Modeller.simulate(source_mechanism=c_coord)
        c_coord = np.array([TP[0],TP[1],TP[2],0*TP[3],0*TP[4],0,0,0,0,0,0.0,0.0])
        Synt_DC = Modeller.simulate(source_mechanism=c_coord)
        
        lw, fsize = 3, 16
        error = []
        for stat_id in self.Observed:
            fig = plt.figure(figsize=(20, 5), facecolor='w', edgecolor='k')
            yaxA,yaxB = [],[]
            for ci,comp in enumerate(['Z','R','T']):
                ax = plt.subplot(1,3,ci+1)
                try:
                    d = self.Observed[stat_id][comp][0].copy()
                    td = np.arange(d.stats.npts)*d.stats.delta
                    plt.plot(td,d.data,'k-',linewidth=lw,label='obs')

                    f0 = Synt_Full['cF_X0'][stat_id][comp][0].copy()
                    f1 = Synt_Dev['cF_X0'][stat_id][comp][0].copy()
                    f2 = Synt_DC['cF_X0'][stat_id][comp][0].copy()
                    tf = np.arange(f0.stats.npts)*f0.stats.delta+0
                    
                    er0 =np.sum((d.data-fac*f0.data)**2)/np.sum(d.data**2) 
                    er1 =np.sum((d.data-fac*f1.data)**2)/np.sum(d.data**2)
                    er2 =np.sum((d.data-fac*f2.data)**2)/np.sum(d.data**2)
                    
                    plt.plot(tf,fac*f0.data,'g--',linewidth=lw,label='full (er='+str(round(er0,4))+')')
                    plt.plot(tf,fac*f1.data,'b--',linewidth=lw,label='dev (er='+str(round(er1,4))+')')
                    plt.plot(tf,fac*f2.data,'r--',linewidth=lw,label='dc (er='+str(round(er2,4))+')')
                    
                    if scaleY:
                        yaxA.append(1.15*np.max(np.asarray([np.max(d.data),np.max(fac*f0.data),np.max(fac*f1.data),np.max(fac*f2.data)])))
                        yaxB.append(1.15*np.min(np.asarray([np.min(d.data),np.min(fac*f0.data),np.min(fac*f1.data),np.min(fac*f2.data)])))
                        #print(yaxA,yaxB)
                    
                    if len(xlim) == 2: 
                        plt.xlim((xlim[0],xlim[1]))
                    plt.title(stat_id,fontsize=fsize)
                    plt.grid(True)
                    plt.legend()
                    plt.xlabel('Time in s',fontsize=fsize)
                    plt.ylabel('Displacement in m',fontsize=fsize)
                    ax.axis('on')
                except:
                    ax.axis('off')
                    continue
            # scale Y axis
            if scaleY: 
                for ci,comp in enumerate(['Z','R','T']):
                    ax = plt.subplot(1,3,ci+1)
                    try:
                        plt.ylim((np.min(np.asarray(yaxB)),np.max(np.asarray(yaxA))))
                    except:
                        continue
                
            if filename == None:
                plt.show()
            else:
                plt.savefig(filename+self.event_id+'_'+stat_id+'_LinInv_waveform.png', 
                        bbox_inches='tight', 
                        transparent=False,
                        pad_inches=0)
    
    def map_amplitudes(self,fi=0,exStat=[]):
        '''
        
        '''
        d_stat, E_list, dmax_List, dmin_List = {}, {'Z':[],'R':[],'T':[]}, {'Z':[],'R':[],'T':[]} ,{'Z':[],'R':[],'T':[]}
        for stat_id in self.Observed:
            if stat_id not in exStat:
                d_stat[stat_id] = {}
                for comp in self.Observed[stat_id]:
                    dmax = np.max(self.Observed[stat_id][comp][fi].data)
                    dmin = np.abs(np.min(self.Observed[stat_id][comp][fi].data))
                    E = np.sum(np.real(hilbert(np.abs(self.Observed[stat_id][comp][fi].data))))
                    E_list[comp].append(E)
                    dmax_List[comp].append(dmax) 
                    dmin_List[comp].append(dmin)
                    d_stat[stat_id][comp] = {'dmax':dmax,'dmin':dmin,
                                            'drel':dmin/dmax,'E':E}
        # plot map
        fig = plt.figure(figsize=(20, 5), facecolor='w', edgecolor='k')
        for ci, comp in enumerate(['Z','R','T']): 
            # normalize vectors
            E_norm = np.sum(np.asarray(E_list[comp]))
            dmax_norm = np.sum(np.asarray(dmax_List[comp]))
            dmin_norm = np.sum(np.asarray(dmin_List[comp]))
            
            # create subplot
            plt.subplot(1,3,ci+1)
            [lat, lon, depth] = self.src_loc
            plt.scatter(lon,lat,s=1500, marker='*', c='k')
            
            for sti, stat_id in enumerate(d_stat):
                [lat,lon,_,_,_,_] = self.STAT_dict[stat_id]
                info = d_stat[stat_id][comp]
            
                plt.scatter(lon,lat, s=1000*info['E']/E_norm, marker='o', c='r')
                plt.title('Energy - '+comp)
            
        plt.show()
        
        
    
    def organizer(self,add_iso=False):
        '''
        
        '''
        self.add_iso = add_iso
        
        # linear inversion
        El2, obs = self.calc_El2()
        MT_opt, a = self.inversion(El2, obs)
        
        # model synthetics
        a_opt = self.get_weights(MT_opt)
        synt = self.SixElementModeller(a_opt)
        
        # get waveform error
        error = self.get_error(synt)
        
        # get source parameter
        strike,dip,rake,clvd,iso,M0_fac = M2Tape(MT_opt)
        
        # get moment magnitude
        M0_ref = 10**((3.0/2.0) * (self.Mw_ref + 10.73))
        if M0_fac > 0.:
            M0_inv = M0_ref*M0_fac
        elif M0_fac < 0.:
            M0_inv = M0_ref/np.abs(M0_fac)
        else:
            M0_inv = M0_ref
        Mw = (2.0 / 3.0) * np.log10(M0_inv) - 10.73
        
        # construct result dictionary (convert numpy to float for json)
        src_params = [float(strike),float(dip),float(rake),float(clvd),float(iso),float(M0_inv)]#float(M0_inv)]
        MT = []
        for mt in MT_opt:
            MT.append(float(mt))
        self.res = {'Error':error,'MT':MT,
               'Src_Param':src_params,'Mw':float(Mw),'Afac':float(M0_fac),'Synt':synt}


##################################################################################################
##################################################################################################
### Xtree algorithm
##################################################################################################
##################################################################################################


class uniXtree:
    
    def __init__(self,Container=None,Observed=None,Fundamentals=None):
        self.Container = Container
        self.Obs = Observed
        self.Fund = Fundamentals
        
        self.V0 = 10**100       
        self.pdf_fac = 1.0
        self.M0 = 1.0
        
        self.nlay_min = Container['inversion']['Xtree']['nlay_min']
        self.nlay_max = Container['inversion']['Xtree']['nlay_max']
        self.multi_core = Container['inversion']['Xtree']['multicore']
        self.half_min = Container['inversion']['Xtree']['half_min']
        self.half_max = Container['inversion']['Xtree']['half_max']
        self.tcrit = Container['inversion']['Xtree']['tcrit']
        self.full_space_Xlayer = Container['inversion']['Xtree']['full_space_Xlayer']
        
        self.simulate_doublet = Container['source']['simulate_doublet']
        self.perform_Tshift = Container['inversion']['time_shift']['perform']
        
        self.inv_function = Container['inversion']['function']
        self.rec_inv_procedure = Container['inversion']['rec_inv_procedure']
        self.magnitude_update = Container['inversion']['magnitude_update']
        
        self.exp_scaling_fac = Container['inversion']['exp_scaling_fac']
        
        self.rmStation = {}
        self.TStation = {}
        for stat_id in self.Obs:
            self.rmStation[stat_id] = []
            #self.TStation[stat_id] = []
        self.test = {'P':[],'NL2':[],'pdf':[]}#,'CVol':[]
        
    def Time_Logger(self,func_name,file_name,run_time):
        debug_message = 'Function %s in file %s with runtime=%s' % (func_name,file_name.split('/')[-1],str(run_time))
        TLOG.debug(debug_message)
    
    def check_termination_criterion(self,top_list,ocii):
        '''
            Criterions:
                max value for ocii
                min value for ocii
                
        '''
        P, pdf, NL2 = [], [], []
        # CVol = []
        for ii in range(8):
            try:
                P.append(top_list[ii][0])
                pdf.append(top_list[ii][1])
                NL2.append(top_list[ii][2])
                #CVol.append(top_list[ii][2]) # 4
            except:
                continue
        self.test['P'].append(np.mean(np.asarray(P)))
        self.test['pdf'].append(np.mean(np.asarray(pdf)))
        self.test['NL2'].append(np.mean(np.asarray(NL2)))
        #self.test['CVol'].append(np.mean(np.asarray(CVol)))
        
        # check termination criterion
        if ocii == self.nlay_max:
            if self.show_progress:
                print('Terminating run: Max ocii reached')
            run = False # I_max is reached
        elif ocii >= self.nlay_min:
            
            # check min tree resolution
            if top_list[ii][2] >= self.half_min: 
                
                # check gradient
                a0 = self.tcrit[1]
                a1 = np.abs(np.diff(np.asarray(self.test[self.tcrit[0]])))[-1]
                a2 = np.abs(np.diff(np.asarray(self.test[self.tcrit[0]])))[-(a0+2):-2]
                #print('abs_diff ',a1,np.mean(a2),a1/np.mean(a2))
                
                if self.tcrit[0] == 'P': 
                    tcrit_cond = a1/np.mean(a2) > self.tcrit[2]
                elif self.tcrit[0] == 'NL2':
                    tcrit_cond = a1/np.mean(a2) < self.tcrit[2]
                else:
                    print('tcrit only usable for P and NL2, you choose '+self.tcrit[0])
                
                if tcrit_cond:
                    if self.show_progress:
                        print('Terminating run: Change in '+self.tcrit[0]+' is below threshold')
                    run = False # change in gradient is below threshold
                else:
                    run = True # gradient is still too steep
            else:
                run = True # interval halfing below threshold (cell is to coarse)  
        else:
            run = True # iteration below I_max and no other criteria meet

        return run
    
    def station_score_handler(self,res,key):
        '''
            input: combined error for full network and each station
            process: select all stations with error above network error (mean of all stations)
            return: list of stations with error larger than mean
            usage: append sublist to larger list over several loops an Xtree layers
                    --> remove stations with most occurence in list from obs dict
                    
            res['TShift']:
            netw:
            [mean, std]
            stat:
            {'BK_CMB': [mean, std],'CI_GRA': [mean, std]}
            trace:
            {'BK_CMB': 'Z':[mean, std],'CI_GRA': 'R':[mean, std]}
        '''
        estat = []
        for stat_id in res['per_station']:
            estat.append([res['per_station'][stat_id],stat_id])
            if self.perform_Tshift:
                self.TStation[key] = res['TShift']
                #for comp in res['TShift']['trace'][stat_id]:
                #    res['TShift']['trace'][stat_id][comp][0]
                #    self.TStation[stat_id].append([ts,key])
        estat.sort(key = lambda x: x[0])   
        for sii,sinfo in enumerate(estat):
            self.rmStation[sinfo[1]].append(sii)

    
    def get_Oc_error(self,c_coord,key):
        '''
        
        '''
        
        #c_coord = [126,67,82, 0.0, 0.0]
        #print('error',c_coord)
        # compute synthetics
        Modeller = Full_MT_modeller(Container=self.Container,Fundamentals=self.Fund)
        Synt = Modeller.simulate(source_mechanism=c_coord)
        # get error
        DM = Data_Modelling(Container=self.Container)
        res = DM.get_pdf(self.Obs, Synt, self.inv_function)
        #print(res)
        #nl2 = res['network'][2]
        #print(nl2)
        #print(asdf)
        # store station wise information for station performance ranking
        self.station_score_handler(res,key)
        
        return res
    
    def Xtree_ranking(self,ranking_list,k_norm,oc_logger,top_fac):
        '''
            ranking_list --> [P,pdf,nl2,key]
            P/k_norm
        '''
        # normalized create ranking list
        ranking = []
        # new Mother_coord
        Mother_coord = []
        # norm P
        #P_test = 0
        for i in ranking_list:
            ranking.append([i[0]/(k_norm*i[4])]+i[1:])
            #P_test += i[0]/k_norm
        #print(P_test)
        # rank based on norm
        sranking = sorted(ranking, reverse=True)
        # get coord of top_fac solutions
        for ii in range(top_fac):
            key = sranking[ii][3]
            Mother_coord.append([oc_logger[key][1],oc_logger[key][0],key])

        return sranking, Mother_coord
    
    def check_limit(self,value,dX,Xmin,Xmax,oclayer,sym):
        '''
        
        '''
        Xout = []
        ifac = 2.**(oclayer) # intervall halfing at oclayer
        for mfac in [1,-1]: # +-dfault
            if dX > 0.: # dfault is larger than zero and positive
                nX = value+mfac*dX/ifac # cfault added with dfault range at oclayer --> child coord
                #print('nx',nX,value,mfac,dX,ifac)
                if Xmin <= nX <= Xmax: # check if child coord is within valid parameter space (Int_boarder)     
                    Xout.append((nX))
                elif nX < Xmin: # case: its smaller than the lowest value  in the parameter space
                    if sym: # check if cfault parameter behaves symetrical (only for strike and rake)
                        Xout.append(round(Xmax+nX,0)) # nX is always negative (strike,rake) --> 180 or 306 +(-nx)
                        #print('small',round(Xmax+nX,0),Xmax,nX)
                    else: # cfault is nor symterical
                        Xout.append(Xmin) # set lower parameter boundary as lower range
                elif nX > Xmax: # case: its larger than the largest value in the parameter space
                    if sym: # compare syntax of lower boundary
                        Xout.append(round(nX-Xmax,0)) 
                    else:
                        Xout.append(Xmax)
            else:
                Xout = [(value)] # retun initial cfault
            
        return Xout
    
    
    def children(self,Mother_coord):
        '''
        
        '''
        iter_list = {}
        # F1: strike dip rake clvd iso
        # F2: strike dip rake clvd iso dM dt
        sym_List = [True,False,True,False,False,
                    True,False,True,False,False,False,False]
        clvd_min = -1./3.#-np.pi/6#
        clvd_max = 1./3.#np.pi/6#
        iso_min = 0. #-3./8. * np.pi #
        iso_max = 3./4. * np.pi #3./8. * np.pi #
        Int_border = [[0.,360.],[0.,1.],[-180.,180.],[clvd_min,clvd_max],[iso_min,iso_max],
                      [0.,360.],[0.,1.],[-180.,180.],[clvd_min,clvd_max],[iso_min,iso_max],
                      [0.,1.],[-self.dtoff,self.dtoff]]
        Cround = [0,6,0,6,6,0,6,0,6,6,3,3]

        
        for ti in range(len(Mother_coord)):
            oclayer = Mother_coord[ti][0]+1
            if oclayer < self.half_max:
                Fault_iter_List = []
                # get central coordinates of children cells
                for Inti in range(len(self.cfault)):
                    if self.dfault[Inti] == 0:
                        Fault_iter_List.append([self.cfault[Inti]])
                    else:
                        #print(Mother_coord[ti][1][Inti])
                        # get child coordinate
                        cfault_child = self.check_limit(Mother_coord[ti][1][Inti],self.dfault[Inti],Int_border[Inti][0],Int_border[Inti][1],oclayer,sym_List[Inti])
                        Fault_iter_List.append([round(cfault_child[0],Cround[Inti]),round(cfault_child[1],Cround[Inti])])
                        #print(cfault_child)
                
                if oclayer not in iter_list.keys(): 
                    iter_list[oclayer] = {}
                iter_list[oclayer][ti] = list(itertools.product(*Fault_iter_List))
        #print(asdf)
        return iter_list
    
  
    def check_c_coord(self,c_coord,Coord_List,Coord_List_child):
        '''
            check if central coordinate (c_coord) was already modeled and is in Coord_List
            or one of the Children in the current run
            if so, check auxillary with same settings
            if that one is also ruled out, c_coord will be scipt (False)
        '''
        
        # check auxiallary plane
        s12,d12,r12 = aux_plane(c_coord[0],c_coord[1],c_coord[2])
        s12,d12,r12 = round(s12,0),round(d12,0),round(r12,0) 
        if self.simulate_doublet:
            s22,d22,r22 = aux_plane(c_coord[5],c_coord[6],c_coord[7])
            s22,d22,r22 = round(s22,0),round(d22,0),round(r22,0) 
            c_coord2b = tuple(np.array([c_coord[0],c_coord[1],c_coord[2],c_coord[3],c_coord[4],
                                       s22,d22,r22,c_coord[8],c_coord[9],c_coord[10],c_coord[11]]))
            c_coord2c = tuple(np.array([s12,d12,r12,c_coord[3],c_coord[4],
                                       c_coord[5],c_coord[6],c_coord[7],c_coord[8],c_coord[9],c_coord[10],c_coord[11]]))
            c_coord2d = tuple(np.array([s12,d12,r12,c_coord[3],c_coord[4],
                                       s22,d22,r22,c_coord[8],c_coord[9],c_coord[10],c_coord[11]]))
            if c_coord not in Coord_List+Coord_List_child:
                return c_coord, True
            #elif c_coord2b not in Coord_List+Coord_List_child:
            #    return c_coord2b, True
            #elif c_coord2c not in Coord_List+Coord_List_child:
            #    return c_coord2c, True
            #elif c_coord2d not in Coord_List+Coord_List_child:
            #    return c_coord2d, True
            else:
                return c_coord, False
        else:
            c_coord2 = tuple(np.array([s12,d12,r12,c_coord[3],c_coord[4]]))
            # check for auxillary or previous c_coord setting
            if c_coord not in Coord_List+Coord_List_child:
                return c_coord, True
            #elif c_coord2 not in Coord_List+Coord_List_child:
            #    return c_coord2, True
            else:
                return c_coord, False
    
    def convet_c_coord(self,c_coord,modus='t2f'):
        '''
        
        '''
        # helper function for isotropic conversion
        def u_function(beta):
            u = 3/4 * beta - 1/2 * np.sin(2*beta) + 1/16 * np.sin(4*beta)
            return 3*np.pi/8 - u

        def diff(x,a):
            ut = u_function(x)
            return (ut - a)**2
        
        # conversion factors
        t2f_clvd = 180./np.pi*100/30
        t2f_iso = 180./np.pi*100/90
        f2t_clvd = np.pi/180.*30/100
        f2t_iso = np.pi/180*90/100
        
        # convert tuple to list
        c_coord = list(c_coord)
        
        if modus == 't2f': # return coord as float with zero digits
            c_coord[1] = round(np.arccos(c_coord[1])*180./np.pi,0)
            c_coord[3] = round((1./3.)*np.arcsin(3*c_coord[3])*t2f_clvd,0)
            c_coord[4] = round(minimize(diff, 1.0, args=(c_coord[4]), 
                                  method='Nelder-Mead', tol=1e-6).x[0]*t2f_iso,0)
            if self.simulate_doublet:
                c_coord[6] = round(np.arccos(c_coord[6])*180./np.pi,0)
                c_coord[8] = round((1./3.)*np.arcsin(3*c_coord[8])*t2f_clvd,0)
                c_coord[9] = round(minimize(diff, 1.0, args=(c_coord[9]), 
                                  method='Nelder-Mead', tol=1e-6).x[0]*t2f_iso,0)                
        elif modus == 'f2t':
            c_coord[1] = np.cos(c_coord[1]*np.pi/180)
            c_coord[3] = (1./3.)*np.sin(3.*c_coord[3])*f2t_clvd
            c_coord[4] = u_function(c_coord[4])*f2t_iso
            if self.simulate_doublet:
                c_coord[6] = np.cos(c_coord[6]*np.pi/180)
                c_coord[8] = (1./3.)*np.sin(3*c_coord[8])*f2t_clvd
                c_coord[9] = u_function(c_coord[9])*f2t_iso
        
        return tuple(c_coord)
    
    def kernel_grid_search_Xtree(self,Mother_coord,Coord_List):
        '''
        
        '''        
        # create lists and dictionaries
        oc_logger, res_logger = {}, {}
        oc_dict = []
        Coord_List_child = []
        itter_ii = 0
        
        # get children
        iter_list = self.children(Mother_coord)
        
        # get pdf of child cell
        for il,half_index in enumerate(iter_list):
            for mcell in iter_list[half_index]:
                for ll, c_coord_tape in enumerate(iter_list[half_index][mcell]):
                    
                    # convert tape parameterization to standard parameterization (e.g. dip 0-90)
                    c_coord = self.convet_c_coord(c_coord_tape,modus='t2f')
                    # check if coord was computed before (incl. aux-solution)
                    c_coord, c_check = self.check_c_coord(c_coord,Coord_List,Coord_List_child)                    
                    
                    if c_check:
                        # define key
                        timestamp = float(UTCDateTime().timestamp)
                        key_c = '_'.join(map(str, c_coord))
                        key = str(round(timestamp,3))+'-'+key_c
                        
                        # log coord info
                        Coord_List_child.append(c_coord)
                        
                        # get error
                        res = self.get_Oc_error(c_coord,key)
                        
                        # add to res list 
                        pdf = res['network'][0]*self.pdf_fac
                        nl2 = res['network'][2]
                        mag = res['Magnitude'][self.magnitude_update]['Mw']
                        
                        # calculate volume probability (absolute)
                        V_sub = self.V0/(2**(self.ocDim*half_index)) # prior volume probability 
                        P = V_sub*pdf # probability density (pdf)               
                        
                        # append to lists
                        oc_dict.append([P,pdf,nl2,key,V_sub])
                        oc_logger[key] = [c_coord_tape,half_index,V_sub,mag]
                        res_logger[key] = res
                        
                        # count number of iteration
                        itter_ii += 1
        return oc_dict,oc_logger,res_logger,itter_ii
    
    def grid_search_Xtree_multi_core(self,ocii,Mother_coord,Coord_List,result):
        '''
            n:      number of layer (subdiscretization of mother cell)
            xspace: [X0,X1] - corner information (sampling space) of parameter X
            yspace: [Y0,Y1] - corner information (sampling space) of parameter Y
        '''
        
        # run grid search kernel
        t_GS_start = UTCDateTime()
        oc_dict,oc_logger,res_logger,itter_ii = self.kernel_grid_search_Xtree(Mother_coord,Coord_List)
        t_GS_end = UTCDateTime()-t_GS_start
        
        # return oc_dict as result (muliprocessing return parameter)
        if len(oc_dict) > 0:
            result[0] = oc_dict
            result[1] = oc_logger
            result[2] = res_logger
            result[3] = self.rmStation
            result[4] = self.TStation
            result[5] = True
            result[6] = [t_GS_end,itter_ii,t_GS_end/itter_ii] # [run time in s, number of loops, time per loop]
        else:
            result[5] = False
            
        
    def grid_search_Xtree_single_core(self,ocii,Mother_coord,Coord_List):
        '''
            n:      number of layer (subdiscretization of mother cell)
            xspace: [X0,X1] - corner information (sampling space) of parameter X
            yspace: [Y0,Y1] - corner information (sampling space) of parameter Y
        '''
        
        # run grid search kernel
        t_GS_start = UTCDateTime()
        oc_dict,oc_logger,res_logger,itter_ii = self.kernel_grid_search_Xtree(Mother_coord,Coord_List)
        t_GS_end = UTCDateTime()-t_GS_start
        
        result = {}
        # return oc_dict as result 
        if len(oc_dict) > 0:
            result[0] = oc_dict
            result[1] = oc_logger
            result[2] = res_logger
            result[3] = self.rmStation
            result[4] = self.TStation
            result[5] = True
            result[6] = [t_GS_end,itter_ii,t_GS_end/itter_ii] # [run time in s, number of loops, time per loop]
        else:
            result[5] = False
        #print(asdf)
        return result
    
    def get_dip_interval_old(self):
        '''
        
        '''
        # convert bad parameterization of full dip space (45+-45 --> 0.707+-0.707)
        # to good parameterization (0.5+-0.5) --> h = cos(dip)
        if self.dfault[1] == 45: # case full dip space
            self.dfault[1] = 0.5
            self.cfault[1] = 0.5
            if self.simulate_doublet:
                self.dfault[6] = 0.5
                self.cfault[6] = 0.5
        else:
            # convert dip
            dip_center = np.cos((self.cfault[1])*np.pi/180.)
            ddip_up = np.cos((self.cfault[1]-self.dfault[1])*np.pi/180.)
            ddip_low = np.cos((self.cfault[1]+self.dfault[1])*np.pi/180.)
            # update center
            self.cfault[1] = dip_center
            # update range
            self.dfault[1] = np.max([dip_center-ddip_low,
                                    ddip_up-dip_center])
            if self.simulate_doublet:
                dip_center = np.cos((self.cfault[6])*np.pi/180.)
                ddip_up = np.cos((self.cfault[6]-self.dfault[6])*np.pi/180.)
                ddip_low = np.cos((self.cfault[6]+self.dfault[6])*np.pi/180.)
                # update center
                self.cfault[6] = dip_center
                # update range
                self.dfault[6] = np.max([dip_center-ddip_low,
                                        ddip_up-dip_center])
    
    def get_dip_interval(self):
        '''
        
        '''
        ddip_up = np.cos((self.cfault[1]-self.dfault[1])*np.pi/180.)
        ddip_low = np.cos((self.cfault[1]+self.dfault[1])*np.pi/180.)
        self.dfault[1] = np.abs(ddip_up - ddip_low)/2
        self.cfault[1] = np.abs(ddip_up-self.dfault[1])
        if self.simulate_doublet:
            ddip_up = np.cos((self.cfault[6]-self.dfault[6])*np.pi/180.)
            ddip_low = np.cos((self.cfault[6]+self.dfault[6])*np.pi/180.)
            self.dfault[6] = np.abs(ddip_up - ddip_low)/2
            self.cfault[6] = np.abs(ddip_up-self.dfault[6])
    
    def get_iso_intervall(self):
        '''
            ISO in [-100,100]
            beta in [0,pi]
            u in [0,3pi/4]
            w in [-3pi/8,3pi/8]
            w = (3*np.pi/8) - u
            
            beta(ISO) = ISO * (3pi/4)/100
            u(beta) = 3/4 * beta - 1/2 * np.sin(2*beta) + 1/16 * np.sin(4*beta)
            beta(u) --> nummerically solved
            
            # original inverse syntax
            https://www.moonbooks.org/Articles/How-to-numerically-compute-the-inverse-function-in-python-using-scipy-/
        '''
        
        if self.dfault[4] != 0.:
            print('!!!!!!!!!!!!!!!!!!!!')
            print('!!!!!!Warning!!!!!!!')
            print('!!!!!!!!!!!!!!!!!!!!')
            print('Sampling for isotropic changes')
            print('has not yet been extensively tested!')
            print('Resulting mechanism might contain errors.')
            print()
        
        
        # helper function for isotropic conversion
        def w_function(beta):
            u = 3/4 * beta - 1/2 * np.sin(2*beta) + 1/16 * np.sin(4*beta)
            return 3*np.pi/8 - u # also known as w
        
        # convert cfault(CLVD) to cfault(gamma)
        self.cfault[4] *= 90/100
        # convert dfault(CLVD) to dfault(gamma)
        self.dfault[4] *= 90/100
        # compute range
        w_val_center = w_function(self.cfault[4]*np.pi/180.)
        w_val_up = w_function((self.cfault[4]+self.dfault[4])*np.pi/180.)
        w_val_down = w_function((self.cfault[4]-self.dfault[4])*np.pi/180.)
        # update center
        self.cfault[4] = w_val_center
        # update range
        self.dfault[4] = round(np.max([w_val_center-w_val_up,
                                       w_val_down-w_val_center]),6)
        
        if self.simulate_doublet:
            # convert cfault(CLVD) to cfault(gamma)
            self.cfault[9] *= 90/100
            # convert dfault(CLVD) to dfault(gamma)
            self.dfault[9] *= 90/100
            # compute range
            # compute range
            w_val_center = w_function(self.cfault[9]*np.pi/180.)
            w_val_up = w_function((self.cfault[9]+self.dfault[4])*np.pi/180.)
            w_val_down = w_function((self.cfault[9]-self.dfault[4])*np.pi/180.)
            # update center
            self.cfault[9] = w_val_center
            # update range
            self.dfault[9] = round(np.max([w_val_center-w_val_up,
                                        w_val_down-w_val_center]),6)

    
    def get_clvd_interval(self):
        '''
            CLVD in [-100,100]
            gamma in [-pi/6,pi/6]
            v in [-1/3,1/3]
            
            gamma(CLVD) = CLVD * (pi/6)/100
            v(gamma) = 1/3 * sin(3gamma)
            gamma(v) = 1/3 * sin^-1(3v)            
        '''
        #print(self.cfault[3])
        # convert cfault(CLVD) to cfault(gamma)
        self.cfault[3] *= 30/100
        # convert dfault(CLVD) to dfault(gamma)
        self.dfault[3] *= 30/100
        # compute range
        clvd_center = (1./3.)*np.sin(3*self.cfault[3]*np.pi/180)
        #(1./3.)*np.sin(3*c_coord[3]*np.pi/180)
        #print(clvd_center)
        dclvd_up = (1./3.)*np.sin(3.*(self.cfault[3]+self.dfault[3])*np.pi/180.)
        dclvd_low = (1./3.)*np.sin(3.*(self.cfault[3]-self.dfault[3])*np.pi/180.)
        # update center
        self.cfault[3] = clvd_center
        # update range
        self.dfault[3] = np.max([np.abs(dclvd_up-clvd_center),
                                 np.abs(clvd_center-dclvd_low)])
        #print('clvd',self.cfault[3],self.dfault[3],dclvd_up,dclvd_low)
        if self.simulate_doublet:
            # convert cfault(CLVD) to cfault(gamma)
            self.cfault[8] *= 30/100
            # convert dfault(CLVD) to dfault(gamma)
            self.dfault[8] *= 30/100
            # compute range
            clvd_center = (1./3.)*np.sin(3.*(self.cfault[8])*np.pi/180.)
            dclvd_up = (1./3.)*np.sin(3.*(self.cfault[8]+self.dfault[8])*np.pi/180.)
            dclvd_low = (1./3.)*np.sin(3.*(self.cfault[8]-self.dfault[8])*np.pi/180.)
            # update center
            self.cfault[8] = clvd_center
            # update range
            self.dfault[8] = np.max([np.abs(dclvd_up-clvd_center),
                                 np.abs(clvd_center-dclvd_low)])
            
    
    def organizer(self,cfault=None,dfault=None,top_fac=None,show_progress=False):
        '''
        
        '''
        # show_progress
        self.show_progress = show_progress
        
        # create Xtree logger dictionary   
        Xtree_log = {}
        Station_log = {}
        TShift_log = {}
        for stat_id in self.Obs:
            Station_log[stat_id] = []
            #TShift_log[stat_id] = []

        
        # get Xtree parameter space
        if cfault == None:
            self.cfault = []
            # use event_id set in input_json
            self.cfault = self.Container['inversion']['Xtree']['starting_solution_F1'].copy()
            if self.simulate_doublet:
                self.cfault += self.Container['inversion']['Xtree']['starting_solution_F2'].copy()
        else:
            self.cfault = cfault
        if dfault == None:
            self.dfault = []
            self.dfault = self.Container['inversion']['Xtree']['Xtree_range_F1'].copy()
            if self.simulate_doublet:
                self.dfault += self.Container['inversion']['Xtree']['Xtree_range_F2'].copy()
                self.dtoff = self.dfault[11]
            else:
                self.dtoff = 0.
        else:
            self.dfault = dfault.copy()
        if top_fac == None:
            top_fac = self.Container['inversion']['Xtree']['top_fac']
        
        # define dip, clvd and iso parameter interval (conversion from degree to uniform Tape parameterization)
        self.get_dip_interval()
        self.get_clvd_interval()
        self.get_iso_intervall()
        #print('Starting_solution ',self.cfault)
        #print('Starting_range ',self.dfault)
        #print('Do I sample the CLVD correctly?')
        #print('Do I sample the ISO correctly?')
        #print(asdf) 
        
        # get Xtree dimension (based on nonzero dfault setting
        self.ocDim = len(np.nonzero(self.dfault)[0])
        
        # create coordinate dictionaries
        Mother_coord = [[0,self.cfault,'000']] # cell half_index, central coordinate (c_coord), key
        Coord_List = [] # list of checked coordinates
        oc_logger, res_logger = {}, {}
        ranking_list = {0:[]}
  
        # Xtree layer loop (change to while loop with break condition function of the pdf
        trun,nRun,tLoop = None, None, None
        run = True
        ocii = 0
        while run:
            if self.full_space_Xlayer == 0:
                child_cells = top_fac
            else:
                if ocii <= self.full_space_Xlayer-1: 
                    child_cells = 2**(self.ocDim*(ocii+1)) # search full space
                else:
                    child_cells = top_fac

            if self.show_progress:
                print(30*'#')
                print('Xtree layer: '+str(ocii+1))
                print('Using cost function: '+self.inv_function+' in '+str(self.ocDim)+' dimensions.')
                print('Number of ranked mother cells: '+str(len(Mother_coord)))
                print('Number of examined coordinates = '+str(child_cells*(2**self.ocDim)))
                #if trun is not None:
                #    print('Estimated run time: '+str(int(tLoop*child_cells*(2**self.ocDim)))+'s')

            if self.multi_core:
                manager = multiprocessing.Manager()
                result = manager.dict()
                pcore = []
                
                # create a new process for given event
                args = [ocii,Mother_coord,Coord_List,result]
                proc = Process(target=self.grid_search_Xtree_multi_core,args=args)
                pcore.append(proc)
                
                # start
                if self.show_progress:
                    print('Start multiprocessing')
                for proc in pcore:
                    proc.start()
                for proc in pcore:
                    proc.join()
                for pi, proc in enumerate(pcore):
                    # this part is needed to delete finishd processes
                    # if they are not deleted, the process is tried to run anew with the same name, this is not possible
                    if proc is not proc.is_alive():
                        del pcore[pi]
                if self.show_progress:
                    print('Collect results from multiprocessing')
                
            else:
                result = self.grid_search_Xtree_single_core(ocii,Mother_coord,Coord_List)
                
            
            # Xtree termination
            if result[5]:
                # update station ranking
                for stat_id in result[3]:
                    Station_log[stat_id] += result[3][stat_id]
                    
                # update station time shift
                TShift_log.update(result[4])
                    
                # update oc_logger
                oc_logger.update(result[1])
                
                # update res_logger
                res_logger.update(result[2])
                
                # get run time
                [trun,nRun,tLoop] = result[6]
                
                if self.show_progress:
                    print('Update true run time: ',trun,' s for ',nRun,' runs. Thats ',tLoop,' s per loop.')
                    print('Length of update list: ',len(result[0]))
                    #print('Length of update list',len(Mother_coord))
                    
                # append new results to ranking list 
                for ii in range(len(result[0])):
                    key = result[0][ii][3] # c_coord_key (timestamp_c_coord)
                    ranking_list[0].append(result[0][ii]) # containes [P,pdf,nl2,key] of all tested coords
                    # add tested coords to coord list
                    Coord_List.append([result[1][key][0]]) # these coords will not be re-evaluated
                
                # remove surveyed Mother_keys from ranking_list
                # get list of Mother keys
                M_List = [el[2] for el in Mother_coord]
                # get initial list of keys in ranking_list
                R_List = [el[3] for el in ranking_list[0]]
                # loop over mother keys
                for M_id in M_List:
                    if M_id != '000': # ingnore initial mother key (will not be in any list)
                        # get index of key in ranking list
                        R_id = R_List.index(M_id)
                        # delete sublist
                        del ranking_list[0][R_id]
                        # get new R_List (else indexing will be wrong)
                        R_List = [el[3] for el in ranking_list[0]]
                
                # normalization factor k (from pdf) 
                k_norm = np.sum(np.asarray([el[1] for el in ranking_list[0]]))
                
                # update Xtree cell information
                if self.show_progress:
                    print('Probability ranking. Displaying top '+str(child_cells)+' solutions (new mother cells)')
                    
                # rank surveyed coordinates
                sranking, Mother_coord = self.Xtree_ranking(ranking_list[0],k_norm,oc_logger,child_cells)
                if self.show_progress:
                    print('P | NL2 | dV | ocii | mech')
                    for ii in range(child_cells):
                        key = sranking[ii][3]
                        mech = self.convet_c_coord(list(oc_logger[key][0]),modus='t2f')
                        print(round(sranking[ii][0],8),' | ',
                              round(sranking[ii][2],4),' | ',
                              round(sranking[ii][4]/self.V0,12),' | ',
                              oc_logger[key][1],' | ',
                              mech)#oc_logger[key][0])#

                #dV, dP = [], []
                #for cont in sranking:
                #    dV.append(cont[4]*10*10)
                #    dP.append(cont[0])
                #print(ocii,'V',np.sum(np.asarray(dV)),'P',np.sum(np.asarray(dP)))
                
                
                # check termination criterion
                if self.show_progress:
                    print('Check termination criterion')
                #if ocii >= 3:
                run = self.check_termination_criterion(sranking,ocii)
                if run:
                    ocii += 1
                    if self.show_progress:
                        print('Proceeding with run ...')
                        
                #print(asdf)
                        
            else:
                if self.show_progress:
                    print('Terminating run: Xtree - Termination')
                run = False
            
            if self.show_progress:
                print(30*'#')
                print()
        
        # update Xtree json file (create new)
        #self.write2json(full_result_out)        
        
        #
        
        # create result_dict to follow format in postprocessing (need to update)
        result_dict = []
        k_norm = np.sum(np.asarray([el[1] for el in ranking_list[0]]))
        #k_normA = np.power(k_norm, 100/self.exp_scaling_fac)
        #k_normB = np.sum(np.asarray([np.power(el[1], 100) for el in ranking_list[0]]))
        #print('K norm',k_norm,k_normB,k_normA)
        for cres in ranking_list[0]: # [P,pdf,nl2,key,dV]
            [c_coord_tape,half_index,V_sub,mag] = oc_logger[cres[3]]
            # content: [P,c_coord,half_index,pdf,V_sub,mag,key,nl2]
            #print(type(cres[0]),type(k_norm),type(V_sub),cres[0],k_norm,V_sub,cres[0]/(k_norm*V_sub))
            c_coord = self.convet_c_coord(c_coord_tape,modus='t2f')
            result_dict.append([cres[0]/(k_norm*V_sub),c_coord,half_index,cres[1],V_sub,mag,cres[3],cres[2]])
            
    
        #dV, dP = [], []
        #for cont in result_dict:
        #    dV.append(cont[4])
        #    dP.append(cont[0])
        #print('P',np.sum(np.asarray(dP)))
        #print('V',np.sum(np.asarray(dV)))
        
        
        # write results in Result dictionary
        Result = {'result':{0:sorted(result_dict, reverse=True)},
                  'oc_logger':oc_logger,
                  'res_Logger':res_logger,
                  'Station_log':Station_log,
                  'TShift_log':TShift_log,
                  'tremination_criterion':self.test}
        
        return Result 


##################################################################################################
##################################################################################################
### Carthesian grid search algorithm
##################################################################################################
##################################################################################################

class Grid_Search:

    def __init__(self,Container=None,Observed=None,Fundamentals=None):
        '''
        
        '''
        self.Container = Container
        self.Observed = Observed
        self.Fundamentals = Fundamentals
    
    def grid_search_multi_core(self,grid,result):
        '''
        
        '''
        res_list = []
        for mech in itertools.product(*grid):
            # simulate mechanism
            c_coord = np.array(mech)
            Modeller = Full_MT_modeller(Container=self.Container,
                                        Fundamentals=self.Fundamentals)
            Synt = Modeller.simulate(source_mechanism=c_coord)
            
            # get error
            DM = Data_Modelling(Container=self.Container)
            res = DM.get_pdf(self.Observed, Synt)
            
            # add to res list 
            netw_res = res['network']
            timestamp = float(UTCDateTime().timestamp)
            res_list.append([netw_res,list(c_coord),res,timestamp])
            
        result[0] = res_list
    
    def organizer(self,grid=None,save2json=False,path2json=''):
        '''
            grid: 
                zero = np.zeros(1)
                S1,D1,R1 = np.arange(0,365,45),np.arange(0,95,45),np.arange(-180,185,45)
                C1,I1,S2,D2,R2,C2,I2,M,T = zero,zero,zero,zero,zero,zero,zero,zero,zero
                grid = [S1,D1,R1,C1,I1,S2,D2,R2,C2,I2,M,T]
        '''
        
        if grid == None: # default grid
            zero = np.zeros(1)
            S1,D1,R1 = np.arange(0,370,10),np.arange(0,100,10),np.arange(-180,190,10)
            C1,I1,S2,D2,R2,C2,I2,M,T = zero,zero,zero,zero,zero,zero,zero,zero,zero
            grid = [S1,D1,R1,C1,I1,S2,D2,R2,C2,I2,M,T]
        
        tstart = UTCDateTime()
        manager = multiprocessing.Manager()
        result = manager.dict()
        pcore = []
                
        # create a new process for given event
        proc = Process(target=self.grid_search_multi_core,args=[grid,result])
        pcore.append(proc)
                
        # start
        Nrun = len(list(itertools.product(*grid)))
        print('Start carthesian grid search for '+str(Nrun)+' samples.')
        for proc in pcore:
            proc.start()
        for proc in pcore:
            proc.join()
        for pi, proc in enumerate(pcore):
            # this part is needed to delete finisched processes
            # if they are not deleted, the process is tried to run anew with the same name, this is not possible
            if proc is not proc.is_alive():
                del pcore[pi]
        run_time = UTCDateTime()-tstart
        hours = int(run_time/3600)
        minutes = int((run_time-hours*3600)/60.)
        seconds = int(run_time-hours*3600-minutes*60)
        print('Finishing run. Total runtime: '+str(hours)+'h '+str(minutes)+'min '+str(seconds)+'s')
        
        
        if save2json:
            with open(path2json+'Grid_Search_'+str(UTCDateTime())+'.json', 'w', encoding='utf-8') as f:
                json.dump(result[0], f, ensure_ascii=False, indent=4)
        
        return result
