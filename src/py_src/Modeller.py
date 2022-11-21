#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    Mike Lindner (mike.lindner@kit.edu), 2018
"""
# obspy functions
from obspy import read, Stream, UTCDateTime, Trace
from obspy.signal.trigger import plot_trigger, classic_sta_lta
from obspy.signal.filter import envelope
from obspy.clients.fdsn import Client
from obspy.imaging.beachball import beach
# standard libaries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal, sparse
import math
# extra libaries
import os
from os import listdir, path
from os.path import isfile, join
import json
import time
import inspect
import copy
# original and wrapper functions
from .util import Tape2M, M2Tape, compute_shift, update_Mw, Time_shift_xcorr_trunc_v2, strip_empties_from_dict
from .Utils.mopad_py3 import MomentTensor
from .Synthetics_Loader import Stream_taper


class Inversion_Preprocessor:

    def __init__(self,Container=None):
        # container
        self.Container = Container
        self.event_id = Container['source']['event_id']
        
        # station information
        self.STAT_dict = Container['network']['STAT_dict']
        self.obs_trace_set = Container['general']['obs_trace_set']
        self.Station_selection = Container['network']['trace_selection']
        
        # reported source time (Event File)
        self.src_time = Container['source']['F1_loc'][0]
        self.t_wind_synt = Container['network']['t_wind_synt']
        self.signal_window = Container['network']['signal_window']
        
        # auto trace selection parameter
        self.E_max = Container['preprocessing']['equalizer']['e_max']
        self.t_unc = Container['preprocessing']['equalizer']['t_uncertainty']
        self.min_coverage = Container['preprocessing']['equalizer']['min_coverage']
        self.S2N_min = Container['preprocessing']['s2n']['s2n_min']
        self.H2V_min = Container['preprocessing']['H2V']
        
        # auto trace selection criteria
        if self.obs_trace_set == 'auto':
            self.N_criteria = Container['preprocessing']['selection_criteria']['N_criteria'] 
        else:
            self.N_criteria = 0
        self.stat_min = Container['preprocessing']['selection_criteria']['stat_min']
        self.trace_min = Container['preprocessing']['selection_criteria']['trace_min']
        self.env_shift = Container['preprocessing']['selection_criteria']['perform_envelope_shift']
        
        # phase window
        self.perfrom_pw_cut = Container['preprocessing']['phase_window']['perform']
        self.pw_width = Container['source']['phase_onset']['width']
        self.pw_taper = Container['preprocessing']['phase_window']['Taper']
        
        # simulation setting
        self.sim_doublet = Container['source']['simulate_doublet']
        
        # bayesian
        self.solve_for_misrot = Container['source']['solve_for_misrot']
        self.solve_for_misloc = Container['source']['solve_for_misloc']
        self.dx_misloc = Container['source']['dx']
        self.dthea_misrot = Container['source']['daz']
        
        # output information
        self.path2fig = ''#Container['path']['figure']
        self.plot = Container['plotter']['equalizer']
        
        # get number of filter pertubations, is set to none only full band is used
        self.Filter = self.get_filter()
        if self.Filter['partition_type'] == None:
            self.FN = [0]
        else:
            self.FN = list(map(int,list(np.arange(0,len(self.Filter['fcut']),1))))
        
        
    def get_filter(self):
        if self.sim_doublet:
            FB1 = self.Container['preprocessing']['doublet_filter']['F1']
            FB2 = self.Container['preprocessing']['doublet_filter']['F2']
            Filter = {'ftype':'bandpass',
                      'freqmin':FB1['freqmin'],
                      'freqmax':FB1['freqmax'],
                      'corners':FB1['corners'],
                      'zerophase':FB1['zerophase'],
                      'partition_type':'lfix',
                      'fcut':len(FB2['fcut'])*[FB1['freqmax']]} 
            
        else:
            Filter = self.Container['preprocessing']['filter'] 
        return Filter
            

    def plot_selection_by_time_uncertainty(self,envelope_container,TShift_Selection):
        '''
        
        '''
        fig = plt.figure(figsize=(30, 10*len(self.FN)), facecolor='w', edgecolor='k')
        gs = gridspec.GridSpec(len(self.FN),3)
        for ci,comp in enumerate(envelope_container.keys()):
            for fii,fi in enumerate(envelope_container[comp]):
                sx = envelope_container[comp][fi][0]
                mb = envelope_container[comp][fi][1]
                lb = envelope_container[comp][fi][2]
                ub = envelope_container[comp][fi][3]
                data = envelope_container[comp][fi][4]
                Station = envelope_container[comp][fi][5]
                
                plt.subplot(gs[fii,ci])
                plt.plot(sx,mb,'k-')
                plt.plot(sx,lb,'k--')
                plt.plot(sx,ub,'k--')
                plt.text(sx[0],ub[-1],'FBNo: '+str(fi))
                
                ex_stat = []
                for si, stat in enumerate(Station):
                    for ii in range(len(TShift_Selection[stat])):
                        if TShift_Selection[stat][comp][fi]:
                            plt.plot(sx[si],data[si],'bo')
                        else:
                            plt.plot(sx[si],data[si],'ro')
                            if stat not in ex_stat:
                                ex_stat.append(stat)
                plt.text(sx[0],ub[-1]*0.75,'Excluded Stations: '+str(ex_stat))        
                plt.title(comp[0])
                plt.ylabel('Time shift in s')
                #plt.xticks(sx,Station)
        #filepath = self.path2fig+'/'+self.region+'/'+self.event_id+'/'+'/Env/'
        #plt.savefig(filepath+'Selection_by_time_uncertainty.png',bbox_inches='tight')
        plt.show()
    
    def display_envelope(self,envelope_container,Shift):
        for comp in ['Z','R','T']:
            fig = plt.figure(figsize=(10*len(self.FN), 5*len(envelope_container)), facecolor='w', edgecolor='k')
            gs = gridspec.GridSpec(len(envelope_container),len(self.FN))
            for oii,obs_id in enumerate(envelope_container.keys()):
                for fii,fi in enumerate(envelope_container[obs_id][comp].keys()):
                    plt.subplot(gs[oii,fii])
                    t_cut = Shift[obs_id][comp][fi][1]
                    t_shift = round(Shift[obs_id][comp][fi][0],2)
                    cover = round(Shift[obs_id][comp][fi][2],2)
                    
                    f=envelope_container[obs_id][comp][fi][0]
                    fe=envelope_container[obs_id][comp][fi][1]
                    ge=envelope_container[obs_id][comp][fi][2]
                    fefull=envelope_container[obs_id][comp][fi][3]
                    gefull=envelope_container[obs_id][comp][fi][4]
                    
                    plt.plot(np.arange(fe.stats.npts)*fe.stats.delta,fe.data,'k-')
                    plt.plot(t_shift+np.arange(ge.stats.npts)*ge.stats.delta,ge.data,'r-')
                    plt.plot(np.arange(fefull.stats.npts)*fefull.stats.delta,fefull.data,'k--')
                    plt.plot(t_shift+np.arange(gefull.stats.npts)*gefull.stats.delta,gefull.data,'r--')
                    plt.plot(np.arange(ge.stats.npts)*ge.stats.delta,ge.data,'r--',alpha=0.2)
                    plt.plot(np.arange(f.stats.npts)*f.stats.delta,f.data/np.max(np.abs(f.data)),'b-',alpha=0.3)
                    plt.plot([t_cut,t_cut],[-1,1],'g--')
                    
                    tx0 = t_shift+np.arange(gefull.stats.npts)*gefull.stats.delta
                    plt.text(tx0[-1]*0.5,0.75,'Masked = '+str(cover))
                    plt.text(tx0[-1]*0.5,-0.75,'tshift = '+str(t_shift)+' s')
                    
                    if cover < self.min_coverage:
                        plt.axvspan(0,tx0[-1], facecolor='r', alpha=0.35)
                    
                    if oii == 0:
                        plt.title('FBNo: '+str(fi))
                    if fii == 0:
                        plt.ylabel(fe.stats.station)
                    if oii == len(envelope_container)-1:
                        plt.xlabel('Time in s')


            #ilepath = self.path2fig+'/'+self.region+'/'+self.event_id+'/'+'/Env/'
            #fig.savefig(filepath+comp+'.png',bbox_inches = 'tight')    
            plt.show()
    
    def selection_by_time_uncertainty(self,Shift):
        '''
            data fi of each comp : stat1 stat2 stat3
        '''
        
        tvar = self.t_unc*(1/self.Filter['freqmin'])
        TShift_Selection = {}
        shift_selection_container = {}
        Station = Shift.keys()
        data,dist={},{}
        
        for oii, obs_id in enumerate(Shift.keys()):
            TShift_Selection[obs_id] = {}
            for cii, comp in enumerate(Shift[obs_id]):
                if comp not in data:
                    data[comp] = {}
                    dist[comp] = {}
                TShift_Selection[obs_id][comp] = {}
                for fii, fi in enumerate(Shift[obs_id][comp]):
                    if fi not in data[comp]:
                        data[comp][fi] = []
                        dist[comp][fi] = []
                    data[comp][fi].append(Shift[obs_id][comp][fi][0])
                    dist[comp][fi].append(self.STAT_dict['F1_X0'][obs_id][3])
                    
        for comp in data:
            shift_selection_container[comp] = {}
            for fi in data[comp]:
                dist_sort = sorted(dist[comp][fi])
                data_sort = [x for _,x in sorted(zip(dist[comp][fi],data[comp][fi]))]
                Station_sort = [x for _,x in sorted(zip(dist[comp][fi],Station))]
                # calculate 1D regression
                sx = np.asarray(dist_sort)
                sb = np.polyfit(sx, np.asarray(data_sort), 1)
                mb = sb[0]*sx+sb[1]
                lb = sb[0]*sx+sb[1]-tvar
                ub = sb[0]*sx+sb[1]+tvar
                # save in dictionary
                shift_selection_container[comp][fi] = [sx,mb,lb,ub,data_sort,Station_sort]
                # select station within time uncertainty range
                for si,stat in enumerate(Station_sort):
                    if lb[si] <= data_sort[si] <= ub[si]:
                        TShift_Selection[stat][comp][fi] = True
                    else:
                        TShift_Selection[stat][comp][fi] = False
        if self.plot:
            self.plot_selection_by_time_uncertainty(shift_selection_container,TShift_Selection)   
        return TShift_Selection
    
    def get_env_coverage(self,fe_in,ge_in,tc):
        '''
            remove data based on time shift and subtract synthetic from observed envelope
            the normalized sum of the difference (whearas negative numbers are set to zero)
            is an indicator fot the synthetic coverage, hence the partion the synthetics are able to describe the observed
        '''
        # copy traces 
        ge = ge_in.copy()
        fe = fe_in.copy()
        
        # get t_cut
        tcint = int(tc*fe.stats.sampling_rate)
        ts = np.abs(tcint)
        
        # cases
        if tcint < 0:
            diff = ge.data[:-ts] - fe.data[ts:]
            diff[diff < 0] = 0
            coverage = np.sum(diff**2)/np.sum(ge.data[:-ts]**2)
        elif tcint > 0:
            diff = ge.data[ts:] - fe.data[:-ts]
            diff[diff < 0] = 0
            coverage = np.sum(diff**2)/np.sum(ge.data[ts:]**2)
        elif tcint == 0 :
            diff = ge.data - fe.data
            diff[diff < 0] = 0
            coverage = np.sum(diff**2)/np.sum(ge.data**2)
            

        return coverage
 
    def get_envelope_information(self,obs,synt):
        '''
            get cut of time from synthetic envelope stack
            --> calculate envelope for all synthetics traces and sum them up
            --> calculate time at which ~99% of the total energy is released
            --> cut-off-time is set at this point
        '''
        # synthetic trace loop
        envelope_container = {}
        Shift = {}
        synt_cutoff = {}
        for oii, obs_id in enumerate(obs.keys()):# self.loc_circ_list.keys()):
            Shift[obs_id] = {}
            envelope_container[obs_id] = {}
            for cii, comp in enumerate(obs[obs_id]):
                Shift[obs_id][comp] = {}
                envelope_container[obs_id][comp] = {}
                for fii, fi in enumerate(obs[obs_id][comp]):
                    if fi not in synt_cutoff:
                        synt_cutoff[fi] = []
                    f = obs[obs_id][comp][fi].copy()
                    fe = obs[obs_id][comp][fi].copy()
                    fefull = obs[obs_id][comp][fi].copy()
                    f_envelope = envelope(f.data)

                    for ii in range(6):
                        fund = 'F'+str(ii+1)
                        g = synt[fund][obs_id][comp]['Source'][fi].copy()
                        ge = synt[fund][obs_id][comp]['Source'][fi].copy()
                        gefull = synt[fund][obs_id][comp]['Source'][fi].copy() 
                        if ii == 0:
                            g_envelope = np.zeros(len(g.data))
                            dt = g.stats.delta
                            t_start = g.stats.starttime
                        g_envelope += envelope(g.data)


                    # calculate cut_off_time based on synthetic envelope energy partion
                    E_total = np.sum(g_envelope)
                    for ti in range(len(g_envelope)-1):
                        if np.sum(g_envelope[:ti+1]) >= E_total*self.E_max:
                            t_cut = ti*dt
                            break
                    # save time length of fundamental synsthetics that contain self.E_max enegry
                    # min/max/mean value will later be used to cut all traces the same langth
                    synt_cutoff[fi].append(t_cut)
                    
                    # get time shift from envelope within relevant time window
                    # note: synthetics and obs onset time have to fit within the 
                    # windowlength of the cut synthetic trace (finding the right peak)
                    ti = int(t_cut/dt)
                    fe.data = f_envelope[:ti+1]/np.max(f_envelope[:ti+1])
                    ge.data = g_envelope[:ti+1]/np.max(g_envelope[:ti+1])
                    shift = compute_shift(fe, ge, 'p2p')
                    
                    # get l2 of cut envelope
                    # coverage = 1 -> obs and synt envelope are identical 
                    fg_diff = ge.stats.starttime - fe.stats.starttime # time difference starttime
                    tsig = ge.stats.starttime+shift-fg_diff # shifting obs and synt based on peak
                    t0 = UTCDateTime(self.src_time) # source time
                    coverage = self.get_env_coverage(fe,ge,tsig-t0)
                    try:
                        coverage = self.get_env_coverage(fe,ge,tsig-t0)
                    except:
                        coverage = -1000.0
                        print('Encountered issue for envelope covarage computation for '+obs_id+'.'+comp)
                        print('Setting coverage to -1000.0')
                    # save full envelope for plot
                    fefull.data = f_envelope/np.max(f_envelope)
                    gefull.data = g_envelope/np.max(g_envelope)
                    
                    # fill output dictionaries
                    #Shift[obs_id][comp][fi] = [tsig-t0,t_end,coverage,t_cut] 
                    Shift[obs_id][comp][fi] = [shift,t_cut,coverage,tsig-t0] 
                    envelope_container[obs_id][comp][fi] = [f,fe,ge,fefull,gefull]
        
        # combine derived information in Shift_Container
        Shift_Container = {}
        Shift_Container['trace_info'] = Shift
        Shift_Container['synt_cutoff'] = synt_cutoff
        
        if self.plot:
            self.display_envelope(envelope_container,Shift)
    
        
        return Shift_Container

    
    def decide(self,H2V,S2N,tvar_test,ecov,obs_id,comp):
        ii = 0
        if S2N >= self.S2N_min:
            ii+=1
        if comp == 'Z':
            if H2V <= self.H2V_min[0]:
                ii+=1
        else:
            if H2V <= self.H2V_min[1]:
                ii+=1
        if ecov >= self.min_coverage:
            ii+=1
        if tvar_test:
            ii+=1
        
        if ii >= self.N_criteria:
            des = True
        else:
            des = False
        # if trace selection is not auto, all traces are choosen
        if self.obs_trace_set != 'auto':
            if comp in self.Station_selection[obs_id]:
                des = True
            else:
                des = False
        
        #print(obs_id,comp,H2V,self.H2V_min,S2N,self.S2N_min,tvar_test,ecov,self.min_coverage,des)
        
        return des

    
    def get_best_selection(self,env_selection,t_selection,winfo):
        '''
            output = {  
                        0:['Stat1-ZRT','Stat2-ZT',...],
                        1:[...]....
                        }
        '''
        Selection_Dict = {}
        FB_counter = {}
        Shift_dict = {}
        for obs_id in env_selection['trace_info']:
            Selection_Dict[obs_id] = {}
            for comp in ['Z','R','T']:
                Selection_Dict[obs_id][comp] = {}
                for fi in t_selection[obs_id][comp]:
                    Selection_Dict[obs_id][comp][fi] = {}
                    if fi not in FB_counter:
                        FB_counter[fi] = 0
                        Shift_dict[fi] = []
                    # get selection parameters
                    H2V = winfo[obs_id][fi]['H2V']
                    S2N = winfo[obs_id][fi][comp]
                    tvar_test = t_selection[obs_id][comp][fi]
                    tshift = env_selection['trace_info'][obs_id][comp][fi][0]
                    ecov = env_selection['trace_info'][obs_id][comp][fi][2]
                    # decide on trace
                    des = self.decide(H2V,S2N,tvar_test,ecov,obs_id,comp)
                    Selection_Dict[obs_id][comp][fi] = [des,tshift]
                    # add fb selection
                    if des:
                        FB_counter[fi] += 1
                        Shift_dict[fi].append(tshift)
            
        return Selection_Dict, FB_counter, Shift_dict
    
    def get_time_window(self,env_selection,Shift_dict,FB_counter):
        '''
        
        '''
        #plt.subplot(1,2,1)
        #_ = plt.hist(env_selection['synt_cutoff'][0], bins='auto')
        #plt.subplot(1,2,2)
        #_ = plt.hist(Shift_dict[0], bins='auto')
        #plt.show()
        
        t_len, t_shift = [],[]
        for fi in Shift_dict:
            if FB_counter[fi] >= self.trace_min:
                if Shift_dict[fi] == Shift_dict[fi]: # check if nan
                    t_len += env_selection['synt_cutoff'][fi]
                    t_shift += Shift_dict[fi]
        
        Tlen = [np.mean(np.asarray(t_len)),np.std(np.asarray(t_len))]
        Tshift = [np.mean(np.asarray(t_shift)),np.std(np.asarray(t_shift))]
        
        return [Tshift,Tlen]
    
    def create_cut_dictionaries(self,synt,Selection_Dict):
        '''
        
        '''
        # create dictionaries
        synt_cut = {}
        # define empty synt_cut dictionary (event_id and fund_id does not exist for obs)
        for event_id in synt:
            synt_cut[event_id] = {}
            for fund_id in synt[event_id]:
                synt_cut[event_id][fund_id] = {}
                for obs_id in Selection_Dict:
                    synt_cut[event_id][fund_id][obs_id] = {}
                    for comp in Selection_Dict[obs_id]:
                        synt_cut[event_id][fund_id][obs_id][comp] = {} 
        return synt_cut
    
    def t_delay_weight(self,obs_id,phase_id,dt,pick_dict,npts):
        '''
            slope = (t_travel_obs - t_travel_synt) / t_travel_obs
        '''
        tt_synt = pick_dict['F1_X0']['t_travel_synt'][obs_id][phase_id]
        tt_obs = pick_dict['F1_X0']['t_travel'][obs_id][phase_id]
        slope = (tt_obs-tt_synt) / tt_obs
        x0 = -self.pw_width[phase_id][0]
        x1 = self.pw_width[phase_id][1]# + dt
        # multiply abs slope with sign of slope to handle data before onset like after
        weight = np.sign(slope)*np.abs(np.linspace(x0,x1,num=npts)*slope)
        # apply taper
        stw = Stream(traces=[Trace(data=weight)])
        stw = Stream_taper(stw,self.pw_taper)
        return stw[0].data
    
    def phase_cutter(self,obs,synt,Selection_Dict,pick_dict):
        '''
        
        '''
        # creat empty synt cut dict
        synt_cut = self.create_cut_dictionaries(synt,Selection_Dict)
        obs_cut = {}
        
        # assignment loop
        for obs_id in self.Station_selection:
            obs_cut[obs_id] = {}
            for comp in Selection_Dict[obs_id]:
                obs_cut[obs_id][comp] = {}
                for fi in Selection_Dict[obs_id][comp]: 
                    obs_cut[obs_id][comp][fi] = {}
                    # check selection
                    if Selection_Dict[obs_id][comp][fi][0]:
                        # phase dictionary
                        ph_dict = pick_dict['F1_X0']['t_onset'][obs_id]
                        #print(ph_dict.keys())
                        st_temp = {} # temporary stream list
                        derv_weight = {} # weight for delay derivative
                        # phase loop
                        for phase_id in list(pick_dict['F1_X0']['t_travel'][obs_id]):#.remove('Source'):
                            # get phase onset time (UTC)
                            t_phase = UTCDateTime(ph_dict[phase_id][0])
                            # create stream object
                            st_temp[phase_id] = Stream(traces=obs[obs_id][comp][fi].copy())
                            dt = obs[obs_id][comp][fi].stats.delta
                            # run synthetics event_id and fundamental loop
                            event_id_list = []
                            traces = [] 
                            for event_id in synt:
                                for fund_id in synt[event_id]:
                                    gc = synt[event_id][fund_id][obs_id][comp][phase_id][fi].copy()
                                    traces.append(gc)
                                    event_id_list.append(event_id+'-'+fund_id)
                            # combine traces to stream
                            for tr in traces:
                                st_temp[phase_id] += Stream(traces=tr)
                            # cut all traces simultaneously to keep constant lenght
                            st_temp[phase_id].trim(t_phase-self.pw_width[phase_id][0],t_phase+self.pw_width[phase_id][1], pad=True)
                            # apply hann taper to smooth cut edges
                            st_temp[phase_id] = Stream_taper(st_temp[phase_id],self.pw_taper)        
                            # weight for delay derivative
                            if event_id == 'F1_X5': # if synt deriv
                                npts = st_temp[phase_id][0].stats.npts
                                derv_weight[phase_id] = self.t_delay_weight(obs_id,phase_id,dt,pick_dict,npts)
                            else:
                                derv_weight[phase_id] = np.ones(st_temp[phase_id][0].stats.npts)
                            
                            
                        # append phase windows to single data array
                        data = st_temp[phase_id][0].data
                        data_temp = {0:np.array([])}
                        weight_temp = {'P':np.array([]),'S':np.array([])}
                        for pi, phase_id in enumerate(self.Station_selection[obs_id][comp]): 
                            weight_temp[phase_id] = np.append(weight_temp[phase_id],derv_weight[phase_id])
                            if pi == 0: t_new = st_temp[phase_id][0].stats.starttime
                            for tri, tr_temp in enumerate(st_temp[phase_id]):
                                if tri not in data_temp: data_temp[tri] = np.array([])
                                data_temp[tri] = np.append(data_temp[tri],tr_temp.data)
 
                        # reassign to dictionaries
                        tr_o = obs[obs_id][comp][fi].copy()
                        tr_o.data = data_temp[0]
                        
                        tr_o.stats.starttime = t_new
                        obs_cut[obs_id][comp][fi] = tr_o
                        
                        for f_key_i, f_key in enumerate(event_id_list):
                            [event_id,fund_id]  = f_key.split('-')
                            tr_s = obs[obs_id][comp][fi].copy()
                            tr_s.data = data_temp[f_key_i+1]
                            tr_s.stats.starttime = t_new
                            if event_id == 'F1_X5':
                                #t = np.arange(tr_s.stats.npts)*dt
                                #plt.plot(t,tr_s.data,'k-')
                                #plt.plot(t,weight_temp[phase_id],'r-')
                                #plt.show()
                                tr_s.data *= weight_temp[phase_id]
                            synt_cut[event_id][fund_id][obs_id][comp][fi] = tr_s

        return obs_cut, synt_cut
    
    def trace_cutter(self,obs,synt,Shift,Selection_Dict):
        '''
        
        '''
        # get time window information
        if self.env_shift:
            t_shift = Shift[0][0]
        else:
            t_shift = 0
        if self.obs_trace_set == 'auto':
            t_len = Shift[1][0]
        else:
            t_len = np.sum(np.asarray(self.t_wind_synt)) 
        
        # creat empty synt cut dict
        synt_cut = self.create_cut_dictionaries(synt,Selection_Dict)
        obs_cut = {}
        
        # assignment loop
        for obs_id in Selection_Dict:
            obs_cut[obs_id] = {}
            for comp in Selection_Dict[obs_id]:
                obs_cut[obs_id][comp] = {}
                for fi in Selection_Dict[obs_id][comp]: 
                    obs_cut[obs_id][comp][fi] = {}
                    # check selection
                    if Selection_Dict[obs_id][comp][fi][0]:
                        # define time window
                        dt = obs[obs_id][comp][fi].stats.delta
                        dtc = abs(int(round(t_shift/dt,0)))
                        # create trace object
                        traces = []
                        st = Stream(traces=obs[obs_id][comp][fi])
                        # run synthetics event_id and fundamental loop
                        event_id_list = []
                        for event_id in synt:
                            for fund_id in synt[event_id]:
                                gc = synt[event_id][fund_id][obs_id][comp]['Source'][fi].copy()
                                # add or subtract data from synthetics to shift rel. 
                                # to observation; Note: no time update performed as
                                # dtc comes from the source time update, synthetics are computed rel.
                                # to the initial errerous one
                                if t_shift > 0: # add zeros
                                    tadd = np.zeros(np.abs(dtc))#np.random.normal(gc.data[0], np.abs(np.mean(gc.data)*0.01), dtc)
                                    gc.data = np.append(tadd,gc.data)
                                    #gc.stats.starttime -= dtc*dt
                                elif t_shift < 0: # remove data
                                    gc.data = gc.data[dtc:]
                                    #gc.stats.starttime += dtc*dt
                                elif t_shift == 0:
                                    gc.data = gc.data
                                else:
                                    print('What happend?')
                                traces.append(gc)
                                event_id_list.append(event_id+'-'+fund_id)
                        # combine traces to stream
                        for tr in traces:
                            st += Stream(traces=tr)
                        if self.signal_window is None:
                            # cut all traces simultaneously to keep constant lenght
                            t0 = np.max([st[0].stats.starttime,st[1].stats.starttime])
                            st.trim(t0,t0+t_len, pad=True)
                        else:
                            [t0,t1] = self.signal_window[self.event_id][obs_id]
                            #print(t0,t1)
                            st.trim(UTCDateTime(t0),UTCDateTime(t1), pad=True)
                        # multiply data
                        #for tr in st:
                        #    tr.data *= 10**6
                        # apply hann taper to smooth cut edges
                        #st.taper(type='hann',max_percentage=0.01)
                        # reassign to dictionaries
                        obs_cut[obs_id][comp][fi] = st[0]
                        
                        for f_key_i, f_key in enumerate(event_id_list):
                            [event_id,fund_id]  = f_key.split('-')
                            synt_cut[event_id][fund_id][obs_id][comp][fi] = st[f_key_i+1]                                

        return obs_cut, synt_cut
    
    def create_Trace_selection_file_format(self,Selection_Dict):
        '''
        
        '''
        # compile temporary dictionary
        temp = {}
        for obs_id in Selection_Dict:
            for comp in Selection_Dict[obs_id]:
                for fi in Selection_Dict[obs_id][comp]:
                    # get selection
                    if Selection_Dict[obs_id][comp][fi][0]:
                        # create dictionary
                        if fi not in temp.keys():
                            temp[fi] = {}
                        if obs_id not in temp[fi].keys():
                            temp[fi][obs_id] = []
                        temp[fi][obs_id].append(comp)
                        
        # create Trace selection file format per fi
        Trace_selection_List = {}
        for fi in temp.keys():
            Trace_selection_List[fi] = []
            for obs_id in temp[fi].keys():
                comp_list = ''.join(map(str, temp[fi][obs_id]))
            Trace_selection_List[fi].append(obs_id+'-'+comp_list)
        
        return Trace_selection_List
                    
    
    def simulate_Selection_dict(self):
        '''
        {'XX_DP03': ['Z', 'R', 'T'], 
         'XX_DP04': ['Z', 'R', 'T'], 
         'XX_DP16': ['Z', 'R', 'T']}
        '''
        Selection_Dict_custom = {}
        for obs_id in self.Station_selection.keys():
            Selection_Dict_custom[obs_id] = {}
            for comp in self.Station_selection[obs_id]:
                Selection_Dict_custom[obs_id][comp] = {}
                for fi in self.FN:
                    Selection_Dict_custom[obs_id][comp][fi] = [True]
                
        return Selection_Dict_custom
        
    
    def organizer(self,obs,synt,winfo,pick_dict):
        '''
        
        '''
        
        # case: trace selection auto or custom
        if self.obs_trace_set == 'auto':
            # perform envelope selection and get initial source time offset
            env_selection = self.get_envelope_information(obs,synt['F1_X0']) 
            # rate traces based on peak-time-distance diagram
            t_selection = self.selection_by_time_uncertainty(env_selection['trace_info'])
            # get selection dict
            Selection_Dict, FB_counter, Shift_dict = self.get_best_selection(env_selection,t_selection,winfo)
            # get constant time window for all traces
            twind = self.get_time_window(env_selection,Shift_dict,FB_counter)
            # get auto selection in Trace_selection_file format
            auto_trace_list = self.create_Trace_selection_file_format(Selection_Dict)
        else:
            # overwrite Selection_Dict for selection in Trace_selection_file
            auto_trace_list = None
            twind = None
            env_selection = None
            t_selection = None
            FB_counter = None
            Selection_Dict = self.simulate_Selection_dict()
        
        # cut and select observed and fundamental-synthetic seismogramms  
        if self.perfrom_pw_cut:
            obs_cut, synt_cut = self.phase_cutter(obs,synt,Selection_Dict,pick_dict)
        else:
            obs_cut, synt_cut = self.trace_cutter(obs,synt,twind,Selection_Dict)
        
        # clear distionaries from empty entries
        OBS = strip_empties_from_dict(obs_cut)
        SYNT = strip_empties_from_dict(synt_cut)
        
        # create logger dictionary
        inv_pre = {
            'envelope_selection':env_selection,
            'shift_selection':t_selection,
            'selection_dict':Selection_Dict,
            'FB_counter':FB_counter,
            'auto_trace_list':auto_trace_list,
            'twind':twind,
            'traveltime_correction':pick_dict
        }
       
        return OBS, SYNT, inv_pre




##################################################################################################
##################################################################################################
### Full_MT_modeller
##################################################################################################
##################################################################################################

class Full_MT_modeller:
    
    def __init__(self,Container=None,Fundamentals=None):
        self.M0 = 1.0
        self.Container = Container
        self.simulate_doublet = Container['source']['simulate_doublet']
        self.solve_for_misloc = Container['source']['solve_for_misloc']
        self.solve_for_misrot = Container['source']['solve_for_misrot']
        self.dthea_misrot = Container['source']['daz']
        self.mf_id = Container['source']['misloc_fault_id']
        self.Fundamentals = Fundamentals
        
        self.Stat_selection = Fundamentals['F1_X0']['F1'].keys()
        
            
    def Time_Logger(self,func_name,file_name,run_time):
        debug_message = 'Function %s in file %s with runtime=%s' % (func_name,file_name.split('/')[-1],str(run_time))
        TLOG.debug(debug_message)
    
    def SixElementModeller_doublet_misloc(self,a,tshift):    
        '''
        
        '''        
        u_X0 = self.SixElementModeller_doublet(a,tshift)
        
        u_Xi = {}
        # pertubation loop
        for pert in ['1','2','3']:
            fault_id = self.mf_id+'_X'+pert
            MF = self.Fundamentals[fault_id]
            fault_id_key =  'cF_X'+pert
            u_Xi[fault_id_key] = {}
            for stat in self.Stat_selection:
                u_Xi[fault_id_key][stat] = {}
                for comp in MF['F1'][stat].keys():
                    u_Xi[fault_id_key][stat][comp] = {}
                    for fi in MF['F1'][stat][comp].keys():
                        u_Xi[fault_id_key][stat][comp][fi] = MF['F1'][stat][comp][fi].copy()
                        u_Xi[fault_id_key][stat][comp][fi].data = np.zeros(u_Xi[fault_id_key][stat][comp][fi].stats.npts)
                        # get tshift in samples
                        dt = self.Fundamentals['F1_X0']['F1'][stat][comp][fi].stats.delta
                        toff = int(tshift/dt) 
                        for jj in range(len(a[self.mf_id])):
                            Fault1 = MF['F'+str(jj+1)][stat][comp][fi].copy()
                            ui = a[self.mf_id][jj]*Fault1.data
                            # check if solveing_for_misloc fault is F1 or F2
                            # shifting direction between F1 and F2 is reversed
                            if self.mf_id == 'F2':
                                tfac = -1
                            else:
                                tfac = 1
                            # check toff cases
                            if tfac*toff > 0:
                                u_Xi[fault_id_key][stat][comp][fi].data += ui  
                            elif tfac*toff < 0:
                                tadd = np.zeros(np.abs(toff))#np.random.normal(ui[-np.abs(toff)], ui[-np.abs(toff)]*0.05, np.abs(toff))
                                u_Xi[fault_id_key][stat][comp][fi].data += np.append(tadd,ui[:-np.abs(toff)])
                            elif tfac*toff == 0:
                                u_Xi[fault_id_key][stat][comp][fi].data += ui
                            else:
                                print('What happend? Is toff Nan or imaginary?')
                                
        # add dictionaries together
        #print(u_X0.keys(),u_Xi.keys())
        u = dict(u_X0, **u_Xi)
                
        return u
    

    
    def SixElementModeller_doublet(self,a,tshift):    
        '''
        
        '''        
        u = {}           
        MF1 = self.Fundamentals['F1_X0']
        MF2 = self.Fundamentals['F2_X0']
        fault_id_key =  'cF_X0'
        u[fault_id_key] = {}
        for stat in self.Stat_selection:
            u[fault_id_key][stat] = {}
            for comp in MF1['F1'][stat].keys():
                u[fault_id_key][stat][comp] = {}
                for fi in MF1['F1'][stat][comp].keys():
                    #if comp == 'Z':
                    #    print(stat,MF1['F1'][stat][comp][fi].stats.npts,MF2['F1'][stat][comp][fi].stats.npts)
                    #'''
                    u[fault_id_key][stat][comp][fi] = MF1['F1'][stat][comp][fi].copy()
                    u[fault_id_key][stat][comp][fi].data = np.zeros(MF1['F1'][stat][comp][fi].stats.npts)
                    # get tshift in samples
                    dt = self.Fundamentals['F1_X0']['F1'][stat][comp][fi].stats.delta
                    toff = int(tshift/dt) 
                    for jj in range(len(a['F1'])): 
                        Fault1 = MF1['F'+str(jj+1)][stat][comp][fi].copy()
                        Fault2 = MF2['F'+str(jj+1)][stat][comp][fi].copy()
                        # get FB information
                        if jj == 0:
                            FB_F1 = Fault1.stats.FBand
                            FB_F2 = Fault2.stats.FBand
                            FB_comb = 'F1: '+FB_F1+' | F2: '+FB_F2
                        u1 = a['F1'][jj]*Fault1.data
                        u2 = a['F2'][jj]*Fault2.data
                        #print(len(Fault1.data),len(Fault2.data),a,toff)
                        #fig = plt.figure(figsize=(20, 5), facecolor='w', edgecolor='k')
                        #plt.subplot(1,2,1)
                        #plt.plot(u1,'k-')
                        #plt.plot(u2,'b--')
                        # check toff cases
                        if toff > 0:
                            tadd = np.zeros(np.abs(toff))
                            u[fault_id_key][stat][comp][fi].data += u1+np.append(tadd,u2[:-np.abs(toff)]) 
                            #plt.subplot(1,2,2)
                            #plt.plot(u[fault_id_key][stat][comp][fi].data,'g-')
                            #plt.plot(u1,'k-')
                            #plt.plot(np.append(tadd,u2[:-np.abs(toff)]),'b-')
                            #plt.plot(u2,'r-')
                        elif toff < 0:
                            tadd = np.zeros(np.abs(toff))
                            u[fault_id_key][stat][comp][fi].data += np.append(tadd,u1[:-np.abs(toff)])+u2
                        elif toff == 0:
                            u[fault_id_key][stat][comp][fi].data += u1+u2
                        else:
                            print('What happend? Is toff Nan or imaginary?')
                        #plt.subplot(1,2,1)
                        #plt.plot(u[fault_id_key][stat][comp][fi].data,'r--')
                        #plt.show()
                        #print(asdf)
                    # update FB information
                    u[fault_id_key][stat][comp][fi].stats.FBand = FB_comb
                #'''
        
        return u
    
    def SixElementModeller_station_delay(self,a):    
        '''
        
        '''
        
        u = {}
        
    
    def SixElementModeller(self,a):    
        '''
        
        '''
        
        u = {}
        for fault_id in self.Fundamentals.keys():
            MF = self.Fundamentals[fault_id]
            fault_id_key =  'cF_X'+fault_id[-1]
            u[fault_id_key] = {}
            for stat in self.Stat_selection:
                u[fault_id_key][stat] = {}
                for comp in MF['F1'][stat].keys():
                    u[fault_id_key][stat][comp] = {}
                    for fi in MF['F1'][stat][comp].keys():
                        u[fault_id_key][stat][comp][fi] = MF['F1'][stat][comp][fi].copy()
                        N = u[fault_id_key][stat][comp][fi].stats.npts
                        mu = np.mean(np.abs(u[fault_id_key][stat][comp][fi].data))
                        u[fault_id_key][stat][comp][fi].data *= 0 #np.random.normal(0.0, mu*0.05, N)
                        for jj in range(len(a)):
                            Fault1 = MF['F'+str(jj+1)][stat][comp][fi].copy()
                            u[fault_id_key][stat][comp][fi].data += a[jj]*Fault1.data
        
        return u

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
            
    def write_waveforms(self,data,path2save):
        for st in data:
            for comp in data[st]:
                for fi in data[st][comp]:
                    tr = data[st][comp][fi]
                    tr.write(path2save+tr.id+'.'+str(fi),format='MSEED')
        
    def simulate(self,source_mechanism=None,path2save=None):
        strike = source_mechanism[0]
        dip = source_mechanism[1]
        rake = source_mechanism[2]
        clvd = source_mechanism[3]
        iso = source_mechanism[4]
        
        # simulate single fault or doublet
        if self.simulate_doublet:
            M_fac_F1 = 1.-source_mechanism[10]
            M_fac_F2 = source_mechanism[10]
            tshift = source_mechanism[11]
            M1 = Tape2M(strike,dip,rake,clvd,iso,M_fac_F1)
            M2 = Tape2M(source_mechanism[5],
                       source_mechanism[6],
                       source_mechanism[7],
                       source_mechanism[8],
                       source_mechanism[9],
                       M_fac_F2)
            a1 = self.get_weights(M1)
            a2 = self.get_weights(M2)
            a = {'F1':a1,'F2':a2}            
            if self.solve_for_misloc:
                waveform = self.SixElementModeller_doublet_misloc(a,tshift)
            else:
                waveform = self.SixElementModeller_doublet(a,tshift)
                
        else:
            M = Tape2M(strike,dip,rake,clvd,iso,self.M0)
            a = self.get_weights(M)
            waveform = self.SixElementModeller(a)  
            #if self.solve_for_misrot:
            #    waveform['cF_X4'],_,_ = rand_rot(waveform['cF_X4'],sigma=self.dthea_misrot,dist='constant')

        if path2save is not None:
            print('Save Files at: ', path2save)
            self.write_waveforms(waveform,path2save)   
        
        return waveform


##################################################################################################
##################################################################################################
### Test_Mechanism
##################################################################################################
##################################################################################################


class Test_Mechanism:
    
    def __init__(self,Container=None,Observed=None,Fundamentals=None):
        '''
        
        '''
        
        self.Container = Container        
        self.Observed = Observed
        self.Fundamentals = Fundamentals
        
        self.sim_doublet = Container['source']['simulate_doublet']
        self.Mw_reference = Container['source']['ref_mag']
        self.magnitude_update = Container['inversion']['magnitude_update']
        self.station_Tshift = Container['inversion']['time_shift']['station_Tshift']
        self.perform_Tshift = Container['inversion']['time_shift']['perform']
        # get number of filter pertubations, is set to none only full band is used
        self.Filter = self.get_filter()
        if self.Filter['partition_type'] == None:
            self.FN = [0]
        else:
            self.FN = list(map(int,list(np.arange(0,len(self.Filter['fcut']),1))))
        self.error = None
    
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
    
    def add_STF(self,):
        '''
        
        '''
        
    
    def add_subplot_axes(self,ax,rect,axisbg='w'):
        fig = plt.gcf()
        box = ax.get_position()
        width = box.width
        height = box.height
        inax_position  = ax.transAxes.transform(rect[0:2])
        transFigure = fig.transFigure.inverted()
        infig_position = transFigure.transform(inax_position)    
        x = infig_position[0]
        y = infig_position[1]
        width *= rect[2]
        height *= rect[3]  # <= Typo was here
        subax = fig.add_axes([x,y,width,height])#,axisbg=axisbg)
        x_labelsize = subax.get_xticklabels()[0].get_size()
        y_labelsize = subax.get_yticklabels()[0].get_size()
        x_labelsize *= rect[2]**0.5
        y_labelsize *= rect[3]**0.5
        subax.xaxis.set_tick_params(labelsize=x_labelsize)
        subax.yaxis.set_tick_params(labelsize=y_labelsize)
        return subax
    
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
            print('Magnitude update short cut is not set!')
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
        
        # magnitude
        Magnitude = self.calc_Magnitude(obs,synt_raw)
    
        # time_shift  
        if self.perform_Tshift:
            synt,Tshift = self.time_shift(obs,synt_raw)
        else:
            synt = synt_raw
            Tshift = None

        return obs, synt, Tshift, Magnitude
               
    
    def simulate_synthetics(self,split_doublet):
        '''
        
        '''
        mech = self.mechanism
        # convert mech to c_coord based on different input cases
        if isinstance(mech, list):
            if len(mech) == 3: # case single fault dc
                c_coord = np.array([mech[0],mech[1],mech[2],0,0,0,0,0,0,0,0.0,0.0])
            elif len(mech) == 4: # case single fault dev
                c_coord = np.array([mech[0],mech[1],mech[2],mech[3],0,0,0,0,0,0,0.0,0.0]) 
            elif len(mech) == 5: # case single fault full 
                c_coord = np.array([mech[0],mech[1],mech[2],mech[3],mech[4],0,0,0,0,0,0.0,0.0])    
            elif len(mech) == 12: # case doublet source
                c_coord = np.asarray(mech)
            else:
                print('Wrong length of mechanism list')
                raise SystemExit
        else:
            c_coord = mech
        
        # simulate synthetics
        Modeller = Full_MT_modeller(Container=self.Container,Fundamentals=self.Fundamentals)
        synt_raw = Modeller.simulate(source_mechanism=c_coord)
            
        # preprocessing (shift, taper)
        self.Obs, self.Synt, self.Tshift, self.Magnitude = self.data_preprocess(self.Observed,synt_raw)
        
        # case: split doublet (simulate both sub faults)
        if split_doublet:     
            
            # turn time shift off (else: issue with timing of split waveforms 
            # --> tshift based on single fault)
            tshift_init = self.perform_Tshift
            
            # simulate F1
            self.perform_Tshift = False
            c_coord_F1 = c_coord
            c_coord_F1[10] = 0.0 # change M fac to 0.0 to simulate 100% F1
            Modeller_F1 = Full_MT_modeller(Container=self.Container,Fundamentals=self.Fundamentals)
            synt_raw_F1 = Modeller_F1.simulate(source_mechanism=c_coord_F1)
            self.Obs_F1, self.Synt_F1, _, self.Magnitude_F1 = self.data_preprocess(self.Observed,synt_raw_F1)
            
            # set initial tshift setting
            self.perform_Tshift = tshift_init
            
            # get shifted F1 traces
            _, self.Synt_F1_ts, self.Tshift_F1, self.Magnitude_F1_ts = self.data_preprocess(self.Observed,synt_raw_F1)
            
            # simulate F2
            self.perform_Tshift = False
            c_coord_F2 = c_coord # change M fac to 1.0 to simulate 100% F2
            c_coord_F2[10] = 1.0 
            Modeller_F2 = Full_MT_modeller(Container=self.Container,Fundamentals=self.Fundamentals)
            synt_raw_F2 = Modeller_F2.simulate(source_mechanism=c_coord_F2)
            self.Obs_F2, self.Synt_F2, _, self.Magnitude_F2 = self.data_preprocess(self.Observed,synt_raw_F2)
            
            # set initial tshift setting
            self.perform_Tshift = tshift_init
            
            # get shifted F2 traces
            _, self.Synt_F2_ts, self.Tshift_F2, self.Magnitude_F2_ts = self.data_preprocess(self.Observed,synt_raw_F2)
    
    def display_Magnitude(self,fi=0):
        '''
        
        '''
        # get station dictionary
        Stations = self.Container['network']['STAT_dict']['F1_X0'] # use only F1 information as az and baz is not used
        
        # get network magnitude
        Mfac_netw = self.Magnitude['network']['Mfac']
        Mw_netw = self.Magnitude['network']['Mw']
        
        # get station wise magnification
        fig = plt.figure(figsize=(20, 5), facecolor='w', edgecolor='k')
        ax, Lon, Lat = {}, [], []
        for stat in self.Magnitude['Mlog']:
            [lat,lon,_,_,_,_] = Stations[stat]
            Lon.append(lon)
            Lat.append(lat)
            for ci,comp in enumerate(['Z','R','T']):
                ax[ci] = plt.subplot(1,3,ci+1)
                ax[ci].set_aspect('equal')
                #try: # case comp is not selected
                [Mfac_stat,Mw_stat] = self.Magnitude['Mlog'][stat][comp][fi]
                circ_ref = plt.Circle((lon,lat),1,color='b')
                circ = plt.Circle((lon,lat),Mfac_stat/Mfac_netw,color='r')
                if Mfac_stat/Mfac_netw > 1:
                    ax[ci].add_artist(circ)
                    ax[ci].add_artist(circ_ref)
                else:
                    ax[ci].add_artist(circ_ref)
                    ax[ci].add_artist(circ)
                #except:
                #    continue
                
        # set limits
        for axi in ax:
            ax[axi].set_xlim((55,82))
            ax[axi].set_ylim((55,82))
                
        plt.show()

    def compute_nl2(self,D,S,perform):
        if perform:
            D = np.asarray(D)
            S = np.asarray(S)
            er = np.sum((D-S)**2)/np.sum(D**2)
        else:
            er = None
        return er
    
    def get_error(self,split_doublet=False):
        '''
        
        '''
        self.simulate_synthetics(False)
        
        error = {'Network':{},'Station':{}}
        D_netw, S_netw, S1_netw, S2_netw  = np.array([]), np.array([]), np.array([]), np.array([])
        for stat_id in self.Obs:
            error['Station'][stat_id] = {}
            D_stat, S_stat, S1_stat, S2_stat  = np.array([]), np.array([]), np.array([]), np.array([])
            for comp in self.Obs[stat_id]:
                error['Station'][stat_id][comp] = {}
                D_comp, S_comp, S1_comp, S2_comp  = np.array([]), np.array([]), np.array([]), np.array([])
                for fi in self.Obs[stat_id][comp]:
                    D_fi, S_fi, S1_fi, S2_fi  = np.array([]), np.array([]), np.array([]), np.array([])
                    # access traces
                    d = self.Obs[stat_id][comp][fi].copy()
                    s = self.Synt['cF_X0'][stat_id][comp][fi].copy() 
                    
                    # get mag fac
                    Mfac = self.get_magnitude_update(self.Magnitude,fi)
                    
                    # append traces to network lists
                    D_netw = np.append(D_netw,d.data)
                    S_netw = np.append(S_netw,Mfac[comp]*s.data)
                    
                    # append traces to stations lists
                    D_stat = np.append(D_stat,d.data)
                    S_stat = np.append(S_stat,Mfac[comp]*s.data)
                    
                    # append traces to components lists
                    D_comp = np.append(D_comp,d.data)
                    S_comp = np.append(S_comp,Mfac[comp]*s.data)
                    
                    # append traces to components lists
                    D_fi = np.append(D_fi,d.data)
                    S_fi = np.append(S_fi,Mfac[comp]*s.data)
                    
                    
                    if split_doublet:
                        s1 = self.Synt_F1_ts['cF_X0'][stat_id][comp][fi].copy()
                        s2 = self.Synt_F2_ts['cF_X0'][stat_id][comp][fi].copy()
                        
                        # get mag fac
                        Mfac1 = self.get_magnitude_update(self.Magnitude_F1_ts,fi)
                        Mfac2 = self.get_magnitude_update(self.Magnitude_F2_ts,fi)
                        
                        # append traces to network lists
                        S1_netw = np.append(S1_netw,Mfac1[comp]*s1.data)
                        S2_netw = np.append(S2_netw,Mfac2[comp]*s2.data)
                        
                        # append traces to stations lists
                        S1_stat = np.append(S1_stat,Mfac1[comp]*s1.data)
                        S2_stat = np.append(S2_stat,Mfac2[comp]*s2.data)
                        
                        # append traces to components lists
                        S1_comp = np.append(S1_comp,Mfac1[comp]*s1.data)
                        S2_comp = np.append(S2_comp,Mfac2[comp]*s2.data)
                        
                        # append trace to fi list (single entry)
                        S1_fi = np.append(S1_fi,Mfac1[comp]*s1.data)
                        S2_fi = np.append(S2_fi,Mfac2[comp]*s2.data)
                        
                    # compute nl2 for fi
                    er = np.sum((d.data-Mfac[comp]*s.data)**2)/np.sum(d.data**2)
                    error['Station'][stat_id][comp][fi] = {'comb':self.compute_nl2(D_fi,S_fi,True),
                                                           'F1':self.compute_nl2(D_fi,S1_fi,split_doublet),
                                                           'F2':self.compute_nl2(D_fi,S2_fi,split_doublet)}
                
                # componential error
                error['Station'][stat_id][comp]['full'] = {'comb':self.compute_nl2(D_comp,S_comp,True),
                                                             'F1':self.compute_nl2(D_comp,S1_comp,split_doublet),
                                                             'F2':self.compute_nl2(D_comp,S2_comp,split_doublet)}
                
            # station error
            error['Station'][stat_id]['full'] = {'comb':self.compute_nl2(D_stat,S_stat,True),
                                                   'F1':self.compute_nl2(D_stat,S1_stat,split_doublet),
                                                   'F2':self.compute_nl2(D_stat,S2_stat,split_doublet)}
            
        # network error
        error['Network'] = {'comb':self.compute_nl2(D_netw,S_netw,True),
                              'F1':self.compute_nl2(D_netw,S1_netw,split_doublet),
                              'F2':self.compute_nl2(D_netw,S2_netw,split_doublet)}            
                
        # store in self.error
        self.error = error
    
    def display_beachball(self,filename=None,file_format='.png'):
        '''
        
        '''
        mech = self.mechanism
        # get mechanism
        if len(mech) == 3: # case single fault dc
            m1 = self.mechanism+[0.,0.]
            m2 = None
            dm = 0.0
        elif len(mech) == 4: # case single fault dev
            m1 = self.mechanism+[0.]
            m2 = None
            dm = 0.0
        elif len(mech) == 5: # case single fault full 
            m1 = self.mechanism
            m2 = None
            dm = 0.0
        elif len(mech) == 12: # case doublet source
            m1 = self.mechanism[:5]
            m2 = self.mechanism[5:10]
            dm = self.mechanism[10]
        else:
            print('Wrong length of mechanism list')
            raise SystemExit 
        
        # figure
        fig = plt.figure(figsize=(20, 5), facecolor='w', edgecolor='k')
        ax = plt.subplot(1,1,1)
        MT1 = Tape2M(m1[0],m1[1],m1[2],m1[3],m1[4],1-dm)
        b = beach(MT1, xy=(0.,0), width=150.0*(1-dm), linewidth=1, facecolor='r', alpha=1.0, nofill=False)
        b.set_zorder(10)
        ax.add_collection(b)
        
        xlim_fac = 1
        
        if m2 is not None:
            plt.text(100,0,'+',fontsize=50,ha='center', va='center')
                        
            MT2 = Tape2M(m2[0],m2[1],m2[2],m2[3],m2[4],dm)
            b = beach(MT2, xy=(220.,0), width=150.0*dm, linewidth=1, facecolor='r', alpha=1.0, nofill=False)
            b.set_zorder(10)
            ax.add_collection(b)
            
            plt.text(300,0,'=',fontsize=50,ha='center', va='center')
            
            b = beach(MT1+MT2, xy=(440.,0), width=150.0, linewidth=1, facecolor='r', alpha=1.0, nofill=False)
            b.set_zorder(10)
            ax.add_collection(b)
            strike,dip,rake,clvd,iso,_ = M2Tape(MT1+MT2)
            
            b = beach([strike,dip,rake], xy=(440.,0), width=150.0, linewidth=2, facecolor='r', alpha=1.0, nofill=True)
            b.set_zorder(10)
            ax.add_collection(b)
            
            plt.text(-50,105,'Combined MT with ['+str(int(strike))+'° | '+str(int(dip))+'° | '+str(int(rake))+'° | '+str(int(clvd*100/30))+'% | '+str(int(iso*100/90))+'%]',fontsize=30,va='center')
            er = self.error['Network']
            plt.text(-50,-105,'VR_comb = '+str(round(1.-er['comb'],2))+' | VR_F1 = '+str(round(1.-er['F1'],2))
                     +' | VR_F2 = '+str(round(1.-er['F2'],2)),fontsize=30,va='center')
            
            xlim_fac = 5
            
        ax.set_aspect('equal')
        ax.set_xlim([-110, xlim_fac*110])
        ax.set_ylim([-110, 110])
        #plt.grid(True)
        plt.axis('off')
        if filename == None:
            plt.show()
        else:
            plt.savefig(filename+'Beachball_Combination'+file_format, 
                    bbox_inches='tight', 
                    transparent=False,
                    pad_inches=0)
    
    def az_beach(self,ax,split_doublet):
        subpos = [0.35,0.35,0.3,0.3]
        ax1 = self.add_subplot_axes(ax,subpos)
        b = beach(self.mechanism[:3], xy=(0.,0), width=150.0, linewidth=1, facecolor='r', alpha=1.0, nofill=False)
        b.set_zorder(10)
        ax1.add_collection(b)
        if split_doublet:
            b = beach(self.mechanism[5:8], xy=(0.,0), width=150.0, linewidth=4, facecolor='r', alpha=1.0, nofill=True)
            b.set_zorder(10)
            ax1.add_collection(b)
        ax1.set_aspect('equal')
        ax1.set_xlim([-110, 110])
        ax1.set_ylim([-110, 110])
        plt.axis('off')
    
    def display_azimuthal_error(self,split_doublet=False,ylim=[],fi=0,filename=None,file_format='.png'):
        '''
        
        '''
        
        # get doublet error
        if split_doublet:
            self.get_error(split_doublet=True)
        
        # axis dictionary
        ax = {}
        # error list
        mer = []
        
        # figure
        fig = plt.figure(figsize=(20, 20), facecolor='w', edgecolor='k')
        ax[0] = plt.subplot(2,2,1, projection='polar')
        STAT_dict = self.Container['network']['STAT_dict']['F1_X0']
        for si,stat_id in enumerate(self.error['Station']):
            az = STAT_dict[stat_id][4]*(np.pi/180.)
            er = self.error['Station'][stat_id]['full']['comb']
            mer.append(er)
            #ax.plot([az,az],[0,3.],'gray','-')
            ax[0].plot(az, er,'ko',markersize=5)
            
            if split_doublet:
                er = self.error['Station'][stat_id]['full']['F1']
                mer.append(er)
                ax[0].plot(az, er,'ro',markersize=5)
                er = self.error['Station'][stat_id]['full']['F2']
                mer.append(er)
                ax[0].plot(az, er,'bo',markersize=5)
                ax[0].set_title('comb. error')
            
            # plot beachball(s)
            self.az_beach(ax[0],split_doublet)
        
        for ci, comp in enumerate(['Z','R','T']):
            ax[ci+1] = plt.subplot(2,2,ci+2, projection='polar')
            # plot beachball(s)
            self.az_beach(ax[ci+1],split_doublet)
            for si,stat_id in enumerate(self.error['Station']):
                try:
                    az = STAT_dict[stat_id][4]*(np.pi/180.)
                    er = self.error['Station'][stat_id][comp][fi]['comb']
                    mer.append(er)
                    #ax.plot([az,az],[0,3.],'gray','-')
                    ax[ci+1].plot(az, er,'ko',markersize=5)
                    ax[ci+1].set_title(comp)
                    
                    if split_doublet:
                        er = self.error['Station'][stat_id][comp][fi]['F1']
                        mer.append(er)
                        ax[ci+1].plot(az, er,'ro',markersize=5)
                        er = self.error['Station'][stat_id][comp][fi]['F2']
                        mer.append(er)
                        ax[ci+1].plot(az, er,'bo',markersize=5)                        
                except:
                    continue
                
                
        for ai in range(len(ax)):
            
            # get max radius
            rmax = np.ceil(np.max(np.asarray(mer)))
            # set dr based on rmax
            if 5.0 < rmax < 10.0:
                dr = 1.0
            elif rmax >= 10.:
                dr = 2.0
            elif rmax <= 1.0:
                dr = 0.25
            elif 1.0 < rmax <= 5.0:
                dr = 0.5
            # set axis
            ax[ai].set_rticks(np.arange(0.,rmax,dr))  # Less radial ticks
            ax[ai].set_rmin(-rmax/3.)
            ax[ai].set_rmax(rmax)
            ax[ai].set_theta_zero_location("N") 
            ax[ai].set_theta_direction(-1)
            ax[ai].grid(True)
            
        
        
        if filename == None:
            plt.show()
        else:
            plt.savefig(filename+'Azimuthal_Error'+file_format, 
                    bbox_inches='tight', 
                    transparent=False,
                    pad_inches=0)

        
    
    def display_station_error(self,split_doublet=False,ylim=[],filename=None,file_format='.png'):
        '''
        
        '''
        if split_doublet:
            self.get_error(split_doublet=True)
        
        
        fig = plt.figure(figsize=(20, 5), facecolor='w', edgecolor='k')
        ax = plt.subplot(1,1,1)
        
        for si,stat_id in enumerate(self.error['Station']):
            
            if (si % 2) == 0:
                ax.axvspan(si-0.5, si+0.5, alpha=0.5, color='green')
            else:
                ax.axvspan(si-0.5, si+0.5, alpha=0.5, color='yellow')
            
            er = self.error['Station'][stat_id]['full']['comb']
            l1, = plt.plot(si, er,'ko',  markersize=10)
            #plt.errorbar(si, er, yerr=er, color='k', label='Std NL2')
            
            if self.error['Station'][stat_id]['full']['F1'] is not None:
                er = self.error['Station'][stat_id]['full']['F1']
                l2, = plt.plot(si-0.25, er,'ro',  markersize=10)
                #plt.errorbar(si-0.25, er[0], yerr=er, color='r', label='Std NL2')
                
                er = self.error['Station'][stat_id]['full']['F2']
                l3, = plt.plot(si+0.25, er,'bo',  markersize=10)
                #plt.errorbar(si+0.25, er, yerr=er, color='b', label='Std NL2')
        
        plt.xticks(np.arange(len(self.error['Station'])), list(self.error['Station'].keys()),
                   rotation='vertical',fontsize=18)
        plt.ylabel('NL2',fontsize=18)  
        plt.yticks(fontsize=15)
        try:
            plt.title('NL2 Error - Network: F1='+str(round(self.error['Network']['F1'],2))+
                    ' | F2='+str(round(self.error['Network']['F2'],2))+
                    ' | Comb.='+str(round(self.error['Network']['comb'],2)),fontsize=18)
            plt.legend([l1,l2,l3],['NL2 - comb.','NL2 - F1','NL2 - F2'],fontsize=15)
        except:
            plt.title('NL2 Error - Network: Comb.='+str(round(self.error['Network']['comb'],2)))
            plt.legend([l1],['NL2 - comb.'],fontsize=15)
            pass
        
        plt.grid(True)
        plt.xlim((-0.5,si+0.5))
        if len(ylim) == 2: 
            plt.ylim((ylim[0],ylim[1]))
        if filename == None:
            plt.show()
        else:
            plt.savefig(filename+'Station_error'+file_format, 
                    bbox_inches='tight', 
                    transparent=False,
                    pad_inches=0)
    
    def plot_waveforms(self,xlim=[],filename=None,StatList=[],split_doublet=False,
                       file_format='.png',file_style='content'):
        '''
            file_style: content or publish
                - content: display different information
                - public: keep is simple and readable
        '''
        if self.sim_doublet and split_doublet:
            self.simulate_synthetics(split_doublet)
            self.get_error(split_doublet=split_doublet)
        else:
            split_doublet=False
            self.simulate_synthetics(split_doublet)
            self.get_error()
        
        if len(StatList) == 0:
            Station_ID = self.Obs
        else:
            Station_ID = StatList 
        
        for stat_id in Station_ID:
            # get y lim of station
            y_lim = {}
            for ci,comp in enumerate(['Z','R','T']):
                for fi in range(len(self.FN)):
                    if fi not in y_lim:
                        y_lim[fi] = []
                    try:
                        d = self.Obs[stat_id][comp][fi].copy()
                        y_lim[fi].append(np.max(np.abs(d.data)))
                        if split_doublet:
                            Mfac = self.get_magnitude_update(self.Magnitude,fi)
                            f1 = self.Synt_F1['cF_X0'][stat_id][comp][fi].copy()
                            Afac1 = 1.-self.mechanism[10]
                            y_lim[fi].append(np.max(np.abs(Afac1*Mfac[comp]*f1.data)))
                            Afac2 = self.mechanism[10]
                            f2 = self.Synt_F2['cF_X0'][stat_id][comp][fi].copy()
                            y_lim[fi].append(np.max(np.abs(Afac2*Mfac[comp]*f2.data)))
                        else:
                            Mfac = self.get_magnitude_update(self.Magnitude,fi)
                            f = self.Synt['cF_X0'][stat_id][comp][fi].copy()
                            y_lim[fi].append(np.max(np.abs(Mfac[comp]*f.data)))
                            
                    except:
                        continue
            
            # plot traces of station
            fig = plt.figure(figsize=(20, 5*len(self.FN)), facecolor='w', edgecolor='k')
            for ci,comp in enumerate(['Z','R','T']):
                for fi in range(len(self.FN)):
                    ax = plt.subplot(len(self.FN),3,3*fi+ci+1)
                    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                    try:
                        # obs
                        d = self.Obs[stat_id][comp][fi].copy()
                        td = np.arange(d.stats.npts)*d.stats.delta
                        plt.plot(td,d.data,'k-',linewidth=5,label='obs')
                        
                        # synt
                        Mfac = self.get_magnitude_update(self.Magnitude,fi)
                        if split_doublet:
                            # param
                            Afac1 = 1.-self.mechanism[10]
                            Afac2 = self.mechanism[10]
                            f1 = self.Synt_F1['cF_X0'][stat_id][comp][fi].copy()
                            f2 = self.Synt_F2['cF_X0'][stat_id][comp][fi].copy()
                            # get tshift from combined trace
                            if self.perform_Tshift:
                                t_off = self.Tshift['trace'][stat_id][comp][0]
                            else:
                                t_off = 0.
                            tf = np.arange(f1.stats.npts)*f1.stats.delta - t_off
                            #plt.text(str(round(t_off,2)))
                            
                            # F1
                            er = self.error['Station'][stat_id][comp][fi]['F1']
                            plt.plot(tf,Afac1*Mfac[comp]*f1.data,'g--',linewidth=3,label='synt F1 (er='+str(round(er,2))+')')
                            
                            # F2
                            er = self.error['Station'][stat_id][comp][fi]['F2']
                            plt.plot(tf,Afac2*Mfac[comp]*f2.data,'b--',linewidth=3,label='synt F2 (er='+str(round(er,2))+')')
                            
                            # marker combined error
                            er = self.error['Station'][stat_id][comp][fi]['comb']
                            plt.plot(-10,0,'r*',label='comb.  (er='+str(round(er,2))+')')
                    
                        else:
                            f = self.Synt['cF_X0'][stat_id][comp][fi].copy()
                            tf = np.arange(f.stats.npts)*f.stats.delta
                            er = self.error['Station'][stat_id][comp][fi]['comb']
                            plt.plot(tf,Mfac[comp]*f.data,'r--',linewidth=4,label='synt (er='+str(round(er,2))+')')
                                
                        
                        if len(xlim) == 2: 
                            plt.xlim((xlim[0],xlim[1]))
                        else:
                            plt.xlim((td[0],td[-1]))
                        if self.sim_doublet and split_doublet:
                            ym = 1.1*np.max(np.asarray(y_lim[fi]))
                            if file_style == 'content':
                                plt.text(5,-1.1*ym,f1.stats.FBand)
                            plt.ylim((-1.2*ym,1.2*ym))
                        else:
                            ym = 1.1*np.max(np.asarray(y_lim[fi]))
                            plt.ylim((-1.2*ym,1.2*ym))
                            
                        if file_style == 'content':
                            try:
                                plt.title(stat_id+'-'+comp+' (FB: '+d.stats.FBand+')',fontsize=14)
                            except:
                                plt.title(stat_id+'-'+comp+' (FB_ID: '+str(fi)+')',fontsize=14)
                            plt.legend(loc=1,ncol=2,fontsize=11)    
                            plt.xlabel('Time in s',fontsize=18)
                            plt.ylabel('Displacement in m',fontsize=18)
                            ax.axis('on')
                            plt.grid(True)
                        elif file_style == 'publish':
                            if len(xlim) == 2: # x position of trace ID at 5% of t axis
                                xTit = (xlim[1]-xlim[0])*0.05
                            else:
                                xTit = (td[-1]-td[0])*0.05
                            stat_id_text = '.'.join(stat_id.split('_'))
                            plt.text(xTit,0.95*ym,stat_id_text+'-'+comp,fontsize=24,weight='bold')
                            plt.grid(True)
                            plt.xlabel('Time in s',fontsize=22,weight='bold')
                            if comp == 'Z':
                                plt.ylabel('Displacement in m',fontsize=22,weight='bold')
                            plt.xticks(fontsize=18,weight='bold')
                            plt.yticks(fontsize=18,weight='bold')
                            ax.yaxis.get_offset_text().set_fontsize(18)
                            ax.yaxis.get_offset_text().set_weight('bold')
                            ax.axis('on')
                            for axis in ['top','bottom','left','right']:# change all spines
                                ax.spines[axis].set_linewidth(4)
                            ax.tick_params(width=4)# increase tick width
                    except:
                        ax.axis('off')
                        continue
            if filename == None:
                plt.show()
            else:
                plt.savefig(filename+'_'+stat_id+file_format,#'.pdf',#'.png', 
                        bbox_inches='tight', 
                        transparent=False,
                        pad_inches=0)
    
    
    def run(self,mechanism=None,split_doublet=False):
        '''
        
        '''
        self.mechanism = mechanism
        self.simulate_synthetics(split_doublet)
        self.get_error(split_doublet=split_doublet)








################################################
'''
    functions for linear inversion
'''
################################################

def calc_El2(waveform_fund,waveform_obs,fi=0):
    El2a = np.zeros((0,6))
    obs = np.zeros((0,1))
    for stat in sorted(waveform_fund['F1']):
        for comp in ['Z','R','T']:
            e0 = np.zeros(len(waveform_fund['F1'][stat][comp][fi]))
            E = np.matrix([e0,e0,e0,e0,e0,e0])
            for jj in range(6):
                E[jj] = waveform_fund['F'+str(jj+1)][stat][comp][fi]
            El2a = np.vstack((El2a,np.transpose(E)))
            obs = np.append(obs,waveform_obs[stat][comp][fi])# = np.vstack((obs,np.transpose(np.asarray(s_cap[stat][comp][1]))))
    El2 = np.matmul(np.linalg.inv(np.matmul( np.transpose(El2a),El2a)),np.transpose(El2a))
    return np.transpose(El2), obs

def SixElementModeller(waveform_fund,a,fi=0):
    u = {}
    for stat in waveform_fund['F1']:
        u[stat] = {}
        for comp in ['Z','R','T']:
            trace = waveform_fund['F1'][stat][comp][fi].copy()
            trace.data *= 0
            u[stat][comp] = {fi:trace}
            for jj in range(len(a)):
                u[stat][comp][fi].data += waveform_fund['F'+str(jj+1)][stat][comp][fi].data * a[jj]                                     
    return u

def inversion(El2, obs):
    a = np.matmul(np.transpose(El2),np.transpose(np.asmatrix(obs)))
    a = np.array(np.transpose(a))[0].tolist()
    #a[5] *= 0
    a_opt = [a[3]+a[4]+a[5],-a[3]+a[5],-a[4]+a[5],a[1],a[2],-a[0]]
    return a_opt+(10**-20)*np.random.rand(len(a_opt)), a

def vec2mat(v):
    m = [[v[0],v[3],v[4]],
        [v[3],v[1],v[5]],
        [v[4],v[5],v[2]]]
    return m
        
def get_weightsA(M):
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
    # remove trace (isotropic part)
    MT = vec2mat(M)
    trM = np.trace(MT) # get trace tr(M)
    MT_dev = MT - (1./3.)*trM*np.eye(3) # remove isotropic influence to get deviatoric part
    # write weight vector "a" based on the given moment tensor
    #a = [MT_dev[0][1],MT_dev[0][2],-MT_dev[1][2],-MT_dev[0][0],-MT_dev[1][1],trM/3.] # Aki
    a = [-MT_dev[1][2],MT_dev[0][1],MT_dev[0][2],-MT_dev[1][1],-MT_dev[2][2],trM/3.] # Dziewonski
    return a


def get_weights(M):
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
    # start time counter
    rf = 5
        
    # remove trace (isotropic part)
    MT = vec2mat(M)
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


def MT_decomposition(MT):
    MT_decomp = MomentTensor(M=MT).get_decomposition()
    full_MT = MT_decomp[3]
    iso_per = MT_decomp[5] 
    dev_per = MT_decomp[7] 
    DC_per = MT_decomp[9] 
    CLVD_per = MT_decomp[15] 
    DC_part = np.asarray(MT_decomp[8])
    MT_DC = [DC_part[0][0],DC_part[1][1],DC_part[2][2],DC_part[0][1],DC_part[0][2],DC_part[1][2]]
    #print(MT_decomp)
    #print('ISOTROPIC: ', iso_per)
    #print('Deviatoric:', dev_per)
    #print('DC:        ', DC_per)
    #print('CLVD:      ', CLVD_per)
    return [iso_per,DC_per,CLVD_per,MT_DC]


################################################
'''
    functions for error simulation
'''
################################################

def get_filter_param(Filter):
    '''
    
    '''
    if Filter['partition_type'] == 'lfix':
        freqmax = Filter['fcut'][fi]
        freqmin = Filter['freqmin']
        corners = Filter['corners']
        zerophase = Filter['zerophase']
    elif Filter['partition_type'] == 'ufix':
        freqmin = Filter['fcut'][fi]
        freqmax = Filter['freqmax']
        corners = Filter['corners']
        zerophase = Filter['zerophase']
    else: # case None
        freqmin = Filter['freqmin']
        freqmax = Filter['freqmax']
        corners = Filter['corners']
        zerophase = Filter['zerophase']
    return freqmin,freqmax,corners,zerophase
    

def simulate_Noise(OBS0,Filter,rseed=None,h2v=1.0,relZ=[False,0.3,None],relP=[False,0.1],absNA=10**-6):
    '''
    
    '''
    Obs = {}
    Noise = {}
    param = {}
    # use random seed
    if rseed is not None:
        np.random.seed(rseed)
    for stat_id in OBS0:
        Obs[stat_id] = {}
        Noise[stat_id] = {}
        param[stat_id] = {}
        # get peak amplitude at station (all components)
        if relP[0]:
            d_max = []
            for ci,comp in enumerate(OBS0[stat_id]):
                for fii,fi in enumerate(OBS0[stat_id][comp]):
                    freqmin,freqmax,corners,zerophase = get_filter_param(Filter)
                    d = OBS0[stat_id][comp][fi].copy()
                    d.filter('bandpass',freqmin=freqmin,freqmax=freqmax,
                            corners=corners,zerophase=zerophase)
                    d_max.append(np.max(np.abs(d.data)))
            dmax = np.max(np.array(d_max))
        # comp loop
        for ci,comp in enumerate(['Z','R','T']):
            try:
                Obs[stat_id][comp] = {}    
                Noise[stat_id][comp] = {}
                param[stat_id][comp] = {}
                # trace loop
                for fii,fi in enumerate(OBS0[stat_id][comp]):   
                    d = OBS0[stat_id][comp][fi].copy()
                    dN = d.stats.npts
                    if fii == 0:
                        sig = 2*np.random.rand(2*dN)-1
                    tr = Trace(data=sig)
                    
                    freqmin,freqmax,corners,zerophase = get_filter_param(Filter)
                    tr.filter('bandpass',freqmin=freqmin,freqmax=freqmax,
                            corners=corners,zerophase=zerophase)  
                    
                    # rescale Amplitude
                    tr.data /= np.max(np.abs(tr.data))
                    if comp == 'Z': # add h2v factor to simulate different noise level on the horizontals
                        cNfac = 1.0
                        zdata = d.data 
                    else:
                        cNfac = h2v
                    if relZ[0]:
                        zNoise = np.max(np.abs(zdata))*relZ[1]
                        if relZ[2] is not None: # check if relative noise amplitude is smaller than min thershold noise amplitude
                            if zNoise > np.max(np.abs(d.data))*relZ[2]:
                                zNoise = np.max(np.abs(d.data))*relZ[2]
                        tr.data *= zNoise*cNfac
                    elif relP[0]:
                        zNoise = dmax*relP[1]
                        tr.data *= zNoise*cNfac
                    else:
                        tr.data *= absNA*cNfac
                    
                    # trim to true data length
                    delta = tr.stats.delta
                    t1 = tr.stats.starttime + int(dN*0.5)*delta
                    t2 = tr.stats.starttime + (int(dN*0.5)+dN-1)*delta
                    tr.trim(t1,t2)
                    
                    Anoise_f = np.max(np.sqrt((tr.data**2)/2.))
                    Asignal = np.max(np.sqrt((d.data**2)/2.))
                    AS2N = Asignal/Anoise_f
                    d.data += tr.data
                    Obs[stat_id][comp][fi] = d      
                    Noise[stat_id][comp][fi] = tr
                    param[stat_id][comp][fi] = {'S2N':AS2N,'A_noise':Anoise_f,'A_data':Asignal}
            except:
                continue
    return Obs, Noise, param



def rand_rot(u,sigma=0,display=False,dist='normal',trunc_fac=1.0,rseed=None):
    A = []
    u2 = {}
    Energy = {}
    rad = np.pi/180.
    # use random seed
    if rseed is not None:
        np.random.seed(rseed)
    for stat in u:
        u2[stat] = {'Z':{},'R':{},'T':{}}
        if dist == 'normal':
            a = np.random.normal(0.0, sigma, 1)
        elif dist == 'normal_trunc':
            a = np.random.normal(0.0, sigma, 1)
            if np.abs(a[0]) > sigma * trunc_fac:
                a[0] = (a[0]/np.abs(a[0])) * sigma * trunc_fac
        elif dist == 'uniform':
            a = np.random.uniform(-sigma,sigma,1)
        elif dist == 'constant':
            a = np.asarray([sigma])
        elif dist == 'dict':
            a = sigma[stat]     
        else:
            print('Entry for dist does not exist. Setting alpha = 0')
        A.append(round(float(a[0]),2))
        
        for fii, fi in enumerate(u[stat]['Z']):
            Z = u[stat]['Z'][fi].copy()
            R = u[stat]['R'][fi].copy()
            T = u[stat]['T'][fi].copy()
            tempR = u[stat]['R'][fi].copy()
            tempR.data = np.zeros(tempR.stats.npts)
            tempT = u[stat]['T'][fi].copy()
            tempT.data = np.zeros(tempT.stats.npts)

            u2[stat]['R'][fi] = tempR
            u2[stat]['T'][fi] = tempT
            u2[stat]['Z'][fi] = Z
        
            mrot = np.matrix([[np.cos(a*rad)[0],np.sin(a*rad)[0]],[-np.sin(a*rad)[0],np.cos(a*rad)[0]]])
            [R2,T2] = np.matmul(mrot,[R.data,T.data])
            u2[stat]['R'][fi].data = np.asarray(R2)[0]
            u2[stat]['T'][fi].data = np.asarray(T2)[0]
        
            ER = np.sum(np.abs(np.asarray(R2)[0]))/np.sum(np.abs(R.data))
            ET = np.sum(np.abs(np.asarray(T2)[0]))/np.sum(np.abs(T.data))
            ETR = np.sum(np.abs(T.data))/np.sum(np.abs(R.data))
            dETR = np.sum(np.abs(np.asarray(T2)[0]))/np.sum(np.abs(np.asarray(R2)[0]))
            Energy[stat] = {fi:[ER,ET,ETR,dETR]}
        
               
            if display:
                print(Energy)
                fig = plt.figure(figsize=(20, 5), facecolor='w', edgecolor='k')
                t = np.arange(Z.stats.npts)*Z.stats.delta
                plt.subplot(1,3,1)
                plt.plot(t,Z.data,'r-')
                plt.title('Z')
                plt.subplot(1,3,2)
                plt.plot(t,R,'r-')
                plt.plot(t,np.asarray(R2)[0],'b--')
                plt.title('R (d$\psi$ = '+str(round(float(a[0]),1))+')')
                plt.subplot(1,3,3)
                plt.plot(t,T,'r-')
                plt.plot(t,np.asarray(T2)[0],'b--')
                plt.title('T (d$\psi$ = '+str(round(float(a[0]),1))+')')
                plt.show()

    return u2, A, Energy


def get_error(obs,synt,fi=0):
    '''
    
    '''
    d, u = [], [] 
    for stat in obs:
        for comp in obs[stat]:
            d.append(obs[stat][comp][fi].data)
            u.append(synt[stat][comp][fi].data)
    d, u = np.asarray(d), np.asarray(u)
    nl2 = np.sqrt(np.sum((d-u)**2)/np.sum(d**2))
    VR = 1.-nl2
    pdf = np.exp(-0.5*nl2)
    
    return nl2, VR, pdf




################################################
'''
    data covariance matrix
'''
################################################

def get_Cd(N_data,dt,freqmin,A_n):
    for ii in np.arange(0,N_data,1):
        JJ = np.arange(-ii,N_data-ii,1)
        expi = np.array([A_n*np.exp(-np.abs(JJ*dt)*freqmin)])
        if ii == 0:
            Cd = expi
        else:
            Cd = np.r_[Cd,expi]
    return Cd


def design_data_covariance_matrix(obs,Container,Noise_Dict=None,N_TSampl=None,Noise_ampli=1.0,Arel=0.5,plot=False):
    '''

    '''
    
    # get settings
    Filter = Container['preprocessing']['filter']
    invProc = Container['inversion']['rec_inv_procedure']
    Cd_modus = Container['inversion']['Cd_modus']
    
    # set lists, dicts, etc.
    if invProc == 'full_network':
        CfL = []
        Cd_tupel = ()
    elif invProc == 'per_station':
        CfL = {}
        Cd_tupel = {}
    Ns = {}
    Cd_dict, CdI_dict = {}, {}
    
    # check for noise dict
    if Noise_Dict == None:
        print('No noise dictionary avaiable! Using relative amplitude with Arel='+str(Arel)+' of abs. trace peak signal.')
    
    # get freqmin 
    if Filter['partition_type'] == 'ufix':
        freqmin = Filter['fcut']
    else: # true for both cases ufix and None
        freqmin = [Filter['freqmin']]

    for stat_id in obs:
        Cd_dict[stat_id] = {}
        if invProc == 'per_station':
            CfL[stat_id] = []
            Cd_tupel[stat_id] = ()
        for ci,comp in enumerate(obs[stat_id]):
            Cd_dict[stat_id][comp] = {}
            for fii,fi in enumerate(obs[stat_id][comp]):
                Cd_dict[stat_id][comp][fi] = {}
                # save trace ID (CT matrax will be constructed based on this list)
                if invProc == 'full_network':
                    CfL.append(stat_id+'-'+comp+'-'+str(fi))
                elif invProc == 'per_station':
                    CfL[stat_id].append(stat_id+'-'+comp+'-'+str(fi))
                # get data trace
                d = obs[stat_id][comp][fi].copy()
                # number of samples
                N_data = d.stats.npts
                #print(N_data,(2*np.pi)**N_data)
                # get dt
                dt = d.stats.delta
                # get amplitude
                if Noise_Dict == None:
                    A_n = (Arel*np.max(np.abs(d.data)))**2
                else:
                    if isinstance(Noise_Dict, dict):
                        # get square of noise amplitude multiplied with noise amplifier (default = 1.0)
                        if N_TSampl == None: # use full noise window
                            A_n = (Noise_ampli*np.max(np.abs(Noise_Dict[stat_id][comp][fi].data)))**2
                        else:
                            NNS = int(N_TSampl/dt) # number of samples before source onset time
                            if NNS > Noise_Dict[stat_id][comp][fi].stats.npts:
                                NNS = Noise_Dict[stat_id][comp][fi].stats.npts - 1
                            A_n = (Noise_ampli*np.max(np.abs(Noise_Dict[stat_id][comp][fi].data[-NNS:])))**2
                    else:
                        A_n = Noise_Dict
                # construct data covariance matrix Cd
                if Cd_modus == 'sampling':
                    Cd = get_Cd(N_data,dt,freqmin[fi],A_n)
                elif Cd_modus == 'simple':
                    Cd = A_n*np.identity(N_data)#,dtype=np.float128)
                else:
                    print(Cd_modus+' is not defined!')
                    raise SystemExit

                # append Cd to tuple
                if invProc == 'full_network':
                   Cd_tupel += (Cd,)  
                elif invProc == 'per_station':
                    Cd_tupel[stat_id] += (Cd,) 
                    
        if invProc == 'per_station':
            # construct block diagonal sparce matrix
            Cd_dict[stat_id] = sparse.block_diag(Cd_tupel[stat_id])
            # compute inverse matrix
            CdI_dict[stat_id] = sparse.linalg.inv(Cd_dict[stat_id])
            Cd_tuple = () # emtpy tuple for next station
    
    if invProc == 'full_network':
        # construct block diagonal sparce matrix
        Cd_mat = sparse.block_diag(Cd_tupel)#*10**10
        # compute inverse matrix
        CdI_mat = sparse.linalg.inv(Cd_mat)
        '''
        lu = sparse.linalg.splu(Cd_mat)    
        diagL = lu.L.diagonal()
        diagU = lu.U.diagonal()
        diagL = diagL.astype(np.float128)
        diagU = diagU.astype(np.float128)
        logdet = np.log(diagL).sum() + np.log(diagU).sum()
        det = np.exp(logdet*0.05,dtype=np.float128)

        # compute determinant
        CD = Cd_mat.toarray()
        mdet = np.mean(CD)*1000
        det = np.linalg.det(CD/mdet)
        '''
    elif invProc == 'per_station':  
        Cd_mat = Cd_dict
        CdI_mat = CdI_dict
    else:
        print(invProc+' is not defined!')
        raise SystemExit
    
    # display matrix
    if plot:
        print('Info: White dots in Cd_inv are numerical artifacts produced by overwriting zero to None.')
        print('This action is only performed for the below figure to highlight the block structure.')
        if invProc == 'full_network':
            fig = plt.figure(figsize=(20, 8), facecolor='w', edgecolor='k')
            plt.subplot(1,2,1)
            cd = Cd_mat.toarray()
            cd[cd==0] = None
            plt.imshow(cd)
            plt.colorbar()
            plt.title('Cd')
            plt.subplot(1,2,2)
            cdi = CdI_mat.toarray()
            cdi[cdi==0] = None
            plt.imshow(cdi)
            plt.colorbar()
            plt.title('Cd_inv')
            plt.show()
            fig = plt.figure(figsize=(20, 5), facecolor='w', edgecolor='k')
            mdiag = []
            for ii in range(cdi.shape[0]):
                mdiag.append(cdi[ii,ii])
            plt.semilogy(np.array(mdiag))
            plt.xlabel('Sampel')
            plt.ylabel('Amplitude')
            plt.title('Cd_inv - trace')
            plt.show()
        elif invProc == 'per_station': 
            for stat_id in Cd_mat:
                print(stat_id)
                fig = plt.figure(figsize=(20, 8), facecolor='w', edgecolor='k')
                plt.subplot(1,2,1)
                cd = Cd_mat[stat_id].toarray()
                cd[cd==0] = None
                plt.imshow(cd)
                plt.colorbar()
                plt.title('Cd')
                plt.subplot(1,2,2)
                cdi = CdI_mat[stat_id].toarray()
                cdi[cdi==0] = None
                plt.imshow(cdi)
                plt.colorbar()
                plt.title('Cd_inv')
                plt.show()
    
    # write to dictionary
    Data_o = {'CfL':CfL,
              'C_d':Cd_mat,
              'det_C_d':None,
              'mdet_C_d':None,
              'k':1.0,#np.sqrt(mdet)/np.sqrt(2*np.pi*det),
              'C_dI':CdI_mat,
              'Noise_ampli':Noise_ampli,
              'Arel':Arel,
              'Noise_Dict':Noise_Dict}
    
    return Data_o



