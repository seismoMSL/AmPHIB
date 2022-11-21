#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    Mike Lindner (mike.lindner@kit.edu), 2018
"""
# obspy
from obspy import read, Stream, UTCDateTime
from obspy.signal.trigger import plot_trigger, classic_sta_lta, delayed_sta_lta, recursive_sta_lta
from obspy.geodetics.base import gps2dist_azimuth
# pyrocko
from pyrocko.gf import LocalEngine, Target, DCSource, ExplosionSource, ws, MTSource, TriangularSTF, HalfSinusoidSTF
from pyrocko import trace
from pyrocko.marker import PhaseMarker, EventMarker
from pyrocko import io, util, moment_tensor
from pyrocko.obspy_compat.base import to_obspy_trace
# basic libaries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.fftpack import diff
from scipy import signal
# logger
import time
import inspect
# extra libaries
import os
from os import listdir, path
from os.path import isfile, join
# original functions
from .util import Tape2M, get_event_id_from_time


def Stream_filter(st,Filter,Ifac):
    if len(Filter['fcut']) == 0:
        lfac = Filter['freqmin']
        ufac = Filter['freqmax']
    else:
        if Filter['partition_type'] == 'ufix':
            lfac = Filter['fcut'][Ifac-1]
            ufac = Filter['freqmax']
        elif Filter['partition_type'] == 'lfix':
            lfac = Filter['freqmin']
            ufac = Filter['fcut'][Ifac-1]
        elif Filter['partition_type'] is None:
            lfac = Filter['freqmin']
            ufac = Filter['freqmax']
        else:
            print('partition type does not exist!')
            raise SystemExit
    if Filter['ftype'] == 'bandpass':
        st.filter(Filter['ftype'], 
                freqmin=lfac, 
                freqmax=ufac, 
                corners=Filter['corners'], 
                zerophase=Filter['zerophase'])
        fb_text = str(lfac)+'-'+str(ufac)+' Hz (BP)'
    elif Filter['ftype'] == 'highpass':
        st.filter(Filter['ftype'], 
                freq=lfac,  
                corners=Filter['corners'], 
                zerophase=Filter['zerophase'])
        fb_text = 'HP_'+str(lfac)+' Hz (HP)'
    elif Filter['ftype'] == 'lowpass':
        st.filter(Filter['ftype'],  
                freq=ufac, 
                corners=Filter['corners'], 
                zerophase=Filter['zerophase'])
        fb_text = str(ufac)+' Hz (LP)'
    
    # create new header entry for frequency label 
    try:
        for tr in st:
            tr.stats.FBand = fb_text 
    except:
        st.stats.FBand = fb_text 
        
    return st, fb_text

def Stream_detrend(st,Detrend):
    # default: demean
    #st.detrend(type='demean')
    if Detrend['type'] is not None:
        st.detrend(type=Detrend['type'])
    return st

def Stream_resample(st,Resampling):
    if Resampling['sampling_rate'] is not None:
        st.resample(Resampling['sampling_rate'],
                    window=Resampling['window'], 
                    strict_length=Resampling['strict_length'])
    return st

def Stream_taper(st,Taper):
    if Taper['type'] is not None:
        st.taper(type=Taper['type'],
                max_percentage=Taper['max_percentage'], 
                max_length=Taper['max_length'],
                side=Taper['side'])
    return st
    
def Stream_rotation(st,baz):
    if baz > 360.:
        baz -= 360.
    try:
        st.rotate(method="NE->RT",back_azimuth=baz)
    except:
        print('Having issues rotation station '+st[0].id)
        raise SystemExit
    return st


class waveform_3D:
    
    def  __init__(self,Container=None):
        self.Container = Container
        
        # Fundamental Green`s Function (90SS, 90DS, 45DS)
        self.src_time = Container['source']['F1_loc'][0]
        self.channel_id = Container['network']['3D_fbd_id']
        # path
        self.path2synt = Container['path']['3D_file_path']
        self.F1_set_id = Container['source']['F1_set_id_3d']
        self.F2_set_id = Container['source']['F2_set_id_3d']
        # trace manipulation
        self.stat_az = Container['network']['STAT_dict']
        self.t_add = Container['network']['3D_t_add']
        self.trace_length = Container['network']['3D_trace_length']
        self.save_mseed = Container['writer']['3D_output']
        self.stf_t_half = Container['network']['3D_STF']
        # synthetic P-onset picker
        self.pick_from_comp = Container['preprocessing']['synt_onset']['comp']
        self.sta_lta = Container['preprocessing']['synt_onset']['sta_lta']
        self.pfac = Container['preprocessing']['synt_onset']['thresh'] 
        self.onset_plot = Container['preprocessing']['synt_onset']['plot']
        self.ph_onset_obs = Container['source']['phase_onset']['obs']
        # station selection
        self.Station_Selection = Container['network']['trace_selection']
    
    def STF(self,tr,event_id):
        '''
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html
        '''
        # get STF window length in sample
        stf_samp = self.stf_t_half[event_id.split('_')[0]] / tr.stats.delta
        # copy trace data
        tr_conv = tr.copy()
        # if SFT (half) > 0., else no STF is applied (delta)
        if stf_samp > 0.:
            # simulate STF (hann) --> signal. takes full length of window, we define it by the half length
            win = signal.windows.hann(int(stf_samp/2)+1)
            # copy trace data
            tr_conv = tr.copy()
            # convolve STF with data
            tr_conv.data = signal.convolve(tr.data, win, mode='same') / sum(win)
        
        return tr_conv.data
        
    
    def Picker(self,tr):
        '''
        
        '''
        # prepare trace
        df = tr.stats.sampling_rate
        noise = np.random.normal(0, 0.005*np.max(np.abs(tr.data)), tr.stats.npts)
        tr_env = tr.data + noise#signal.hilbert(np.abs(tr.data + noise))
        
        # set initial picker
        [sta,lta] = self.sta_lta # initial sta, lta
        cft_temp = classic_sta_lta(tr_env, int(sta*df), int(lta*df))
        
        # compute threshold based on prozentual cft peak
        thresA = np.max(cft_temp[:int(len(cft_temp)/3)])*self.pfac
        thresB = np.max(cft_temp[:int(len(cft_temp)/3)])*self.pfac
        
        # compute trigger series
        cft = classic_sta_lta(tr_env, int(sta*df), int(lta*df))
        
        # get Ponset (first time trigger)
        Ponset = np.where(cft>=thresA)[0][0]/df - sta/2#  + lta/2   
        if self.onset_plot:
            plot_trigger(tr, cft, thresA, thresB, show=False)
            plt.show()
        
        return [Ponset]
    
    def process_stream(self,st,event_id,stat_id,t_info):
        '''
            tp_update = synt P-onset - obs P-onset [s]
        '''
        # resample
        #st = Stream_resample(st,self.Resampling)
        
        # rotate stream NE->RT based on direc (-tion) key 
        baz = self.stat_az[event_id][stat_id][5]
        st = Stream_rotation(st,baz)

        for tr in st:
            # taper synthetic_traces
            #tr.taper(0.15,type='hann')
            # add zerotrace to data and update starttime
            S01 = int(round(500./tr.stats.delta,0))
            S02 = int(round(500./tr.stats.delta,0))
            tadd1 = np.random.normal(tr.data[0], np.mean(np.abs(tr.data))*0.075, S01)
            tadd2 = np.random.normal(tr.data[-1], np.mean(np.abs(tr.data))*0.075, S02)
            tr.data = np.append(tadd1,np.append(tr.data,tadd2))
            #tr.detrend(type='demean')
            tr.stats.starttime -= (S01*tr.stats.delta-t_info)
            # set header
            tr.stats.network = stat_id.split('_')[0]
            tr.stats.station = stat_id.split('_')[1]
            tr.stats.location = ''
            tr.stats.channel = self.channel_id+tr.stats.channel
            
        # cut trace to set time interval
        t0 = UTCDateTime(self.src_time)
        st.trim(t0-self.t_add,t0+self.trace_length)

        
        '''
        st1 = st.copy()
        fig = plt.figure(figsize=(20, 5), facecolor='w', edgecolor='k')
        t = np.arange(st1[2].stats.npts)*st1[2].stats.delta
        plt.subplot(1,2,1)
        plt.plot(t,st1[2].data)
        plt.title(st1[2].id+' - '+str(st1[2].stats.starttime))
        plt.subplot(1,2,2)
        st1.filter('bandpass',freqmin=0.05,freqmax=0.1,corners=4,zerophase=False)
        plt.plot(t,st1[2].data)
        plt.title(st1[2].id+' - '+str(st[2].stats.starttime))
        plt.show()
        '''
        
        return st
    
    def construct_fundamental(self,Fund_pert,event_id,stat_id):
        '''

        '''
        synt_Path = self.path2synt[event_id][stat_id] # get path to synthetics
        #print('path to 3D synthetics', synt_Path)
        traces = []
        for ci, comp in enumerate(['Z','N','E']):
            for pi, pertub in enumerate(Fund_pert['pertub']):
                tr_temp = read(synt_Path[comp][pertub])[0]
                if pi == 0:
                    tr = tr_temp.copy() # copy trace
                    tr.stats.starttime = self.src_time #  set event time as starttime
                    tr.data *= Fund_pert['sign'][pi] # multiply with sign of pertub solution
                else:
                    tr.data += Fund_pert['sign'][pi]*tr_temp.data # add further pertub solutions with sign
            traces.append(tr) # append combined trace
        # combine traces to stream
        st = Stream(traces=traces)
        return st

    def simulate(self,event_id,Fund_pert,pick_dict):
        '''

        '''
        synt = {}
        ponset = {}
        for stat_id in self.Station_Selection.keys():
            synt[stat_id] = {}
            ponset[stat_id] = {'abs':[],'rel':[]}
            
            # read and construct fundamentals from pertubed solutions
            st = self.construct_fundamental(Fund_pert,event_id,stat_id)    
                        
            # copy stream
            st_phase = st.copy()
            
            # preprocess stream
            tup = 0.0
            st = self.process_stream(st,event_id,stat_id,tup)
                        
            # assignment loop (source relative)
            for comp in ['Z','R','T']:
                synt[stat_id][comp] = {'Source':{}}
                synt[stat_id][comp]['Source'] = st.select(component=comp)[0].copy()
                        
            # assignment loop (phase relative)
            if self.ph_onset_obs is not None:
                for phase_id in self.ph_onset_obs['t_travel'][stat_id]:
                    st_temp =  self.process_stream(st_phase.copy(),event_id,stat_id,pick_dict[event_id]['t_update'][stat_id][phase_id])
                    for comp in ['Z','R','T']:
                        synt[stat_id][comp][phase_id] = st_temp.select(component=comp)[0]                        
                
            # convolve STF
            for phase_id in synt[stat_id][comp]:
                synt[stat_id][comp][phase_id].data = self.STF(synt[stat_id][comp][phase_id],event_id)

            # pick P-Onset (after conv a STF as this none might change the starttime?)
            if comp in self.pick_from_comp:
                Ppick = self.Picker(synt[stat_id][comp]['Source'])
                ponset[stat_id]['rel'] += Ppick
                timestamp = synt[stat_id][comp]['Source'].stats.starttime + Ppick[0]
                ponset[stat_id]['abs'] += [UTCDateTime(timestamp).timestamp]
            
        return synt, ponset
            
            
                

    def create_6_fund_database(self):
        #Fund_pert = { 'F1':{'pertub':['mrt'],'sign':[1]},
        #              'F2':{'pertub':['mrp'],'sign':[1]},
        #              'F3':{'pertub':['mtp'],'sign':[-1]},
        #              'F4':{'pertub':['mrr','mpp'],'sign':[-1,1]},
        #              'F5':{'pertub':['mtt','mpp'],'sign':[-1,1]},
        #              'F6':{'pertub':['mrr','mtt','mpp'],'sign':[1,1,1]}}
        Fund_pert = { 'F1':{'pertub':['mtp'],'sign':[-1]},
                      'F2':{'pertub':['mrt'],'sign':[1]},
                      'F3':{'pertub':['mrp'],'sign':[1]},
                      'F4':{'pertub':['mrr','mtt'],'sign':[1,-1]},
                      'F5':{'pertub':['mrr','mpp'],'sign':[1,-1]},
                      'F6':{'pertub':['mrr','mtt','mpp'],'sign':[1,1,1]}}
        
        # create dictionary
        Fund_Waveforms = {}
        pick_dict = {}
        
        # event_id loop
        for event_id in self.path2synt:
            Fund_Waveforms[event_id] = {}
            # get get_Ppicks
            if self.ph_onset_obs != None:
                pick_dict[event_id] = self.ph_onset_obs
                pick_dict[event_id]['t_update'] = {}
                pick_dict[event_id]['t_travel_synt'] = {}
                pick_dict[event_id]['t_onset_synt'] = {}
            else:
                pick_dict[event_id] = {'t_update':{},'t_travel_synt':{},'t_onset_synt':{}}
            # fundamentals loop
            ponset_stat = {} # save station onset info for each fundamental solution
            for fund_mech in Fund_pert:
                fund, ponset = self.simulate(event_id,Fund_pert[fund_mech],pick_dict)
                Fund_Waveforms[event_id][fund_mech] = fund
                
                for stat_id in ponset:
                    if stat_id not in ponset_stat: 
                        ponset_stat[stat_id] = []
                    ponset_stat[stat_id] += ponset[stat_id]['abs']
                
                # save mseed
                 #if self.save_mseed is not None:
                 #   self.write_mseed(fund,event_id,fund_mech)
                    
            # get average ponset info
            if self.ph_onset_obs != None:
                for stat_id in ponset_stat:
                    tpm, tps = np.mean(np.asarray(ponset_stat[stat_id])),np.std(np.asarray(ponset_stat[stat_id]))
                    t_onset = UTCDateTime(self.src_time) + tpm-UTCDateTime(self.src_time)
                    t_synth = t_onset - UTCDateTime(tr.stats.starttime).timestamp
                    print(stat_id,tpm,tps,UTCDateTime(t_onset))
                    
                    
                    '''
                    tr = Fund_Waveforms[event_id][fund_mech][stat_id]['Z']['Source']
                    df = tr.stats.sampling_rate
                    t = np.arange(tr.stats.npts)/df
                    fig = plt.figure(figsize=(20, 5), facecolor='w', edgecolor='k')
                    plt.plot(t,tr.data,'k-')
                    plt.plot(t_synth,0,'bo',markersize=10)
                    plt.title(tr.id+str(t_synth))
                    plt.xlim((t_synth-20,t_synth+50))
                    plt.show()
                    '''
                                    
                    #t_obs = round(pick_dict[event_id]['t_average'][stat_id]['P'],self.digit)
                    #t_synth = round(np.mean(np.asarray(ponset_stat[stat_id])),self.digit)
                    #print(t_obs,t_synt)
                    
                    #pick_dict[src_id]['t_update'][stat_id] = t_obs - t_synth
                    #pick_dict[src_id]['t_travel_synt'][stat_id] = t_synth #round(store.t(self.table_id[phase_id], (depth, dist)),self.digit)
                    #pick_dict[src_id]['t_onset_synt'][stat_id] = t_onset #str(UTCDateTime(self.src_time) + t_synth[phase_id])
        return Fund_Waveforms




class CAP:
    
    def __init__(self,Container=None):
        self.Container = Container
        
        # Fundamental Green`s Function (90SS, 90DS, 45DS)
        self.src_time = Container['source']['F1_loc'][0]
        self.path2CAP_fund = Container['path']['CAP_file_path']
        self.CAP_pattern = self.Container['path']['CAP_pattern']
        self.stat_az = Container['network']['STAT_dict']
        self.t_add = Container['network']['CAP_t_add']
        self.trace_length = Container['network']['trace_length']
        self.save_mseed = Container['writer']['CAP_output']
        self.stf_t_half = Container['network']['CAP_STF']
        self.ph_onset_obs = Container['source']['phase_onset']['obs']
        
        # synthetic P-onset picker
        self.perform_picker = Container['preprocessing']['synt_onset']['perform']
        self.pick_from_comp = Container['preprocessing']['synt_onset']['comp']
        self.sta_lta = Container['preprocessing']['synt_onset']['sta_lta']
        self.pfac = Container['preprocessing']['synt_onset']['thresh'] 
        self.onset_plot = Container['preprocessing']['synt_onset']['plot'] 
        
        # station selection
        self.Station_Selection = Container['network']['trace_selection']     
    
    def Time_Logger(self,func_name,file_name,run_time):
        debug_message = 'Function %s in file %s with runtime=%s' % (func_name,file_name.split('/')[-1],str(run_time))
        TLOG.debug(debug_message)
    
    def STF(self,tr,event_id):
        '''
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html
        '''
        # get STF window length in sample
        stf_samp = self.stf_t_half[event_id.split('_')[0]] / tr.stats.delta
        # copy trace data
        tr_conv = tr.copy()
        # if SFT (half) > 0., else no STF is applied (delta)
        if stf_samp > 0.:
            # simulate STF (hann) --> signal. takes full length of window, we define it by the half length
            win = signal.windows.hann(int(stf_samp/2)+1)
            # copy trace data
            tr_conv = tr.copy()
            # convolve STF with data
            tr_conv.data = signal.convolve(tr.data, win, mode='same') / sum(win)
        
        return tr_conv.data
        
    
    def Picker(self,tr):
        '''
        
        '''
        # copy trace
        tr_temp = tr.copy()
        # prepare trace
        df = tr_temp.stats.sampling_rate
        noise = np.random.normal(0, 0.005*np.max(np.abs(tr_temp.data)), tr_temp.stats.npts)
        tr_env = tr_temp.data + noise#signal.hilbert(np.abs(tr.data + noise))
        
        # set initial picker
        [sta,lta] = self.sta_lta # initial sta, lta
        cft_temp = classic_sta_lta(tr_env, int(sta*df), int(lta*df))
        
        # compute threshold based on prozentual cft peak
        thresA = np.max(cft_temp[:int(len(cft_temp)/3)])*self.pfac
        thresB = np.max(cft_temp[:int(len(cft_temp)/3)])*self.pfac
        
        # compute trigger series
        cft = classic_sta_lta(tr_env, int(sta*df), int(lta*df))
        
        # get Ponset (first time trigger)
        Ponset = np.where(cft>=thresA)[0][0]/df - sta/2#  + lta/2 
        if self.onset_plot:
            plot_trigger(tr, cft, thresA, thresB, show=False)
            plt.show()
        
        return [Ponset]
    
    def trace_sorter(self,comp,mech,direc):
        '''
            create new dictionary structure based in the CAP definition
            Z BCBB 90SS 90DS 45DS ISO
            R BCBB 90SS 90DS 45DS ISO
            T AA B 90SS 90DS      ISO
        '''
        if comp == 'Z' and mech == '90SS' and direc == 'B': return 0 
        if comp == 'Z' and mech == '90DS' and direc == 'C': return 1 
        if comp == 'Z' and mech == '45DS' and direc == 'B': return 2 
        if comp == 'Z' and mech == 'ISO' and direc == 'B': return 5 
        
        if comp == 'R' and mech == '90SS' and direc == 'B': return 0 
        if comp == 'R' and mech == '90DS' and direc == 'C': return 1 
        if comp == 'R' and mech == '45DS' and direc == 'B': return 2 
        if comp == 'R' and mech == 'ISO' and direc == 'B': return 5
    
        if comp == 'T' and mech == '90SS' and direc == 'A': return 3 
        if comp == 'T' and mech == '90DS' and direc == 'A': return 4 
        if comp == 'T' and mech == 'ISO' and direc == 'B': return 5
    
        return None
    
    def calc_A(self,event_id,fault):
        '''
        
        '''
        
        strike_f, dip, rake = \
            fault[0]*np.pi/180,fault[1]*np.pi/180,fault[2]*np.pi/180          
        A = {}
        for stat_id in self.Station_Selection.keys():
            az_st = self.stat_az[event_id][stat_id][4]*np.pi/180
            Afac = np.zeros(5)  
            strike = (az_st-strike_f)            
            Afac[0] = np.sin(2*strike)*np.cos(rake)*np.sin(dip) + 0.5*np.cos(2*strike)*np.sin(rake)*np.sin(2*dip)
            Afac[1] = np.cos(strike)*np.cos(rake)*np.cos(dip) - np.sin(strike)*np.sin(rake)*np.cos(2*dip)
            Afac[2] = 0.5*np.sin(rake)*np.sin(2*dip)
            Afac[3] = np.cos(2*strike)*np.cos(rake)*np.sin(dip) - 0.5*np.sin(2*strike)*np.sin(rake)*np.sin(2*dip)
            Afac[4] = -np.sin(strike)*np.cos(rake)*np.cos(dip) - np.cos(strike)*np.sin(rake)*np.cos(2*dip)
            A[stat_id] = Afac
            
        return A         
    
    def read_waveforms(self,event_id,stat_id):
        '''
            CAP_Path[stat_id][mech_key][comp][direc]
            fundamentals[stat_id[comp][fund_id]
        '''
        CAP_Path = self.path2CAP_fund[event_id][stat_id]
        az_direc = {'A':0.,'B':45.,'C':90.}
        Fund = {'Z':{},'R':{},'T':{}}
        for mech_key in CAP_Path:
            # component loop
            for direc in self.CAP_pattern[mech_key]:#['A','B','C']:
                for ci, comp in enumerate(['Z','N','E']):
                    path = CAP_Path[mech_key][comp][direc]
                    # read trace and add to stream
                    if ci == 0:
                        st = read(path)
                    else:
                        st += read(path)
                
                # rotate stream NE->RT based on direc (-tion) key
                baz = az_direc[direc]+180.
                st = Stream_rotation(st,baz)
                                
                # update starttime to event time
                for tr in st:
                    tr.stats.starttime = self.src_time
                
                # get fund_id for selected trace
                for comp2 in ['Z','R','T']:
                    fund_id = self.trace_sorter(comp2,mech_key,direc)
                    if fund_id is not None:
                        st2 = st.select(component=comp2).copy()
                        # add zerotrace to data and update starttime
                        S01 = int(round(500./st2[0].stats.delta,0))
                        S02 = int(round(500./st2[0].stats.delta,0))
                        #tadd1 = st2[0].data[0]*np.ones(S01)
                        #tadd2 = st2[0].data[-1]*np.ones(S02)
                        tadd1 = np.random.normal(st2[0].data[0], np.max(np.abs(st2[0].data[:10])), S01)
                        tadd2 = np.random.normal(st2[0].data[0], np.max(np.abs(st2[0].data[:10])), S02)
                        st2[0].data = np.append(tadd1,np.append(st2[0].data,tadd2))
                        st2[0].stats.starttime -= S01*st2[0].stats.delta
                        if self.t_add is not None:
                            # cut trace to set time interval
                            t0 = UTCDateTime(self.src_time)
                            st2.trim(t0-self.t_add,t0+self.trace_length)
                        else:
                            # cut trace to set time interval
                            t0 = UTCDateTime(self.src_time)
                            st2.trim(t0,t0+self.trace_length)
                        Fund[comp2][fund_id] = st2[0]                        

        return Fund
    
    def calc_synt(self,event_id,A,mech):
        '''
            weigthed trace combination following
            Lian-She Zhao and Donald V. Helmberger 1994
            Source Estimation from Broadband Regional Seismograms
        '''
        synt = {}
        ponset = {}
        pID = {'Z':[0,1,2],'R':[0,1,2],'T':[3,4]}
        for stat_id in self.Station_Selection.keys():
            synt[stat_id] = {}
            ponset[stat_id] = {'abs':[],'rel':[]}
            Fund = self.read_waveforms(event_id,stat_id)
            for comp in ['Z','R','T']:
                synt[stat_id][comp] = {'Source':{}}
                synt[stat_id][comp]['Source'] = Fund[comp][5].copy() 
                synt[stat_id][comp]['Source'].data = np.zeros(synt[stat_id][comp]['Source'].stats.npts) 
                if mech == 'DC':
                    for fund_id in pID[comp]:
                        synt[stat_id][comp]['Source'].data += A[stat_id][fund_id]*Fund[comp][fund_id].data
                elif mech == 'ISO':
                    synt[stat_id][comp]['Source'].data = Fund[comp][5].data
                elif mech == 'default':
                    synt[stat_id][comp]['Source'].data = np.zeros(synt[stat_id][comp].stats.npts) 
                
                # convolve STF
                #t = np.arange(synt[stat_id][comp]['Source'].stats.npts)*synt[stat_id][comp]['Source'].stats.delta
                #fig = plt.figure(figsize=(20, 5), facecolor='w', edgecolor='k')
                #plt.plot(t,synt[stat_id][comp]['Source'].data,'k-')
                synt[stat_id][comp]['Source'].data = self.STF(synt[stat_id][comp]['Source'],event_id)
                #plt.plot(t,synt[stat_id][comp]['Source'].data,'r--')
                #plt.xlim((50,150))
                #plt.show()
                #print(asdf)
                
                # pick P-Onset (after conv a STF as this none might change the starttime?)
                if self.perform_picker:
                    if comp in self.pick_from_comp:
                        Ppick = self.Picker(synt[stat_id][comp]['Source'])
                        ponset[stat_id]['rel'] += Ppick
                        timestamp = synt[stat_id][comp]['Source'].stats.starttime + Ppick[0]
                        ponset[stat_id]['abs'] += [UTCDateTime(timestamp).timestamp]
            
            '''
            tr = synt[stat_id]['Z']['Source']
            noise = np.random.normal(0, 0.005*np.max(np.abs(tr.data)), tr.stats.npts)
            tr_env = tr.data+noise#signal.hilbert(np.abs(tr.data))
            df = tr.stats.sampling_rate
            t = np.arange(tr.stats.npts)/df
            fig = plt.figure(figsize=(20, 5), facecolor='w', edgecolor='k')
            plt.plot(t,tr_env,'k-')
            for tp0 in ponset[stat_id]['rel']:
                plt.plot(tp0,0,'ro',markersize=5)
            Ponset = np.mean(np.asarray(ponset[stat_id]['rel']))
            plt.plot(Ponset,0,'bo',markersize=10)
            plt.title(tr.id+str(Ponset))
            plt.xlim((Ponset-20,Ponset+50))
            plt.show()
            '''
              
        return synt, ponset

    def simulate(self,event_id,fault):
        '''
        
        '''
        # create dictionary
        synt_data = {}
        
        if fault == 'ISO':
            synt_data, ponset = self.calc_synt(event_id,None,'ISO')
        elif fault == 'default':
            synt_data, ponset = self.calc_synt(event_id,None,'default')
        else:
            # simulate weight A for true setting
            A = self.calc_A(event_id,fault)
            # simulate traces for true weight A
            synt_data, ponset = self.calc_synt(event_id,A,'DC')
                
        return synt_data, ponset
    
    def write_mseed(self,fund,event_id,fund_mech):
        '''
        
        '''
        path = self.save_mseed
        if not os.path.exists(path+event_id):
            os.makedirs(path+event_id)
        if not os.path.exists(path+event_id+'/'+fund_mech):
            os.makedirs(path+event_id+'/'+fund_mech)
            
        filepath = path+event_id+'/'+fund_mech+'/'
        for stat_id in fund:
            for comp in fund[stat_id]:
                tr = fund[stat_id][comp]
                tr.stats.station =  stat_id.split('_')[1]
                tr.stats.network =  stat_id.split('_')[0]
                tr.write(filepath+tr.id,format='MSEED')
    
    def get_mechanism(self,fault_dict=None):
        Fund_Waveforms = {}
        for event_id in self.path2CAP_fund:
            Fund_Waveforms[event_id] = {}
            for f_key in fault_dict:
                Fund_Waveforms[event_id][f_key],_ = self.simulate(event_id,fault_dict[f_key])
        return Fund_Waveforms
    
    def create_6_fund_database(self):
        Fund_DC = { 'F1':[0.,90.,0.,0.],
                    'F2':[90.,90.,90.,0.],
                    'F3':[0.,90.,90.,0.],
                    'F4':[90.,45.,90.,0.],
                    'F5':[0.,45.,90.,0.],
                    'F6':'ISO'}#,
                    #'default':'default'}
        
        # create dictionary
        Fund_Waveforms = {}
        pick_dict = {}
        
        # event_id loop
        for event_id in self.path2CAP_fund:
            Fund_Waveforms[event_id] = {}
            # get get_Ppicks
            if self.ph_onset_obs != None:
                pick_dict[event_id] = self.ph_onset_obs
                pick_dict[event_id]['t_update'] = {}
                pick_dict[event_id]['t_travel_synt'] = {}
                pick_dict[event_id]['t_onset_synt'] = {}
            else:
                pick_dict[event_id] = {'t_update':{},'t_travel_synt':{},'t_onset_synt':{}}
            # fundamentals loop
            ponset_stat = {} # save station onset info for each fundamental solution
            for fund_mech in Fund_DC:
                fund, ponset = self.simulate(event_id,Fund_DC[fund_mech])
                Fund_Waveforms[event_id][fund_mech] = fund
                
                if self.perform_picker:
                    for stat_id in ponset:
                        if stat_id not in ponset_stat: 
                            ponset_stat[stat_id] = []
                        ponset_stat[stat_id] += ponset[stat_id]['abs']
                
                # save mseed
                if self.save_mseed is not None:
                    self.write_mseed(fund,event_id,fund_mech)
                    
            # get average ponset info
            if self.ph_onset_obs != None and self.perform_picker:
                for stat_id in ponset_stat:
                    tpm, tps = np.mean(np.asarray(ponset_stat[stat_id])),np.std(np.asarray(ponset_stat[stat_id]))
                    t_onset = UTCDateTime(self.src_time) + tpm-UTCDateTime(self.src_time)
                    t_synth = t_onset - UTCDateTime(tr.stats.starttime).timestamp
                    #print(stat_id,tpm,tps,UTCDateTime(t_onset))
                    
                    
                    '''
                    tr = Fund_Waveforms[event_id][fund_mech][stat_id]['Z']['Source']
                    df = tr.stats.sampling_rate
                    t = np.arange(tr.stats.npts)/df
                    fig = plt.figure(figsize=(20, 5), facecolor='w', edgecolor='k')
                    plt.plot(t,tr.data,'k-')
                    plt.plot(t_synth,0,'bo',markersize=10)
                    plt.title(tr.id+str(t_synth))
                    plt.xlim((t_synth-20,t_synth+50))
                    plt.show()
                    '''
                                    
                    #t_obs = round(pick_dict[event_id]['t_average'][stat_id]['P'],self.digit)
                    #t_synth = round(np.mean(np.asarray(ponset_stat[stat_id])),self.digit)
                    #print(t_obs,t_synt)
                    
                    #pick_dict[src_id]['t_update'][stat_id] = t_obs - t_synth
                    #pick_dict[src_id]['t_travel_synt'][stat_id] = t_synth #round(store.t(self.table_id[phase_id], (depth, dist)),self.digit)
                    #pick_dict[src_id]['t_onset_synt'][stat_id] = t_onset #str(UTCDateTime(self.src_time) + t_synth[phase_id])
                

        return Fund_Waveforms
        
        
class Synthetics_Data_Loader:
    
    def __init__(self,Container=None):
        self.Container = Container
        self.synt_source = Container['general']['synt_source']
        #self.Pyrocko_t_add = Container['network']['Pyrocko_t_add']
        #self.CAP_t_add = Container['network']['CAP_t_add']
        self.t_add = Container['network']['t_wind_synt'][0]
        
        # path
        self.sim_doublet = Container['source']['simulate_doublet']
        
        # source information
        self.src_time = Container['source']['F1_loc'][0]
        
        # preprocessing
        self.Filter = self.get_filter()
        self.Resampling = Container['preprocessing']['resampling']
        self.Detrend = Container['preprocessing']['detrend']
        self.Taper = Container['preprocessing']['taper']
        
        # phase window
        #self.perfrom_pw_cut = Container['preprocessing']['phase_window']['perform']
        #self.pw_width = Container['preprocessing']['phase_window']['width']
        
        # case solve for misrotation
        self.solve_for_misrot = Container['source']['solve_for_misrot']
        self.daz = Container['source']['daz']
        
        # case solve for station delay
        self.solve_for_station_delay = Container['source']['solve_for_delay']
        self.t_del = Container['source']['t_del']
        
    def get_filter(self):
        Filter = {}
        if self.sim_doublet:
            FB1 = self.Container['preprocessing']['doublet_filter']['F1']
            FB2 = self.Container['preprocessing']['doublet_filter']['F2']
            Filter['F1'] = {
                    'ftype':'bandpass',
                    'freqmin':FB1['freqmin'],
                    'freqmax':FB1['freqmax'],
                    'corners':FB1['corners'],
                    'zerophase':FB1['zerophase'],
                    'partition_type':'lfix',
                    'fcut':len(FB2['fcut'])*[FB1['freqmax']]} 
            Filter['F2'] = {
                    'ftype':'bandpass',
                    'freqmin':FB1['freqmin'],
                    'freqmax':FB2['freqmax'],
                    'corners':FB1['corners'],
                    'zerophase':FB1['zerophase'],
                    'partition_type':'ufix',
                    'fcut':FB2['fcut']} 
                
        else:
            Filter['F1'] = self.Container['preprocessing']['filter'] 
        return Filter
    
    def mrot_deriv(self,stat_id,trZ,trR,trT,trR0,trT0):
        '''
            d_Z’ = 0    
            d_R’ = (R(-a)-R(a)) / 2*a           
            d_T’ = (T(-a)-T(a)) / 2*a
        '''
        if isinstance(self.daz, dict):
            daz = self.daz[stat_id.split('_')[1]][1]*np.pi/180.
        else: # station wise orientation informations are not available
            # daz is a scalar
            daz = self.daz*np.pi/180.
        # compute derivative
        trZ.data *= 0
        # rotate (left/right rotation)
        trRr1 =  trR0.data*np.cos(daz) + trT0.data*np.sin(daz)
        trTr1 =  -trR0.data*np.sin(daz) + trT0.data*np.cos(daz)
        trRr2 =  trR0.data*np.cos(-daz) + trT0.data*np.sin(-daz)
        trTr2 =  -trR0.data*np.sin(-daz) + trT0.data*np.cos(-daz)
        # drot
        trR.data = (trRr2-trRr1)/(2*daz)
        trT.data = (trTr2-trTr1)/(2*daz)
        
        return trZ, trR, trT
    
    def delay_deriv(self,trZ,trR,trT):
        '''
            time derivative of signal
        '''
        trZ.data = diff(trZ.data)
        trR.data = diff(trR.data)
        trT.data = diff(trT.data)
        
        return trZ, trR, trT
    
    def get_derivative(self,waveform,src_key):
        '''
                
        '''
        for fund in waveform[src_key]:
            for stat_id in waveform[src_key][fund]:    
                if src_key == 'F1_X4':
                    for phase_id in waveform[src_key][fund][stat_id]['Z']:
                        for fi in waveform[src_key][fund][stat_id]['Z'][phase_id]:
                            # copy original traces
                            trZ0 = waveform[src_key][fund][stat_id]['Z'][phase_id][fi].copy()
                            trR0 = waveform[src_key][fund][stat_id]['R'][phase_id][fi].copy()
                            trT0 = waveform[src_key][fund][stat_id]['T'][phase_id][fi].copy()
                            trZ = waveform[src_key][fund][stat_id]['Z'][phase_id][fi].copy()
                            trR = waveform[src_key][fund][stat_id]['R'][phase_id][fi].copy()
                            trT = waveform[src_key][fund][stat_id]['T'][phase_id][fi].copy()
                            # calc derivatives
                            trZ, trR, trT = self.mrot_deriv(stat_id,trZ,trR,trT,trR0,trT0)
                            # overwrite dict content
                            waveform[src_key][fund][stat_id]['Z'][phase_id][fi] = trZ
                            waveform[src_key][fund][stat_id]['R'][phase_id][fi] = trR
                            waveform[src_key][fund][stat_id]['T'][phase_id][fi] = trT
                elif src_key == 'F1_X5':
                    for phase_id in waveform[src_key][fund][stat_id]['Z']:
                        for fi in waveform[src_key][fund][stat_id]['Z'][phase_id]:
                            trZ = waveform[src_key][fund][stat_id]['Z'][phase_id][fi].copy()
                            trR = waveform[src_key][fund][stat_id]['R'][phase_id][fi].copy()
                            trT = waveform[src_key][fund][stat_id]['T'][phase_id][fi].copy()
                            # calc derivatives
                            trZ, trR, trT = self.delay_deriv(trZ,trR,trT)                    
                            # overwrite dict content
                            waveform[src_key][fund][stat_id]['Z'][phase_id][fi] = trZ
                            waveform[src_key][fund][stat_id]['R'][phase_id][fi] = trR
                            waveform[src_key][fund][stat_id]['T'][phase_id][fi] = trT
        return waveform
        
    def create_multiband_database(self,Fund):
        '''
            Load synthetics from local directory
        '''
        # initiat function
        waveform  = {}
        
        
        # get number of filter pertubations, is set to none only full band is used
        if self.Filter['F1']['partition_type'] == None:
            FN = [0]
        else:
            FN = list(map(int,list(np.arange(0,len(self.Filter['F1']['fcut']),1))))
        
        for event_id in Fund:
            waveform[event_id] = {}
            for fund_id in Fund[event_id]:
                waveform[event_id][fund_id] = {}
                for stat_id in Fund[event_id][fund_id]:
                    waveform[event_id][fund_id][stat_id] = {}                   
                    for comp in Fund[event_id][fund_id][stat_id]:
                        waveform[event_id][fund_id][stat_id][comp] = {}
                        for phase_id in Fund[event_id][fund_id][stat_id][comp]:
                            waveform[event_id][fund_id][stat_id][comp][phase_id] = {}
                            for fi in FN:
                                tr = Fund[event_id][fund_id][stat_id][comp][phase_id].copy()
                                tr, fb_text = Stream_filter(tr,self.Filter[event_id[:2]],fi)
                                #fb_text = 'raw'
                                #tr.stats.FBand = 'raw'
                                tr = Stream_resample(tr,self.Resampling)                                
                                tr = Stream_taper(tr,self.Taper)
                                
                                t0 = tr.stats.starttime - self.t_add
                                tend = tr.stats.endtime
                                tr.trim(t0,tend)
                                '''
                                if self.synt_source == 'Pyrocko':
                                    t0 = tr.stats.starttime - self.Pyrocko_t_add
                                    tend = tr.stats.endtime
                                    tr.trim(t0,tend)
                                elif self.synt_source == 'CAP':
                                    t0 = tr.stats.starttime - self.CAP_t_add
                                    tend = tr.stats.endtime
                                    tr.trim(t0,tend)
                                '''
                                
                                waveform[event_id][fund_id][stat_id][comp][phase_id][fi] = tr
        
        if self.solve_for_misrot:
            waveform = self.get_derivative(waveform,'F1_X4')
        if self.solve_for_station_delay:
            waveform = self.get_derivative(waveform,'F1_X5')
        
        return waveform
        

class pyrocko_database:
    
    def __init__(self,Container=None):
        self.M_ref = Container['source']['ref_mag'][0]
        self.src_time = Container['source']['F1_loc'][0]
        self.src_loc = Container['source']['SRC_dict']
        self.event_list = Container['source']['event_list']
        self.event_id = Container['source']['event_id']
        self.Station_Selection = Container['network']['trace_selection']
        self.Stations = Container['network']['stat_global']
        self.stat_dict = Container['network']['STAT_dict']
        self.path2fomosto_database = Container['path']['fomosto_database']
        self.store_id = Container['source']['pyrocko_store_id']
        self.store_id_subnetwork = Container['network']['store_id_subnetwork']
        self.channel_id = Container['network']['pyrocko_fbd_id'] 
        self.trace_length = Container['network']['trace_length']
        self.t_add = Container['network']['Pyrocko_t_add']
        self.stf_t_half = Container['network']['Pyrocko_STF']
        self.table_id = Container['source']['phase_onset']['table_id']
        self.digit = 2
        self.Resampling = Container['preprocessing']['resampling']
        # phase window
        self.perfrom_pw_cut = Container['preprocessing']['phase_window']['perform']
        self.ph_onset_obs = Container['source']['phase_onset']['obs']
    
    def get_source(self,Source,Location,t_half):
        '''
        
        '''
        M0 = moment_tensor.magnitude_to_moment(self.M_ref)
        MT = Tape2M(Source[0],Source[1],Source[2],Source[3],Source[4],M0)
        tref = str(self.src_time).split('T')[0]+' '+str(self.src_time).split('T')[1]
        if t_half is not None:
            stf = HalfSinusoidSTF(t_half*2.)
        else:
            stf = None
        source = MTSource(time=util.str_to_time(tref),
                    lat=Location[0],lon=Location[1],depth=Location[2]*1000.,
                    mnn=MT[1],mee=MT[2],mdd=MT[0],
                    mne=-MT[5],mnd=MT[3],med=-MT[4],
                    stf=stf)    
        
        '''
        if Source[4] == 90.:
            source = MTSource(time=util.str_to_time(tref),
                    lat=Location[0],lon=Location[1],depth=Location[2]*1000.,
                    mnn=MT[1],mee=MT[2],mdd=MT[0],
                    mne=-MT[5],mnd=MT[3],med=-MT[4],
                    stf=stf)
        else:
            source = DCSource(time=util.str_to_time(tref),
                lat=Location[0],lon=Location[1],depth=Location[2]*1000.,
                strike=Source[0],dip=Source[1],rake=Source[2],
                magnitude=self.M_ref,stf=stf)
        '''
        return source

    def get_Target(self,store_id,lat,lon):
        '''
        
        '''
        channel_codes = 'ENZ'
        targets = [
            Target(
                lat=lat,
                lon=lon,
                store_id=store_id,
                codes=('', 'STA', '', channel_code))
            for channel_code in channel_codes]
        return targets
    
    def process_stream(self,st,stat_id,baz,t_info):
        '''
            tp_update = synt P-onset - obs P-onset [s]
        '''
        # resample
        #st = Stream_resample(st,self.Resampling)
        
        # rotate stream
        st = Stream_rotation(st,baz)
        
        for tr in st:
            # taper synthetic_traces
            #tr.taper(0.15,type='hann')
            # add zerotrace to data and update starttime
            S01 = int(round(500./tr.stats.delta,0))
            S02 = int(round(500./tr.stats.delta,0))
            tadd1 = np.random.normal(tr.data[0], np.mean(np.abs(tr.data))*0.075, S01)
            tadd2 = np.random.normal(tr.data[-1], np.mean(np.abs(tr.data))*0.075, S02)
            tr.data = np.append(tadd1,np.append(tr.data,tadd2))
            #tr.detrend(type='demean')
            tr.stats.starttime -= (S01*tr.stats.delta-t_info)
            # set header
            tr.stats.network = stat_id.split('_')[0]
            tr.stats.station = stat_id.split('_')[1]
            tr.stats.location = ''
            tr.stats.channel = self.channel_id+tr.stats.channel
            
        # cut trace to set time interval
        t0 = UTCDateTime(self.src_time)
        st.trim(t0-self.t_add,t0+self.trace_length)
        '''
        st1 = st.copy()
        fig = plt.figure(figsize=(20, 5), facecolor='w', edgecolor='k')
        t = np.arange(st1[2].stats.npts)*st1[2].stats.delta
        plt.subplot(1,2,1)
        plt.plot(t,st1[2].data)
        plt.title(st1[2].id+' - '+str(st1[2].stats.starttime))
        plt.subplot(1,2,2)
        st1.filter('bandpass',freqmin=0.05,freqmax=0.1,corners=4,zerophase=False)
        plt.plot(t,st1[2].data)
        plt.title(st1[2].id+' - '+str(st[2].stats.starttime))
        plt.show()
        '''
        
        return st
    
    def simulate_fund(self,fault_dict=None):
        '''
        
        '''
        # get current directory
        curr_dir = os.getcwd()
        # change location to fomosto store directory
        os.chdir(self.path2fomosto_database)
        
        # fundamentals
        if fault_dict is None:
            fault_dict = {'F1':[0.,90.,0.,0.,0.],
                          'F2':[90.,90.,90.,0.,0.],
                          'F3':[0.,90.,90.,0.,0.],
                          'F4':[90.,45.,90.,0.,0.],
                          'F5':[0.,45.,90.,0.,0.],
                          'F6':[0.,0.,0.,0.,100.]}
        else:
            fault_dict = fault_dict
        
        # set fomosto engine
        engine = LocalEngine(store_superdirs=['.'])
        
        # dictionaries
        Fund_Waveforms = {}
        pick_dict = {}
        
        # phase list
        Phase_List = ['Source']
        
        # trace loop
        for src_id in self.src_loc:
            Fund_Waveforms[src_id] = {}
            # get get_Ppicks
            if self.ph_onset_obs != None:
                pick_dict[src_id] = self.ph_onset_obs
                pick_dict[src_id]['t_update'] = {}
                pick_dict[src_id]['t_travel_synt'] = {}
                pick_dict[src_id]['t_onset_synt'] = {}
            else:
                pick_dict[src_id] = {'t_update':{},'t_travel_synt':{},'t_onset_synt':{}}
            # store_id loop
            for store_id in self.store_id:
                print(store_id)
                # check if multiple store_id`s are selected
                if len(self.store_id) > 1:
                    Station_Selection = self.store_id_subnetwork[store_id]
                else:
                    Station_Selection = self.Station_Selection
                # src_id loop
                for fmi, fund_mech in enumerate(fault_dict):
                    Fund_Waveforms[src_id][fund_mech] = {}
                    t_half = self.stf_t_half[src_id.split('_')[0]]
                    source = self.get_source(fault_dict[fund_mech],self.src_loc[src_id],t_half)
                
                    # 6 fundamental loop
                    for stat_id in Station_Selection:
                        Fund_Waveforms[src_id][fund_mech][stat_id] = {}
                        [lat,lon] = self.Stations[stat_id][:2]
                        targets = self.get_Target(store_id,lat,lon)
                        # Processing that data will return a pyrocko.gf.Reponse object.
                        response = engine.process(source, targets)
                        # This will return a list of the requested traces:
                        synthetic_traces = response.pyrocko_traces()
                        
                        # creating stream object
                        traces = []
                        for ii in range(3):
                            traces.append(to_obspy_trace(synthetic_traces[ii]))
                        st = Stream(traces=traces)
                        
                        # get phase onset
                        if self.ph_onset_obs != None and fmi == 0:
                            t_synth, t_update, t_onset = {}, {}, {}
                            store = engine.get_store(store_id)
                            depth = self.src_loc[src_id][2]*1000.
                            dist = self.stat_dict[src_id][stat_id][3]
                            for phase_id in self.table_id:
                                t_synth[phase_id], t_update[phase_id], t_onset[phase_id] = 0., 0., 0.
                                if phase_id not in Phase_List: Phase_List.append(phase_id) # append new phases
                                if stat_id in pick_dict[src_id]['t_average']: # check for stat
                                    if phase_id in pick_dict[src_id]['t_average'][stat_id]: # check for phase
                                        t_synth[phase_id] = round(store.t(self.table_id[phase_id], (depth, dist)),self.digit)
                                        t_obs = round(pick_dict[src_id]['t_average'][stat_id][phase_id],self.digit)
                                        t_update[phase_id] = t_obs - t_synth[phase_id] 
                                        t_onset[phase_id] = str(UTCDateTime(self.src_time) + t_synth[phase_id])
                            # save t_update
                            pick_dict[src_id]['t_update'][stat_id] = t_update
                            pick_dict[src_id]['t_travel_synt'][stat_id] = t_synth
                            pick_dict[src_id]['t_onset_synt'][stat_id] = t_onset
                        
                        # preprocess stream (add header, set time information etc.)
                        baz = self.stat_dict[src_id][stat_id][5]
                        trace_dict = {}
                        for phase_id in Phase_List:
                            st_temp = st.copy()
                            if phase_id == 'Source':
                                for tr in self.process_stream(st_temp,stat_id,baz,0.):
                                    comp = tr.stats.channel[-1]
                                    trace_dict[comp] = {phase_id:tr} 
                            else: # other phases defined by cnv file
                                for tr in self.process_stream(st_temp,stat_id,baz,pick_dict[src_id]['t_update'][stat_id][phase_id]):
                                    comp = tr.stats.channel[-1]
                                    trace_dict[comp].update({phase_id:tr})
                        '''
                        if fmi == 0:
                            fig = plt.figure(figsize=(20, 5), facecolor='w', edgecolor='k')
                            col = {'Source':'k-','P':'b--','S':'r--'}
                            for ci,comp in enumerate(trace_dict):
                                for phase_id in trace_dict[comp]:
                                    t = np.arange(trace_dict[comp][phase_id].stats.npts)*trace_dict[comp][phase_id].stats.delta
                                    data = trace_dict[comp][phase_id].data/np.max(np.abs(trace_dict[comp][phase_id].data)) - ci*2
                                    if phase_id != 'Source' and ci == 0:
                                        plt.plot(t,data,col[phase_id],label=phase_id+' (dt='+str(round(pick_dict[src_id]['t_update'][stat_id][phase_id],2))+'s)')
                                    elif phase_id == 'Source' and ci == 0:
                                        plt.plot(t,data,col[phase_id],label=phase_id)
                                    else:
                                        plt.plot(t,data,col[phase_id])
                                        
                            plt.title(stat_id,fontsize=20)
                            plt.xlim((180,195))
                            plt.xlabel('Time in s',fontsize=18)
                            #plt.ylabel('Displacement in m',fontsize=18)
                            plt.legend(fontsize=16,loc=1)
                            plt.grid(True)
                            plt.yticks([0, -2, -4], ['Z', 'R', 'T'],fontsize=18)
                            plt.savefig(stat_id+'.png',bbox_inches='tight',transparent=False,pad_inches=0)
                        '''
                        
                        # fill Fund_Waveforms dictionary
                        Fund_Waveforms[src_id][fund_mech][stat_id] = trace_dict

        # change back to notebook directory
        os.chdir(curr_dir)
        
        return Fund_Waveforms, pick_dict




class Observable_Data_Loader:
    
    '''
        load observables from local source 
    '''
    
    def __init__(self,Container=None):
        #
        self.Container = Container
        self.synt_source = Container['general']['synt_source']
        
        # path
        self.data_path = Container['path']['observation_data_path']
        self.sim_doublet = Container['source']['simulate_doublet']
        
        # source information
        self.src_time = Container['source']['F1_loc'][0]
        self.t_add = Container['network']['t_wind_synt'][0] 
        '''
        if self.synt_source == 'CAP':
            self.t_add = Container['network']['CAP_t_add']
        elif self.synt_source == 'Pyrocko':
            self.t_add = Container['network']['Pyrocko_t_add']
        '''
        
        # station selection
        self.Station_Selection = Container['network']['trace_selection'] 
        
        # preprocessing
        self.Filter = self.get_filter()
        self.Resampling = Container['preprocessing']['resampling']
        self.Detrend = Container['preprocessing']['detrend']
        self.Taper = Container['preprocessing']['taper']
        self.pick = Container['preprocessing']['picker']['perform']
        self.pFilter = Container['preprocessing']['picker']['filter']
        self.pWind = Container['preprocessing']['picker']['STA-LTA']
        self.pThres = Container['preprocessing']['picker']['threshold']
        self.t_noise = Container['preprocessing']['picker']['t_noise']
        self.snWind = Container['preprocessing']['s2n']['window']
        self.stat_az = Container['network']['STAT_dict'] 
        
        # plot
        self.plot = Container['plotter']['waveforms_preproc']
        self.pplot = Container['plotter']['picker']
    
    def plot_raw_waveforms(self,gs,st):
        for tri,tr in enumerate(st):
            npts = tr.stats.npts
            sr = tr.stats.sampling_rate
            t = np.arange(npts)/sr
            
            ax = plt.subplot(gs[0, tri])
            plt.plot(t,tr.data)
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.ylabel('Displacement in m')
            plt.title(tr.id)
             
            ax = plt.subplot(gs[1, tri])
            freqs = np.fft.fftfreq(npts, 1/sr)
            idx = np.argsort(freqs)
            ps = np.abs(np.fft.fft(tr.data))**2
            plt.loglog(freqs[idx], ps[idx],'k')   
            #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.axvspan(self.Filter['freqmin'], self.Filter['freqmax'], facecolor='r', alpha=0.25)
            plt.ylabel('Power spectral density')
            plt.xlabel('Time in s')
            
    def plot_filtered_waveforms(self,gs,st,fi,fb_text):
        for tri,tr in enumerate(st):
            npts = tr.stats.npts
            sr = tr.stats.sampling_rate
            t = np.arange(npts)/sr
            
            ax = plt.subplot(gs[3, tri])
            plt.plot(t,fi*2+tr.data/np.max(tr.data))
            plt.text(len(tr.data)*0.75/sr,fi*2+0.5,fb_text,fontsize=5,bbox=dict(facecolor='red', alpha=0.25))
            if fi == 0:
                plt.ylabel('Displacement in m')
                plt.xlabel('Time in s')
    
    def get_filter(self):
        if self.sim_doublet:
            FB1 = self.Container['preprocessing']['doublet_filter']['F1']
            FB2 = self.Container['preprocessing']['doublet_filter']['F2']
            Filter = {'ftype':'bandpass',
                      'freqmin':FB1['freqmin'],
                      'freqmax':FB2['freqmax'],
                      'corners':FB1['corners'],
                      'zerophase':FB1['zerophase'],
                      'partition_type':'ufix',
                      'fcut':len(FB2['fcut'])*[FB1['freqmin']]} 
            
        else:
            Filter = self.Container['preprocessing']['filter'] 
        return Filter
    
    def Picker(self,st,gs,tcutoff):
        ftype=self.pFilter['ftype']
        freqmin=self.pFilter['freqmin']
        freqmax=self.pFilter['freqmax']
        corners=self.pFilter['corners']
        zerophase=self.pFilter['zerophase']
        [sta,lta] = self.pWind
        [thresA,thresB] = self.pThres
        
        if self.pick:
            st.filter(ftype, freqmin=freqmin, freqmax=freqmax, corners=corners, zerophase=zerophase)   
            df = st[0].stats.sampling_rate
            if np.any(st[0].data):
                cft = classic_sta_lta(st[0].data, int(sta*df), int(lta*df))
                if self.pplot:
                    plot_trigger(st[0], cft, thresA, thresB)
                try:
                    Ponset = np.where(cft>=thresA)[0][0]/df
                except:
                    print('Warning: Auto-picker could not find p onset.')
                    print('Set assumed noise duration as onset time.')
                    Ponset = self.t_noise-tcutoff
            else:
                Ponset = st[0].stats.starttime+self.t_noise-tcutoff # st[0].stats.starttime
        else:
            Ponset = st[0].stats.starttime+self.t_noise-tcutoff # st[0].stats.starttime+self.t_noise
        
        # save onset time in onset stream (only used in weight
        st[0].stats.onset = Ponset
        
        if self.plot:
            for tri,tr in enumerate(st):
                ax = plt.subplot(gs[2, tri])
                t = np.arange(tr.stats.npts)*tr.stats.delta
                plt.plot(t,tr.data/np.max(tr.data))
                plt.plot(Ponset,0,'ko')
        return Ponset, st
    
    def get_weight(self,st,gs,fii): 
        '''
            TODO
            s2n at least 3
        '''
        weight = {}
        onset = {}
        stat_S2N = 0.0
        
        # taper waveform
        st = Stream_taper(st,self.Taper)
        
        # get signal onset time
        df = st[0].stats.sampling_rate
        onset_UTC = st[0].stats.onset
        if self.pick:
            # onset_UTC is of type float not obspy utcdatetime
            onset = int((onset_UTC)*df)
        else:
            starttime_UTC = st[0].stats.starttime
            onset = int((onset_UTC-starttime_UTC)*df)
        comp_sum = 0.0
        t_safty = 15
        
        # calc S2N for each trace 
        for tri,tr in enumerate(st):
            # we assume pure noise signal before onset plus additional 5 seconds
            Noise = tr.data[0:onset-int((self.snWind[0]+t_safty)*df)]
            # we assume the transient signal to be up to 1 min after the onset
            Signal = tr.data[onset-int(t_safty*df):onset+int(self.snWind[1]*df)]

            # we also include the full window to account for misspicking
            Full_Wave = tr.data
            # calc Signal to Noice ration (S2N) --> S2N = 1-S/N = (S-N)/S
            #S2N = 1-np.mean(np.sqrt(Noise**2))/np.mean(np.sqrt(Signal**2))
            S2N = np.mean(np.sqrt(Signal**2))/np.mean(np.sqrt(Noise**2))
            # calc full window weight
            Ww = 1-np.mean(np.sqrt(Full_Wave**2))/S2N
            # create weight dictionary
            weight[tr.stats.channel[-1]] = S2N 
            comp_sum += S2N
            
            if self.plot:
                if fii == 0:
                    ax = plt.subplot(gs[2, tri])
                    plt.axvspan(0, len(Noise)/df, facecolor='r', alpha=0.5)
                    plt.axvspan(len(Noise)/df,(len(Noise)+len(Signal))/df, facecolor='b', alpha=0.5)
                    plt.text(0,0.85,'S2N = '+str(round(S2N,2)),bbox=dict(facecolor='white', alpha=0.25))
                    
        # component wise S2N weight (at the respected station)
        weight['full'] = comp_sum/3
                
        return weight
    
    def process_traces(self,st):
        '''
            trace processing workflow
        '''
        # create empty dictionary
        waveform = {}
        noise = {}
        winfo = {}

        # get number of filter pertubations, is set to none only full band is used
        if self.Filter['partition_type'] == None:
            FN = [0]
        else:
            FN = list(map(int,list(np.arange(0,len(self.Filter['fcut']),1))))
        
        # plotting
        if self.plot:
            fig = plt.figure(figsize=(20, 25), facecolor='w', edgecolor='k')
            gs = gridspec.GridSpec(4,3)
            self.plot_raw_waveforms(gs,st)
        else:
            gs = None
        
        # preprocess data
        st = Stream_detrend(st,self.Detrend)
        st_onset = st.copy()
        
        # get onset
        tcutoff = UTCDateTime(self.src_time) - st[0].stats.starttime
        Ponset, st_onset = self.Picker(st_onset,gs,tcutoff)  
        st[0].stats.onset = Ponset
        
        # trace counter
        tcounter = 0
        
        # create multi-filter dictionary
        for fii,fi in enumerate(FN): # +1 for full bandpass range
            st1 = st.copy()
            st1, fb_text = Stream_filter(st1,self.Filter,fi)
            #fb_text='raw'
            st1 = Stream_resample(st1,self.Resampling)
            st1 = Stream_taper(st1,self.Taper)
                        
            # plot information
            if self.plot:
                self.plot_filtered_waveforms(gs,st1,fi,fb_text)
            
            #get signal to noise
            winfo[fi] = self.get_weight(st1,gs,fii)
            if len(st1) == 3:
                V=np.abs(st1[0].data)
                H=(np.abs(st1[1].data)+np.abs(st1[2].data))/2
                winfo[fi]['H2V'] = round(np.sum(H)/np.sum(V),3)
            elif len(st1) == 1:
                winfo[fi]['H2V'] = 100
            
            # noice stream
            st1_noise = st1.copy()
            
            # check starttime and cut to event origin time (same time will be set for the synthetics)
            if st1[0].stats.starttime < UTCDateTime(self.src_time):
                st1.trim(UTCDateTime(self.src_time)-self.t_add,st1[0].stats.endtime)
                st1_noise.trim(st1_noise[0].stats.starttime,UTCDateTime(self.src_time)-self.t_add)
            elif st1[0].stats.starttime > UTCDateTime(self.src_time):
                print('Time issue for station '+st[0].id)
                print(str(st1[0].stats.starttime)+' v.s. '+str(UTCDateTime(self.src_time)))
                print('Observed trace is cut after onset time. Please cut longer time window!')
                raise SystemExit
            
            # create waveform distionary
            for tri, tr in enumerate(st1):
                comp = tr.stats.channel[-1] 
                if comp not in waveform.keys():
                    waveform[comp] = {}
                waveform[comp][fi] = tr
                tcounter += 1
            
            # create noise distionary
            for tri, tr in enumerate(st1_noise):
                comp = tr.stats.channel[-1] 
                if comp not in noise.keys():
                    noise[comp] = {}
                noise[comp][fi] = tr
            
        winfo['tcounter'] = tcounter
        if self.plot:
            filepath = self.path2plot
            fig.savefig(filepath+st[0].id+'.png',bbox_inches = 'tight')
            
            
        return waveform, noise, winfo
    
    def load_obs(self):
        '''
            Load observations from local directory
        '''
        
        # initiat function
        waveform = {}
        noise = {}
        winfo = {}
        
        # get file list
        wave_files = [f for f in listdir(self.data_path) if isfile(join(self.data_path, f)) and f[-1] == 'Z']
        
        if not wave_files:
            print('Observation files not found!')
            print('Code searches for files ending with "Z"')
            print('Check it naming follows the pattern: XX.ABC..BHZ')
            raise SystemExit
        
        # station index counter
        stat_counter = 0
        
        for wfi, files in enumerate(wave_files):
            info = files.split('.')
            # load only pre-selected stations (from Station_List_Hand)
            if info[0]+'_'+info[1] in self.Station_Selection.keys():
                # define stat_id
                stat_id = info[0]+'_'+info[1]
                # read in waveforms station-wise
                st = read("".join([self.data_path,files[:-1]+"Z"]))
                try:
                    # try to read horizontal components
                    st += read("".join([self.data_path,files[:-1]+"N"]))
                    st += read("".join([self.data_path,files[:-1]+"E"]))                     
                    # 3 components
                    st_dim = 3
                except:
                    # 1 component (vertical)
                    st_dim = 1
                
                if st_dim == 3:
                    baz = self.stat_az['F1_X0'][stat_id][5]
                    st = Stream_rotation(st,baz)
                
                data, ndata, info = self.process_traces(st)
                if info['tcounter'] > 0:
                    stat_counter += 1
                    waveform[stat_id], noise[stat_id], winfo[stat_id] = data, ndata, info
                   
        winfo['stat_counter'] = stat_counter

        return waveform, noise, winfo
    
    
    

def Quick_Loader(Json_File):
    '''
    
    '''
    try:
        from py_src.Modeller import Inversion_Preprocessor
    except:
        from .Modeller import Inversion_Preprocessor
    
    # case: dictionary or path
    if isinstance(Json_File, str):  
        import json
        with open(Json_File) as json_data_file:
            Container = json.load(json_data_file)
    else:
        Container = Json_File

    gen = Container['general']
    if gen['synt_source'] == 'CAP':
        # local CAP database
        fundamentals = CAP(Container=Container).create_6_fund_database() 
        # pick dictionary (not implemented for CAP)
        pick_dict = None
    elif gen['synt_source'] == 'Pyrocko':
        # pyrocko
        fundamentals, pick_dict = pyrocko_database(Container=Container).simulate_fund()
    elif gen['synt_source'] == '3D':
        # local 3D waveform database
        fundamentals = waveform_3D(Container=Container).create_6_fund_database()
        # pick dictionary TODO
        pick_dict = None
        
    # prepare database (incl. preprocessing corresponding to obs)
    fsynt = Synthetics_Data_Loader(Container=Container).create_multiband_database(fundamentals)
        
    # load observables
    waveform, noise, winfo = Observable_Data_Loader(Container=Container).load_obs()    
    
    '''
    tr = waveform['CU_ANWB']['Z'][0]
    t = np.arange(tr.stats.npts)*tr.stats.delta
    plt.plot(t,tr.data,'k-')
    print(tr.stats.starttime)
    '''
    
    # preprocess traces
    wave_preproc = Inversion_Preprocessor(Container=Container)
    OBS, SYNT, inv_pre = wave_preproc.organizer(waveform,fsynt,winfo,pick_dict)
    
    '''
    tr = OBS['CU_ANWB']['Z'][0]
    t = np.arange(tr.stats.npts)*tr.stats.delta
    plt.plot(t,tr.data,'r--')
    plt.xlim((0,260))
    plt.grid(True)
    plt.show()
    '''
    
    return Container, OBS, SYNT, noise, inv_pre, winfo
    
    
    
