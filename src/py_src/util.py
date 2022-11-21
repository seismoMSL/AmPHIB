#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    Mike Lindner (mike.lindner@kit.edu), 2018
"""
# standart libaries
import numpy as np
import matplotlib
from matplotlib.widgets import Slider
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fft2, ifft2, fftshift
from scipy import signal,stats
from scipy.signal import hilbert
import time
import json
# tkinter
from functools import partial
import tkinter as tk
# Obspy
from obspy import UTCDateTime, Stream, read, Trace
from obspy.geodetics.base import gps2dist_azimuth
# os
import os
from os import listdir
from os.path import isfile, join, isdir
# pathlib
from pathlib import Path
# external functions and libaries
#from Utils.mopad_py3 import MomentTensor
#from Utils.Logger import *
#from Utils.calc_Kagan import get_kagan_angle
from .Utils.tape2015 import cmt2tt15, tt152cmt, tt2cmt, cmt2tt
from .Utils.cnv import read_cnv


################################################
'''
    JSON-Creator helper functions
'''
################################################


def get_fund_depth_list_CAP(data_path,out=False):
    '''
        get list of available synthetic depth information
    '''
    if data_path != None: 
        synt_files = [f for f in listdir(data_path)]
        database = {}
        for sf in synt_files:
            depth = float(sf.split('_')[-1])
            database[depth] = data_path+'Fund_Events_'+str(depth)+'/'
            if out:
                print('Key: ',depth,' (event depth)')
                print('Path: ',database[depth])
                for fund_mech in [f for f in listdir(database[depth])]:
                    dsize = sum(d.stat().st_size for d in os.scandir(database[depth]+'/'+fund_mech+'/') if d.is_file())
                    print('Subfolder:',fund_mech,' (Size: ',round(dsize/(2**20),2),'MB)')
                print()
        return database            
    else:
        return None
    
def get_event_id_path(data_path,out=False):
    '''
        get path to choosen event
    '''
    event_files = [f for f in listdir(data_path)]
    database = {}
    for ev in event_files:
        if isdir(data_path+ev):
            ev_id = ev.split('_')[0]
            database[ev_id] = "".join([data_path,ev])+'/'
            if out:
                _, _, files = next(os.walk(database[ev_id]))
                NFiles = len(files)
                dsize = sum(d.stat().st_size for d in os.scandir(database[ev_id]+'/') if d.is_file())
                print('Key: ',ev_id,' (event id)',' | Number of files: ',NFiles,' | Size: ',round(dsize/(2**20),2),'MB')
                print('Path: ',database[ev_id])
    return database

def event_reader(filename,out=False,plot=False,depth=[0.,1000.],lat=[-90.,90.],lon=[-180.,180.],mag=[-5.,10.]):
    # open txt file
    with open(filename, "r") as f:
        content = f.readlines()
    f.close()
    # get info
    event_list = {}
    for ci, info in enumerate([line.rstrip('\n') for line in content]):
        time_raw = str(UTCDateTime(info.split()[6]))
        key = info.split()[0]#+'_'+time_raw
        time = time_raw
        loc = [float(info.split()[2]),float(info.split()[3]),float(info.split()[4])]
        magnitude = float(info.split()[5])
        event_list[key] = [time,loc,magnitude] 
        if out:
            if lat[0] <= loc[0] <= lat[1] and lon[0] <= loc[1] <= lon[1] and depth[0] <= loc[2] <= depth[1] and mag[0] <= magnitude <= mag[1]:
                print(info)
        if plot:
            if lat[0] <= loc[0] <= lat[1] and lon[0] <= loc[1] <= lon[1] and depth[0] <= loc[2] <= depth[1] and mag[0] <= magnitude <= mag[1]:
                print('TODO: diplay map of events. (colder-coding, etc.?)')
    return event_list

def station_reader(filename,event_list,out=False,plot=False,event_id=None,dist_p=[]):
    # open txt file
    with open(filename, "r") as f:
        content = f.readlines()
    f.close()
    # get info
    station_list = {}
    network_dict = {}
    for info in [line.rstrip('\n') for line in content]:
        key = info.split()[1]+'_'+info.split()[0]
        station_list[key] = [float(info.split()[2]),float(info.split()[3]),float(info.split()[5])/1000] 
        if info.split()[1] not in network_dict:
            network_dict[info.split()[1]] = []
        network_dict[info.split()[1]].append(info.split()[0])
        if out:
            print(info)           
    if plot:
        fig = plt.figure(figsize=(20, 20), facecolor='w', edgecolor='k')
        ax = plt.subplot()
        for key in station_list:
            plt.plot(station_list[key][1],station_list[key][0],'kd')
            plt.text(station_list[key][1],station_list[key][0],key)
        if event_id == None:
            # case plot all events from event list
            for event_id in event_list:
                plt.plot(event_list[event_id][3],event_list[event_id][2],'kd')
                plt.text(event_list[event_id][3],event_list[event_id][2],event_id) 
        else:
            # case select one event and add distance circle to the plot
            src_lat,src_lon = event_list[event_id][1][0],event_list[event_id][1][1]
            plt.plot(src_lon,src_lat,'ro')
            plt.text(src_lon,src_lat,event_id) 
            # get circle dimension
            DIST = []
            Ndist = 5
            for key in station_list:
                stat_lat,stat_lon = station_list[key][0],station_list[key][1]
                distance, _,_ = gps2dist_azimuth(src_lat,src_lon,stat_lat,stat_lon) 
                DIST.append(distance)
            DIST = np.asarray(DIST)
            # plot evenly spaced distance plots
            if len(dist_p) == 0: 
                dist_p = np.linspace(np.min(DIST), np.max(DIST), Ndist, endpoint=True)
            else:
                dist_p = list(np.asarray(dist_p)*1000.)
            for dist in dist_p:
                dist /= (1000.*111.)
                circle = plt.Circle((src_lon,src_lat), dist, color='b', fill=False)
                ax.add_artist(circle)
                plt.text(src_lon+dist,src_lat,str(round(dist*111.,1))+' km',color='b',alpha=0.5,ha='center')
        plt.xlabel('Lon')
        plt.ylabel('Lat')
        #plt.legend()
        plt.axis('scaled')
        plt.grid(True)
        plt.show()
    return station_list, network_dict


def load_dataset_identifier(filename,out=False):
    '''
        load content of Database_Identifier
        --> get reference set_id of 3D synthetics
        --> folder of synthetic waveforms should be named after the set_id
    '''
    if filename is not None:
        # open txt file
        with open(filename, "r") as f:
            content = f.readlines()
        f.close()
        # get info
        id_list = {}
        for ci, info in enumerate([line.rstrip('\n') for line in content]):
            set_id = info.split()[0]
            loc = [float(info.split()[1]),float(info.split()[2]),float(info.split()[3])]
            ref_mag =  info.split()[4]
            M0 = info.split()[5]
            id_list[set_id] = {'location':loc,'Mw':ref_mag,'M0':M0}
        if out:
            print('set_id | [lat,lon,depth]     reference_Mw    ref_moment')
            for set_id in id_list:
                print(set_id,' | ',id_list[set_id]['location'],id_list[set_id]['Mw'],id_list[set_id]['M0'])
    else:
        id_list = None
    return id_list
        

def recomment_set(DSI,path2synt,F1_loc,F2_loc=None,out=False):
    '''
    
    '''
    # get list of available synthetics
    avail_synt = os.listdir(path2synt)
    
    # recommendation loop
    rec_dict = {'F1':[],'F2':[]}
    for set_id in DSI:
        loc = DSI[set_id]['location']
        ed_F1,_,_ = gps2dist_azimuth(loc[0],loc[1],F1_loc[0],F1_loc[1])
        hp_F1 = np.sqrt(ed_F1**2 + (1000.*(loc[2]-F1_loc[2]))**2)
        if set_id in avail_synt: # check if set is available
            rec_dict['F1'].append([hp_F1,set_id])
        if F2_loc is not None:
            ed_F2,_,_ = gps2dist_azimuth(loc[0],loc[1],F2_loc[0],F2_loc[1])
            hp_F2 = np.sqrt(ed_F2**2 + (1000.*(loc[2]-F2_loc[2]))**2)
            if set_id in avail_synt: # check if set is available
                rec_dict['F2'].append([hp_F2,set_id])
    try:
        if F2_loc is not None:
            recommendation = {'F1':sorted(rec_dict['F1'], key=lambda x: x[0]),'F2':sorted(rec_dict['F2'], key=lambda x: x[0])}
        else:
            recommendation = {'F1':sorted(rec_dict['F1'], key=lambda x: x[0]),'F2':[[None,None]]}
    except:
        print('No data found!')
        raise SystemExit
    
    if out:
        print('Recommended synthetics: ')
        print('F1: ',recommendation['F1'][0][1],' with a distance to the real hypocenter of ',round(recommendation['F1'][0][0]/1000.,2),' km')
        if F2_loc is not None:
            print('F2: ',recommendation['F2'][0][1],' with a distance to the real hypocenter of ',round(recommendation['F2'][0][0]/1000.,2),' km')
    
    return recommendation
    
    

def selected_Stations0(Stat_file_selected):
    '''
        This function returns a dictionary of the hand checked stations,
        in the simulation and error calculation, only those stations are used
    '''
    selected_Station_list = {}
    with open(Stat_file_selected, 'r') as f:
        StatInfo = f.readlines()
    f.close()
    # get info
    for line in [line.rstrip('\n') for line in StatInfo]:
        info = line.split()
        st_id = []
        for st in info[2:]:
            st_id.append(st)
        # create sub dict for different components at the same event id 
        if info[0] not in selected_Station_list.keys(): 
            # check if event id already exists
            selected_Station_list[info[0]] = {}
        selected_Station_list[info[0]][info[1]] = st_id #info[2:]

    return selected_Station_list

def strip_empties_from_list(data):
    '''
        https://stackoverflow.com/questions/33529312/remove-empty-dicts-in-nested-dictionary-with-recursive-function
    '''
    new_data = []
    for v in data:
        if isinstance(v, dict):
            v = strip_empties_from_dict(v)
        elif isinstance(v, list):
            v = strip_empties_from_list(v)
        if v not in (None, str(), list(), dict(),):
            new_data.append(v)
    return new_data

def strip_empties_from_dict(data):
        '''
            https://stackoverflow.com/questions/33529312/remove-empty-dicts-in-nested-dictionary-with-recursive-function
        '''
        new_data = {}
        for k, v in data.items():
            if isinstance(v, dict):
                v = strip_empties_from_dict(v)
            elif isinstance(v, list):
                v = strip_empties_from_list(v)
            if v not in (None, str(), list(), dict(),):
                new_data[k] = v
        return new_data


def selected_traces_from_file(filename,event_list,station_list,
                              distance_range=[0,10**6],rm_stations=[],out=False,out_auto=False):
    '''
        This function returns a dictionary of the hand checked stations,
        in the simulation and error calculation, only those stations are used
    '''
    selected_Station_list = {}
    with open(filename, 'r') as f:
        StatInfo = f.readlines()
    f.close()
    # get info
    for line in [line.rstrip('\n') for line in StatInfo]:
        info = line.split()
        # create sub dict for different components at the same event id 
        if info[0] not in selected_Station_list.keys(): 
            # check if event id already exists
            selected_Station_list[info[0]] = {'auto':{}}
        # phase selection identifier
        pwsel = lambda x: x == '1'
        st_id = {}    
        for st in info[2:]:
            cont = st.split('-')
            if len(cont) == 2: # trace selection
                st_id[cont[0]] = list(cont[1])
            elif len(cont) == 3: # phase selection
                st_id[cont[0]] = {}
                for ci,comp in enumerate(list(cont[1])):
                    st_id[cont[0]][comp] = []
                    if pwsel(cont[2][2*ci]): 
                        st_id[cont[0]][comp].append('P')
                    if pwsel(cont[2][2*ci+1]):
                        st_id[cont[0]][comp].append('S')
        
        # append to selection dichtionary
        selected_Station_list[info[0]][info[1]] = st_id
    
        if out:
            print(info[0],' ',info[1])
            print(info[2:])
            print()
            
    # add default entry for automatic selection (default entry is later reduced to the auto selection)
    for event_id in event_list:
        # check and add new event entry
        [src_lat,src_lon,src_z] = event_list[event_id][1]
        if event_id not in selected_Station_list.keys():
            selected_Station_list[event_id] = {'auto':{}}
        # add default trace selection to auto 
        for stat_id in station_list.keys():
            if stat_id not in rm_stations:
                stat_lat,stat_lon = station_list[stat_id][0],station_list[stat_id][1]
                distance, _,_ = gps2dist_azimuth(src_lat,src_lon,stat_lat,stat_lon) 
                if distance_range[0] <= distance/1000 <= distance_range[1]:
                    selected_Station_list[event_id]['auto'][stat_id] = ['Z','R','T']
        if out_auto:
            print(event_id,'  auto')
            print(selected_Station_list[event_id]['auto'])
            print()
            
    # clean dictionary from empty lists/dicts
    selected_Station_list = strip_empties_from_dict(selected_Station_list)
            
    return selected_Station_list
    
def get_event_id_from_time(event_list,time):
    '''
        input
        event_list: dict following src['event_list']
        time: string z.B. '2020-01-31T21:05:33.000000Z'
        output
        [event_id, difference to listed time in sec]
    '''
    time_list, id_list = [], []
    for event_id in event_list:
        id_list.append(event_id)
        time_list.append(np.abs(UTCDateTime(event_list[event_id][0])-UTCDateTime(time)))
    idx_min = np.argmin(np.asarray(time_list))
    return [id_list[idx_min],time_list[idx_min]]


def get_orientations(filename):
    '''
    # Station_code Nominal_north_component_azimuth(degCW) Error(deg), NumMeasurements
        DP01 63.68 4.58 160 41
        DP03 205.58 4.02 172 41
    # Note:
    # DEPAS instruments use the “left-hand rule” so BH2 is 90 degrees clockwise of BH1, and BH1 is the nominally north component. Therefore, the azimuth is # given for *BH1*
    # SIO instruments use the “right-hand rule” so BH2 is 90 degrees anti-clockwise of BH1, and BH2 is the nominally north component. Therefore, the azimuth below is given for *BH2*
    '''

    with open(filename, "r") as f:
        content = f.readlines()
    f.close()
    # get info
    orientations = {}
    for ci, info in enumerate([line.rstrip('\n') for line in content]):
        if not info.startswith('#'): # comment
            [stat,alpha,dalpha,N,xxx] = info.split() # xxx undefined entry
            orientations[stat] = [float(alpha),float(dalpha)]
            ## aloha is negative as we are rotationg back
            #if stat[:2] == 'DP': # DEPAS instruments
            #    orientations[stat] = [-float(alpha),float(dalpha)] 
            #elif stat[:2] == 'SI': # SIO instruments 
            #    orientations[stat] = [-float(alpha),float(dalpha)]
    return orientations
    

def update_3D_source_location(Container):
    '''
    
    '''
    F1_info = Container['source']['F1_loc']
    F1_id = Container['source']['F1_set_id_3d']
    F1_info[1] = Container['path']['DSI_dict'][F1_id]['location']
    Mw = Container['path']['DSI_dict'][F1_id]['Mw']
    if Container['source']['simulate_doublet']:
        F2_info = Container['source']['F2_loc']
        F2_id = Container['source']['F2_set_id_3d']
        F2_info[1] = Container['path']['DSI_dict'][F1_id]['location']
        if Container['path']['DSI_dict'][F2_id]['Mw'] != Mw:
            print('Warning: Synthetic Mw of F1 and F2 are different!')
    else:
        F2_info = F1_info
    
    return {'F1_loc':F1_info,'F2_loc':F2_info,'ref_mag':Mw}
    

class Load_Picks:
    
    def __init__(self,Container=None):
        '''
        
        '''
        self.src_info = Container['source']['F1_loc']
        self.Networks = Container['network']['netw_id']
        self.cnv_file = Container['path']['cnv_file']
        self.t_cluster = Container['source']['phase_onset']['src_cluster']
        self.pw_width = Container['source']['phase_onset']['width']
        

    def get_ev_sub_list(self,catalog):
        '''
        
        '''
        [slat,slon,sdepth] = self.src_info[1]
        ev_sub_list = [self.src_info[0],]
        for ii,cat in enumerate(catalog):
            time = cat.origins[0].time
            lat = cat.origins[0].latitude
            lon = cat.origins[0].longitude
            depth = cat.origins[0].depth
            Epi_dist,_,_ = gps2dist_azimuth(slat,slon,lat,lon)
            Hypo_dist = np.sqrt(Epi_dist**2 + (1000.*sdepth-depth)**2)
            if Hypo_dist <= self.t_cluster['radius']*1000.:
                ev_sub_list.append(str(time))
        return ev_sub_list
    
    def get_station_id(self,stat):
        '''
        
        '''
        # remove whitespaces in stat string
        stat = stat.replace(" ", "")
        # check if stat is in network list
        for netw in self.Networks:
            if stat in self.Networks[netw]:
                return netw+'_'+stat
            else:
                continue
        print('Station '+stat+' has no network id.')
        raise SystemExit       
    
    def from_cnv(self,display_tt,display_catalog):
        '''
        
        '''
        # read cnv file
        try:
            catalog = read_cnv(self.cnv_file, yr_prefix=2000)
        except:
            print('File does not exist!')
            raise SystemExit
        
        # display content of .cnv
        if display_catalog:
            print(catalog)
        
        # get event sub list
        ev_sub_list = self.get_ev_sub_list(catalog)
        # create dictionary
        t_travel = {} # phase travel time
        ev_weight_dict = {} # station weight
        t_onset = {} # onset time in UTC
        # loop cataloge
        for ii,cat in enumerate(catalog):
            time = cat.origins[0].time
            lat = cat.origins[0].latitude
            lon = cat.origins[0].longitude
            depth = cat.origins[0].depth
            if str(time) in ev_sub_list:
                for pi, pick in enumerate(cat.picks):
                    stat = self.get_station_id(pick.waveform_id['station_code'])
                    tp_id = pick.phase_hint
                    tp = pick.time
                    if stat not in t_travel:
                        t_travel[stat] = {}
                        ev_weight_dict[stat] = {}
                        t_onset[stat] = {}
                    if tp_id not in t_travel[stat]:
                        t_travel[stat][tp_id] = []
                        ev_weight_dict[stat][tp_id] = []
                        t_onset[stat][tp_id] = []
                    t_travel[stat][tp_id].append(tp-time)
                    if str(time) == str(self.src_info[0]):
                        ev_weight_dict[stat][tp_id].append(self.t_cluster['weight'][0])
                        t_onset[stat][tp_id].append(str(tp))
                    else:
                        ev_weight_dict[stat][tp_id].append(self.t_cluster['weight'][1])
                        
                    
        # get weighted traveltime of picked p phase
        tp_dict = {}
        tpv_dict = {}
        t_N = {}
        for stat in t_travel:
            tp_dict[stat] = {}
            tpv_dict[stat] = {}
            t_N[stat] = {}
            for tp_id in t_travel[stat]:
                weights = ev_weight_dict[stat][tp_id]
                if len(t_onset[stat][tp_id]) == 0: # case pick is not available
                    tp_dict[stat][tp_id] = np.average(t_travel[stat][tp_id], weights=weights)
                    tpv_dict[stat][tp_id] = np.average((t_travel[stat][tp_id]-tp_dict[stat][tp_id])**2, weights=weights)
                    t_N[stat][tp_id] = len(t_travel[stat][tp_id])
                    t_onset[stat][tp_id].append(str(UTCDateTime(self.src_info[0]) + tp_dict[stat][tp_id]))
                else: # case: pick exists
                    indx = np.argwhere(np.asarray(weights)==self.t_cluster['weight'][0])[0][0]
                    tp_dict[stat][tp_id] = t_travel[stat][tp_id][indx]
                    tpv_dict[stat][tp_id] = 0
                    t_N[stat][tp_id] = 1
                
        # output
        output = {'t_travel':t_travel,
                  'weights':ev_weight_dict,
                  'event_times':ev_sub_list,
                  't_average':tp_dict,
                  't_variance':tpv_dict,
                  't_N':t_N,
                  't_onset':t_onset,
                  'pw_window':self.pw_width}
        
        if display_tt:
            self.display_travel_times(output)
        
        return output

    def display_travel_times(self,output):
        '''
        
        '''
        fig = plt.figure(figsize=(20, 5), facecolor='w', edgecolor='k')
        ax = plt.subplot(1,1,1)
        
        tt_list = []
        for si,stat_id in enumerate(output['t_average']):
            if (si % 2) == 0:
                ax.axvspan(si-0.5, si+0.5, alpha=0.25, color='green')
            else:
                ax.axvspan(si-0.5, si+0.5, alpha=0.25, color='yellow')
            
            try:
                key = output['t_average'][stat_id]['P']
            except:
                key = output['t_average'][stat_id]['S']
            cont = []
            for phase_id in output['t_average'][stat_id]:
                cont.append(output['t_average'][stat_id][phase_id])
                cont.append(output['t_variance'][stat_id][phase_id])
                cont.append(output['t_N'][stat_id][phase_id])
            tt_list.append([key,cont,stat_id])
        
        stat_list = []
        for ci, lcont in enumerate(sorted(tt_list, key=lambda x: x[0])):
            text_N = ''
            l1, = plt.plot(ci, lcont[1][0],'bo',  markersize=10) # P
            plt.errorbar(ci, lcont[1][0], yerr=lcont[1][1], color='k')
            text_N += str(lcont[1][2])
            try:
                l2, = plt.plot(ci, lcont[1][3],'ro',  markersize=10) # S
                plt.errorbar(ci, lcont[1][3], yerr=lcont[1][4], color='k')
                text_N += '|'+str(lcont[1][5])
            except:
                continue
            stat_list.append(lcont[2]+' (N='+text_N+')')

        
        plt.xticks(np.arange(len(stat_list)), stat_list, rotation='vertical',fontsize=18)
        plt.ylabel('Travel time in s',fontsize=18)  
        plt.yticks(fontsize=15)
        
        plt.legend([l1,l2],['P-Onset','S-Onset'],fontsize=15)
        
        plt.grid(True)
        plt.xlim((-0.5,si+0.5))

        plt.show()


    
    def get_picks(self,file_type=None,display_tt=False,display_catalog=False):
        '''
        
        '''
        if file_type == 'cnv':
            if self.cnv_file != None:
                return self.from_cnv(display_tt,display_catalog)
            else:
                print('cnv file does not exist!')
                return None
        else:
            return None


def load_signal_window_file(path2PWfile):
    '''
        
    '''
    cont = {}
    with open(path2PWfile, 'r') as f:
        content = f.readlines()
        content = [line.rstrip('\n') for line in content]
    f.close()
    for li, line in enumerate(content):
        info = line.split()
        cont[info[0]] = {}
        for stat_info in info[1:]:
            [key,t1,t2] = stat_info.split('=')
            cont[info[0]][key] = [t1,t2]
    return cont


def create_output_folder(output_folder,sub_folder):
    '''
    
    '''
    # check if output_folder exists
    if not os.path.exists(output_folder):
        print(output_folder+' does not exist!')
        return None
    else:
        if not os.path.exists(output_folder+'/'+sub_folder):
            sub_dir = os.makedirs(output_folder+'/'+sub_folder)
            print('Creating '+output_folder+sub_folder)
            return output_folder+sub_folder+'/'
        else:
            print(sub_folder+' already exists.')
            return output_folder+sub_folder+'/'


################################################
'''
    manage station settings
'''
################################################
    
    
    
class get_Station_information:
    
    def __init__(self,Container=None):
        # data container and event information
        self.Container = Container
        self.Networks = Container['network']['netw_id']
        self.synt_source = Container['general']['synt_source']
        self.trace_selection = self.Container['network']['trace_selection'].keys()
        
        # simulate doublet
        self.sim_doublet = Container['source']['simulate_doublet']
        
        # gobal station file
        self.stat_glob = Container['network']['stat_global']
        self.distance_range = Container['network']['distance_range']
        
        # event location and reference information (use copy() else loc list will be overwritten)
        self.misloc_fault_id = Container['source']['misloc_fault_id']
        self.event_depth_F1 = Container['source']['event_depth']
        self.event_depth_F2 = Container['source']['doublet_event_depth']
        self.src_loc_dict = self.get_src_loc()
        
        # magnitude dict
        self.src_mag_dict = {'F1_X0':self.Container['source']['F1_loc'][2],
                             'F2_X0':self.Container['source']['F2_loc'][2]}
        
        # origin time dict
        self.src_time_dict = {'F1_X0':self.Container['source']['F1_loc'][0],
                              'F2_X0':self.Container['source']['F2_loc'][0]}
        
        # station misrotation
        self.daz = Container['source']['daz']
        
        # station delay time
        self.t_del = Container['source']['t_del']

        # get global information
        self.stat_loc_dict = self.get_stat2src_loc()
        
        
        if self.synt_source == 'CAP':
            # CAP profil settings
            #self.src_loc_circ = [75000, 75000] #Container['network']['synthetics_source_offset']
            self.offset = Container['network']['profil']['offset']
            self.profil_lenght = Container['network']['profil']['profil_lenght']
            self.station_sampling = Container['network']['profil']['station_sampling']
            self.stat_CAP = self.CAP_system_profil()
            # CAP reference list
            self.CAP_LIST = self.get_loc_circ_list()
            self.CAP_path = self.get_path_CAP()
            # set path info of 3D synthetics to None
            self.PATH_3D = None
        elif self.synt_source == '3D':
            # Container info
            self.path2synt = Container['path']['3D_synthetic_data_path']
            self.DSI_dict = Container['path']['DSI_dict']
            self.F1_ID = Container['source']['F1_set_id_3d']
            self.F2_ID = Container['source']['F2_set_id_3d']
            self.fbd_id =  Container['network']['3D_fbd_id']
            # set path info of 3D synthetics
            self.PATH_3D = self.get_3D_path()
            # set CAP to None
            self.CAP_LIST = None
            self.CAP_path = None
        else: # pyrocko
            # set path info of 3D synthetics to None
            self.PATH_3D = None
            # set CAP to None
            self.CAP_LIST = None
            self.CAP_path = None
    
    def check_3D_file(self,path_dict):
        '''
        
        '''
        ISSUE, ICount = [], 0
        for stat_id in path_dict:
            for comp in path_dict[stat_id]:
                for pert in path_dict[stat_id][comp]:
                    path2file = path_dict[stat_id][comp][pert]
                    file_check = Path(path2file)
                    if file_check.exists():
                        ICount += 1
                    else:
                        ISSUE.append(path2file)

        # notification
        if len(ISSUE) > 0:
            print('Caution! Some files are missing.')
            #print(str(int(100*ICount/len(ISSUE)))+' % files are missing:')
            for misfile in ISSUE:
                print(misfile)
    
    
    def get_3D_path(self):
        '''
           return {'F1_X0':{'XX_AB01':{'Z':{'mrr':'path/to/mseed','mtt':'path/to/mseed',...}...}...}...}
        '''
        temp_F1, temp_F2 = {},{}
        for stat_id in self.trace_selection:
            temp_F1[stat_id], temp_F2[stat_id] = {}, {}
            [netw,stat] = stat_id.split('_')  
            for comp in ['Z','N','E']:
                temp_F1[stat_id][comp], temp_F2[stat_id][comp] = {}, {}
                for pert in ['mrr','mtt','mpp','mrt','mrp','mtp']:
                    tr_id = netw+'.'+stat+'..'+self.fbd_id+comp+'.'+pert
                    temp_F1[stat_id][comp][pert] = self.path2synt+self.F1_ID+'/'+tr_id
                    if self.sim_doublet:
                        temp_F2[stat_id][comp][pert] = self.path2synt+self.F2_ID+'/'+tr_id
        
        if self.sim_doublet:
            self.check_3D_file(temp_F1)
            self.check_3D_file(temp_F2)
            return {'F1_X0':temp_F1,'F2_X0':temp_F2}
        else:
            self.check_3D_file(temp_F1)
            return {'F1_X0':temp_F1}

    def get_dX(self,src_loc_init,pertub_id):
        '''
            add pertubation to initial source location
        '''
        dx = self.Container['source']['dx']
        src_loc = src_loc_init.copy()
        if pertub_id == 'dX':
            src_loc[1] += dx[0]/111.
        elif pertub_id == 'dY':
            src_loc[0] += dx[0]/111.
        elif pertub_id == 'dZ':
            src_loc[2] += dx[1]
        return src_loc
    
    
    def get_src_loc(self):
        '''
            returns dictionary for main, duplet and perturbed source location
        '''
        SRC_dict = {}
        
        # sanity-check settings
        if self.Container['source']['solve_for_misloc']:
            if self.sim_doublet:
                if self.misloc_fault_id not in ['F1','F2']:
                    print('Encountered setting issue!')
                    print('Misloc_fault_id does not exist: '+self.misloc_fault_id)
                    print('Change settings in JSON file. Aborting run...')
                    raise SystemExit
            else:
                if self.misloc_fault_id != 'F1':
                    print('Doublet simulation is set to False.')
                    print('Solving for mislocation con only be done for F1.')
                    print('Change settings in JSON file. Aborting run...')
                    raise SystemExit
            
        # F1 location
        src_loc_F1 = self.Container['source']['F1_loc'][1]
        src_loc_F1[2] = self.event_depth_F1 # update depth based on synth depth
        SRC_dict['F1_X0'] = src_loc_F1
        
        
        # solving for misrotation of F1
        if self.Container['source']['solve_for_misrot']:
            SRC_dict['F1_X4'] = src_loc_F1
        elif self.Container['source']['solve_for_delay']:
            SRC_dict['F1_X5'] = src_loc_F1
        else:
            # F1 XYZ-pertubation
            if self.Container['source']['solve_for_misloc']:
                if self.misloc_fault_id == 'F1': 
                    pertub_id = {'1':'dX','2':'dY','3':'dZ'}
                    for pert in pertub_id:
                        src_loc_F1_pert = self.get_dX(src_loc_F1,pertub_id[pert])
                        SRC_dict['F1_X'+pert] =  src_loc_F1_pert
            
            # F2 location
            if self.sim_doublet: 
                src_loc_F2 = self.Container['source']['F2_loc'][1]
                src_loc_F1[2] = self.event_depth_F2 # update depth based on synth depth
                SRC_dict['F2_X0'] = src_loc_F2
                
                # F2 XYZ-pertubation
                if self.Container['source']['solve_for_misloc']:
                    if self.misloc_fault_id == 'F2':
                        pertub_id = {'1':'dX','2':'dY','3':'dZ'}
                        for pert in pertub_id:
                            src_loc_F2_pert = self.get_dX(src_loc_F2,pertub_id[pert])
                            SRC_dict['F2_X'+pert] =  src_loc_F2_pert
                    
        return SRC_dict
        
        
    def get_stat2src_loc(self):
        Station = {}
        for src_id in self.src_loc_dict:
            Station[src_id] = {}
            [src_lat,src_lon,src_depth] = self.src_loc_dict[src_id]
            for stat_id in self.stat_glob:
                [stat_lat,stat_lon,stat_depth] = self.stat_glob[stat_id]
                distance, az, baz_init = gps2dist_azimuth(src_lat,src_lon,stat_lat,stat_lon)  
                # check distance range
                if self.distance_range[0] <= distance/1000 <= self.distance_range[1]:
                    # case: solve for misrotation
                    if self.Container['source']['solve_for_misrot'] and src_id == 'F1_X4':
                        if isinstance(self.daz, dict):
                            baz = baz_init + self.daz[stat_id.split('_')[1]][1] # add pertubation to azimut
                        else: # station wise orientation informations are not available
                            # daz is a scalar
                            baz = baz_init + self.daz # add pertubation to azimut
                        if baz > 360.:
                            baz -= 360
                        elif baz < 0:
                            baz += 360.
                    else:
                        baz = baz_init
                    # add to Station dictionary
                    Station[src_id][stat_id] = [stat_lat,stat_lon,stat_depth,distance,az,baz]
        return Station
         
    

    
    def create_circ_station_dict(self,X,Y,distance):
        Buri = 0.0 # add entry in json to set this parameter for SPECFEM3D input format
        Elev = 0.0 # add entry in json to set this parameter for SPECFEM3D input format
        CIRC = {}
        az_List = ['B','C','A'] #[45,90,0]
        for direc in np.arange(0,len(X[:,1]),1):
            for Stat in np.arange(1,len(X[1,:])+1,1):
                if az_List[direc] == 'A': az = 0.0
                if az_List[direc] == 'B': az = 45.0
                if az_List[direc] == 'C': az = 90.0
                key = 'XX_'+az_List[direc]+'-'+str(Stat)
                CIRC[key] = [az_List[direc]+'-'+str(Stat), 'XX', Y[direc,Stat-1], X[direc,Stat-1], Elev, Buri,distance[Stat-1],az]
        return CIRC

    def CAP_system_profil(self):
        dist_List = np.arange(self.offset*10**3,self.profil_lenght*10**3,self.station_sampling*10**3)
        distance = np.zeros(len(dist_List))
        X,Y=np.zeros([3,len(dist_List)]),np.zeros([3,len(dist_List)])
        for dLi, dist in enumerate(dist_List):
            distance[dLi] = dist
            # 45° for 90SS and 45DS on Z and R
            X[0,dLi], Y[0,dLi] = int(dist/np.sqrt(2)), int(dist/np.sqrt(2))
            #X[0,dLi], Y[0,dLi] = int(self.src_loc_circ[1])+int(dist/np.sqrt(2)), int(dist/np.sqrt(2))
            # 90° for 90DS on Z and R
            X[1,dLi], Y[1,dLi] = int(dist), int(self.offset*10**3)
            #X[1,dLi], Y[1,dLi] = int(self.src_loc_circ[1])+int(dist), int(self.src_loc_circ[0])
            # 0° for 90SS and 90 DS on T
            X[2,dLi], Y[2,dLi] = int(self.offset*10**3), int(dist)  
            #X[2,dLi], Y[2,dLi] = int(self.src_loc_circ[1]), int(self.src_loc_circ[0])+int(dist)
        stat_CAP = self.create_circ_station_dict(X,Y,distance)
        return stat_CAP

    def get_loc_circ_list(self):
        '''
            Output example:
            {'XX_DP06': 405, 'XX_SI07': 1252, 'XX_DP08': 85, ...}
        '''
        loc_circ_list = {}
        dist_loc, dist_circ = [],[]
        
        for ev_id in self.stat_loc_dict:
            loc_circ_list[ev_id] = {}
            
            # get distance infomation of global station network
            for sti, st in enumerate(self.stat_loc_dict[ev_id].keys()):
                if st in self.trace_selection:
                    dist_loc.append([int(self.stat_loc_dict[ev_id][st][3]),st])
                
            # get distance infomation of circular station network     
            for sti, st in enumerate(self.stat_CAP.keys()):
                dist_circ.append([int(self.stat_CAP[st][6]),st])
                
            # get CAP distance index for corresponding source-station pairing  
            CAP_dist = np.arange(self.offset*10**3,self.profil_lenght*10**3,self.station_sampling*10**3)
            for [dloc, st_loc] in dist_loc: 
                idx_min = np.argmin([np.abs(dloc-item) for item in CAP_dist])
                loc_circ_list[ev_id][st_loc] = int(dist_circ[idx_min][1].split('-')[-1])

        return loc_circ_list

    def get_path_CAP(self):
        '''
            create cap path dictionary
        '''
        # path and depth indormation
        path = self.Container['path']['CAP_synthetic_data_path']
        mech = self.Container['path']['CAP_subfolder_structure']
        CAP_pattern = self.Container['path']['CAP_pattern']
        fbd_id = self.Container['network']['cap_fbd_id']
        dz = self.Container['source']['dx'][1]
        F1_depth = self.Container['source']['event_depth']
        F2_depth = self.Container['source']['doublet_event_depth']
        
        # create path dictionary
        CAP_Path = {}
        ISSUE, ICount = [], 0
        
        # path assignment loop
        for src_id in self.CAP_LIST:
            CAP_Path[src_id] = {}
            # manage synthetic depth information
            if src_id[:2] == 'F1':
                d0 = F1_depth
            else:
                d0 = F2_depth
            if src_id[-1] == '3':
                depth = d0+dz
            else:
                depth = d0
            # get path id for selected src_id
            try:
                path_id = path[depth]
            except: # case: reload json, key is a string
                path_id = path[str(depth)]
            # set paths to synthetics
            for stat_id in self.CAP_LIST[src_id]:
                CAP_Path[src_id][stat_id] = {}
                CAP_id = self.CAP_LIST[src_id][stat_id]
                for subf in mech:
                    '''
                        90SS: A,B
                        90DS: A,C
                        45DS: B
                        ISO:  A or B or C (?)
                    '''
                    CAP_Path[src_id][stat_id][subf] = {}
                    for comp in ['Z','N','E']:
                        CAP_Path[src_id][stat_id][subf][comp] = {}
                        for direc in CAP_pattern[subf]:#['A','B','C']:
                            filename = 'XX.'+direc+'-'+str(CAP_id)+'..'+fbd_id+comp
                            path2file = path_id+subf+'/'+filename
                            # check if file exists
                            file_check = Path(path2file)
                            if file_check.exists():
                                CAP_Path[src_id][stat_id][subf][comp][direc] = path2file
                                ICount += 1
                            else:
                                ISSUE.append(path2file)
        
        # notification
        if ICount < len(ISSUE):
            print('Caution! Some files are missing.')
            print(str(int(100*ICount/len(ISSUE)))+' % files are missing:')
            for misfile in ISSUE:
                print(misfile)
                                
                            
        return CAP_Path
    

################################################
'''
    time shift
'''
################################################



def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    Source: 
    https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return average, np.sqrt(variance)


def p2p_shift(x,y):
    # shift synthetic waveform relative to g_max - f_max offset       
    # peak value shift
    shift = x.data.argmax() - y.data.argmax()   
    return shift

def cross_correlation_using_fft(x, y):
    f1 = fft(x)
    f2 = fft(np.flipud(y))
    cc = np.real(ifft(f1 * f2))
    return fftshift(cc)

def equalize_length_of_traces(tr1,tr2):
    '''
        
    '''
    starttime = np.max([tr1.stats.starttime,tr2.stats.starttime])
    endtime = np.min([tr1.stats.endtime,tr2.stats.endtime])
    st = Stream(traces=[tr1, tr2])
    st.trim(tr1.stats.starttime,endtime, pad=True)
    return st[0],st[1]

def compute_shift(x0, y0, mode):
    x = x0.copy()
    y = y0.copy()
  
    if mode == 'fft':
        if len(x.data) != len(y.data):
            x,y = equalize_length_of_traces(x,y)
        c = cross_correlation_using_fft(x.data, y.data)
        assert len(c) == len(x.data)
        zero_index = int(len(x.data) / 2) - 1
        shift = zero_index - np.argmax(c)
        return shift*x.stats.delta
    
    if mode == 'p2p':
        shift = p2p_shift(x, y)
        return shift*x.stats.delta



def get_dt_old(y1,y2,dt_max):
    n = y1.stats.npts
    sr = y1.stats.sampling_rate
    cz = signal.correlate(y2.data, y1.data, mode='same')
    cn = np.sqrt(signal.correlate(y1.data, y1.data, mode='same')[int(n/2)] * signal.correlate(y2.data, y2.data, mode='same')[int(n/2)])
    corr =  cz/cn 
    smax = dt_max*sr
    corr_cut = corr[int(0.5*n-smax):int(0.5*n+smax+1)]
    ncut = len(corr_cut)
    delay_arr = np.linspace(-0.5*ncut/sr, 0.5*ncut/sr, ncut)
    
    '''
    plt.subplot(2,1,1)
    t = np.arange(y1.stats.npts)*y1.stats.delta
    plt.plot(t,y1.data,'k-')
    plt.plot(t,y2.data,'r--')
    plt.title(y1.id)
    plt.subplot(2,1,2)
    c_arr = np.linspace(-0.5*n/sr, 0.5*n/sr, n)
    plt.plot(c_arr,corr,'k-')
    plt.plot(delay_arr,corr_cut,'r--')
    plt.xlim((-15,15))
    plt.show()
    '''
    return corr, delay_arr[np.argmax(corr_cut)]


def get_corr(y1,y2):
    n = y1.stats.npts
    sr = y1.stats.sampling_rate
    cz = signal.correlate(y2.data, y1.data, mode='same')
    cn = np.sqrt(signal.correlate(y1.data, y1.data, mode='same')[int(n/2)] * signal.correlate(y2.data, y2.data, mode='same')[int(n/2)])
    corr =  cz/cn 
    return corr, sr, n

def get_dt(cdt,sr,n,sleft,sright):
    corr_cut = cdt[int(0.5*n-np.abs(sleft)):int(0.5*n+sright+1)]
    ncut = len(corr_cut)
    delay_arr = np.linspace(-0.5*ncut/sr, 0.5*ncut/sr, ncut)  
    #print(int(0.5*n-sleft),int(0.5*n+sright+1))
    dt = delay_arr[np.argmax(corr_cut)]
    #print(dt)
    #try:
    #    dt = delay_arr[np.argmax(corr_cut)]
    #except:
    #    dt = 0
    return dt


def lag_finder_v2(obs_raw,synt_raw,cweight,dt_max_netw,dt_max_stat,dt_max_tr):
    # source: https://stackoverflow.com/questions/41492882/find-time-shift-of-two-signals-using-cross-correlation
    delay = {'netw':[],'stat':{},'trace':{},'logger':{}}
    corr_dict = {}
    
    # get network shift (mean and std)
    trace_fi = [] 
    trace_weight = []
    for sti,stat in enumerate(obs_raw.keys()):
        delay['logger'][stat] = {}
        corr_dict[stat] = {} # append corr trace of a station
        for comp in obs_raw[stat]:
            delay['logger'][stat][comp] = {}
            corr_dict[stat][comp] = {}
            for fi in obs_raw[stat][comp]:
                y1 = obs_raw[stat][comp][fi].copy()
                y2 = synt_raw['cF_X0'][stat][comp][fi].copy()
                sr = y1.stats.sampling_rate
                n = y1.stats.npts
                # get corr trace and sampling rate sr and trace length n
                cdt, sr, n = get_corr(y1,y2)
                # compute dt (truncated) for network shift of given trace pair
                if dt_max_netw > 0.:
                    sleft = dt_max_netw*sr #  left sided time shift
                    sright = dt_max_netw*sr #  right sided time shift 
                    dt = get_dt(cdt,sr,n,sleft,sright)
                else:
                    dt = 0.
                # add to logger dictionary
                delay['logger'][stat][comp][fi] = dt
                # add to netw list (weighted mean network shift)
                trace_fi.append(dt)
                trace_weight.append(cweight[comp])# denominator weighted mean
                # add to corr, sr and n to corr_dict (will be accessed later for station and trace shift) 
                corr_dict[stat][comp][fi] = [cdt,sr,n]
        
    # get weighted mean and std for network time shift
    ave_netw, std_netw = weighted_avg_and_std(trace_fi, trace_weight)
    delay['netw'] = [ave_netw, std_netw]   
        
    # get station shift (mean and std)
    for sti,stat in enumerate(corr_dict):
        trace_fi = [] 
        trace_weight = []
        delay['stat'][stat] = []
        for comp in corr_dict[stat]:
            for fi in corr_dict[stat][comp]:
                [cdt,sr,n] = corr_dict[stat][comp][fi]
                if dt_max_stat > 0.:
                    sleft = ave_netw*sr - dt_max_stat*sr #  left sided time shift
                    sright = ave_netw*sr + dt_max_stat*sr #  right sided time shift 
                    #print(stat,comp,dt_max_stat,ave_netw*sr,dt_max_stat*sr,sleft,sright,sr,n)
                    dt = get_dt(cdt,sr,n,sleft,sright)
                else:
                    dt = ave_netw
                trace_fi.append(dt+ave_netw)
                trace_weight.append(cweight[comp])
                    
        # get weighted mean and std for station time shift
        ave_stat, std_stat = weighted_avg_and_std(trace_fi, trace_weight)
        delay['stat'][stat] = [ave_stat, std_stat]
        
    
    # get trace shift (mean and std)
    for sti,stat in enumerate(corr_dict):
        delay['trace'][stat] = {}
        for comp in corr_dict[stat]:
            trace_fi = [] 
            trace_weight = []
            delay['trace'][stat][comp] = []
            for fi in corr_dict[stat][comp]:
                [cdt,sr,n] = corr_dict[stat][comp][fi]
                if dt_max_tr > 0.:
                    sleft = ave_netw*sr + ave_stat*sr - dt_max_tr*sr #  left sided time shift
                    sright = ave_netw*sr + ave_stat*sr + dt_max_tr*sr #  right sided time shift
                    dt = get_dt(cdt,sr,n,sleft,sright)
                else:
                    dt = 0.
                trace_fi.append(dt+delay['stat'][stat][0])
                trace_weight.append(cweight[comp])
                    
            # get weighted mean and std for trace time shift
            ave_tr, std_tr = weighted_avg_and_std(trace_fi, trace_weight)
            delay['trace'][stat][comp] = [ave_tr, std_tr]

    return delay

def cut_traces_based_on_time_shift_v2(synt_raw,delay):
    '''
        set convert traces to stream and set pad=True in trim to avoid different lenght traces  
    '''    
    synt_shift = {}
   

    for event_id in synt_raw:
        synt_shift[event_id] = {}
        for sti,stat in enumerate(synt_raw[event_id].keys()):
            synt_shift[event_id][stat] = {}
            for comp in synt_raw[event_id][stat]:
                synt_shift[event_id][stat][comp] = {}
                # get t_shift
                t_shift = delay['trace'][stat][comp][0]
                # fb loop
                for fi in synt_raw[event_id][stat][comp]:
                    # copy trace
                    tr2 = synt_raw[event_id][stat][comp][fi].copy()
                    dtc = abs(int(t_shift/tr2.stats.delta))
                    N_init = tr2.stats.npts
                    
                    # cases
                    if dtc != 0: # case: dtc is zero but t_shift is not --> [:-0] == []
                        if t_shift < 0:
                            tr2.data = np.append(np.zeros(dtc),tr2.data[:-dtc])
                        elif t_shift > 0:
                            tr2.data = np.append(tr2.data[dtc:],np.zeros(dtc))     
                        #elif t_shift == 0:
                        #    tr2.data = tr2.data
                        else:
                            print('Issue in cut_traces_based_on_time_shift_v2')
                    synt_shift[event_id][stat][comp][fi] = tr2


    return synt_shift

def Time_shift_xcorr_trunc_v2(obs_raw,synt_raw,station_Tshift):
    # get settings
    cweight = station_Tshift['cweight']
    dt_max_netw = station_Tshift['dt_max_netw']
    dt_max_stat = station_Tshift['dt_max_stat']
    dt_max_tr = station_Tshift['dt_max_tr']
    # get dt
    dt = lag_finder_v2(obs_raw,synt_raw,cweight,dt_max_netw,dt_max_stat,dt_max_tr)
    # cut traces based on time shift
    synt_shift = cut_traces_based_on_time_shift_v2(synt_raw,dt)

    '''
    event_id = list(synt_raw.keys())[0]
    stat, comp, fi = 'BK_CMB', 'Z', 0
    fig = plt.figure(figsize=(20, 5), facecolor='w', edgecolor='k')
    for ci, comp in enumerate(obs_raw[stat]):
        plt.subplot(1,3,ci+1)
        t = np.arange(obs_raw[stat][comp][fi].stats.npts)*obs_raw[stat][comp][fi].stats.delta
        plt.plot(t,obs_raw[stat][comp][fi].data/np.max(obs_raw[stat][comp][fi].data),'k-')
        plt.plot(t,synt_raw[event_id][stat][comp][fi].data/np.max(synt_raw[event_id][stat][comp][fi].data),'b-')
        plt.plot(t,synt_shift[event_id][stat][comp][fi].data/np.max(synt_shift[event_id][stat][comp][fi].data),'r-')
    plt.show()
    print(dt)
    print(asdf)
    '''
    
    return synt_shift,dt


def plot_phase_picks(OBS,inv_pre,fname=None):
    '''
    
    '''
    for stat in OBS:
        fig = plt.figure(figsize=(20, 5), facecolor='w', edgecolor='k')
        info = ''
        for ci, comp in enumerate(['Z','R','T']):
            obs = OBS[stat][comp][0]
            t0 = obs.stats.starttime
            t = np.arange(obs.stats.npts)*obs.stats.delta
            d = obs.data
            plt.plot(t,d/np.max(np.abs(d))-2*ci,'k-')
        for phase_id in ['P','S']:
            try:
                t_obs = inv_pre['traveltime_correction']['F1_X0']['t_average'][stat][phase_id]
                t_onset = inv_pre['traveltime_correction']['F1_X0']['t_onset'][stat][phase_id][0]
                t_synt = inv_pre['traveltime_correction']['F1_X0']['t_travel_synt'][stat][phase_id]
                dt = inv_pre['traveltime_correction']['F1_X0']['t_update'][stat][phase_id]
                info += 'dt'+phase_id+'='+str(round(dt,3))+'s ' 
                plt.plot([t_obs,t_obs],[-5,1],'r-')
                plt.plot([t_synt,t_synt],[-5,1],'b--')
            except:
                continue
        plt.title(stat+' - '+info,fontsize=18)    
        plt.grid(True)
        plt.yticks([0, -2, -4], ['Z', 'R', 'T'],fontsize=18)
        plt.xlabel('Time since '+str(t0)+' in s',fontsize=18)
        if fname is not None:
            plt.savefig(fname+'_'+stat+'.png',bbox_inches='tight',transparent=False,pad_inches=0)
        else:
            plt.show()




################################################
'''
    Modeller and Sampler
'''
################################################

def update_Mw(Mw_reference,f,g):
    '''
    
    '''
    
    # get Mw_reference
    [Mw_ref,perform_mag_update,Mw_lit,update_Mode] = Mw_reference
    
    if perform_mag_update:
        if update_Mode == 'P2P':
            Mfac = np.max(np.abs(f))/np.max(np.abs(g))
            M0 = 10**((3.0/2.0) * (Mw_ref + 10.7))*Mfac
            Mw = (2.0 / 3.0) * np.log10(M0) - 10.7
        elif update_Mode == 'LinReg':
            g_temp = np.argwhere(np.abs(g) >= 0.75*np.max(np.abs(g)))
            g_sig = [item[0] for item in g_temp.tolist()]
            # get amplitude factor
            Mfac, intercept, r_value, p_value, std_err = stats.linregress(g[g_sig],f[g_sig])    
            Mfac = np.abs(Mfac)
            # update moment
            M0 = 10**((3.0/2.0) * (Mw_ref + 10.7))*Mfac
            # update magnitude
            Mw = (2.0 / 3.0) * np.log10(M0) - 10.7
        elif update_Mode == 'AMean':
            g_temp = np.argwhere(np.abs(g) >= 0.75*np.max(np.abs(g)))
            g_sig = [item[0] for item in g_temp.tolist()]
            Mfac = np.mean(np.abs(f[g_sig])/np.abs(g[g_sig]))
            M0 = 10**((3.0/2.0) * (Mw_ref + 10.7))*Mfac
            Mw = (2.0 / 3.0) * np.log10(M0) - 10.7
        else: # use P2P as fallback
            print('update_Mode is not known. Using P2P.')
            Mfac = np.max(np.abs(f))/np.max(np.abs(g))
            M0 = 10**((3.0/2.0) * (Mw_ref + 10.7))*Mfac
            Mw = (2.0 / 3.0) * np.log10(M0) - 10.7
    else:
        M0_ref = 10**((3.0/2.0) * (Mw_ref + 10.7))
        M0_lit = 10**((3.0/2.0) * (Mw_lit + 10.7))
        Mfac = M0_lit/M0_ref
        Mw = Mw_lit

    return Mfac, Mw



def construct_c_coord_array(strike_F1=0.,dip_F1=0.,rake_F1=0.,
                            clvd_F1=0.,iso_F1=0.,
                            strike_F2=0.,dip_F2=0.,rake_F2=0.,
                            clvd_F2=0.,iso_F2=0.,
                            M_fac=0.,tint=0.):
    c_coord = np.array([strike_F1,dip_F1,rake_F1,
                            clvd_F1,iso_F1,
                            strike_F2,dip_F2,rake_F2,
                            clvd_F2,iso_F2,
                            M_fac,tint])
    return c_coord



def get_P_old(Result):
    pdf = []
    nl2 = []
    for fi in range(len(Result['result'][0])):
        pdf.append(Result['result'][0][fi][0])
        temp = Result['result'][0][fi][7]
        if temp >= 1.:
            nl2.append(1.0)
        else:
            nl2.append(temp)
    pdf = np.asarray(pdf)    
    nl2 = np.asarray(nl2)
    VR = 1.-nl2
    cdf = pdf#/np.sum(pdf)
    
    P = []
    for ii in range(len(cdf)-1):
        pp = np.sum(cdf[:ii])
        P.append(pp)
    P = np.asarray(P)
    
    return P,cdf,nl2,VR

def get_P(Result):
    '''
    
    '''
    out_res = Result['result'][0]
    # cdf
    cdf_sum = np.sum(np.asarray([el[0] for el in out_res]))
    cdf = np.asarray([el[0] for el in out_res])/cdf_sum
    # nl2
    nl2 = np.asarray([el[7] for el in out_res])
    nl2[nl2 > 1.] = 1.
    # VR
    VR = 1.-nl2
    # P
    P = []
    for ii in range(len(cdf)-1):
        pp = np.sum(cdf[:ii])
        P.append(pp)
    P = np.asarray(P)    
    
    return P,cdf,nl2,VR

################################################
'''
    basic external and original helper functions for moment tensor processing
'''
################################################

def ang2M(fault,M0,MTnota):
    '''
        Input:
            fault: [strike,dipt,slip,iso_fac]
            M0: Seismic Moment (scalar value)
            MTnota: Aki or Dziewonski
        Output:
            MT elements
    '''
    rad=np.pi/180
    strike, dip, rake = fault[0]*rad, fault[1]*rad, fault[2]*rad
    ISO_A = 1-np.abs(fault[3])
    ISO_B = fault[3]
    M = np.zeros(6)
    M[0] =  ISO_A*M0*(np.sin(2*dip)*np.sin(rake)) + M0*ISO_B                                                                                
    M[1] = -ISO_A*M0*(np.sin(dip)*np.cos(rake)*np.sin(2*strike) + np.sin(2*dip)*np.sin(rake)*np.sin(strike)*np.sin(strike)) + M0*ISO_B                    
    M[2] =  ISO_A*M0*(np.sin(dip)*np.cos(rake)*np.sin(2*strike) - np.sin(2*dip)*np.sin(rake)*np.cos(strike)*np.cos(strike)) + M0*ISO_B                     
    M[3] = -ISO_A*M0*(np.cos(dip)*np.cos(rake)*np.cos(strike)  + np.cos(2*dip)*np.sin(rake)*np.sin(strike))
    if MTnota == "Aki": 
        M[4] = -ISO_A*M0*(np.cos(dip)*np.cos(rake)*np.sin(strike)  - np.cos(2*dip)*np.sin(rake)*np.cos(strike))                       
        M[5] =  ISO_A*M0*(np.sin(dip)*np.cos(rake)*np.cos(2*strike) + 0.5*np.sin(2*dip)*np.sin(rake)*np.sin(2*strike))                
    if MTnota == "Dziewonski":     
        M[4] = ISO_A*M0*(np.cos(dip)*np.cos(rake)*np.sin(strike)  - np.cos(2*dip)*np.sin(rake)*np.cos(strike))                       
        M[5] = -ISO_A*M0*(np.sin(dip)*np.cos(rake)*np.cos(2*strike) + 0.5*np.sin(2*dip)*np.sin(rake)*np.sin(2*strike))
    return M

def Tape2M(strike,dip,rake,clvd,iso,M0):
    M = tt2cmt(clvd*30/100,iso*90/100,M0,strike,dip,rake)
    return M

def M2Tape(M):
    clvd,iso,M0,strike,dip,rake = cmt2tt(M)
    return strike,dip,rake,clvd*100/30,iso*100/90,M0

def Tape2M_uniform(strike,h,rake,v,w,M0):
    '''
    input Tape2M_uniform
    strike = kappa, rake = sigma
    h = cos(dip)^-1
    v = 1/3 sin(3*clvd)^-1  # |clvd| < pi/6
    w = 0.75*iso - 0.5+sin(2*iso) + 1/16 sin(4*iso) # 0 < iso < pi
    input: tt152cmt
    kappa, sigma, M0, v, w, h
    '''
    M = tt152cmt(strike,rake,M0,v,w,h)
    return M

def M2Tape_uniform(M):
    #output: kappa, sigma, M0, v, w, h
    rho,v,w,kappa,sigma,h = cmt2tt15(M)
    return kappa, sigma, h, v, w, rho

def get_aux(strike,dip,rake):
    from obspy.imaging.beachball import aux_plane
    fault2 = aux_plane(strike,dip,rake)
    return fault2[0],fault2[1],fault2[2]
    
def get_Kagan(f1,f2):
    k = get_kagan_angle(f1[0],f1[1],f1[2],f2[0],f2[1],f2[2])
    return k


            
def add_Result2json(Result,key,filename='Result'):
    '''
    
    '''
    import json
    
    # open existing result json file
    try:
        with open(filename+'.json') as json_data_file:
            res0 = json.load(json_data_file)
    except:
        res0 = {}
    
    # create new entry
    res_new = {key:Result}
    
    # update dictionary
    res0.update(res_new)
    
    with open(filename+'.json', 'w', encoding='utf-8') as f:
        json.dump(res0, f, ensure_ascii=False, indent=4)   


##################################################################################################
##################################################################################################
### Trace Selection Tool
##################################################################################################
##################################################################################################

class Trace_Selector:
    
    def __init__(self,path2data=None,
                 Station_file=None,Event_file=None,
                 path2file=None,path2PWfile=None,phase_picks=None,
                 distance_range=[0,1000.],azimuth_range=[0.,360.],
                 f_init=[0.04,0.06],df=0.005,twind=[],Pwind=[],
                 path2fig='',update_file='a',
                 set_tag='Set1',event_id='evXXX',
                 taper=[],zerophase=False):
        
        # change backend (will be switched back when closing)
        matplotlib.use('TKAgg')
        
        # set init info
        self.path2data = path2data
        self.Station_file = Station_file
        self.Event_file = Event_file
        self.phase_picks = phase_picks
        self.distance_range = distance_range
        self.azimuth_range = azimuth_range
        self.path2fig = path2fig
        self.path2file = path2file
        self.path2PWfile = path2PWfile
        self.update_file = update_file
        self.twind = twind # time window
        self.Pwind = Pwind # time window signal (rel to peak)
        self.set_tag = set_tag
        self.event_id = event_id
        self.taper = taper
        self.bcolor = 'white'
        self.A_scale = 'Asin'
        self.zerophase = zerophase
        
        # implemented phase onset
        self.Phase_ID = ['P','S']
        self.pw_sheme = 'PSPSPS'
        self.Peak_Signal_Window = {} # signal window rel. to peak
        
        # load waveform list
        self.get_station_info()
        self.get_waveforms()
               
        # initiate tk window
        self.APP_TITLE = 'Trace Selector: Event '+self.event_id
        self.app_win = tk.Tk()
        self.app_win.title(self.APP_TITLE)
        
        # create figure
        self.fig = plt.Figure(figsize=(20,10))
        self.ax = {}
        for ii in range(6):
            if ii <= 2: jj,jj_suf = ii, '0'
            elif ii > 2: jj,jj_suf = ii-3, '1'
            self.ax[self.Comp_List[jj]+jj_suf] = self.fig.add_subplot(2,3,ii+1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.app_win)
        
        # define buttons
        self.button_prev = tk.Button(self.app_win, text="Back", command=self.action_button_prev)
        self.button_next = tk.Button(self.app_win, text="Next", command=self.action_button_next)  
        self.button_plot = tk.Button(self.app_win, text="Plot", command=self.action_button_plot)
        self.button_save = tk.Button(self.app_win, text="Save", command=self.action_button_save)
        self.button_header = tk.Button(self.app_win, text="Header", command=self.action_button_header)
        self.button_help = tk.Button(self.app_win, text="Help", command=self.action_button_help)
        self.button_Arel = tk.Button(self.app_win, text="Ampl. (rel.)", command=self.action_button_Arel)
        self.button_Asin = tk.Button(self.app_win, text="Ampl. (single)", command=self.action_button_Asin)
        self.button_prev.place(x=20 , y=5, width=120, height=60)
        self.button_next.place(x=150, y=5, width=120, height=60)
        self.button_plot.place(x=280, y=5, width=120, height=60)
        self.button_save.place(x=890, y=5, width=120, height=60)   
        self.button_header.place(x=20 , y=675, width=120, height=40)
        self.button_help.place(x=150, y=675, width=120, height=40)
        self.button_Arel.place(x=280, y=675, width=120, height=18)
        self.button_Asin.place(x=280, y=697, width=120, height=18)
        
        # set frequency information
        self.df = df
        df = str(df)
        self.df_r = df[::-1].find('.')
        [self.fmin, self.fmax] = f_init
        
        # set frequency labelling
        title_fmin = tk.Label(self.app_win, fg="dark green")
        title_fmax = tk.Label(self.app_win, fg="dark green")
        title_fmin.place(x=410 , y=5, width=230, height=25)
        title_fmax.place(x=650 , y=5, width=230, height=25)
        title_fmin.config(text='Lower Frequency')
        title_fmax.config(text='Upper Frequency')
        self.label_fmin = tk.Label(self.app_win, fg="dark green")
        self.label_fmax = tk.Label(self.app_win, fg="dark green")
        self.label_fmin.place(x=480 , y=35, width=90, height=25)
        self.label_fmax.place(x=720 , y=35, width=90, height=25)
        self.label_fmin.config(text=str(round(self.fmin,self.df_r))+' Hz')
        self.label_fmax.config(text=str(round(self.fmax,self.df_r))+' Hz')
        
        # define frequency buttons
        self.button_fmin_m = tk.Button(self.app_win, text="-", command=self.action_button_fmin_m)
        self.button_fmin_p = tk.Button(self.app_win, text="+", command=self.action_button_fmin_p)   
        self.button_fmax_m = tk.Button(self.app_win, text="-", command=self.action_button_fmax_m)
        self.button_fmax_p = tk.Button(self.app_win, text="+", command=self.action_button_fmax_p) 
        
        self.button_fmin_m.place(x=410 , y=35, width=60, height=30)
        self.button_fmin_p.place(x=580, y=35, width=60, height=30)
        self.button_fmax_m.place(x=650 , y=35, width=60, height=30)
        self.button_fmax_p.place(x=820, y=35, width=60, height=30)
            
        # display initial plot
        self.key = self.wdir[self.sid].split('.')[0]+'_'+self.wdir[self.sid].split('.')[1]
        self.plot() 

        # start tk loop
        self.app_win.mainloop()
    
    def get_station_info(self):
        if self.Station_file != None:
            if self.Event_file != None:
                self.Station_dict = True
                
                # open event file
                with open(self.Event_file, "r") as f:
                    content_event = f.readlines()
                f.close()
                
                # open station file
                with open(self.Station_file, "r") as f:
                    content_station = f.readlines()
                f.close()
                
                # get event info
                self.event_list = {}
                for info in [line.rstrip('\n') for line in content_event]:
                    time_raw = info.split()[6]
                    key = info.split()[0]#+'_'+time_raw
                    time = time_raw
                    glob_loc = [float(info.split()[2]),float(info.split()[3]),float(info.split()[4])]
                    mag = float(info.split()[5])
                    self.event_list[key] = [time,glob_loc,mag]
                    
                # get station info
                self.station_list = {}
                for info in [line.rstrip('\n') for line in content_station]:
                    key = info.split()[1]+'_'+info.split()[0]
                    distance, az, baz = gps2dist_azimuth(
                        self.event_list[self.event_id][1][0],self.event_list[self.event_id][1][1],
                        float(info.split()[2]),float(info.split()[3]))
                    self.station_list[key] = [float(info.split()[2]),float(info.split()[3]),float(info.split()[5])/1000,
                                             distance,az,baz]
            else:
                self.Station_dict = False
        else:
            self.Station_dict = False
        
    
    def get_selection(self):
        '''
            input: self.trace_dict[XX.S1][Z] = True ....
            output: XX_S1-ZRT XX_S2-RT ...
            output (phase): XX_S1-ZRT-111111
            --> numbering: 
                        - digit 1,2: --> Z as P,S where 1 is True and 0 is False
                        - digit 3,4: --> R as P,S where 1 is True and 0 is False
                        - digit 5,6: --> T as P,S where 1 is True and 0 is False
        '''
        selection_format = ''
        for key in self.trace_dict:
            comp_list = ''
            phase_list = '-'
            for comp in self.trace_dict[key]:
                if self.trace_dict[key][comp]:
                    comp_list += comp
                if self.phase_picks != None:
                    for phase_id in self.Phase_ID:
                        if self.trace_dict[key][comp][phase_id]:
                            phase_list += '1'
                        else:
                            phase_list += '0'
                else:
                    phase_list = ''
            if len(comp_list) >= 1:
                selection_format += key+'-'+comp_list+phase_list+' '
        return selection_format
    
    def get_PWselection(self,cdict=None):
        '''
        
        '''
        if cdict != None:
            CDICT = cdict
        else:
            CDICT = self.Peak_Signal_Window
            
        sf = ''
        for key in CDICT:
            sf += key+'='+CDICT[key][0]+'='+CDICT[key][1]+' '
        return sf
    
    def load_trace_file(self):
        '''
            output: cont[ev001_Set1] = 'ev001 Set1 XX_S1-ZRT XX_S2-RT ...' 
        '''
        cont = {}
        sel = {}
        with open(self.path2file, 'r') as f:
            content = f.readlines()
            content = [line.rstrip('\n') for line in content]
        f.close()
        for li, line in enumerate(content):
            info = line.split()
            key = info[0]+'_'+info[1]
            cont[key] = ' '.join(info[2:])#info #  get dict in output format
            # define  lists and dictionaries
            sel[key] = {}
            stations = [] # stations list (if a station is missing in the file, add from self.wdir with False)
            for stat in range(len(info)-2):
                sinfo = info[stat+2].split('-')
                sel[key][sinfo[0]] = {}
                #'''
                # fill default components with False
                if self.phase_picks == None:
                    if self.Station_dict: # if no Station_dict is given, use default ZNE
                        sel_temp = {'Z':False,'R':False,'T':False}
                    else:
                        sel_temp = {'Z':False,'N':False,'E':False}
                else:
                    pw_sel_def = {}
                    for phase_id in ['P','S']:#self.Phase_ID:
                        pw_sel_def[phase_id] = False # phase window selection (default)
                    if self.Station_dict: # if no Station_dict is given, use default ZNE
                        sel_temp = {'Z':pw_sel_def.copy(),'R':pw_sel_def.copy(),'T':pw_sel_def.copy()}
                    else:
                        sel_temp = {'Z':pw_sel_def.copy(),'N':pw_sel_def.copy(),'E':pw_sel_def.copy()}
                # update selection flag based on file content
                for ci in range(len(sinfo[1])):
                    try:
                        if self.phase_picks == None:
                            sel_temp[sinfo[1][ci]] = True
                            stations.append(sinfo[0])
                        else:
                            for pi in np.arange(2*ci,2*ci+2,1):
                                stations.append(sinfo[0])
                                if sinfo[2][pi] == '1':
                                    sel_temp[sinfo[1][ci]][self.pw_sheme[pi]] = True
                                elif sinfo[2][pi] == '0':
                                    sel_temp[sinfo[1][ci]][self.pw_sheme[pi]] = False
                    except:
                        print('Selection is in RT system, please provide Station_dict')
                        raise SystemExit
                sel[key][sinfo[0]] = sel_temp
            # add missing stations
            for s_wdir in self.wdir:
                wkey = s_wdir.split('.')[0]+'_'+s_wdir.split('.')[1]
                if wkey not in stations:
                    if self.phase_picks == None:
                        if self.Station_dict:
                            sel[key][wkey] = {'Z':False,'R':False,'T':False}
                        else:
                            sel[key][wkey] = {'Z':False,'N':False,'E':False}
                    else:
                        pw_sel_def = {}
                        for phase_id in self.Phase_ID:
                            pw_sel_def[phase_id] = False # phase window selection (default)
                        if self.Station_dict:
                            sel[key][wkey] = {'Z':pw_sel_def.copy(),'R':pw_sel_def.copy(),'T':pw_sel_def.copy()}
                        else:
                            sel[key][wkey] = {'Z':pw_sel_def.copy(),'R':pw_sel_def.copy(),'T':pw_sel_def.copy()}

        return cont, sel
    
    def load_pw_file(self):
        '''
        
        '''
        cont = {}
        with open(self.path2PWfile, 'r') as f:
            content = f.readlines()
            content = [line.rstrip('\n') for line in content]
        f.close()
        for li, line in enumerate(content):
            info = line.split()
            cont[info[0]] = {}
            for stat_info in info[1:]:
                [key,t1,t2] = stat_info.split('=')
                cont[info[0]][key] = [t1,t2]
        return cont
    
    def write_trace_file(self):
        # get selection format
        selection_format = self.get_selection() 
        # output
        if self.update_file == 'append': # append selection to existing list
            with open(self.path2file, 'a') as f:
                f.write('%s %s %s\n' %(self.event_id,self.set_tag,selection_format))
                f.close()
            print('Append selection to selection list.')
        elif self.update_file == 'update': # update selection in existing list
            # load existing trace selection file
            cont,_ = self.load_trace_file()
            key = self.event_id+'_'+self.set_tag
            if key in cont:
                cont[key] = selection_format
                with open(self.path2file, 'w') as f:
                    for keyi in cont:
                        f.write('%s %s %s\n' %(keyi.split('_')[0],keyi.split('_')[1],cont[keyi]))
                f.close()
            else:
                print('Selection key does not exist, appending new selection to selection list.')
                with open(self.path2file, 'a') as f:
                    f.write('%s %s %s\n' %(self.event_id,self.set_tag,selection_format))
                f.close()
            print('Updating selection list.')
        else: # do nothing
            print('Exiting without writing to file.')
    
    def write_p_window_file(self):
        '''
        
        '''
        # output
        if self.update_file == 'append': # append selection to existing list
            # get selection format
            selection_format = self.get_PWselection()
            with open(self.path2PWfile, 'a') as f:
                f.write('%s %s\n' %(self.event_id,selection_format))
                f.close()
            print('Append selection to selection list.')
        elif self.update_file == 'update': # update selection in existing list
            # load existing trace selection file
            cont = self.load_pw_file()
            if self.event_id in cont:
                # compare loaded and current content
                for stat in cont[self.event_id]:
                    if stat not in self.Peak_Signal_Window:
                        self.Peak_Signal_Window[stat] = cont[self.event_id][stat]
                # save new self.Peak_Signal_Window to cont
                cont[self.event_id] = self.Peak_Signal_Window
                # write selections
                with open(self.path2PWfile, 'w') as f:
                    for keyi in cont:
                        selection_format = self.get_PWselection(cdict=cont[keyi])
                        f.write('%s %s\n' %(keyi,selection_format))
                f.close()
            else:
                print('Selection key does not exist, appending new selection to selection list.')
                # get selection format
                selection_format = self.get_PWselection()
                with open(self.path2PWfile, 'a') as f:
                    f.write('%s %s\n' %(self.event_id,selection_format))
                f.close()
            print('Updating selection list.')
        else: # do nothing
            print('Exiting without writing to file.')
        
        
    
    def get_indx_of_subwindow(self,st_i):
        '''
        
        '''
        if len(self.twind) == 2:
            # get time marker
            event_time = self.event_list[self.event_id][0]
            starttime = st_i[0].stats.starttime
            dt = st_i[0].stats.delta
            t0 = int((UTCDateTime(event_time) - UTCDateTime(starttime))/dt)
            return t0-int(self.twind[0]/dt), t0+int(self.twind[1]/dt)
        else:
            return 0, -1
    
    def action_button_header(self):
        header = self.st_filt[0].stats
        newWindow = tk.Toplevel(self.app_win) 
        newWindow.title("Trace Header") 
        newWindow.geometry("500x800") 
        tk.Label(newWindow,text=str(header), justify='left').place(x=0 , y=5)
        
    def action_button_help(self):
        '''
            https://www.geeksforgeeks.org/open-a-new-window-with-a-button-in-python-tkinter/
        '''
        newWindow = tk.Toplevel(self.app_win) 
        newWindow.title("Help") 
        newWindow.geometry("600x600") 
        bz_text = ''' 
        Help:
        
        Background color sheme of AZ label:
        red: 45 > az <= 325
        blue: 45 <= az < 135
        green: 135 <= az < 225
        yellow: 225 <= az < 325 
        
        Vertical lines:
        yellow: window of filtered traces
        green: origin time
        red: Pwind range (only these data will be used)      
        '''
        tk.Label(newWindow,text=bz_text, justify='left').place(x=0 , y=0)
        tk.Label(newWindow,text='(c) Mike Lindner, 2020', justify='right').place(x=500 , y=0)
    
    def action_button_Arel(self):
        self.A_scale = 'Arel'
        # raw time window
        A_min, A_max = [], []
        for ii in range(3):
            st_i = self.st.select(component=self.Comp_List[ii])
            A_min.append(np.min(st_i[0].data))
            A_max.append(np.max(st_i[0].data))
        for ii in range(3):
            axii = self.Comp_List[ii]+'0'
            self.ax[axii].set_ylim((np.min(np.asarray(A_min))*1.05, 1.05*np.max(np.asarray(A_max))))
        # filtered time window
        A_min, A_max = [], []
        ta, tb = self.get_indx_of_subwindow(st_i)
        for ii in range(3):
            st_i = self.st_filt.select(component=self.Comp_List[ii])
            A_min.append(np.min(st_i[0].data[ta:tb]))
            A_max.append(np.max(st_i[0].data[ta:tb]))
        for ii in range(3):
            axii = self.Comp_List[ii]+'1'
            self.ax[axii].set_ylim((np.min(np.asarray(A_min))*1.05, 1.05*np.max(np.asarray(A_max))))
        self.canvas.draw()
        self.canvas.flush_events()
    
    def action_button_Asin(self):
        self.A_scale = 'Asin'
        # raw time window
        for ii in range(3):
            axii = self.Comp_List[ii]+'0'
            # select trace
            st_i = self.st.select(component=self.Comp_List[ii])
            self.ax[axii].set_ylim((np.min(st_i[0].data)*1.05, 1.05*np.max(st_i[0].data)))
        # filtered time window
        ta, tb = self.get_indx_of_subwindow(st_i)
        for ii in range(3):
            axii = self.Comp_List[ii]+'1'
            # select trace
            st_i = self.st_filt.select(component=self.Comp_List[ii])
            self.ax[axii].set_ylim((np.min(st_i[0].data[ta:tb])*1.05, 1.05*np.max(st_i[0].data[ta:tb])))
        self.canvas.draw()
        self.canvas.flush_events()
    
    def action_button_plot(self):
        fname = self.key+'_fmin-'+str(round(self.fmin,self.df_r))+'_fmax-'+str(round(self.fmax,self.df_r))
        self.fig.savefig(self.path2fig+fname+'.png', 
                bbox_inches='tight', 
                    transparent=False,
                    pad_inches=0)
        print('Saving plot as: '+self.path2fig+self.key+'.png')
    
    def action_button_fmin_m(self):
        if self.fmin - self.df > 0.:
            self.fmin -= self.df
        self.label_fmin.config(text=str(round(self.fmin,self.df_r))+' Hz')
        self.filter_trace()
        
    def action_button_fmin_p(self):
        self.fmin += self.df
        self.label_fmin.config(text=str(round(self.fmin,self.df_r))+' Hz')
        self.filter_trace()
        
    def action_button_fmax_m(self):
        if self.fmax - self.df > 0.:
            self.fmax -= self.df
        self.label_fmax.config(text=str(round(self.fmax,self.df_r))+' Hz')
        self.filter_trace()
        
    def action_button_fmax_p(self):
        self.fmax += self.df
        self.label_fmax.config(text=str(round(self.fmax,self.df_r))+' Hz')
        self.filter_trace()
    
    def action_button_save(self):
        time.sleep(0.5)
        self.write_trace_file()
        if self.path2PWfile is not None:
            self.write_p_window_file()
        self.app_win.destroy()
        matplotlib.use('module://ipykernel.pylab.backend_inline')

    def action_button_prev(self):
        if self.sid == 0:
            self.sid = 0
        else:
            self.sid -= 1            
        self.key = self.wdir[self.sid].split('.')[0]+'_'+self.wdir[self.sid].split('.')[1]
        self.plot()
            
    def action_button_next(self):
        if self.sid == len(self.wdir):
            self.sid = len(self.wdir)
        else:
            self.sid += 1
            
        self.key = self.wdir[self.sid].split('.')[0]+'_'+self.wdir[self.sid].split('.')[1]
        self.plot() 
    
    
    def filter_trace(self):
        
        
        self.st_filt = self.st.copy()
        
        self.st_filt.filter('bandpass',
                            freqmin=self.fmin,
                            freqmax=self.fmax,
                            corners=4,zerophase=self.zerophase)
        
        if self.A_scale == 'Arel':
            A_min, A_max = [], []
            for ii in range(3):
                st_i = self.st_filt.select(component=self.Comp_List[ii])
                A_min.append(np.min(st_i[0].data))
                A_max.append(np.max(st_i[0].data))
        
        
        for ii in range(3):
        
            axii = self.Comp_List[ii]+'1'
            
            # select trace
            st_i = self.st_filt.select(component=self.Comp_List[ii])
            
            t = np.arange(st_i[0].stats.npts)*st_i[0].stats.delta
            self.l[axii].set_ydata(st_i[0].data)
            
            if self.A_scale == 'Asin':
                self.ax[axii].set_ylim((np.min(st_i[0].data)*1.05, 1.05*np.max(st_i[0].data)))
            elif self.A_scale == 'Arel':
                self.ax[axii].set_ylim((np.min(np.asarray(A_min))*1.05, 1.05*np.max(np.asarray(A_max))))
            
            self.canvas.draw()
            self.canvas.flush_events()
    
    def load_trace_selection_format(self):
        '''
        
        '''
        _,trace_dict = self.load_trace_file()
        try: # try if given key already exist
            self.trace_dict = trace_dict[self.event_id+'_'+self.set_tag]
        except: # if not, create default dict based on wdir content
            self.trace_dict = {}
            for loc,stat_id in enumerate(self.wdir):
                key = stat_id.split('.')[0]+'_'+stat_id.split('.')[1]
                self.trace_dict[key] = {}
                for comp in self.Comp_List:
                    self.trace_dict[key][comp] = True
    
    def load_phase_selection_format(self):
        '''
        
        '''
        _,trace_dict = self.load_trace_file()
        try: # try if given key already exist
            self.trace_dict = trace_dict[self.event_id+'_'+self.set_tag]
        except: # if not, create default dict based on wdir content
            self.trace_dict = {}
            for loc,stat_id in enumerate(self.wdir):
                key = stat_id.split('.')[0]+'_'+stat_id.split('.')[1]
                self.trace_dict[key] = {}
                for comp in self.Comp_List:
                    self.trace_dict[key][comp] = {}
                    for phase_id in self.Phase_ID:
                        try:
                            if phase_id in self.phase_picks['t_onset'][key]: # check if pick exists
                                self.trace_dict[key][comp][phase_id] = True
                            else:
                                self.trace_dict[key][comp][phase_id] = False
                        except:
                            self.trace_dict[key][comp][phase_id] = False
                            continue
                    
    def get_waveforms(self):
        self.sid = 0
        self.wdir = sorted([f for f in listdir(self.path2data) if isfile(join(self.path2data, f)) and f[-1] == 'Z'])
        self.key = self.wdir[self.sid] 
        
        if self.Station_dict: # if no Station_dict is given, use default ZNE
            self.Comp_List = ['Z','R','T']
        else:
            self.Comp_List = ['Z','N','E']
        
        # load initial selection from file or create default dict with True flag
        if self.phase_picks == None:
            self.load_trace_selection_format()
        else:
            self.load_phase_selection_format()
            
        #print(self.trace_dict)
        #print(asdf)
        
    def callback(self,event):
        ax = event.inaxes
        canvas = event.canvas
        if self.phase_picks == None:
            if event.guiEvent.num == 1:
                ax.set_facecolor('red')
                sel = False
            elif event.guiEvent.num == 3:
                ax.set_facecolor(self.bcolor)
                sel = True
            
            canvas.draw()
        
            # get subplot number
            subplot_num = ax._subplotspec.num1
            
            # get component from subplot number
            if subplot_num in [0,3]: comp = self.Comp_List[0]
            elif subplot_num in [1,4]: comp = self.Comp_List[1]
            elif subplot_num in [2,5]: comp = self.Comp_List[2]
            
            # update selection dict
            self.trace_dict[self.key][comp] = sel
     
        else:
            if self.key in self.phase_picks['t_onset']:
                pPicks = self.phase_picks['t_onset'][self.key]
            else:
                pPicks = []
            pw_window = self.phase_picks['pw_window']
            td_temp = {} # non self object
            for comp in self.trace_dict[self.key]:
                td_temp[comp] = {}
                for phase_id in self.trace_dict[self.key][comp]:
                    td_temp[comp][phase_id] = self.trace_dict[self.key][comp][phase_id]
            for phase_id in pPicks:
                t_onset = UTCDateTime(pPicks[phase_id][0]) - self.st_filt[0].stats.starttime
                xa,xb = t_onset-pw_window[phase_id][0], t_onset+pw_window[phase_id][1]
                if xa < event.xdata < xb: # check if click is in phase window
                    ax.axvspan(xa, xb, alpha=1.0, color='white')
                    if event.guiEvent.num == 1:
                        ax.axvspan(xa, xb, alpha=0.5, color='red')
                        sel = False
                    elif event.guiEvent.num == 3:
                        ax.axvspan(xa, xb, alpha=0.5, color='green')
                        sel = True
                    
                    canvas.draw()
                    
                    # get subplot number
                    subplot_num = ax._subplotspec.num1
                    
                    # get component from subplot number
                    if subplot_num in [0,3]: comp = self.Comp_List[0]
                    elif subplot_num in [1,4]: comp = self.Comp_List[1]
                    elif subplot_num in [2,5]: comp = self.Comp_List[2]
                    
                    # update temporary selection dict
                    td_temp[comp][phase_id] = sel
                    
            # update selection dict
            self.trace_dict[self.key] = td_temp

    def check_components(self,st):
        '''
            check if all the components are available
        '''
        comp = []
        for tr in st:
            tr_ = tr.copy()
            comp.append(tr.stats.channel[-1])
            N = tr_.stats.npts #  all traces should be the same length
            A = np.max(np.abs(tr_.data))*0.01
            stats = tr_.stats # compy default stats (comp will be overwritten)
            
        if len(comp) != 3: # components are missing
            traces = []
            for ci in ['Z','N','E']:
                if ci not in comp:
                    trc = Trace(data=np.random.normal(0., A, N))
                    trc.stats = stats
                    trc.stats.channel = trc.stats.channel[:-1]+ci
                    traces.append(trc.copy())
            st += Stream(traces)

        return st
        
    def plot(self):
        '''
        
        '''
            
        # read raw waveforms for station id self.key
        st = read(self.path2data+self.wdir[self.sid][:-1]+'*')
        
        st = self.check_components(st)
        
        st.detrend(type='demean')
        st.detrend(type='simple')
        
        if len(self.taper) > 0:
            st.taper(type=self.taper[0],max_percentage=self.taper[1])
        
        # rotate stream if Station_dict is available
        if self.Station_file != None:
            if self.Event_file != None:
                st.rotate(method='NE->RT',back_azimuth=self.station_list[self.key][5])
                # display additional station info
                title_fmin = tk.Label(self.app_win, fg="dark green")
                title_fmax = tk.Label(self.app_win, fg="dark green")
                title_fmin.place(x=20 , y=80, width=90, height=25)
                title_fmax.place(x=20 , y=110, width=90, height=25)
                az = int(self.station_list[self.key][4])
                if 45 > az <= 325: bcol = 'red'
                elif 45 <= az < 135: bcol = 'blue'
                elif 135 <= az < 225: bcol = 'green'
                elif 225 <= az < 325: bcol = 'yellow'
                else: bcol = 'white'
                title_fmin.config(text='AZ='+str(az)+'°',background=bcol)
                title_fmax.config(text='Dist='+str(round(self.station_list[self.key][3]/1000.,2))+' km')
                
                # exclude station outside of ditance range
                dist_cond = self.distance_range[0] <= self.station_list[self.key][3]/1000 <= self.distance_range[1]
                
                # exclude stations outside of azimuth range
                if self.azimuth_range[0] > self.azimuth_range[1]: # case lower boudary is below 0 (or 360) e.g. [270,90]
                    az_cond = self.azimuth_range[0] >= self.station_list[self.key][4] <= self.azimuth_range[1]
                else: # case lower boundary is larger than 0 e.g. [90,270]
                    az_cond = self.azimuth_range[0] <= self.station_list[self.key][4] <= self.azimuth_range[1]
                
                #print(dist_cond,az_cond,dist_cond and az_cond)
                # add or remove station
                if dist_cond and az_cond:
                    self.bcolor = 'white'
                else:
                    self.bcolor = 'grey'
                    for comp in ['Z','R','T']:
                        if self.phase_picks == None:
                            self.trace_dict[self.key][comp] = False
                        else:
                            for phase_id in self.Phase_ID:
                                self.trace_dict[self.key][comp][phase_id] = False
                        
        
        self.st = st.copy()
        self.st_filt = st.copy()
        self.dt = st[0].stats.delta
        
        self.st_filt.filter('bandpass',
                            freqmin=self.fmin,
                            freqmax=self.fmax,
                            corners=4,zerophase=self.zerophase)
        # waveform dict
        self.l = {}
        
        # get peak time of filtered waveforms
        Ptime_temp, Ptime = [], []
        for tr in self.st_filt:
            Ptime_temp.append([np.max(tr.data),np.argmax(tr.data)])
        for [Amp,Idx] in sorted(Ptime_temp):
            if Amp > 0.1*sorted(Ptime_temp)[0][0]: # remove all peaks smaller than 10% of the largest peak
                Ptime.append(Idx)
        
        
        # get amplitude range
        A_minR, A_maxR = [], []
        for ii in range(3):
            st_i = self.st.select(component=self.Comp_List[ii])
            A_minR.append(np.min(st_i[0].data))
            A_maxR.append(np.max(st_i[0].data))
        if self.A_scale == 'Arel':
            A_min, A_max = [], []
            for ii in range(3):
                st_i = self.st_filt.select(component=self.Comp_List[ii])
                A_min.append(np.min(st_i[0].data))
                A_max.append(np.max(st_i[0].data))
                
        
        # subplot loop
        for ii in range(6):
            if ii <= 2: jj,jj_suf = ii, '0'
            elif ii > 2: jj,jj_suf = ii-3, '1'
            
            axii = self.Comp_List[jj]+jj_suf
            # clear axis
            self.ax[axii].clear()
            
            # get and set time marker
            event_time = self.event_list[self.event_id][0]
            starttime = st_i[0].stats.starttime
            self.t0 = float(UTCDateTime(event_time) - UTCDateTime(starttime))
            
            if jj_suf == '0':
                # select trace
                st_i = st.select(component=self.Comp_List[jj])
                t = np.arange(st_i[0].stats.npts)*self.dt
                self.l[axii], = self.ax[axii].plot(t,st_i[0].data,'k-')
                # t0 marker
                self.ax[axii].plot([self.t0,self.t0],[np.min(st_i[0].data)*1.05, 1.05*np.max(st_i[0].data)],'g--',linewidth=2)
                # twind marker
                if len(self.twind) == 2:
                    self.ax[axii].plot([self.t0-self.twind[0],self.t0-self.twind[0]],[np.min(st_i[0].data)*1.05, 1.05*np.max(st_i[0].data)],'y--',linewidth=2)
                    self.ax[axii].plot([self.t0+self.twind[1],self.t0+self.twind[1]],[np.min(st_i[0].data)*1.05, 1.05*np.max(st_i[0].data)],'y--',linewidth=2)
                if self.A_scale == 'Asin':
                    self.ax[axii].set_ylim((np.min(st_i[0].data)*1.05, 1.05*np.max(st_i[0].data)))
                elif self.A_scale == 'Arel':
                    self.ax[axii].set_ylim((np.min(np.asarray(A_minR))*1.05, 1.05*np.max(np.asarray(A_maxR))))  
                # time window
                if len(self.twind) == 2 and self.phase_picks != None:
                        self.ax[axii].set_xlim((self.t0-self.twind[0],self.t0+self.twind[1]))
                    
            elif jj_suf == '1':
                # select trace
                st_i = self.st_filt.select(component=self.Comp_List[jj])
                t = np.arange(st_i[0].stats.npts)*self.dt
                self.l[axii], = self.ax[axii].plot(t,st_i[0].data,'k-')
                # t0 marker
                self.ax[axii].plot([self.t0,self.t0],[np.min(st_i[0].data)*1.05, 1.05*np.max(st_i[0].data)],'g--',linewidth=2)
                # phase marker
                if self.phase_picks != None:
                    if self.key in self.phase_picks['t_onset']:
                        pPicks = self.phase_picks['t_onset'][self.key]
                    else:
                        pPicks = []
                        xb = self.twind[1]-2.5
                    pw_window = self.phase_picks['pw_window']
                    for phase_id in pPicks:
                        t_onset = UTCDateTime(pPicks[phase_id][0]) - self.st_filt[0].stats.starttime
                        xa,xb = t_onset-pw_window[phase_id][0], t_onset+pw_window[phase_id][1]
                        self.ax[axii].plot([t_onset,t_onset],[np.min(st_i[0].data)*1.05, 1.05*np.max(st_i[0].data)],'b--',linewidth=2)
                        #self.ax[axii].plot([xa,xa],[np.min(st_i[0].data)*1.05, #1.05*np.max(st_i[0].data)],'c--',linewidth=1)
                        #self.ax[axii].plot([xb,xb],[np.min(st_i[0].data)*1.05, #1.05*np.max(st_i[0].data)],'c--',linewidth=1)
                    self.ax[axii].set_xlim((self.t0,xb+2.5))
                        
                else:
                    # time window
                    if len(self.twind) == 2:
                        self.ax[axii].set_xlim((self.t0-self.twind[0],self.t0+self.twind[1]))
                        # peak amplitude window marker
                        if len(self.Pwind) == 2:
                            Pindx = int(np.mean(np.array(Ptime)))
                            tPeak = t[Pindx]
                            if self.t0 < tPeak-self.Pwind[0]:
                                self.ax[axii].plot([tPeak-self.Pwind[0],tPeak-self.Pwind[0]],[np.min(st_i[0].data)*1.05, 1.05*np.max(st_i[0].data)],'r--',linewidth=2)
                                tp1 = tPeak-self.Pwind[0]
                            else:
                                self.ax[axii].plot([self.t0-self.twind[0],self.t0-self.twind[0]],[np.min(st_i[0].data)*1.05, 1.05*np.max(st_i[0].data)],'r--',linewidth=2)
                                tp1 = self.t0
                            if self.t0+self.twind[1] > tPeak+self.Pwind[1]:
                                self.ax[axii].plot([tPeak+self.Pwind[1],tPeak+self.Pwind[1]],[np.min(st_i[0].data)*1.05, 1.05*np.max(st_i[0].data)],'r--',linewidth=2)
                                tp2 = tPeak+self.Pwind[1]
                            else:
                                self.ax[axii].plot([self.t0+self.twind[1],self.t0+self.twind[1]],[np.min(st_i[0].data)*1.05, 1.05*np.max(st_i[0].data)],'r--',linewidth=2)
                                tp2 = self.t0+self.twind[1]
                            # save time window 
                            self.Peak_Signal_Window[self.key] = [str(self.st_filt[0].stats.starttime+tp1),
                                                                str(self.st_filt[0].stats.starttime+tp2)]
                        #else:
                        #    self.ax[axii].set_xlim((self.t0,xb+2.5))   
        
                # set y_scale
                if self.A_scale == 'Asin':
                    self.ax[axii].set_ylim((np.min(st_i[0].data)*1.05, 1.05*np.max(st_i[0].data)))
                elif self.A_scale == 'Arel':
                    self.ax[axii].set_ylim((np.min(np.asarray(A_min))*1.05, 1.05*np.max(np.asarray(A_max))))
   
            
            
            # axis label
            if jj_suf == '0':
                self.ax[axii].set_title(st_i[0].id, fontsize=14)
            elif jj_suf == '1':
                if self.Comp_List[jj] == 'R': 
                    starttime = str(st_i[0].stats.starttime)
                    self.ax[axii].set_xlabel('Time since '+event_time+' in s', fontsize=14)
            if self.Comp_List[jj] == 'Z':
                self.ax[axii].set_ylabel("Amplitude in m", fontsize=14)
            self.ax[axii].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            self.ax[axii].grid(True)
            
            
            # set background color
            if self.phase_picks == None:
                if self.trace_dict[self.key][self.Comp_List[jj]]:
                    self.ax[axii].set_facecolor(self.bcolor)
                else:
                    self.ax[axii].set_facecolor('red')
            else:
                pw_window = self.phase_picks['pw_window']
                if self.key in self.phase_picks['t_onset']:
                    self.ax[axii].set_facecolor('white')
                    pPicks = self.phase_picks['t_onset'][self.key]
                    for phase_id in pPicks:
                        t_onset = UTCDateTime(pPicks[phase_id][0]) - self.st_filt[0].stats.starttime
                        xa,xb = t_onset-pw_window[phase_id][0], t_onset+pw_window[phase_id][1]
                        if self.trace_dict[self.key][self.Comp_List[jj]][phase_id]:
                            if jj_suf == '1':
                                self.ax[axii].axvspan(xa, xb, alpha=0.5, color='green')
                        else:
                            if jj_suf == '1':
                                self.ax[axii].axvspan(xa, xb, alpha=0.5, color='red')
                else:
                    self.ax[axii].set_facecolor('grey')
                
                
        
        # update figure
        self.canvas.get_tk_widget().pack()
        self.canvas.draw()
        self.canvas.callbacks.connect('button_press_event', self.callback)
        self.canvas.flush_events()


  

