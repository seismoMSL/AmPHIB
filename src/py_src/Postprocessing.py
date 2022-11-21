from .util import Tape2M, compute_shift, get_P
from .Modeller import Full_MT_modeller, Test_Mechanism
from .Utils.calc_Kagan import get_kagan_angle
from .Utils.FMC import get_Tenary, circles

from decimal import Decimal
import matplotlib.pyplot as plt      
import numpy as np
import math
from obspy.imaging.beachball import beach, aux_plane
from obspy import UTCDateTime
from scipy import integrate
import copy
import scipy.stats as stats

##################################################################################################
##################################################################################################
### Cluster_Result
##################################################################################################
##################################################################################################

class Cluster_Result:
    
    def __init__(self,Container=None,OBS=None,SYNT=None):
        '''
        
        '''
        self.Container = Container
        self.OBS = OBS
        self.SYNT = SYNT
        self.event_id = Container['source']['event_id']
        self.magnitude_update = Container['inversion']['magnitude_update']
        self.mspace_List = np.asarray(Container['postprocessing']['mspace_List'])
        self.cmax = Container['postprocessing']['cmax']
        self.rem_fac = Container['postprocessing']['rem_fac']
        self.dfault = self.Container['inversion']['Xtree']['Xtree_range_F1'].copy()
        self.ocDim = len(np.nonzero(self.dfault)[0])
    
    def get_cluster(self,Result=None,method='Kagan',mrad=[10.0,15.0]):
        '''
        
        '''
        if Result == None:
            print('Result dictionary is missing!')
            raise SystemExit
        if method == 'Kagan':
            self.get_Kagan_cluster(Result,mKagan=mrad)
        elif method == 'sdr':
            if Container['source']['simulate_doublet']:
                print('Doublet clustering is not implemented for sdrci. Use Kagan instead.')
            else:
                self.get_sdrci_cluster(Result,msdr=mrad)
        else:
            print('Method does not exist!')
            raise SystemExit
        return output
    
    def split_ranking_list(self,ranking_list):
        '''
            
        '''
        # create dictionary of split lists
        ranking_list1, ranking_list2 = {0:[]}, {0:[]}
        
        for ii, res_list in enumerate(ranking_list[0]):
            # get comb fm
            fm = ranking_list[0][ii][1] 
            
            # deep copy of res_list (else it will be overwritten)
            res_list1 = copy.deepcopy(res_list)
            # copy comb ranking list to f1 sublist
            ranking_list1[0].append(res_list1)
            # delete parameter of f2 in mech tuple
            ranking_list1[0][ii][1] = tuple(list(fm[:5])+list(fm[-2:])) # append dt, dm
            
            # deep copy of res_list (else it will be overwritten)
            res_list2 = copy.deepcopy(res_list)
            # copy comb ranking list to f2 sublist
            ranking_list2[0].append(res_list2)
            # delete parameter of f1 in mech tuple
            ranking_list2[0][ii][1] = fm[5:] 
            
        return ranking_list1, ranking_list2    
    
    def clustering_sdrci(self,ranking_list,fm_top,fm_aux):
        res_cont = []
        rem_cont = {0:[],1:[]}

        mspace0 = self.mspace_List*self.msdr[0]
        mspace1 = self.mspace_List*self.msdr[-1]
        for res_list in ranking_list[0]:
            fm = res_list[1]
            c_check = []
            for indx in range(len(fm_top)):
                if self.dfault[indx] > 0.:
                    if fm_top[indx]-mspace0[indx] <= fm[indx] <= fm_top[indx]+mspace0[indx]:
                        c_check.append(0)
                    elif fm_aux[indx]-mspace0[indx] <= fm[indx] <= fm_aux[indx]+mspace0[indx]:
                        c_check.append(0)
                    elif fm_top[indx]-mspace1[indx] <= fm[indx] <= fm_top[indx]+mspace1[indx]:
                        c_check.append(1)
                    elif fm_aux[indx]-mspace1[indx] <= fm[indx] <= fm_aux[indx]+mspace1[indx]:
                        c_check.append(1)
                    else:
                        c_check.append(2)
                    #print(indx,fm_top[indx],fm_aux[indx],mspace0[indx],fm[indx],c_check)
            if c_check.count(0) >= self.ocDim:
                res_cont.append(res_list)
                des = 0
            elif c_check.count(1) == self.ocDim:
                rem_cont[1].append(res_list)
                des = 1
            else:
                rem_cont[0].append(res_list)
                des = 2
            #print(des) 
            #print()
        return res_cont, rem_cont
       
    def get_sdrci_cluster(self,Result,msdr=[1.0,2.0]):
        '''
        
        '''
        self.res_logger = Result['res_Logger']
        self.msdr = msdr
        import copy
        
        # copy dictionary
        ranking_list = copy.deepcopy(Result['result']) #  deepcopy as normal copy does not work
        
        # set Parameter and create lists and dictionaries
        res_cont = {}
        Ninit = len(ranking_list[0])
        
        pdf = []
        for fi in range(len(ranking_list[0])):
            pdf.append(ranking_list[0][fi][0])
        pdf = np.asarray(pdf)    
        pdf_sum = np.sum(pdf)
        
        # get global top result
        mech_top = ranking_list[0][0][1]
        s2,d2,r2 = aux_plane(mech_top[0],mech_top[1],mech_top[2])
        mech_top_aux = [s2,d2,r2,mech_top[3],mech_top[4]]
        
        clust = True
        c_indx = 0
        while clust:
            res_cont[c_indx], ranking_list = self.clustering_sdrci(ranking_list,mech_top,mech_top_aux)
            try:
                # get new top mechanism with auxillary setting
                mech_top = ranking_list[0][0][1]
                s2,d2,r2 = aux_plane(mech_top[0],mech_top[1],mech_top[2])
                mech_top_aux = [s2,d2,r2,mech_top[3],mech_top[4]]

                c_indx += 1
                if len(ranking_list[0]) < Ninit*self.rem_fac:
                    clust = False
                elif c_indx == self.cmax:
                    clust = False
                    
            except:
                clust = False
        
        # display results and create pobability key list (pobability of key in res_cont)
        prob_key_list = [] # probability key list
        for iii,ii in enumerate(res_cont):
            # get probability
            pdf = []
            for jj in range(len(res_cont[ii])):
                pdf.append(res_cont[ii][jj][0])
            P = 100*np.sum(np.asarray(pdf))/pdf_sum
            prob_key_list.append([P,ii])
        
        # create output dict in Result format
        output = {'f1':{'cluster':{},'key':sorted(prob_key_list, reverse=True)},
                  'TShift_log':Result['TShift_log']}
        for ii in range(len(res_cont)):
            output['f1']['cluster'][ii] = {'result':{0:res_cont[ii]}}
            
        # save as class variable
        self.output = output        
       
    def get_Kagan_cluster(self,Result,mKagan=[10.0,15.0]):
        '''
        
        '''
        if self.Container['source']['simulate_doublet']:
            self.get_Kagan_cluster_doublet(Result,mKagan=mKagan)
        else:
            self.get_Kagan_cluster_single(Result,mKagan=mKagan)
       
       
    def clustering_kagan(self,ranking_list,fm_top):
        res_cont = []
        rem_cont = {0:[],1:[]}
        N = len(ranking_list[0])
        for res_list in ranking_list[0]:
            fm = res_list[1]
            Kagan = get_kagan_angle(fm_top[0], fm_top[1], fm_top[2], fm[0], fm[1], fm[2])           
            if Kagan != Kagan: # check for nan
                ranking_list[0].remove(res_list)
            else:
                if Kagan <= self.mKagan[0]:
                    res_cont.append(res_list)
                elif self.mKagan[0] < Kagan < self.mKagan[1]:
                    rem_cont[1].append(res_list)
                elif Kagan >= self.mKagan[1]: 
                    rem_cont[0].append(res_list)
        return res_cont, rem_cont
    
    def get_Kagan_cluster_single(self,Result,mKagan=[10.0,15.0]):
        '''
        
        '''
        self.res_logger = Result['res_Logger']
        self.mKagan = mKagan
        
        import copy

        # copy dictionary
        ranking_list = copy.deepcopy(Result['result']) #  deepcopy as normal copy does not work

        # set Parameter and create lists and dictionaries
        res_cont = {}
        Ninit = len(ranking_list[0])

        pdf = []
        for fi in range(len(ranking_list[0])):
            pdf.append(ranking_list[0][fi][0])
        pdf = np.asarray(pdf)    
        pdf_sum = np.sum(pdf)

        # get global top result
        fm_top = ranking_list[0][0][1]

        clust = True
        c_indx = 0
        while clust:           
            res_cont[c_indx], ranking_list = self.clustering_kagan(ranking_list,fm_top)
            
            try:
                # get new top mechanism with auxillary setting
                fm_top = ranking_list[0][0][1]

                c_indx += 1
                if len(ranking_list[0]) < Ninit*self.rem_fac:
                    clust = False
                if c_indx == self.cmax:
                    clust = False
            except:
                clust = False
        
        prob_key_list = [] # probability key list
        for iii,ii in enumerate(res_cont):
            # get probability
            pdf = []
            for jj in range(len(res_cont[ii])):
                pdf.append(res_cont[ii][jj][0])
            P = 100*np.sum(np.asarray(pdf))/pdf_sum
            prob_key_list.append([P,ii])
        
        # create output dict in Result format
        output = {'f1':{'cluster':{},'key':sorted(prob_key_list, reverse=True)},
                  'TShift_log':Result['TShift_log']}
        for ii in range(len(res_cont)):
            output['f1']['cluster'][ii] = {'result':{0:res_cont[ii]}}
        
        # save as class variable
        self.output = output
    
    def get_Kagan_cluster_doublet(self,Result,mKagan=[10.0,15.0]):
        '''
        
        '''
        self.res_logger = Result['res_Logger']
        self.mKagan = mKagan
        
        import copy

        # copy dictionary
        ranking_list = copy.deepcopy(Result['result']) #  deepcopy as normal copy does not work
        
        # set Parameter and create lists and dictionaries
        res_cont1, res_cont2 = {}, {}
        Ninit = len(ranking_list[0])

        pdf = []
        for fi in range(len(ranking_list[0])):
            pdf.append(ranking_list[0][fi][0])
        pdf = np.asarray(pdf)    
        pdf_sum = np.sum(pdf)
        
        # split ranking_list
        ranking_list1, ranking_list2 = self.split_ranking_list(ranking_list)  
        
        #print(ranking_list1[0][0])
        #print()
        #print(ranking_list2[0][0])
        #print(asdf)
        
        # get global top result
        fm1_top = ranking_list1[0][0][1]
        fm2_top = ranking_list2[0][0][1]
        
        ####################################################################
        # cluster fm1
        clust1 = True
        c_indx1 = 0
        while clust1:           
            res_cont1[c_indx1], ranking_list1 = self.clustering_kagan(ranking_list1,fm1_top)

            
            try:
                # get new top mechanism with auxillary setting
                fm1_top = ranking_list1[0][0][1]

                c_indx1 += 1
                if len(ranking_list1[0]) < Ninit*self.rem_fac:
                    clust1 = False
                if c_indx1 == self.cmax:
                    clust1 = False
            except:
                clust1 = False
        
        # get probability for fm1 cluster
        prob_key_list1 = [] # probability key list
        for iii,ii in enumerate(res_cont1):
            # get probability
            pdf = []
            for jj in range(len(res_cont1[ii])):
                pdf.append(res_cont1[ii][jj][0])
            P = 100*np.sum(np.asarray(pdf))/pdf_sum
            prob_key_list1.append([P,ii])
        
        ####################################################################
        # cluster fm2
        clust2 = True
        c_indx2 = 0
        while clust2:           
            res_cont2[c_indx2], ranking_list2 = self.clustering_kagan(ranking_list2,fm2_top)
            
            try:
                # get new top mechanism with auxillary setting
                fm2_top = ranking_list2[0][0][1]

                c_indx2 += 1
                if len(ranking_list2[0]) < Ninit*self.rem_fac:
                    clust2 = False
                if c_indx2 == self.cmax:
                    clust2 = False
            except:
                clust2 = False
        
        # get probability for fm2 cluster
        prob_key_list2 = [] # probability key list
        for iii,ii in enumerate(res_cont2):
            # get probability
            pdf = []
            for jj in range(len(res_cont2[ii])):
                pdf.append(res_cont2[ii][jj][0])
            P = 100*np.sum(np.asarray(pdf))/pdf_sum
            prob_key_list2.append([P,ii])
        
        ####################################################################
        # create output dict in Result format
        output = {'f1':{'cluster':{},'key':sorted(prob_key_list1, reverse=True)},
                  'f2':{'cluster':{},'key':sorted(prob_key_list2, reverse=True)},
                  'fcomb':ranking_list,
                  'TShift_log':Result['TShift_log']}
        for ii in range(len(res_cont1)):
            output['f1']['cluster'][ii] = {'result':{0:res_cont1[ii]}}
        for ii in range(len(res_cont2)):
            output['f2']['cluster'][ii] = {'result':{0:res_cont2[ii]}}
        
        # save as class variable
        self.output = output
        
    
    def display_cluster(self,fx='f1',Ncmin=10,filename=None):
        '''
        
        '''        
        ###########################
        ### collect cluster infos
        p, nl2, mw, t0, Kagan, CLVD, ISO, MT, MTb, MTb2 = {},{},{},{},{},{},{},{},{},{}
        for cidx, sc_key in enumerate(self.output[fx]['cluster']):
            p[sc_key], nl2[sc_key], mw[sc_key] = [], [], []
            t0[sc_key], Kagan[sc_key], CLVD[sc_key], ISO[sc_key] = [],[], [], []
            MT[sc_key] = []
            for rii, res in enumerate(self.output[fx]['cluster'][sc_key]['result'][0]):
                try:
                    p[sc_key].append(res[0])
                    nl2[sc_key].append(res[7])
                    mw[sc_key].append(self.res_logger[res[6]]['Magnitude'][self.magnitude_update]['Mw'])
                    fm = res[1]
                    if fm[3] == fm[3]:
                        CLVD[sc_key].append(fm[3])
                    if fm[4] == fm[4]:
                        ISO[sc_key].append(fm[4])
                    if len(self.output['TShift_log']) < 1:
                        t0[sc_key].append(0.0)
                    else:
                        time = self.output['TShift_log'][res[6]]['netw'][0]
                        t0[sc_key].append(time)
                    
                    MT[sc_key].append(Tape2M(fm[0],fm[1],fm[2],fm[3],fm[4],1.0))
                    if rii == 0:
                        nofill = False
                        MTb[sc_key] = fm
                        s2,d2,r2 = aux_plane(MTb[sc_key][0], MTb[sc_key][1], MTb[sc_key][2])
                        MTb2[sc_key] = [np.round(s2,1),np.round(d2,1),np.round(r2,1),MTb[sc_key][3],MTb[sc_key][4]]
                    else:
                        nofill = True
                        kagan = get_kagan_angle(MTb[sc_key][0], MTb[sc_key][1], MTb[sc_key][2],
                                    fm[0],fm[1],fm[2])
                        if kagan == kagan: # check if number
                            Kagan[sc_key].append(kagan)
                    
                except:
                    continue
            
        ###########################
        ### quality check
        sc_key_list = []
        for sc_key in p:
            if len(p[sc_key]) >= Ncmin:
                sc_key_list.append(sc_key)
        
        
        ###########################
        ### create figure
        # number of cluster
        fx_dim = math.ceil(len(sc_key_list)/2)
        fig = plt.figure(figsize=(20,5*fx_dim), facecolor='w', edgecolor='k')
        for cidx, sc_key in enumerate(sc_key_list):
            ax = plt.subplot(fx_dim,4,2*(cidx+1)-1) # beachball
            for fmi in range(len(p[sc_key])):
                if fmi == 0:
                    nofill = False
                else:
                    nofill = True
                try: # beach can be instable for zeros in MT
                    b = beach(MT[sc_key][fmi].tolist(), width=200, xy=(0, 0),linewidth=5, 
                    facecolor='y',edgecolor='k' , alpha=1.0, nofill=nofill)
                    b.set_zorder(50)
                    ax.add_collection(b)
                except:
                    continue
            ax.set_xlim([-103, 103])
            ax.set_ylim([-103, 103])
            plt.axis('off')
            
            ax = plt.subplot(fx_dim,4,2*(cidx+1)) # info
            ax.text(0.1,10,'P = '+str(int(100*np.sum(np.asarray(p[sc_key]))))+'%',fontsize=15)
            ax.text(0.1,9,'NL2 = '+str(np.round(np.mean(np.asarray(nl2[sc_key])),3))+'+-'+str(np.round(np.std(np.asarray(nl2[sc_key])),3)),fontsize=15)
            ax.text(0.1,8,'(s,d,r,clvd,iso) = ',fontsize=15)
            ax.text(0.1,7,'('+str(MTb[sc_key][0])+','+str(MTb[sc_key][1])+','+str(MTb[sc_key][2])+') \ ',fontsize=15)
            ax.text(0.1,6,'('+str(MTb2[sc_key][0])+','+str(MTb2[sc_key][1])+','+str(MTb2[sc_key][2])+')',fontsize=15)
            ax.text(0.1,5,'CLVD = '+str(int(np.mean(np.asarray(CLVD[sc_key]))))+'%+-'+str(int(np.std(np.asarray(CLVD[sc_key]))))+'%',fontsize=15)
            ax.text(0.1,4,'ISO = '+str(int(np.mean(np.asarray(ISO[sc_key]))))+'%+-'+str(int(np.std(np.asarray(ISO[sc_key]))))+'%',fontsize=15)
            ax.text(0.1,3,'Kagan = '+str(np.round(np.mean(np.asarray(Kagan[sc_key])),1))+'+-'+str(np.round(np.std(np.asarray(Kagan[sc_key])),1)),fontsize=15)                
            ax.text(0.1,2,'t0-shift = '+str(round(np.mean(np.asarray(t0[sc_key])),2))+'+-'+str(round(np.std(np.asarray(t0[sc_key])),2)),fontsize=15)
            ax.text(0.1,1,'Mw = '+str(round(np.mean(np.asarray(mw[sc_key])),2))+'+-'+str(round(np.std(np.asarray(mw[sc_key])),2)),fontsize=15)

            ax.set_xlim([0, 12])
            ax.set_ylim([0, 10])
            plt.axis('off')
            
        if filename == None:
            plt.show()
        else:
            plt.savefig(filename+'.png', 
                        bbox_inches='tight', 
                        transparent=False,
                        pad_inches=0)
    
    
    def get_dFM(self,fm0,fm1):
        '''
        
        '''
        # get aux plane of fm1
        s2,d2,r2 = aux_plane(fm1[0],fm1[1],fm1[2])
        fm2 = [s2,d2,r2]
        
        dfm, dfmi, fx_diff = [], [] ,[]
        for ii in range(3):
            fx_diff_temp = [np.abs(fm0[ii]-fm1[ii]),np.abs(fm0[ii]-fm2[ii])]
            fx_diff.append(fx_diff_temp)
            dfmi.append(np.argmin(fx_diff_temp))
        
        # get dfm of plane nearest to fm0 
        if sum(dfmi) >= 2: # aux_plane fm2
            for ii in range(3):
                dfm.append(fx_diff[ii][1])
        elif sum(dfmi) <= 1: # original plane fm1
            for ii in range(3):
                dfm.append(fx_diff[ii][0])
        
        return dfm
    
    def weighted_avg_and_std(self,values, weights):
        """
        Return the weighted average and standard deviation.

        values, weights -- Numpy ndarrays with the same shape.
        """
        average = np.average(values, weights=weights)
        # Fast and numerically precise:
        variance = np.average((values-average)**2, weights=weights)
        return round(average,4), round(np.sqrt(variance),4)
    
    
    def plot_simple_STF(self,cl_print=0,filename=None):
        '''
            plot dt and dm per cluster
        '''
        # access cluster
        P, DM, DT = {}, {}, {}
        for cl in self.output['f1']['cluster']:
            P[cl], DM[cl], DT[cl] = [], [], []
            for ires in range(len(self.output['f1']['cluster'][cl]['result'][0])):
                P[cl].append(self.output['f1']['cluster'][cl]['result'][0][ires][0])
                DM[cl].append(self.output['f1']['cluster'][cl]['result'][0][ires][1][5])
                DT[cl].append(self.output['f1']['cluster'][cl]['result'][0][ires][1][6])
        
        # get P weight
        Pa = np.array(P[cl_print])
        Pa = Pa/np.sum(Pa)
        # convet to numpy array
        DTa = np.array(DT[cl_print])
        DMa = np.array(DM[cl_print])
        # compute weighted average and std
        DTav, DTst = self.weighted_avg_and_std(DTa, Pa)
        DMav, DMst = self.weighted_avg_and_std(DMa, Pa)
        
        # create t vector
        x = np.linspace(0,1.5*DTav,100)
        
        # compute norm dist. time
        y = stats.norm.pdf(x, DTav, DTst)
        y /= np.max(y)
        
        # compute uncertanity range 
        y1 = ((1.-DMav)-DMst)*y
        y2 = ((1.-DMav)+DMst)*y
        ym = (1.-DMav)*y
        
        # plot
        fig = plt.figure(figsize=(20, 7), facecolor='w', edgecolor='k')
        plt.fill_between(x, y1, y2, alpha=0.4)
        plt.plot(x, ym, 'r-',linewidth=5)
        plt.plot([0,0],[0,DMav],'r-',linewidth=8)
        plt.errorbar(0, DMav, DMst, marker='o', zorder = 10,linewidth=4)
        plt.grid(True)
        plt.xticks(fontsize=20,weight='bold')
        plt.yticks(fontsize=20,weight='bold')
        plt.xlabel('Time in s',fontsize=30,weight='bold')
        plt.ylabel('rel. Moment in 1',fontsize=30,weight='bold')
        plt.title('dt = '+str(round(DTav,2))+'+-'+str(round(DTst,2))
          +' | dm = '+str(round(DMav,2))+'+-'+str(round(DMst,2)),
          fontsize=40,weight='bold')
        
        if filename == None:
            plt.show()
        else:
            plt.savefig(filename+'.png'), 
                        #bbox_inches='tight', 
                        #transparent=False,
                        #pad_inches=0)       
            
    
    def plot_waveforms(self,filename=None):
        '''
            uses the Test_Mechanism class
        '''
        # check input
        if self.OBS == None or self.SYNT == None:
            print('Please provide observables and synthetics.')
            print('Cluster_Result(Container,OBS=OBS,SYNT=SYNT)')
            raise SystemExit
        
        # plot waveforms
        TestMech = Test_Mechanism(Container=self.Container,Observed=self.OBS,Fundamentals=self.SYNT)
        if self.Container['source']['simulate_doublet']:
            # get filename
            if filename is not None:
                filename_split = filename+'_split'
                filename_comb = filename+'_comb'
            else:
                filename_split = filename
                filename_comb = filename
            # get mechanism
            fm1 = list(self.output['f1']['cluster'][0]['result'][0][0][1][:5])
            fm2 = list(self.output['f2']['cluster'][0]['result'][0][0][1])
            mechanismD = fm1+fm2 # both fmx have the same dt and dm          
            # runt Test_mechanism waveform plotter
            TestMech.run(mechanism=mechanismD,split_doublet=True)
            TestMech.plot_waveforms(split_doublet=True,filename=filename_split)
            TestMech.plot_waveforms(split_doublet=False,filename=filename_comb)
        else:
            mechanismD = list(self.output['f1']['cluster'][0]['result'][0][0][1][:5])
            TestMech.run(mechanism=mechanismD,split_doublet=False)
            TestMech.plot_waveforms(split_doublet=False,filename=filename)
    
    
    def plot_station_delay_map(self,filename=None):
        '''
        
        
        
        for stat in Result_DC_Cd['TShift_log'][key]['stat'].keys():            
            DT = []
            for ii in range(750):
                key = Result_DC_Cd['result'][0][ii][6]
                dt_netw = Result_DC_Cd['TShift_log'][key]['netw']
                DT.append(Result_DC_Cd['TShift_log'][key]['stat'][stat][0] - dt_netw[0])
            dt = np.mean(np.asarray(DT))
            dt_std = np.std(np.asarray(DT))
            
            lon, lat = m(slist[stat][1],slist[stat][0])
            
            if dt < 0:
                col = 'b'
            elif dt > 0:
                col = 'r'
            
            circ_A = plt.Circle((lon,lat), np.abs(dt)*0.1, color=col)
            circ_B = plt.Circle((lon,lat), np.abs(dt_std)*0.1, color='k')
            
            if dt >= dt_std:
                ax.add_patch(circ_A)
                ax.add_patch(circ_B)
            elif dt < dt_std:
                ax.add_patch(circ_B)
                ax.add_patch(circ_A)
            
            dlat = np.abs(dt)*0.135
            ax.text(lon,lat+dlat,stat.split('_')[1],fontsize=18,ha="center",weight='bold')
        #plt.ylim((46,52))
        #plt.xlim((6,12))
        #plt.axis('square')
        plt.show()
        '''
        return None
    
    def print_result(self,fx='f1',ev_suffix='',filename=None):
        '''
        
        '''
        cl_dict = {}
        for cidx, sc_key in enumerate(self.output[fx]['cluster']):
            p, nl2, mw, t0, Kagan, CLVD, ISO = [],[],[],[],[], [], []
            DM, DT = [], []
            ds,dd,dr= [], [], []
            for rii, res in enumerate(self.output[fx]['cluster'][sc_key]['result'][0]):
                try:
                    # get P
                    p.append(res[0])
                    # get NL2
                    nl2.append(res[7])
                    # get fault
                    fm = res[1]
                    if self.Container['source']['simulate_doublet']:
                        # get DM
                        DM.append(res[1][5])
                        # get DT
                        DT.append(res[1][6])
                    # get CLVD and ISO
                    if fm[3] == fm[3]:
                        CLVD.append(fm[3])
                    if fm[4] == fm[4]:
                        ISO.append(fm[4])
                    # get Mw
                    mw.append(self.res_logger[res[6]]['Magnitude'][self.magnitude_update]['Mw'])
                    # get time
                    if len(self.output['TShift_log']) < 1:
                        t0.append(0.0)
                    else:
                        time = self.output['TShift_log'][res[6]]['netw'][0]
                        t0.append(time)
                    # get Kagan
                    if rii == 0:
                        MTb = fm
                    else:
                        kagan = get_kagan_angle(MTb[0], MTb[1], MTb[2],fm[0],fm[1],fm[2])
                        if kagan == kagan: # check if number
                            Kagan.append(kagan)
                    # get dfm
                    if rii == 0:
                        MTb = fm
                        # check if MTb is within dStrike to 0 or 360
                        # --> avoid issues with periodicity
                        dStrike = 25
                        if 360-dStrike > MTb[0] < dStrike:
                            s2,d2,r2 = aux_plane(MTb[0],MTb[1],MTb[2])
                            MTb = [s2,d2,r2] # use aux_plane
                    else:
                        dfm = self.get_dFM(MTb,fm)
                        ds.append(dfm[0])
                        dd.append(dfm[1])
                        dr.append(dfm[2])
                except:
                    continue
            
            if self.Container['source']['simulate_doublet']:
                # get mean DT,DM
                # get P weight
                Pa = np.array(p)
                Pa = Pa/np.sum(Pa)
                # convet to numpy array
                DTa = np.array(DT)
                DMa = np.array(DM)
                # compute weighted average and std
                DTav, DTst = self.weighted_avg_and_std(DTa, Pa)
                DMav, DMst = self.weighted_avg_and_std(DMa, Pa)
            else:
                DTav, DTst = 0.0, 0.0
                DMav, DMst = 1.0, 0.0
            
            
            # save info to cluster dict
            cl_dict[sc_key] = {'P':int(100*np.sum(np.asarray(p))),
                               'NL2':[np.round(np.mean(np.asarray(nl2)),3),np.round(np.std(np.asarray(nl2)),3)],
                               'Mw':[np.round(np.mean(np.asarray(mw)),3),np.round(np.std(np.asarray(mw)),3)],
                               'fault':[MTb[0],MTb[1],MTb[2]],
                               'dfault':[np.mean(np.asarray(ds)),np.mean(np.asarray(dd)),np.mean(np.asarray(dr))],
                               'CLVD':[np.round(np.mean(np.asarray(CLVD)),0),np.round(np.std(np.asarray(CLVD)),0)],
                               'ISO':[np.round(np.mean(np.asarray(ISO)),0),np.round(np.std(np.asarray(ISO)),0)],
                               'Kagan':[np.round(np.mean(np.asarray(Kagan)),1),np.round(np.std(np.asarray(Kagan)),1)],
                               'T_shift':[round(np.mean(np.asarray(t0)),2),round(np.std(np.asarray(t0)),2)],
                               'DT':[DTav, DTst],
                               'DM':[DMav, DMst]
                                }
            
        if filename != None:
            with open(filename, 'a') as f:  
                for sc_key in cl_dict:
                    cont = cl_dict[sc_key]
                    text_format = '%s %i %.3f %.3f %.2f %.2f %.2f %.2f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.4f %.4f %.4f %.4f %s\n'
                    f.write(text_format %(self.event_id,
                                            cont['P'],
                                            cont['NL2'][0],cont['NL2'][1],
                                            cont['Mw'][0],cont['Mw'][1],
                                            cont['T_shift'][0],cont['T_shift'][1],
                                            cont['fault'][0],cont['dfault'][0],cont['fault'][1],
                                            cont['dfault'][1],cont['fault'][2],cont['dfault'][2],
                                            cont['CLVD'][0],cont['CLVD'][1],
                                            cont['ISO'][0],cont['ISO'][1],
                                            cont['Kagan'][0],cont['Kagan'][1],
                                            cont['DT'][0],cont['DT'][1],
                                            cont['DM'][0],cont['DM'][1],
                                            ev_suffix
                                            ))
            f.close()
        
        return cl_dict

   
##################################################################################################
##################################################################################################
### Plot functions
##################################################################################################
##################################################################################################

def display_iteration_process(Result,log=False):
    fig = plt.figure(figsize=(20, 10), facecolor='w', edgecolor='k')
    for ki,key in enumerate(Result['tremination_criterion']):
        plt.subplot(2,2,ki+1)
        plt.plot(Result['tremination_criterion'][key])
        if log:
            plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel(key)
        plt.grid(True)
    plt.show()



    
def plot_result_barcharts(Result,filename=None):
    
    strike,dip,rake,clvd = [],[],[],[]
    for fii,fi in enumerate(range(len(Result['result'][0]))):
        fm = Result['result'][0][fi][1]
        strike.append(fm[0])
        dip.append(fm[1])
        rake.append(fm[2])
        clvd.append(fm[3])

    info = {
            0:['Strike','Angle in degree','Occurence',[0.,360.]],
            1:['Dip','Angle in degree','Occurence',[0.,90.]],
            2:['Rake','Angle in degree','Occurence',[-180.,180.]],
            3:['CLVD','Percent (Tape)','Occurence',[-30,30]]
            }
    fig = plt.figure(figsize=(20, 10), facecolor='w', edgecolor='k')
    for bi,bar_data in enumerate([strike,dip,rake,clvd]):#,alpha,CLVD,ISO,sl,dl,rl]):

        ax = plt.subplot(2,2,bi+1)
        n, bins, patches = ax.hist(x=np.asarray(bar_data), bins='auto', color='#0504aa',
                                   alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel(info[bi][1])
        plt.ylabel(info[bi][2])
        plt.title(info[bi][0])
        maxfreq = n.max()
        ax.set_ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
        if info[bi][3] is not None:
            ax.set_xlim((info[bi][3]))

    if filename == None:
        plt.show()
    else:
        plt.savefig(filename+'.png', 
                    bbox_inches='tight', 
                    transparent=False,
                    pad_inches=0)
    
    
def get_FN_label(Container,fi):
    '''
    
    '''
    fdict = Container['preprocessing']['filter']
    
    if fdict['partition_type'] == 'lfix':
        FN_label = 'FB = '+str(fdict['freqmin'])+'-'+str(fdict['fcut'][fi])+' Hz'
    elif fdict['partition_type'] == 'lfix':
        FN_label = 'FB = '+str(fdict['fcut'][fi])+'-'+str(fdict['freqmin'])+' Hz'
    elif fdict['partition_type'] == None:
        FN_label = 'FB = '+str(fdict['freqmin'])+'-'+str(fdict['freqmax'])+' Hz'
    return FN_label
    
    
def plot_result_waveforms(OBS,Result,Container,SYNT,inv_pre,
                          prob=0.97,filename=None,
                          display_sigma=False,N_TSampl=-1,
                          Noise_Dict=None,Noise_ampli=1.0,Arel=0.5):
    
    inv = Container['inversion']
    pre = Container['preprocessing']
    # modus
    solve_for_misrot = Container['source']['solve_for_misrot']
    solve_for_misloc = Container['source']['solve_for_misloc']
    solve_for_delay = Container['source']['solve_for_delay']
    #pick_dict = inv_pre['traveltime_correction']['F1_X0']['t_update']
    dt = 1./pre['resampling']['sampling_rate']
    
    if pre['filter']['partition_type'] is not None:
        FN = len(pre['filter']['fcut'])
    else:
        FN = 1
    
    # get P
    P, cdf, nl2, VR = get_P(Result)
    if prob <= 1.0:
        MTN = (P < prob).sum()
    else:
        MTN = prob

    # tshift onsettime
    if inv['time_shift']['perform']:
        t0_tshift = []
        for MTi in range(MTN):
            key = Result['result'][0][MTi][6]
            t0_tshift.append(Result['TShift_log'][key]['netw'][0])
        event_time_update = round(np.mean(np.asarray(t0_tshift)),2)
    elif pre['selection_criteria']['perform_envelope_shift']:
        event_time_update = round(inv_pre['twind'][0][0],2)
    else:
        event_time_update = 0.
    
    # magnitude update
    magnitude_update = inv['magnitude_update']
    
    
    for stat_id in OBS:
        
        # create figure
        fig = plt.figure(figsize=(20, 5*FN), facecolor='w', edgecolor='k')
        Peak = []
        Pmax, Pmin = [], []
        NoSf = 0
        for ci, comp in enumerate(OBS[stat_id]): 
            NC = 3#len(OBS[stat_id])
            for fii,fi in enumerate(OBS[stat_id][comp]):
                NoSf += 1
                d = OBS[stat_id][comp][fi].copy()
                t = np.arange(d.stats.npts)*d.stats.delta
                ax = plt.subplot(FN,NC,ci+1+NC*fii)
                plt.plot(t,d.data,'k-',linewidth=5,label='Obs (with noise)')
                Pmax += [np.max(d.data)]
                Pmin += [np.min(d.data)]
                tshift_list = []
                for MTi in range(MTN):
                    fm = Result['result'][0][MTi][1]
                    c_coord = np.array([fm[0],fm[1],fm[2],fm[3],0,
                                        0,0,0,0,0,0.0,0.0])
                    Modeller = Full_MT_modeller(Container=Container,Fundamentals=SYNT)
                    Synt = Modeller.simulate(source_mechanism=c_coord)
                    
                    # get time shift
                    if inv['time_shift']['perform']:
                        key = Result['result'][0][MTi][6]
                        tshift = Result['TShift_log'][key]['trace'][stat_id][comp][0]
                        #tshift = list(filter(lambda x:x[1]==key,res_data))[ci][0]
                        tshift_list.append(tshift)
                        t_synt = t-tshift
                    else:
                        t_synt = t
                    
                    # get amplitude factor
                    key = Result['result'][0][MTi][6]
                    mag = Result['res_Logger'][key]['Magnitude'][magnitude_update]['Mfac']
                    
                    uT = Synt['cF_X0'][stat_id][comp][fi].copy()
                    Pmax += [np.max(mag*uT.data)]
                    Pmin += [np.min(mag*uT.data)]
                    plt.plot(t_synt,mag*uT.data,'r--')
                    if MTi == 0:
                        plt.plot(t_synt,mag*uT.data,'r--',label='Synthetics')
                        # calc NL2 and VR
                        nl2 = np.sum((d.data-mag*uT.data)**2)/np.sum(d.data**2)
                        VR = 1.-nl2
                        if display_sigma:
                            '''
                            if solve_for_misloc:
                                u1 = synt['cF_X1'][stat_id][comp][fi].copy()
                                u2 = synt['cF_X2'][stat_id][comp][fi].copy()
                                u3 = synt['cF_X3'][stat_id][comp][fi].copy()
                                du1 = list(Mfac[comp]*(u1.data - u0.data) / self.dx_misloc[0])
                                du2 = list(Mfac[comp]*(u2.data - u0.data) / self.dx_misloc[0])
                                du3 = list(Mfac[comp]*(u3.data - u0.data) / self.dx_misloc[1])
                            
                            if solve_for_misrot:
                                dthea_misrot = Container['source']['daz']
                                u4 = Synt['cF_X4'][stat_id][comp][fi].copy()
                                # get offset
                                # u' = d/da * u --> d = u'*da / u 
                                du = u4.data*(dthea_misrot*(np.pi/180.))/uT.data
                                
                                dA1a = mag*uT.data*(1-du) - Anoise*0.6827
                                dA2a = mag*uT.data*(1+du) + Anoise*0.6827
                                dA1b = mag*uT.data*(1-2*du) - 1*Anoise*0.9545
                                dA2b = mag*uT.data*(1+2*du) + 1*Anoise*0.9545
                            '''
                            
                            if solve_for_delay:
                                #sign = np.sign(pick_dict[stat_id])
                                u5 = Synt['cF_X5'][stat_id][comp][fi].copy()
                                # get offset
                                # u' = d/da * u --> d = u'*da / u 
                                du = (u5.data*dt)/uT.data
                            
                            if Noise_Dict == None:
                                dA1a = (Arel*np.max(np.abs(d.data)))
                                dA2a = dA1a
                                dA1b, dA2b = 2*dA1a, 2*dA1a
                            else:                            
                                Nfac = (Noise_ampli*np.max(np.abs((Noise_Dict[stat_id][comp][fi].data[-100:]))))
                                dA1a = mag*uT.data + Nfac*np.ones(uT.stats.npts)
                                dA2a = mag*uT.data - Nfac*np.ones(uT.stats.npts)
                                dA1b = mag*uT.data + 2*Nfac*np.ones(uT.stats.npts)
                                dA2b = mag*uT.data - 2*Nfac*np.ones(uT.stats.npts)
                            
                            ax.fill_between(t_synt, dA1b, dA2b, 
                                            facecolor='yellow', alpha=0.35,
                                            label='2$\sigma$')
                            ax.fill_between(t_synt, dA1a, dA2a, 
                                            facecolor='green', alpha=0.35,
                                            label='$\sigma$')
                # add FB label 
                FN_label = 'FB = '+d.stats.FBand
                ax.text(0.5, 0.95,FN_label,fontsize=12, transform=ax.transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.75))   
                
                # add tshift label
                if inv['time_shift']['perform']:
                    tshift_list = np.asarray(tshift_list)
                    ts_label = 'tshift = '+str(round(np.mean(tshift_list)-event_time_update,2))+' +- '+str(round(np.std(tshift_list),2))
                    ax.text(0.5, 0.875,ts_label,fontsize=12, transform=ax.transAxes,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.75)) 
                # add phase info
                #if ['phase_window']['perform']:
                    
                
                # display NL2 and VR
                #plt.title(d.id+' (Nl2='+str(round(nl2,2))+' | VR='+str(round(VR,2))+')',fontsize=15)
                plt.title(stat_id+'_'+comp+' (Nl2='+str(round(nl2,2))+')',fontsize=18)
                plt.xlabel('Time in s',fontsize=18)
                plt.ylabel('Displacement in m',fontsize=18)
                ax.tick_params(axis='both', which='major', labelsize=15)
                plt.xlim((np.max([np.min(t_synt),np.min(t)]),np.min([np.max(t),np.max(t_synt)])))
                plt.grid(True)
                #if ci+1+3*fii == FN*3-1:
                #if ci == 0:   
                    #plt.legend(bbox_to_anchor=(0.25, 1.05, NC, .102), 
                    #ncol=2, mode="expand", borderaxespad=0.,fontsize=15)
                plt.legend(loc=4,ncol=2,fontsize=15)
                    
        for sfig in range(NoSf):
            ax = plt.subplot(FN,NC,sfig+1)
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            ax.yaxis.get_offset_text().set_fontsize(15)
            ax.set_ylim((1.85*np.min(np.asarray(Pmin)),1.25*np.max(np.asarray(Pmax))))
            #ax.set_ylim((-1.5*10**-6,1.5*10**-6))
            
                    
        if filename == None:
            plt.show()
        else:
            plt.savefig(filename+'_'+stat_id+'.png', 
                        bbox_inches='tight', 
                        transparent=False,
                        pad_inches=0)
            
    
    
    
    
def plot_result_beachball(Result,inv_pre,Container,prob=[0.15,0.3],filename=None,MT_True=[],suffix=''):
    '''
    
    '''
    # get P
    inv_method = Container['inversion']['function']
    scale_fac = Container['inversion']['exp_scaling_fac']
    P, cdf, nl2, VR = get_P(Result)
    if prob[-1] <= 1.0:
        nP = [(P < prob[0]).sum(),(P < prob[-1]).sum()]
    else:
        nP = [int(prob[0]),int(prob[1])]
        
    ecp = 0
    ecolor = {0:'r',1:'k',2:'gray'}
    Zorder = {0:20,1:15,2:10}
    
    # get time information
    event_id = Container['source']['event_id']
    event_time = Container['source']['F1_loc'][0]
    if Container['inversion']['time_shift']['perform']:
        t0_tshift = []
        for MTi in range(nP[-1]):
            key = Result['result'][0][MTi][6]
            t0_tshift.append(Result['TShift_log'][key]['netw'][0])
        event_time_update = np.mean(np.asarray(t0_tshift))
        event_time_update_uncertainty = np.std(np.asarray(t0_tshift))
    elif Container['preprocessing']['selection_criteria']['perform_envelope_shift']:
        event_time_update = inv_pre['twind'][0][0]
        event_time_update_uncertainty = inv_pre['twind'][0][1]
    else:
        event_time_update = 0.
        event_time_update_uncertainty = 0.        
    event_location = Container['source']['F1_loc'][1]
    event_model_depth = Container['source']['SRC_dict']['F1_X0'][2] #Container['source']['event_depth']
    sfilter = Container['preprocessing']['filter']
    trace_selection = Container['network']['trace_selection']
    nComp = {'Z':0,'R':0,'T':0}
    for stat_id in trace_selection:
        for ci in trace_selection[stat_id]:
            nComp[ci] += 1
    

    # figure
    fig = plt.figure(figsize=(20, 15), facecolor='w', edgecolor='k')
    ax = plt.subplot(3,2,1)
    strike,dip,rake,clvd,iso = [],[],[],[],[]
    strike2,dip2,rake2,clvd2,iso2, dM,dT = [],[],[],[],[],[],[]
    mag = {0:[],1:[],2:[]}
    NL2 = {0:[],1:[],2:[]}
    
    if len(MT_True) > 0:
        mt = Tape2M(MT_True[0],MT_True[1],MT_True[2],MT_True[3],MT_True[4],1)
        b = beach(mt, width=200, xy=(0, 0),linewidth=1, 
                  facecolor='y',edgecolor=ecolor[ecp] , alpha=1.0, nofill=False)
        b.set_zorder(5)
        ax.add_collection(b)

        
    # add black circle around beachball (as top solution is red)
    b = beach([1,1,1,0,0,0], width=200, xy=(0, 0),linewidth=5, 
                facecolor='y',edgecolor='k' , alpha=1.0, nofill=True)
    b.set_zorder(50)
    ax.add_collection(b)
    
    FM = get_fault(Result)
       
    for fi in range(nP[-1]):
        #fm = Result['result'][0][fi][1]
        fm = FM[fi]
        p = Result['result'][0][fi][0]
        
        if ecp == 0:
            strike.append(fm[0])
            dip.append(fm[1])
            rake.append(fm[2])
            clvd.append(fm[3])
            iso.append(fm[4])
            
            if len(fm) > 5:
                strike2.append(fm[5])
                dip2.append(fm[6])
                rake2.append(fm[7])
                clvd2.append(fm[8])
                iso2.append(fm[9])
                dM.append(fm[10])
                dT.append(fm[11])
            
        if fi == 0:
            if len(MT_True) > 0:
                nofill = True
            else:
                nofill = False
        else:
            nofill=True
            
        #if fi > int(len(Result['result'][0])*prob[ecp]):
        if fi > nP[ecp]:
            if len(prob) > ecp:
                ecp += 1   
        
        
        # get nl2
        nl2 = Result['result'][0][fi][7]
        if nl2 == nl2:
            NL2[ecp].append(nl2)
            
        # magnitude update
        magnitude_update = Container['inversion']['magnitude_update']
        key = Result['result'][0][fi][6]
        Mw = Result['res_Logger'][key]['Magnitude'][magnitude_update]['Mw']
        if Mw == Mw: # if false, Mw is nan
            mag[ecp].append(Mw)

            
        mt = Tape2M(fm[0],fm[1],fm[2],fm[3],fm[4],1)
        try:
            b0 = beach(mt, width=200, xy=(0, 0),linewidth=1, 
                      facecolor='y',edgecolor=ecolor[ecp] , alpha=1.0, nofill=nofill) 
        except:
            continue
            
            
        
        if len(fm) > 5:
            mt = Tape2M(fm[5],fm[6],fm[7],fm[8],fm[9],1)
            try:
                b1 = beach(mt, width=200, xy=(200, 0),linewidth=1, 
                          facecolor='y',edgecolor=ecolor[ecp] , alpha=1.0, nofill=nofill) 
            except:
                continue
        
            
        if fi == 0:
            if len(MT_True) > 0:
                b0.set_zorder(Zorder[ecp])
                if len(fm) > 5:
                    b1.set_zorder(Zorder[ecp])
            else:
                b0.set_zorder(5)
                if len(fm) > 5:
                    b1.set_zorder(5)
        else:
            b0.set_zorder(Zorder[ecp])
            if len(fm) > 5:
                b0.set_zorder(Zorder[ecp])
                
        ax.add_collection(b0)
        if len(fm) > 5:
            ax.add_collection(b1)
        
    ax.set_aspect('equal')
    #ax.set_aspect('equal')
    if len(FM[0]) > 5:
        ax.set_xlim([-103, 303])
    else:
        ax.set_xlim([-103, 103])
    ax.set_ylim([-103, 103])
    plt.axis('off')
    
    
    ax = plt.subplot(3,2,2)
    lns1 = ax.plot(cdf,'b-',label='pdf')
    plt.grid(True)
    ax.set_ylabel('pdf')
    ax.set_xlabel('Inversion number')
    
    ax2 = ax.twinx()
    for pi in range(len(prob)):
        nPx = nP[pi]
        ax2.plot([nPx,nPx],[0,1],ecolor[pi]+'--')
    lns2 = ax2.plot(P,'g-',label='probability')
    lns3 = ax2.plot(VR,'y-',label='VR')
    ax2.set_ylabel('P, VR')
    
    title_text = 'PDF and Probability (exp_scaling_fac: %.2E)' % Decimal(scale_fac)
    plt.title(title_text)
    plt.grid(True)
    # added these three lines
    lns = lns1+lns2+lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, bbox_to_anchor=(-0.1, 0.8),fontsize=12)
    
    
    ax = plt.subplot(3,2,3)
    plt.axis('off')
    fontsize = 14
    incre = 0.10
    pos = 0.99
    xpos = 0.05
    deg = u'\xb0'
    
    text = "Event information: %s"  % (event_id)
    plt.text(xpos, pos, text, fontsize=fontsize*1.5)
    pos -= incre*1.2 
    
    text = "t0: %s $\pm$ %.2f sec."  % (
        str(UTCDateTime(event_time)+event_time_update),event_time_update_uncertainty)
    plt.text(xpos, pos, text, fontsize=fontsize*1.5)
    pos -= incre*1.2 

    text = "Lat: %.4f | Lon: %.4f"  % (
        event_location[0],event_location[1])
    plt.text(xpos, pos, text, fontsize=fontsize*1.5)
    pos -= incre*1.2 
    
    text = "Depth: %.2f km (obs) / %.2f km (synth)"  % (
        event_location[2],event_model_depth)
    plt.text(xpos, pos, text, fontsize=fontsize*1.5)
    pos -= incre*1.2 
    
    text = "Number of Stations: %i  (Z:%i, R:%i, T:%i)"  % (
        len(trace_selection),nComp['Z'],nComp['R'],nComp['T'])
    plt.text(xpos, pos, text, fontsize=fontsize*1.5)
    pos -= incre*1.2    
    if sfilter['partition_type'] == 'lfix':
        text = "Frequency Band(s): %.3f - %s Hz"  % (sfilter['freqmin'],sfilter['fcut'])
    elif sfilter['partition_type'] == 'ufix':
        text = "Frequency Band(s): %s - %.3f Hz"  % (sfilter['fcut'],sfilter['freqmax'])
    else:
        text = "Frequency Band(s): %.3f - %.3f Hz"  % (sfilter['freqmin'],sfilter['freqmax'])
    plt.text(xpos, pos, text, fontsize=fontsize*1.5)
    pos -= incre*1.2
        
    ax = plt.subplot(3,2,5)
    plt.axis('off')
    fontsize = 14
    incre = 0.10
    pos = 1.25
    xpos = 0.05
    deg = u'\xb0'
        
    text = "Inversion results:"
    plt.text(xpos, pos, text, fontsize=fontsize*1.5)
    pos -= incre*1.2   
    text = "Number of Inversions: %i"  % (len(Result['result'][0]))
    plt.text(xpos, pos, text, fontsize=fontsize*1.5)
    pos -= incre*1.2
    
    if len(prob) == 1:
        nP0 = nP[0]
        text = 'Probability: '+str(round(100*P[nP0],1))+ '% (red)' 
    elif len(prob) == 2:
        nP0 = nP[0]
        nP1 = nP[1]
        text = 'Probability: '+str(round(100*P[nP0],1))+ '% (red) | '+str(round(100*P[nP1],1))+ '% (black)'
    plt.text(xpos, pos, text, fontsize=fontsize*1.5)
    pos -= incre*1.2
    
    if len(prob) == 1:
        nl2a = 1.-np.asarray(NL2[0])
        text = "VR: %.2f $\pm$ %.2f (red)" %(np.mean(nl2a),np.std(nl2a)) 
    elif len(prob) == 2:
        nl2a = 1.-np.asarray(NL2[0])
        nl2b = 1.-np.asarray(NL2[1])
        text = "VR: %.2f $\pm$ %.2f (red) | %.2f $\pm$ %.2f (black)" %(
            np.mean(nl2a),np.std(nl2a),np.mean(nl2b),np.std(nl2b)) 
    plt.text(xpos, pos, text, fontsize=fontsize*1.5)
    pos -= incre*1.2
    
    msig1 = np.asarray(mag[0])
    text = "Magnitude (Mw): %.2f $\pm$ %.2f" %(np.mean(msig1),np.std(msig1))  
    plt.text(xpos, pos, text, fontsize=fontsize*1.5)
    pos -= incre*1.2
    
    MTA = [np.mean(np.asarray(strike)),np.mean(np.asarray(dip)),np.mean(np.asarray(rake)),
           np.mean(np.asarray(clvd)),np.mean(np.asarray(iso))]
    MTB = aux_plane(MTA[0],MTA[1],MTA[2])
    
    
    if len(MT_True) > 0:
        MTA_lit = MT_True
        MTB_lit = aux_plane(MTA_lit[0],MTA_lit[1],MTA_lit[2]) 
        Kagan_lit = get_kagan_angle(MTA_lit[0], MTA_lit[1], MTA_lit[2], 
                        MTA[0], MTA[1], MTA[2])
        sRef = " (Lit. %i/%i)" %(MTA_lit[0],MTB_lit[0])
        dRef = " (Lit. %i/%i)" %(MTA_lit[1],MTB_lit[1])
        rRef = " (Lit. %i/%i)" %(MTA_lit[2],MTB_lit[2])
        cRef = " (Lit. %i%%)" %(MTA_lit[3])
        iRef = " (Lit. %i%%)" %(MTA_lit[4])
        kRef = " (Lit. %.2f)" %(Kagan_lit)
    else:
        sRef,dRef,rRef,cRef,iRef,kRef = '','','','','',''
    

    text = "Strike ($\psi$): %i/%i $\pm$ %.1f"  %(MTA[0],MTB[0],np.std(np.asarray(strike)))
    plt.text(xpos, pos, text+sRef , fontsize=fontsize*1.5)
    pos -= incre*1.2
    text = "Dip ($\delta$): %i/%i $\pm$ %.1f"  %(MTA[1],MTB[1],np.std(np.asarray(dip)))
    plt.text(xpos, pos, text+dRef, fontsize=fontsize*1.5)
    pos -= incre*1.2
    text = "Rake ($\lambda$): %i/%i $\pm$ %.1f" %(MTA[2],MTB[2],np.std(np.asarray(rake)))
    plt.text(xpos, pos, text+rRef, fontsize=fontsize*1.5)
    pos -= incre*1.2
    text = "CLVD: %i $\pm$ %.1f"  %(np.mean(np.asarray(clvd)),np.std(np.asarray(clvd)))
    plt.text(xpos, pos, text+cRef, fontsize=fontsize*1.5)
    pos -= incre*1.2
    text = "ISO: %i $\pm$ %.1f" %(np.mean(np.asarray(iso)),np.std(np.asarray(iso)))
    plt.text(xpos, pos, text+iRef, fontsize=fontsize*1.5)
    pos -= incre*1.2
    
    Kagan_inv1, Kagan_inv2 = [], []
    if len(prob) == 1:
        nP0 = nP[0]
        for fi in range(nP0):
            fm = FM[fi]
            Kagan_inv1.append(get_kagan_angle(MTA[0], MTA[1], MTA[2],
                                    fm[0],fm[1],fm[2]))
        text = "Kagan: %.2f $\pm$ %.2f" %(np.mean(np.asarray(Kagan_inv)),np.std(np.asarray(Kagan_inv)))
    elif len(prob) == 2:
        nP0 = nP[0]
        nP1 = nP[1]
        for fi in range(nP0):
            fm = FM[fi]
            Kagan_inv1.append(get_kagan_angle(MTA[0], MTA[1], MTA[2],
                                    fm[0],fm[1],fm[2]))
        for fi in range(nP1):
            fm = FM[fi]
            Kagan_inv2.append(get_kagan_angle(MTA[0], MTA[1], MTA[2],
                                    fm[0],fm[1],fm[2]))
        text = "Kagan: %.2f $\pm$ %.2f (red) | %.2f $\pm$ %.2f (black)" %(
            np.mean(np.asarray(Kagan_inv1)),np.std(np.asarray(Kagan_inv1)),
            np.mean(np.asarray(Kagan_inv2)),np.std(np.asarray(Kagan_inv2)))
    
    plt.text(xpos, pos, text+kRef, fontsize=fontsize*1.5)
    pos -= incre*1.2
    
    
    ax = plt.subplot(3,2,4, projection='polar')
    STAT_dict = Container['network']['STAT_dict']['F1_X0']
    TSelection = Container['network']['trace_selection']
    D = []
    for stat_id in STAT_dict:
        if stat_id in TSelection.keys():
            col = 'r*'
            CN = len(TSelection[stat_id])
            if CN == 1: ms = 3
            elif CN == 2: ms = 5
            elif CN == 3: ms = 7
        else:
            col = 'k*'
            ms = 3
        dist = STAT_dict[stat_id][3]/1000.
        az = STAT_dict[stat_id][4]*(np.pi/180.)
        D.append(dist)
        ax.plot(az, dist,col,markersize=ms)
    ax.set_rmax(1.1*np.max(np.asarray(D)))
    ax.set_rticks(np.array([0.25, 0.5, 0.75, 1.0])*np.max(np.asarray(D)))  # Less radial ticks
    #ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.set_theta_zero_location("N") 
    ax.set_theta_direction(-1)
    ax.grid(True)
    #ax.set_title("Regional Map", va='bottom')
    ax.plot(0,0,'r*',label='Selected')
    ax.plot(0,0,'k*',label='Not Selected')
    ax.plot(0,0,'b*',label='Source')
    ax.legend(bbox_to_anchor=(-0.1, 0.8),fontsize=12)
    
  
    ax = plt.subplot(3,2,6)
    #X,Y,size,color,zorder,alpha = get_Tenary(Result['result'],P=P,prob=prob)
    XY,Prob,size,color,zorder,alpha = get_Tenary(Result['result'],P=P,prob=prob)
    #circles(ax,X,Y,size,color,zorder,alpha,P=P,plotname=None,prob=prob,inv_method=inv_method)
    circles(ax,XY,Prob,size,color,zorder,alpha,P=P,plotname=None,prob=prob,inv_method=inv_method)
    
    
    if filename == None:
        plt.show()
    else:
        fname = filename+suffix
        plt.savefig(fname+'.png', 
                bbox_inches='tight', 
                    transparent=False,
                    pad_inches=0)



def get_fault(Result):
    
    FM_ref = Result['result'][0][0][1]
    FM_ref_aux = aux_plane(FM_ref[0],FM_ref[1],FM_ref[2])
    MT_res = [FM_ref]
    for fmi in range(len(Result['result'][0])):
        FM = Result['result'][0][fmi][1]
        F1 = np.array([np.abs(FM[0]-FM_ref[0]),np.abs(FM[1]-FM_ref[1]),np.abs(FM[1]-FM_ref[1])])
        F2 = np.array([np.abs(FM[0]-FM_ref_aux[0]),np.abs(FM[1]-FM_ref_aux[1]),np.abs(FM[1]-FM_ref_aux[1])])
        
        if np.mean(F1) < np.mean(F2):
            MT_res.append(FM)
        else:
            FM_aux = aux_plane(FM[0],FM[1],FM[2])
            MT_res.append([FM_aux[0],FM_aux[1],FM_aux[2],FM[3],FM[4]])
        
    return MT_res
    
    
    
    
