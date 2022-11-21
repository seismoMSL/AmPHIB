# Defining functions for FMC
#
# FMC, Focal Mechanisms Classification
# Copyright (C) 2015  Jose A. Alvarez-Gomez
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#
# Some of this functions are python adaptations from the
# Gasperini and Vannucci (2003) FORTRAN subroutines:
# Gasperini P. and Vannucci G., FPSPACK: a package of simple Fortran subroutines
# to manage earthquake focal mechanism data, Computers & Geosciences (2003)
#
# Version 1.01
#
#
import numpy as np
import matplotlib.pyplot as plt
from obspy.imaging.beachball import beach
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

plt.rc('pdf', fonttype=3)

from numpy import zeros, asarray, sin, cos, sqrt, dot, deg2rad, rad2deg, arccos, arcsin, arctan2, mod, genfromtxt, column_stack, atleast_2d, shape, savetxt, where, linalg, trace, log10, pi


def norm(wax, way, waz):
	"""This function Computes Euclidean norm and normalized components of a vector."""
	a=asarray((wax,way,waz))
	anorm=sqrt(dot(a,a.conj()))
	if anorm == 0:
		ax=0
		ay=0
		az=0
	else:
		ax=wax/anorm
		ay=way/anorm
		az=waz/anorm
	return ax, ay, az
	
def ca2ax(wax, way, waz):
	(ax,ay,az)=norm(wax,way,waz)
	if az<0:
		ax=-ax
		ay=-ay
		az=-az
	if ay!=0 or ax!=0:
		trend=rad2deg(arctan2(ay,ax))
	else:
		trend=0
	trend=mod(trend+360,360)
	plunge=rad2deg(arcsin(az))
	return trend, plunge
	
def nd2pl(wanx,wany,wanz,wdx,wdy,wdz):
	(anX,anY,anZ)=norm(wanx,wany,wanz)
	(dx,dy,dz)=norm(wdx,wdy,wdz)
	
	if anZ>0:
		anX=-anX
		anY=-anY
		anZ=-anZ
		dx=-dx
		dy=-dy
		dz=-dz
	if anZ==-1:
		wdelta=0
		wphi=0
		walam=arctan2(-dy,dx)
	else :
		wdelta=arccos(-anZ)
		wphi=arctan2(-anX,anY)
		walam=arctan2(-dz/sin(wdelta),dx*cos(wphi)+dy*sin(wphi))
		
	phi=rad2deg(wphi)
	delta=rad2deg(wdelta)
	alam=rad2deg(walam)
	phi=mod(phi+360,360)
	dipdir=phi+90
	dipdir=mod(dipdir+360,360)
	return phi,delta,alam,dipdir

def pl2nd(strike,dip,rake):
	""" compute Cartesian components of outward normal and slip vectors from strike, dip and rake 
	strike         strike angle in degrees (INPUT)
    dip            dip angle in degrees (INPUT)
    rake           rake angle in degrees (INPUT)
    anx,any,anz    components of fault plane outward normal vector in the 
                   Aki-Richards Cartesian coordinate system (OUTPUT)
    dx,dy,dz       components of slip versor in the Aki-Richards 
                   Cartesian coordinate system (OUTPUT)"""
	
	wstrik=deg2rad(strike)
	wdip=deg2rad(dip)
	wrake=deg2rad(rake)

	anX=-sin(wdip)*sin(wstrik)
	anY=sin(wdip)*cos(wstrik)
	anZ=-cos(wdip)
	dx=cos(wrake)*cos(wstrik)+cos(wdip)*sin(wrake)*sin(wstrik)
	dy=cos(wrake)*sin(wstrik)-cos(wdip)*sin(wrake)*cos(wstrik)
	dz=-sin(wdip)*sin(wrake)

	return anX, anY, anZ, dx, dy, dz

def pl2pl(strika,dipa,rakea):

	anX,anY,anZ,dx,dy,dz = pl2nd(strika,dipa,rakea)
	strikb, dipb, rakeb, dipdirb = nd2pl(dx,dy,dz,anX,anY,anZ)
	
	return strikb, dipb, rakeb, dipdirb
	     
def nd2pt(wanx,wany,wanz,wdx,wdy,wdz):
	"""compute Cartesian component of P, T and B axes from outward normal and slip vectors."""
	(anX,anY,anZ)=norm(wanx,wany,wanz)
	(dx,dy,dz)=norm(wdx,wdy,wdz)
	px=anX-dx
	py=anY-dy
	pz=anZ-dz
	(px,py,pz)=norm(px,py,pz)
	if pz<0:
			px=-px
			py=-py
			pz=-pz
	tx=anX+dx
	ty=anY+dy
	tz=anZ+dz
	(tx,ty,tz)=norm(tx,ty,tz)
	if tz<0:
			tx=-tx
			ty=-ty
			tz=-tz
	bx=py*tz-pz*ty
	by=pz*tx-px*tz
	bz=px*ty-py*tx
	if bz<0:
			bx=-bx
			by=-by
			bz=-bz
			
	return px, py, pz, tx, ty, tz, bx, by, bz

def nd2ar(anX,anY,anZ,dx,dy,dz,am0):
	
	wanx, wany, wanz = norm(anX,anY,anZ)
	wdx, wdy, wdz = norm(dx,dy,dz)
	
	if am0==0:
		aam0=1.0
	else:
		aam0=am0
	
	am=zeros((3,3))
	am[0][0]=aam0*2.0*wdx*wanx
	am[0][1]=aam0*(wdx*wany+wdy*wanx)
	am[1][0]=am[0][1]
	am[0][2]=aam0*(wdx*wanz+wdz*wanx)
	am[2][0]=am[0][2]
	am[1][1]=aam0*2.0*wdy*wany
	am[1][2]=aam0*(wdy*wanz+wdz*wany)
	am[2][1]=am[1][2]
	am[2][2]=aam0*2.0*wdz*wanz
      
	return am
	
def ar2ha(am):
	amo=zeros((3,3))
	amo[0][0]=am[0][0]
	amo[0][1]=-am[0][1]
	amo[0][2]=am[0][2]
	amo[1][0]=-am[1][0]
	amo[1][1]=am[1][1]
	amo[1][2]=-am[1][2]
	amo[2][0]=am[2][0]
	amo[2][1]=-am[2][1]
	amo[2][2]=am[2][2]
	
	return amo
	
def kave(plungt,plungb,plungp):
	"""x and y for the Kaverina diagram"""
	zt=sin(deg2rad(plungt))
	zb=sin(deg2rad(plungb))
	zp=sin(deg2rad(plungp))
	L=2*sin(0.5*arccos((zt+zb+zp)/sqrt(3)))
	N=sqrt(2*((zb-zp)**2+(zb-zt)**2+(zt-zp)**2))
	x=sqrt(3)*(L/N)*(zt-zp)
	y=(L/N)*(2*zb-zp-zt)
	return x, y

def mecclass(plungt,plungb,plungp):

	plunges=asarray((plungp,plungb,plungt))
	P=plunges[0]
	B=plunges[1]
	T=plunges[2]
	maxplung,axis = plunges.max(0),plunges.argmax(0)
	if maxplung >= 67.5:
		if axis == 0: # P max
			clase = 'N' # normal faulting
		elif axis == 1: # B max
			clase = 'SS' # strike-slip faulting
		elif axis == 2: # T max
			clase = 'R' # reverse faulting
	else:
		if axis == 0: # P max
			if B > T :
				clase = 'N-SS' # normal - strike-slip faulting
			else:
				clase = 'N' # normal faulting  
		if axis == 1: # B max
			if P > T :
				clase = 'SS-N' # strike-slip - normal faulting
			else:
				clase = 'SS-R' # strike-slip - reverse faulting
		if axis == 2: # T max
			if B > P :
				clase = 'R-SS' # reverse - strike-slip faulting
			else:
				clase = 'R' # reverse faulting
	return clase

def moment(am):
	# To avoid problems with cosines
	ceros = where(am==0)
	am[ceros]=0.000001
	
	# Eigenvalues and Eigenvectors
	val,vect = linalg.eig(am)
	# Ordering of eigenvalues and eigenvectors (increasing eigenvalues)
	idx = val.argsort()   
	val = val[idx]
	vect = vect[:,idx]
	
	# G&V ar2pt
	e=trace(am)/3
	dval=val-e
	
	# fclvd, seismic moment and Mw
	fclvd=(abs(dval[1]/(max((abs(dval[0])),(abs(dval[2]))))))
	am0=(abs(dval[0])+abs(dval[2]))/2
	
	return am0, fclvd, dval, vect



def baseplot(ax,plotname=None,fsize=15):
    # border	
    #fig=plt.figure()
    #plt.axes().set_aspect('equal')
    ax.set_aspect('equal')
    
    X=zeros((1,101))
    Y=zeros((1,101))
    for a in range (0,101):
        T=arcsin(sqrt(a/100.0))/(pi/180)
        B=0.0
        P=arcsin(sqrt(1-(a/100.0)))/(pi/180)
        X[0][a],Y[0][a]=kave(T,B,P)

    tickx,ticky=kave(range(0,91,10),zeros((1,10)),range(90,-1,-10))
    plt.plot(X[0],Y[0],color='black',linewidth=2)
    plt.scatter(tickx,ticky,marker=3,c='black',linewidth=2)
    for i in range(0,10):
        plt.text(tickx[0][i],ticky[0][i]-0.04,i*10,fontsize=fsize-3,verticalalignment='top')
    plt.text(0,-0.85,'T axis plunge',fontsize=fsize-3,horizontalalignment='center')

    X=zeros((1,101))
    Y=zeros((1,101))
    for a in range (0,101):
        B=arcsin(sqrt(a/100.0))/(pi/180)
        P=0.0
        T=arcsin(sqrt(1-(a/100.0)))/(pi/180)
        X[0][a],Y[0][a]=kave(T,B,P)
    
    tickx,ticky=kave(zeros((1,10)),range(0,91,10),range(90,-1,-10))
    plt.plot(X[0],Y[0],color='black',linewidth=2)
    plt.scatter(tickx,ticky,marker=0,c='black',linewidth=2)
    for i in range(0,10):
        plt.text(tickx[0][i]-0.04,ticky[0][i],i*10,fontsize=fsize-3,horizontalalignment='right')
    plt.text(-0.5,0.5,'B axis plunge',fontsize=fsize-3,horizontalalignment='center',rotation=60)

    
    X=zeros((1,101))
    Y=zeros((1,101))
    for a in range (0,101):
        P=arcsin(sqrt(a/100.0))/(pi/180)
        T=0.0
        B=arcsin(sqrt(1-(a/100.0)))/(pi/180)
        X[0][a],Y[0][a]=kave(T,B,P)
    
    plt.plot(X[0],Y[0],color='black',linewidth=1)
    
    # inner lines
    # class fields
    X=zeros((1,51))
    Y=zeros((1,51))
    for a in range(0,51):
        B=arcsin(sqrt((a/50.0)*0.14645))/(pi/180)
        T=67.5
        P=arcsin(sqrt((1-(a/50.0))*0.14645))/(pi/180)
        X[0][a],Y[0][a]=kave(T,B,P)
    
    xf=X[0][25]
    yf=Y[0][25]
    plt.plot([0,xf],[0,yf],color='grey',linewidth=1)
    plt.plot(X[0][25:51],Y[0][25:51],color='grey',linewidth=1)
    
    X=zeros((1,51))
    Y=zeros((1,51))
    for a in range(0,51):
        B=arcsin(sqrt((a/50.0)*0.14645))/(pi/180)
        P=67.5
        T=arcsin(sqrt((1-(a/50.0))*0.14645))/(pi/180)
        X[0][a],Y[0][a]=kave(T,B,P)
    
    xf=X[0][25]
    yf=Y[0][25]
    plt.plot([0,xf],[0,yf],color='grey',linewidth=1)
    plt.plot(X[0][25:51],Y[0][25:51],color='grey',linewidth=1)
    
    X=zeros((1,51))
    Y=zeros((1,51))
    for a in range(0,51):
        T=arcsin(sqrt((a/50.0)*0.14645))/(pi/180)
        B=67.5
        P=arcsin(sqrt((1-(a/50.0))*0.14645))/(pi/180)
        X[0][a],Y[0][a]=kave(T,B,P)

    plt.plot(X[0],Y[0],color='grey',linewidth=1)
    
    plt.plot([0,0],[0.555221438,-0.605810893],color='grey',linewidth=1)
    plt.plot([0,0.52481139],[0,0.303],color='grey',linewidth=1)
    plt.plot([0,-0.52481139],[0,0.303],color='grey',linewidth=1)
    # Labels
    if plotname is not None:
        plt.text(0,-0.9,plotname,horizontalalignment='center',fontsize=fsize+3)
    plt.text(-0.9,-0.5,'NF',horizontalalignment='right',fontsize=fsize)
    plt.text(0.9,-0.5,'TF',horizontalalignment='left',fontsize=fsize)
    plt.text(0,1,'SS',horizontalalignment='center',fontsize=fsize)
    
    plt.axis('off')
    
    #return fig

def SS0():
    '''

    '''
    
    # SS lower boundary:
    X_SS=zeros((1,51))
    Y_SS=zeros((1,51))
    for a in range(0,51):
        T=arcsin(sqrt((a/50.0)*0.14645))/(pi/180)
        B=67.5
        P=arcsin(sqrt((1-(a/50.0))*0.14645))/(pi/180)
        X_SS[0][a],Y_SS[0][a]=kave(T,B,P)
    
    # B axis (sub)
    X_b=zeros((1,101))
    Y_b=zeros((1,101))
    for a in range (0,101):
        P=arcsin(sqrt(a/100.0))/(pi/180)
        T=0.0
        B=arcsin(sqrt(1-(a/100.0)))/(pi/180)
        X_b[0][a],Y_b[0][a]=kave(T,B,P)
    
    # combine lines
    X = np.asarray(list(X_b[0][:15])+list(X_SS[0])+list(np.flip(-X_b[0][:15])))
    Y = np.asarray(list(Y_b[0][:15])+list(Y_SS[0])+list(np.flip(Y_b[0][:15])))
    
    # define SS area (list of tuples)
    SS_area = zip(X, Y)
    
    # create polygon
    SS_polygon = Polygon(SS_area)
    
    return SS_polygon
    
def SSNFTF2SS():
    '''
    
    '''
    
    # SS lower boundary:
    X_SS=zeros((1,51))
    Y_SS=zeros((1,51))
    for a in range(0,51):
        T=arcsin(sqrt((a/50.0)*0.14645))/(pi/180)
        B=67.5
        P=arcsin(sqrt((1-(a/50.0))*0.14645))/(pi/180)
        X_SS[0][a],Y_SS[0][a]=kave(T,B,P)
    
    # vertical axis: 
    X_v, Y_v = [0,0],[0.555221438,0]
    
    # V left:
    X_vl, Y_vl = [0,-0.52481139],[0,0.303]
    
    # B axis (sub)
    X_b=zeros((1,101))
    Y_b=zeros((1,101))
    for a in range (0,101):
        P=arcsin(sqrt(a/100.0))/(pi/180)
        T=0.0
        B=arcsin(sqrt(1-(a/100.0)))/(pi/180)
        X_b[0][a],Y_b[0][a]=kave(T,B,P)

    # combine lines
    X = np.asarray(list(X_SS[0][:25])+X_v+X_vl+list(np.flip(X_b[0][15:51])))
    Y = np.asarray(list(Y_SS[0][:25])+Y_v+Y_vl+list(np.flip(Y_b[0][15:51])))
    
    # define left area (list of tuples)
    SSNFTF2SS_left = zip(X, Y)
    
    # define right area (list of tuples)
    SSNFTF2SS_right = zip(-X, Y)
    
    # create polygon
    SSNFTF2SS_left_polygon = Polygon(SSNFTF2SS_left)
    SSNFTF2SS_right_polygon = Polygon(SSNFTF2SS_right)
    
    return SSNFTF2SS_left_polygon, SSNFTF2SS_right_polygon

def NFTF2SSNFTF():
    '''
    
    '''
    
    # V left:
    X_vl, Y_vl = [-0.52481139,0],[0.303,0]
    
    # bottom line
    X=zeros((1,51))
    Y=zeros((1,51))
    for a in range(0,51):
        B=arcsin(sqrt((a/50.0)*0.14645))/(pi/180)
        T=67.5
        P=arcsin(sqrt((1-(a/50.0))*0.14645))/(pi/180)
        X[0][a],Y[0][a]=kave(T,B,P)
    
    # linear line
    X_lin, Y_lin = [-X[0][25],0],[Y[0][25],0]

    # curved line
    X_cur, Y_cur = -X[0][25:51],Y[0][25:51]
    
    # B axis
    X_b=zeros((1,101))
    Y_b=zeros((1,101))
    for a in range (0,101):
        P=arcsin(sqrt(a/100.0))/(pi/180)
        T=0.0
        B=arcsin(sqrt(1-(a/100.0)))/(pi/180)
        X_b[0][a],Y_b[0][a]=kave(T,B,P)
    
    # combine lines
    X = np.asarray(X_vl+X_lin+list(X_cur)+list(np.flip(X_b[0][51:87])))
    Y = np.asarray(Y_vl+Y_lin+list(Y_cur)+list(np.flip(Y_b[0][51:87])))
    
    # define left area (list of tuples)
    NFTF2SSNFTF_left = zip(X, Y)
    
    # define right area (list of tuples)
    NFTF2SSNFTF_right = zip(-X, Y)
    
    # create polygon
    NFTF2SSNFTF_left_polygon = Polygon(NFTF2SSNFTF_left)
    NFTF2SSNFTF_right_polygon = Polygon(NFTF2SSNFTF_right)
    
    #x,y = NFTF2SSNFTF_left_polygon.exterior.xy
    #plt.plot(x,y,'r')
    #x,y = NFTF2SSNFTF_right_polygon.exterior.xy
    #plt.plot(x,y,'b')
    
    return NFTF2SSNFTF_left_polygon, NFTF2SSNFTF_right_polygon

def NFTF():
    '''
    
    '''
    
    # top line
    X=zeros((1,51))
    Y=zeros((1,51))
    for a in range(0,51):
        B=arcsin(sqrt((a/50.0)*0.14645))/(pi/180)
        T=67.5
        P=arcsin(sqrt((1-(a/50.0))*0.14645))/(pi/180)
        X[0][a],Y[0][a]=kave(T,B,P)
    
    # linear line
    X_lin, Y_lin = [-X[0][25],0],[Y[0][25],0]

    # curved line
    X_cur, Y_cur = -X[0][25:51],Y[0][25:51]
    
    # B axis
    X_b=zeros((1,101))
    Y_b=zeros((1,101))
    for a in range (0,101):
        P=arcsin(sqrt(a/100.0))/(pi/180)
        T=0.0
        B=arcsin(sqrt(1-(a/100.0)))/(pi/180)
        X_b[0][a],Y_b[0][a]=kave(T,B,P)
    
    # vertical axis
    X_v, Y_v = [0,0],[-0.605810893,0]

    # B axis
    X_b=zeros((1,101))
    Y_b=zeros((1,101))
    for a in range (0,101):
        P=arcsin(sqrt(a/100.0))/(pi/180)
        T=0.0
        B=arcsin(sqrt(1-(a/100.0)))/(pi/180)
        X_b[0][a],Y_b[0][a]=kave(T,B,P)

    # T axis 
    X_t=zeros((1,101))
    Y_t=zeros((1,101))
    for a in range (0,101):
        T=arcsin(sqrt(a/100.0))/(pi/180)
        B=0.0
        P=arcsin(sqrt(1-(a/100.0)))/(pi/180)
        X_t[0][a],Y_t[0][a]=kave(T,B,P)
        
    # combine lines
    X = np.asarray(list(X_b[0][87:])+list(X_t[0][:51])+X_v+X_lin+list(X_cur))
    Y = np.asarray(list(Y_b[0][87:])+list(Y_t[0][:51])+Y_v+Y_lin+list(Y_cur))
    
    # define left area (list of tuples)
    NFTF_left = zip(X, Y)
    
    # define right area (list of tuples)
    NFTF_right = zip(-X, Y)
    
    # create polygon
    NFTF_left_polygon = Polygon(NFTF_left)
    NFTF_right_polygon = Polygon(NFTF_right)
    
    #x,y = NFTF_left_polygon.exterior.xy
    #plt.plot(x,y,'r')
    #x,y = NFTF_right_polygon.exterior.xy
    #plt.plot(x,y,'b')
    
    return NFTF_left_polygon, NFTF_right_polygon

def get_areas():
    '''
        point = Point(0., 0.8)
        print(polygon.contains(point))
        x,y = polygon.exterior.xy
        plt.plot(x,y,'r')
    '''
    
    SS = SS0()
    SSNF, SSTF = SSNFTF2SS()
    NFSS, TFSS = NFTF2SSNFTF()
    NF, TF = NFTF()
    
    return {'SS':SS,'SSNF':SSNF,'SSTF':SSTF,'NFSS':NFSS,'TFSS':TFSS,'NF':NF,'TF':TF}

#def circles(ax,X,Y,size,color,zorder,alpha,P=None,plotname=None,prob=[0.95,0.975],inv_method='L2'):
def circles(ax,XY,Prob,size,color,zorder,alpha,P=None,plotname=None,prob=[0.95,0.975],inv_method='L2'):
    #plotname=plotname
    #fig=baseplot(plotname)
    baseplot(ax,plotname=plotname)
    
    if prob[-1] <= 1.0 and P is not None:
        nP = [(P < prob[0]).sum(),(P < prob[-1]).sum()]
        ptes = True
    else:
        nP = [int(prob[0]),int(prob[1])]
        ptes = False
    
    fsize = 12
    
    if True:#inv_method == 'L2':   
        for ii in range(len(XY)):
            [x,y] = XY[ii]
            plt.scatter(x,y,s=size[ii],c=color[ii],zorder=zorder[ii],alpha=alpha[ii],
                        linewidth=1.5,edgecolor='black')
            if color[ii] == 'yellow' and alpha[ii] == 1.0:
                plt.scatter(x,y,s=145,c='yellow',zorder=1,alpha=alpha[ii],linewidth=1.5,edgecolor='black',
                            label='Best solution (F1)')
                if ptes:
                    plt.scatter(x,y,s=120,c='red',zorder=1,alpha=alpha[ii],linewidth=1.5,edgecolor='black',
                                label='Top '+str(int(prob[0]*100))+'  (F1)')
                    plt.scatter(x,y,s=85,c='black',zorder=1,alpha=alpha[ii],linewidth=1.5,edgecolor='black', label='Top '+str(int(prob[1]*100))+'')
                else:
                    plt.scatter(x,y,s=120,c='red',zorder=1,alpha=alpha[ii],linewidth=1.5,edgecolor='black',
                                label='Top '+str(nP[0])+'  (F1)')
                    plt.scatter(x,y,s=85,c='black',zorder=1,alpha=alpha[ii],linewidth=1.5,edgecolor='black', label='Top '+str(nP[1])+'')
                plt.scatter(x,y,s=45,c='gray',zorder=1,alpha=alpha[ii],linewidth=1.5,edgecolor='black',
                            label='Remainder')       
            if color[ii] == 'green' and alpha[ii] == 1.0:
                plt.scatter(x,y,s=145,c='green',zorder=1,alpha=alpha[ii],linewidth=1.5,edgecolor='black',
                            label='Best solution (F2)')
                if ptes:
                    plt.scatter(x,y,s=120,c='blue',zorder=1,alpha=alpha[ii],linewidth=1.5,edgecolor='black', label='Top '+str(int(prob[0]*100))+'  (F2)')
                else:
                    plt.scatter(x,y,s=120,c='red',zorder=1,alpha=alpha[ii],linewidth=1.5,edgecolor='black',
                                label='Top '+str(nP[0])+'  (F1)')
        ax.legend(bbox_to_anchor=(-0.1, 0.85),fontsize=fsize)
    else:    
        XY_strip,Prob_strip = strip_list(XY,Prob)
        x = list(zip(*XY_strip))[0]
        y = list(zip(*XY_strip))[1]
        norm = plt.Normalize(vmax=np.asarray(Prob_strip).max(),vmin=np.asarray(Prob_strip).min())
        heatmap = plt.tricontourf(x,y,Prob_strip,20,alpha=1.0,cmap='hot',norm=norm)
        cbar = plt.colorbar(heatmap,fraction=0.055, pad=0.17, orientation="horizontal")
        cbar.set_label('Probability', fontsize=fsize) #,rotation=270
        cbar.ax.tick_params(labelsize=fsize*0.75,rotation=0) 
    
    

def get_Tenary_loc(strike1,dip1,rake1):

    anX, anY, anZ, dx, dy, dz = pl2nd(strike1,dip1,rake1)
    px, py, pz, tx, ty, tz, bx, by, bz = nd2pt(anX,anY,anZ,dx,dy,dz)

    trendp,plungp=ca2ax(px,py,pz)
    trendt,plungt=ca2ax(tx,ty,tz)
    trendb,plungb=ca2ax(bx,by,bz)

    # x, y Kaverina diagram
    x_kav,y_kav=kave(plungt,plungb,plungp)
    
    return x_kav, y_kav 
 
 
def get_marker(fi,nP0,nP1,fmi):
    if fi == 0:
        siz = 150
        if fmi == 0: col = 'yellow'
        elif fmi == 1: col = 'green'
        zor = 50
        alp = 1.0
    elif 0 < fi <= nP0:
        siz = 125
        if fmi == 0: col = 'red'
        elif fmi == 1: col = 'blue'
        zor = 40
        alp = 0.75
    elif nP0 < fi <= nP1:
        siz = 100
        col = 'black'
        zor = 30
        alp = 0.5
    else:
        siz = 50
        col = 'gray'
        zor = 20
        alp = 0.25
    return siz, col, zor, alp 
 

def strip_list(XY,Prob):
    XY_strip = []
    Prob_strip = []
    Prob = np.asarray(Prob)
    for aa, xy in enumerate(XY):
        indx = []
        if xy not in XY_strip:
            for bb, xy2 in enumerate(XY):
                if xy == xy2:
                    indx.append(bb)
            pmax = np.argwhere(Prob[indx]==Prob[indx].max())[0][0]
            XY_strip.append(xy)
            Prob_strip.append(Prob[indx[pmax]])
    return XY_strip,Prob_strip
        

def get_Tenary(full_result_out,P=None,prob=[0.95,0.975]):
    
    try:
        test = full_result_out[0]
        Rnid = 0
    except:
        test = full_result_out['0']
        Rnid = '0'
    
    N = len(full_result_out[Rnid])
    
    if prob[-1] <= 1.0:
        nP = [(P < prob[0]).sum(),(P < prob[-1]).sum()]
        ptes = True
    else:
        nP = [int(prob[0]),int(prob[1])]
        ptes = False
    
    XY,Prob=[],[]
    size,color,zorder,alpha=[],[],[],[]
    
    for fi in range(N):
        # get focal mechanism
        fm = full_result_out[Rnid][fi][1]
        
        for fmi, [i0,i1] in enumerate([[0,3],[5,8]]):
            try:
                [strike1,dip1,rake1] = fm[i0:i1]
            
                # get location in diagram
                x_kav,y_kav = get_Tenary_loc(strike1,dip1,rake1)
                # get marker information
                siz, col, zor, alp = get_marker(fi,nP[0],nP[1],fmi)
            
                # append to list
                #xplot.append(x_kav)
                #yplot.append(y_kav)
                Prob.append(P[fi])
                XY.append([x_kav,y_kav])
                size.append(siz)
                color.append(col)
                zorder.append(zor)
                alpha.append(alp)
                
            except:
                continue

    #return xplot,yplot,size,color,zorder,alpha
    return XY,Prob,size,color,zorder,alpha

