# AmPHIB

AmPHIB - Amphibious Bayesian is a RMT inversion roution using a uniform X-dimensional tree importance sampling algorithm: uniXtree.
In its current version it is able to consider uncertanities by high noise, station misalignments and onset-times.

Simulation can be done using 1D and 3D synthetics:
- CAP (1D)
- fomosto by pyrocko (1D)
- 6 elements (3D)

### Example    
This repository includes one example dataset
of the 2016-10-18 22:08:14 M 5.6 - 14 km WSW of Pointe Michel, Dominica event.
Open Main.ipynb and navigate through the different sub-notebook. 
Run JSON-Creator to renew input_ev123_Example.json for your local environment.


### Requirements
- Obspy 1.2.2         
- Pyrocko 2019.06.06 (optional)     
- Matplotlib 3.2.2       
- Numpy 1.18.5        


### AmPHIB depends on several external python codes:
- FMC https://github.com/Jose-Alvarez/FMC                
  FMC.py was updated to return the polygone information of a coordinate.       
- UAF Geophysics Tools https://github.com/uafgeotools/mtuq               
  tape2015.py is used without additional changes.             
- MoPaD https://github.com/geophysics/MoPaD            
  mopad.py was updated from py2 to py3.             
- usgs/strec https://github.com/usgs/strec/tree/master/strec           
  kagan.py is used without additional changes             

