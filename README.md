# AmPHIB

AmPHIB - Amphibious Bayesian is an RMT inversion routine using a uniform X-dimensional tree importance sampling algorithm: uniXtree.
In its current version, it can consider uncertainties by high noise, station misalignments, and onset times.

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
- obspy 1.2.2         
- Pyrocko 2019.06.06 (optional)     
- Matplotlib 3.2.2       
- Numpy 1.18.5        

### Test Version May 2023

### AmPHIB uses selected functions of four external python codes:
- FMC https://github.com/Jose-Alvarez/FMC under the licence GPL-3.0 license                 
  FMC.py was updated to return the polygon information of a coordinate.       
- UAF Geophysics Tools https://github.com/uafgeotools/mtuq under the licence BSD-2-Clause license               
  tape2015.py is used without additional changes.             
- MoPaD https://github.com/geophysics/MoPaD under the licence LGPL-3.0 license             
  mopad.py was updated from py2 to py3.             
- usgs/strec https://github.com/usgs/strec/tree/master/strec under the USGS licence statement                
  kagan.py is used without additional changes             

