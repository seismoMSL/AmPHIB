# AmPHIB

AmPHIB - Amphibious Bayesian is a RMT inversion roution using a uniform X-dimensional tree importance sampling algorithm: uniXtree.
In its current version it is able to consider uncertanities by high noise, station misalignments and onset-times.

Simulation can be done using 1D and 3D synthetics:
- CAP (1D)
- fomosto by pyrocko (1D)
- 6 elements (3D)


### Requirements
- Obspy
- Pyrocko (optional)
- Matplotlib
- Numpy


### AmPHIB depends on several external python codes:
- FMC https://github.com/Jose-Alvarez/FMC
  FMC.py was updated to return the polygone information of a coordinate.
- UAF Geophysics Tools https://github.com/uafgeotools/mtuq
  type2015.py is used without additional changes.
- MoPaD https://github.com/geophysics/MoPaD
  mopad.py was updated from py2 to py3.
- usgs/strec https://github.com/usgs/strec/tree/master/strec
  kagan.py is used without additional changes

