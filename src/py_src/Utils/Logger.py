import logging
import sys
import time
import inspect
from obspy import UTCDateTime

def log_debug(time,fname,pyname):
    info_message = 'Function %s in file %s with runtime=%s' % (fname,pyname.split('/')[-1],str(time))
    LOG.debug(info_message)
    
    
def log_info_Octree(probability,c_coord,pdf):
    info_message = 'p=%1.2f, VR=%1.2f, coord=(%s)' % (probability,1-pdf,str(list(c_coord)))
    LOG.info(info_message)
    
def log_info(log):
    py_file = log[1]
    func_name = log[3]
    line = log[2]
    
    info_message = log[1]+'_'+str(log[2])+'_'+log[3]
    LOG.info(info_message)
    
def log_warning(log):
    py_file = log[1]
    func_name = log[3]
    line = log[2]
    
    info_message = log[1]+'_'+str(log[2])+'_'+log[3]
    LOG.info(info_message)



LOG = logging.getLogger('RMT_uniXPy')
LOG.setLevel(logging.DEBUG)
#handler = logging.FileHandler('RMT_UniOcPy.'+str(UTCDateTime())+'.log')
handler = logging.FileHandler('RMT_uniXPy.log')
format = logging.Formatter('%(asctime)s  %(name)s%(levelname)s: %(message)s')
handler.setFormatter(format)
LOG.addHandler(handler)


TLOG = logging.getLogger('Time_Logger_RMT_uniXPy')
TLOG.setLevel(logging.DEBUG)
#handler = logging.FileHandler('RMT_UniOcPy.'+str(UTCDateTime())+'.log')
handler = logging.FileHandler('Time_Logger_RMT_uniXPy.log')
format = logging.Formatter('%(asctime)s  %(name)s%(levelname)s: %(message)s')
handler.setFormatter(format)
TLOG.addHandler(handler)
flog = inspect.stack()[0]
start_time = UTCDateTime()
debug_message = 'Function Logger in file %s at starttime=%s' % (flog[1].split('/')[-1],str(start_time))
TLOG.debug(debug_message)



