from obspy import Catalog, UTCDateTime
from obspy.core.event import (
    Event,
    Origin,
    Pick,
    WaveformStreamID,
)

import warnings

uncert_mapping = {
    0 : 0.05,
    1 : 0.1,
    2 : 0.2,
    3 : 0.5,
    4: 1,
}

'''
Source 
/scratch/gpi/seis/project_data/Albania/utility/obspy-utils/obspyutils/
'''

def _set_year_prefix(utc_dt, yr_prefix=2000):
    utc_dt = utc_dt.datetime
    utc_dt = utc_dt.replace(year=int(yr_prefix) + utc_dt.year)
    return UTCDateTime(utc_dt)

def _latlon_cnv_parser(lat, lon):
    """
    Parse CNV formatted latitude and longitude strs.
    
    E.g. ('70.1234S', '12.4567E')  = (-70.1234, 12.4567)

    Parameters:
    -----------
    lat : str
        CNV formatted long str '%8.6f'$hemisphere where hemisphere is
        abbreviated as one of 'S', 'N'.
    lon : str
        CNV formatted long str '%8.6f'$hemisphere where hemisphere is
        abbreviated as one of 'E', 'N'.
    
    Returns:
    --------
    (lat, lon) : 2D tuple of floats
        The parsed latitude and longitude information as floats. 

    """

    if lat[-1] == 'S':
        lat = -float(lat[:-1])
    elif lat[-1] == 'N':
        lat = float(lat[:-1])
    else:
        raise ValueError(
            f"CNV latitude format expects hemisphere information as last character"
            f"e.g. '12.3456N'. Latitude value {lat} is invalid."
        )
    
    if lon[-1] == 'W':
        lon = -float(lon[:-1])
    elif lon[-1] == 'E':
        lon = float(lon[:-1])
    else:
        raise ValueError(
            f"CNV longitude format expects hemisphere information as last character"
            f"e.g. '12.3456E'. Longitude value {lon} is invalid."
        )
    
    return lat, lon
    
def _parse_cnv_hline(headerline):
    """
    Parse CNV formatted headerline. 

    Parameters:
    -----------
    headerline : str
        CNV headerline. 

    Returns:
    --------
    (ev_onset, lat, lon, depth, mag, rms) : 6D-tuple of UTCDateTime, and floats
        Individual elements of the parsed headerline information.
    
    """
    
    yr = int(headerline[:2])
    mth = int(headerline[2:4])
    day = int(headerline[4:6])
    hr = int(headerline[7:9])
    mins = int(headerline[9:11])
    sec = float(headerline[11:17])

    *_, lat, lon, depth, mag, gap, rms = headerline.strip().split()

    ev_onset = UTCDateTime(yr, mth, day, hr, mins, sec)
    ev_onset = _set_year_prefix(ev_onset)

    lat, lon = _latlon_cnv_parser(lat, lon)
    depth, mag, rms = [float(i) for i in (depth, mag, rms)]

    return ev_onset, lat, lon, depth, mag, rms

def _get_nonadjacent_blanklines(content):
    """
    Get location of all non-adjacent blanklines in the CNV file.

    Parameters:
    -----------
    content : list
        Content of the CNV file as a list of strs for each line.

    Returns:
    --------
    blanklines : list
        List of all non-adjacent blank line numbers. 
    
    """
    blanklines = []
    prev = None

    for line_n, line in enumerate(content):
        if not line.strip() and line_n - 1 != prev:
            blanklines.append(line_n)    
            prev = line_n

    return blanklines

def _parse_cnv_pick(pick_str, ev_onset):
    """
    Parse CNV format pick str. 

    Parameters:
    -----------
    pick_str : str
        12 element CNV str of pick information.
    ev_onset : obspy.UTCDateTime 
        The event onset time.
    
    Returns:
    --------
    pk : obspy.core.event.Pick
        Pick object containing parsed Pick information.

    """
    if len(pick_str) != 12:
        raise ValueError(
            f"Error parsing picks, CNV pick str format expects str of length 12. "
            f"Got {pick_str}"
        )
    sta = pick_str[:4]
    phase = pick_str[4]
    uncert = uncert_mapping[int(pick_str[5])]
    pick_dt  = float(pick_str[6:])

    wvfm_id = WaveformStreamID(network_code="", station_code=sta)
    pk = Pick(
        time=ev_onset + pick_dt,
        phase_hint=phase,
        waveform_id=wvfm_id,
    )

    return pk

def _parse_cnv_pickline(pickline, ev_onset):
    """
    Parse line of picks from CNV file. 

    Parameters:
    -----------
    pickline : str
        The CNV formatted pick information.
    ev_onset : obspy.UTCDateTime 
        The event onset time.
    
    Returns:
    --------
    pks : list
        List of parsed picks from CNV file.

    """
    pks = []
    for i in range(len(pickline) // 12):
        pickinfo = pickline[(i * 12):(i + 1) * 12]
        pks.append(_parse_cnv_pick(pickinfo, ev_onset))
    
    return pks

def _parse_cnv_event(event_lines):
    """
    Parse event information from CNV file.

    Parameters:
    -----------
    event_lines : list
        List of the raw CNV formatted event information, where each line 
        is stored as an individual str in the event list.

    Returns:
    --------
    ev : obspy.core.event.Event
        The event object conttaining parsed event information.

    """
    ev_onset, lat, lon, depth, mag, rms = _parse_cnv_hline(event_lines[0])
    
    ev = Event()
    o = Origin()

    o.longitude = lon
    o.latitude = lat
    o.depth = depth * 1e3
    o.time = ev_onset
    ev.origins = [o]
    ev.magnitudes = None # [mag]

    for line in event_lines[1:]:
        [ev.picks.append(pk) for pk in _parse_cnv_pickline(line, ev_onset)]
    
    return ev

def read_cnv(cnv_file, yr_prefix=2000):
    """
    Read CNV file. 

    Parameters:
    -----------
    cnv_file : str
        Path to CNV  file.
    yr_prefix : str / int
        The prefix for year property of datetimes.
    
    Returns:
    --------
    cat : obspy.core.Catalog
        Catalog object containing all parsed event information and phase information.

    """
    warnings.warn(
        f"As CNV format does not store the full year (e.g. 2014 is stored are 14). "
        f"Using yr_prefix={yr_prefix} to ensure correct dates. "
    )
    cat = Catalog()

    with open(cnv_file, 'r') as f:
        content = f.readlines()

    blanklines = _get_nonadjacent_blanklines(content)
    
    for idx, _ in enumerate(blanklines[1:]):
        
        start_ev_line = blanklines[idx]
        end_ev_line   = blanklines[idx+1]
        
        try:
            cat += _parse_cnv_event(content[start_ev_line + 1: end_ev_line])
        except ValueError as e:
            raise ValueError(
                f"Error parsing event at lines {start_ev_line} : {end_ev_line}. "
                f"Is there an extra blank line in the CNV file?\n"
                f"Recieved: {e}"
            )

    return cat        



