from typing import List
import json

def load_json(path: str):
    '''
    load json file

    Parameters
    ----------
    path : str
        .json file path

    Returns
    -------
    dict
        json object as dictionary
    '''    
    with open(path) as f:
        json_object = json.load(f)
    return json_object

def pixel_configuration(config_path: str):
    '''
    loads the DMD resolution and provides options to hardcode a ROI for the
    DMD operation. by calling this function from other codes that generate
    patterns, they can set the regions out of the ROI to always off. In
    practice this function is being used to get the DMD resolution only.

    Parameters
    ----------
    config_path
        path for config.json file
    device_pixel 
        The number of device pixels, [rows, columns] 
    active_pixel 
        The number of active pixels, [rows, columns] 
    active_offset
        offset for active pixels, [rows, columns] 
    ''' 
    config = load_json(config_path)
    device_pixel = config['device_pixel']
    active_pixel = config['active_pixel']
    active_offset = config['active_offset']

    if len(active_offset)==0 or all([i >= 0 for i in active_offset]) or active_col_offset==None:
        active_row_offset, active_col_offset = [0, 0]
    else:
        active_row_offset, active_col_offset = active_offset
    device_rows, device_cols = device_pixel
    active_rows, active_cols = active_pixel
    
    active_rows = min(active_rows, device_rows - active_row_offset)
    active_cols = min(active_cols, device_cols - active_col_offset)

    return (device_rows, device_cols, active_rows, active_cols, active_row_offset, active_col_offset)

