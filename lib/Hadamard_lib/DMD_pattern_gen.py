from typing import List
import os
import numpy as np
from .Hadamard_gen import *
from .utils import *

nlocations_and_offset = [11, 3]
# nlocations_and_offset = [19 4];  [n offset]
# nlocations_and_offset = [27 6];  [n offset]
# nlocations_and_offset = [35 10];  [n offset]
# nlocations_and_offset = [59 09];   [n offset]
# nlocations_and_offset = [63 14];  [n offset]


# hadamard_patterns = vm(alp_btd_to_logical(hadamard_patterns_scramble_nopermutation(nlocations_and_offset)));
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
pixel_config = pixel_configuration(config_path)

def hadamard_dmd(rows: int, cols: int, n: int, separation: int, scale_factor: int):
    # device_rows, device_cols, active_rows, \
    # active_cols, active_row_offset, active_col_offset = pixel_config

    device_rows = rows
    device_cols = cols
    
    # device row = matrix column
    # device col = matrix row
    scaled_col = np.ceil(device_rows / scale_factor).astype(int)
    scaled_row = np.ceil(device_cols / scale_factor).astype(int)
    # print('sc sc', scaled_col, scaled_row)
    col_idx, row_idx = np.meshgrid(np.array(range(scaled_col)), np.array(range(scaled_row)))

    hadamard_bincode = hadamard_bincode_nopermutation(n)

    hadamard_code_index = ((row_idx*separation+col_idx)%n + 1).astype(int)
    hadamard_code_index = hadamard_code_index.repeat(scale_factor, axis=0).repeat(scale_factor, axis=1)[:device_cols, :device_rows]

    col_idx, row_idx = np.meshgrid(np.array(range(device_rows)), np.array(range(device_cols)))
   
    dmd_pattern = np.ones((device_cols, device_rows, hadamard_bincode.shape[1]))
    # print(dmd_pattern.shape, np.prod(dmd_pattern.shape)/12, row_idx.reshape(-1).shape, col_idx.reshape(-1).shape)
    dmd_pattern[row_idx.reshape(-1), col_idx.reshape(-1), :] = hadamard_bincode[hadamard_code_index.reshape(-1)-1, :].astype(np.int8)
    return dmd_pattern

def hadamard_patterns_scramble_nopermutation(nblock_and_step: List,
    projection_element: int=1, super_mask: np.ndarray=None):
    '''
    [alp_pattern_data, bincode] = hadamard_patterns(block_dimensions,...
                                    projection_element,super_mask)  
    Input
        block_dimensions: [block_rows, block_cols], or just one number
        that means both.
        projection_element: 'binning' of the pattern, projection element is
        a multiplier for the pixel size in the pattern
        super_mask: overall mask that will be applied to the pattern.
    Output
        alp_pattern_data: 3D matrix with the desired patterns. number of
        frames should be block_rows*block_cols rounded to the next multiple
        of four. this matrix is in binary_top_down format.
        bincode: a matrix that fully defines the pattern encoding that was
        used. 
    Parameters
    ----------
    block_and_step : _type_
        _description_
    projection_element : _type_
        _description_
    super_mask : _type_
        _description_
    '''    
    device_rows, device_cols, active_rows, \
    active_cols, active_row_offset, active_col_offset = pixel_config

    if super_mask==None:
        super_mask = np.ones((device_rows, device_cols))
    

    nblock = nblock_and_step[0]
    blockstep = nblock_and_step[1]
    bincode = hadamard_bincode_nopermutation(nblock)
    bitplanes = bincode.shape[1]
    plind = (np.array([range(nblock)]) + blockstep*np.array([range(nblock)]).T)%nblock+1
    bincode = hadamard_bincode_nopermutation(nblock)
    bitplanes = bincode.shape[1]
    plind = (np.array(range(nblock)) + blockstep*np.array([range(nblock)]).T)%nblock+1
    rr, cc = np.argwhere(plind>=0)[:, 1], np.argwhere(plind>=0)[:, 0]

    xs = np.zeros((nblock, nblock, nblock))
    xs[rr, cc, np.reshape(plind,-1, 'F')-1] = True
    xs = np.reshape(xs, (nblock**2, nblock), 'F')
    xs = np.dot(xs, bincode)
    xs = np.reshape(xs, (nblock, nblock, bitplanes), 'F')
    xs = np.multiply(xs*2-1, (np.array(range(1, nblock+1))%2*2-1).reshape(nblock, 1, 1))/2+0.5
    xs = np.vstack([xs, xs])
    # xs = np.block([[[xs]], [[xs]]])
    xs = xs.repeat(projection_element, axis=0).repeat(projection_element, axis=1)

    repeat_factor = np.ceil([device_rows/nblock/2/projection_element, device_cols/nblock/projection_element, 1]).astype(int)
    xspat = np.tile(xs, repeat_factor)

    # random_size = nblock*np.array([2, 1])*np.ceil([device_rows/nblock/2/projection_element, device_cols/nblock/projection_element]).astype(int)
    # xsrand = np.sign(np.random.randn(random_size[0], random_size[1]))
    # xsfullrand = xsrand.repeat(projection_element, axis=0).repeat(projection_element, axis=1)
    return xspat

def hadamard_bincode_nopermutation(nblock):
    '''
    hadamard_bincode_nopermutation Generate a Hadamard binary encoding matrix. 
    bincode = hadamard_bincode(length) generates a zeros and ones matrix.

    Output
        bincode: A truncated normalized Hadamard matrix.  
        For length > 3, this matrix has more columns than rows, the first
        column is all ones. For length < 256, it has as many columns as the
        next multiple of 4 after length. For length > 255, it has less than
        about sqrt(length) excess columns. Its rows are orthogonal:
        (bincode-.5)*(bincode-.5)' is an identity matrix. Also, its rows
        have equal sum, and its columns have close to length/2 sum, except
        for column one which has sum length. 

    See also hadamard_patterns.

    2016 Vicente Parot
    Cohen Lab - Harvard University

    Parameters
    ----------
    nblock : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    '''    
    bincode = nextHadamard(nblock)/2+.5
    bincode = bincode[-nblock:, :]
    return bincode

