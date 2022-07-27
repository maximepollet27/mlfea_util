import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

if __name__ == '__main__':

    # Input path
    ROOT_PATH = Path.cwd().parent

    # Load all sample names
    #all_samples = os.listdir(ROOT_PATH.joinpath('./ansys_analysis/output'))
    all_samples = os.listdir(ROOT_PATH.joinpath('./mlfea_util/data/imperfect_shells'))

    # Load perfect samples

    # Loop over all samples
    all_inputs = []
    all_outputs = []
    for isample in all_samples:

        # Inputs
        span = float(isample.split('_')[0][1:])
        height = float(isample.split('_')[1][1:])
        thickness = float(isample.split('_')[2][1:])
        thick_spread = float(isample.split('_')[3][2:])
        thick_centre = float(isample.split('_')[4][2:])
        surface_spread = float(isample.split('_')[5][2:])
        surface_centre = float(isample.split('_')[6][2:])
        surface_distrib = int(float(isample.split('_')[7][2:]))
        psample = 'span{:.6f}_height{:.6f}_radius{:.6f}_thick{:.6f}'.format(span, height, span/10, thickness)

        # Load defs
        #with ROOT_PATH.joinpath('./data_backup/training_data_def/{:s}/landmarks_defs.pickle'.format(isample)).open('rb') as f:
        with ROOT_PATH.joinpath('./mlfea_util/data/imperfect_shells/{:s}/landmarks_defs.pickle'.format(isample)).open('rb') as f:
            landmark_defs = pickle.load(f)
        
        # Store mid defs and thick defs in separate lists in landmark order
        mid_defs = []
        thick_defs = []
        for i in range(6417):
            mid_defs.append(landmark_defs[i][0])
            thick_defs.append(landmark_defs[i][1])
        
        # Load resulting stresses and weights
        #res_stresses = pd.read_csv(ROOT_PATH.joinpath('./ansys_analysis/output/{:s}/nodal_results.csv'))
        res_stresses = pd.read_csv(ROOT_PATH.joinpath('./mlfea_util/data/imperfect_shells/{:s}/nodal_results.csv'.format(isample)), dtype={'NNUM':np.int16})
        #with ROOT_PATH.joinpath('./data_backup/training_data_def/{:s}/weights.pickle'.format(isample)).open('rb') as f:
        with ROOT_PATH.joinpath('./mlfea_util/data/imperfect_shells/{:s}/weights.pickle'.format(isample)).open('rb') as f:
            weights = pickle.load(f)

        # Calc landmark stresses
        s_x_mid = []
        s_y_mid = []
        s_z_mid = []
        s_xy_mid = []
        s_yz_mid = []
        s_xz_mid = []
        s_x_top = []
        s_y_top = []
        s_z_top = []
        s_xy_top = []
        s_yz_top = []
        s_xz_top = []
        s_x_bot = []
        s_y_bot = []
        s_z_bot = []
        s_xy_bot = []
        s_yz_bot = []
        s_xz_bot = []
        for i in range(6417):
            sxm = 0
            sym = 0
            szm = 0
            sxym = 0
            syzm = 0
            sxzm = 0
            sxt = 0
            syt = 0
            szt = 0
            sxyt = 0
            syzt = 0
            sxzt = 0
            sxb = 0
            syb = 0
            szb = 0
            sxyb = 0
            syzb = 0
            sxzb = 0
            w_sum = 0
            for node, weight in weights[i].items():

                w_sum += weight
                sxm += weight*res_stresses.iloc[node]['SX']
                sym += weight*res_stresses.iloc[node]['SY']
                szm += weight*res_stresses.iloc[node]['SZ']
                sxym += weight*res_stresses.iloc[node]['SXY']
                syzm += weight*res_stresses.iloc[node]['SYZ']
                sxzm += weight*res_stresses.iloc[node]['SXZ']
                sxt += weight*res_stresses.iloc[int(node+(res_stresses['NNUM'].max()/3))]['SX']
                syt += weight*res_stresses.iloc[int(node+(res_stresses['NNUM'].max()/3))]['SY']
                szt += weight*res_stresses.iloc[int(node+(res_stresses['NNUM'].max()/3))]['SZ']
                sxyt += weight*res_stresses.iloc[int(node+(res_stresses['NNUM'].max()/3))]['SXY']
                syzt += weight*res_stresses.iloc[int(node+(res_stresses['NNUM'].max()/3))]['SYZ']
                sxzt += weight*res_stresses.iloc[int(node+(res_stresses['NNUM'].max()/3))]['SXZ']
                sxb += weight*res_stresses.iloc[int(node+(2*res_stresses['NNUM'].max()/3))]['SX']
                syb += weight*res_stresses.iloc[int(node+(2*res_stresses['NNUM'].max()/3))]['SY']
                szb += weight*res_stresses.iloc[int(node+(2*res_stresses['NNUM'].max()/3))]['SZ']
                sxyb += weight*res_stresses.iloc[int(node+(2*res_stresses['NNUM'].max()/3))]['SXY']
                syzb += weight*res_stresses.iloc[int(node+(2*res_stresses['NNUM'].max()/3))]['SYZ']
                sxzb += weight*res_stresses.iloc[int(node+(2*res_stresses['NNUM'].max()/3))]['SXZ']

            sxm /= w_sum   
            sym /= w_sum   
            szm /= w_sum   
            sxym /= w_sum   
            syzm /= w_sum   
            sxzm /= w_sum
            sxt /= w_sum   
            syt /= w_sum   
            szt /= w_sum   
            sxyt /= w_sum   
            syzt /= w_sum   
            sxzt /= w_sum
            sxb /= w_sum   
            syb /= w_sum   
            szb /= w_sum   
            sxyb /= w_sum   
            syzb /= w_sum   
            sxzb /= w_sum 
            
            s_x_mid.append(sxm)
            s_y_mid.append(sym)
            s_z_mid.append(szm)
            s_xy_mid.append(sxym)
            s_yz_mid.append(syzm)
            s_xz_mid.append(sxzm)
            s_x_top.append(sxt)
            s_y_top.append(syt)
            s_z_top.append(szt)
            s_xy_top.append(sxyt)
            s_yz_top.append(syzt)
            s_xz_top.append(sxzt)
            s_x_bot.append(sxb)
            s_y_bot.append(syb)
            s_z_bot.append(szb)
            s_xy_bot.append(sxyb)
            s_yz_bot.append(syzb)
            s_xz_bot.append(sxzb)
        
        # Compile lists
        all_inputs.append([span, height, thickness, height/span, thickness/span, thick_spread, thick_centre, surface_spread, surface_centre, surface_distrib]+mid_defs+thick_defs)
        all_outputs.append(s_x_mid+s_y_mid+s_z_mid+s_xy_mid+s_yz_mid+s_xz_mid+s_x_top+s_y_top+s_z_top+s_xy_top+s_yz_top+s_xz_top+s_x_bot+s_y_bot+s_z_bot+s_xy_bot+s_yz_bot+s_xz_bot)
    
    # Create column names
    input_col_names = ['span', 'height', 'thickness', 'height/span', 'thickness/span', 'ts', 'tc', 'ss', 'sc', 'sd']
    mid_cols = []
    thick_cols = []
    sx_mid = []
    sy_mid = []
    sz_mid = []
    sxy_mid = []
    syz_mid = []
    sxz_mid = []
    sx_top = []
    sy_top = []
    sz_top = []
    sxy_top = []
    syz_top = []
    sxz_top = []
    sx_bot = []
    sy_bot = []
    sz_bot = []
    sxy_bot = []
    syz_bot = []
    sxz_bot = []

    for i in range(6417):
        mid_cols.append('mid_def_{:d}'.format(i))
        thick_cols.append('thick_def_{:d}'.format(i))
        sx_mid.append('sx_mid_{:d}'.format(i))
        sy_mid.append('sy_mid_{:d}'.format(i))
        sz_mid.append('sz_mid_{:d}'.format(i))
        sxy_mid.append('sxy_mid_{:d}'.format(i))
        syz_mid.append('syz_mid_{:d}'.format(i))
        sz_mid.append('sxz_mid_{:d}'.format(i))
        sx_top.append('sx_top_{:d}'.format(i))
        sy_top.append('sy_top_{:d}'.format(i))
        sz_top.append('sz_top_{:d}'.format(i))
        sxy_top.append('sxy_top_{:d}'.format(i))
        syz_top.append('syz_top_{:d}'.format(i))
        sz_top.append('sxz_top_{:d}'.format(i))
        sx_bot.append('sx_bot_{:d}'.format(i))
        sy_bot.append('sy_bot_{:d}'.format(i))
        sz_bot.append('sz_bot_{:d}'.format(i))
        sxy_bot.append('sxy_bot_{:d}'.format(i))
        syz_bot.append('syz_bot_{:d}'.format(i))
        sz_bot.append('sxz_bot_{:d}'.format(i))
    input_col_names = input_col_names + mid_cols + thick_cols
    output_col_names = sx_mid+sy_mid+sz_mid+sxy_mid+syz_mid+sxz_mid+sx_top+sy_top+sz_top+sxy_top+syz_top+sxz_top+sx_bot+sy_bot+sz_bot+sxy_bot+syz_bot+sxz_bot

    # Make and store dfs
    input_df = pd.DataFrame(data=all_inputs, columns=input_col_names)
    output_df = pd.DataFrame(data=all_outputs, columns=output_col_names)
    input_df.to_csv(ROOT_PATH.joinpath('mlfea_util/data/ml_dfs/input_df.csv'), index=False)
    output_df.to_csv(ROOT_PATH.joinpath('mlfea_util/data/ml_dfs/output_df.csv'), index=False)    