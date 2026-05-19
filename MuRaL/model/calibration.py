import math
import numpy as np
import pandas as pd

def lambda_calib(prob0, prob):
    lambda_calib = -math.log(prob0)
    prob_calib = lambda_calib * prob / (1 - prob0)
    return prob_calib

def poisson_calibrate(df_pred):

    prob0 = np.clip(df_pred['prob0'], 1e-10, 1.0)
    lambda_calib = -np.log(prob0)
    denominator = 1 - prob0
    prob_cols = [col for col in df_pred.columns if col.startswith('prob') and col != 'prob0']

    df_calib = df_pred.copy()
    # Apply calibration to each probability column
    for col in prob_cols:
        df_calib[col] = lambda_calib * df_pred[col] / denominator

    df_calib['prob0'] = 1 - lambda_calib
    return df_calib

def poisson_calibrate_file(pred_file):
    """Apply Poisson calibration to an existing prediction file.
    
    Args:
        pred_file: path to prediction TSV (gzipped or plain)
    """
    for ext in ['.tsv.gz', '.tsv']:
         if pred_file.endswith(ext):
            out_file = pred_file.replace(ext, f'.poisson_cal{ext}')
            break
    
    print(f'Poisson calibration only mode')
    print(f'  Input:  {pred_file}')
    print(f'  Output: {out_file}')
    
    df = pd.read_csv(pred_file, sep='\t')
    prob_cols = [c for c in df.columns if c.startswith('prob')]
    
    prob_df = poisson_calibrate(df[prob_cols])
    df[prob_cols] = prob_df[prob_cols]
    
    compression = 'gzip' if out_file.endswith('.gz') else None
    df.to_csv(out_file, sep='\t', index=False,
              float_format='%.4g', compression=compression)
    print(f'  Done.')