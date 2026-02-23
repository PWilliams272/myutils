import pandas as pd
import numpy as np
import datetime

def kernel_smooth_with_uncertainty(df, x_col, y_col, yerr_col=None, bandwidth=1.0, kernel='gaussian', x_out=None):
    x = df[x_col].values
    y = df[y_col].values
    
    # Robust conversion for datetime, date, or timedelta columns
    is_datetime = False
    base = None
    if np.issubdtype(x.dtype, np.datetime64):
        is_datetime = True
        x = pd.to_datetime(x).astype('int64') / 1e9 / 86400  # days since epoch
    elif np.issubdtype(x.dtype, np.timedelta64):
        x = pd.to_timedelta(x).astype('timedelta64[D]').astype(float)
    elif isinstance(x[0], (pd.Timestamp, pd.Timedelta)):
        is_datetime = isinstance(x[0], pd.Timestamp)
        x = np.array([xi.days for xi in x])
    elif isinstance(x[0], (datetime.date, datetime.datetime)):
        is_datetime = True
        base = pd.to_datetime(x[0])  # Ensure base is always a Timestamp
        x = np.array([(pd.to_datetime(xi) - base).days for xi in x])
    elif isinstance(x[0], datetime.timedelta):
        x = np.array([xi.days for xi in x])
    else:
        x = x.astype(float)

    if yerr_col is not None:
        yerr = df[yerr_col].values
    else:
        yerr = np.ones_like(y, dtype=np.float64)  # Use equal weights if no yerr provided

    # Allow smoothing at arbitrary x_out points, including datetime
    if x_out is None:
        x_out_num = x
    else:
        x_out = np.array(x_out)
        if is_datetime:
            # Convert x_out to numeric days since epoch or base
            if base is not None:
                x_out_num = np.array([(pd.to_datetime(xi) - base).days for xi in x_out])
            else:
                x_out_num = pd.to_datetime(x_out).astype('int64') / 1e9 / 86400
        else:
            x_out_num = x_out.astype(float)

    smoothed = np.full_like(x_out_num, np.nan, dtype=np.float64)
    smoothed_err = np.full_like(x_out_num, np.nan, dtype=np.float64)
    for i, xi in enumerate(x_out_num):
        dists = x - xi
        if yerr_col is not None:
            mask = ~np.isnan(y) & ~np.isnan(yerr)
        else:
            mask = ~np.isnan(y)
        if kernel == 'gaussian':
            try:
                kernel_weights = np.exp(-0.5 * (dists[mask] / bandwidth) ** 2)
            except Exception as e:
                print(f"[DEBUG] Error in kernel_weights calculation at i={i}: {e}")
        elif kernel in ['uniform', 'boxcar']:
            kernel_weights = (np.abs(dists[mask]) <= bandwidth).astype(float)
        else:
            raise ValueError("Unknown kernel")
        # Inverse variance weights
        if yerr_col is not None:
            inv_var = 1.0 / (yerr[mask] ** 2)
        else:
            inv_var = np.ones_like(kernel_weights)
        weights = np.array(kernel_weights) * np.array(inv_var)
        if weights.sum() > 0:
            smoothed[i] = np.sum(weights * y[mask]) / np.sum(weights)
            smoothed_err[i] = np.sqrt(1.0 / np.sum(weights))
    return smoothed, smoothed_err