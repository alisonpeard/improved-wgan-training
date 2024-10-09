# %%
import os
import numpy as np
import xarray as xr
# %%
CHANNELS = ['u10', 'tp']
datapath = '~/Documents/DPhil/paper1.nosync/training/18x22/data_pretrain.nc'
data = xr.open_dataset(datapath).sel(channel=CHANNELS)
arr_unif = data.uniform.values
arr_gumbel = -np.log(-np.log(arr_unif))
np.savez('train/gumbel.npz', data=arr_gumbel)
np.savez('train/uniform.npz', data=arr_unif)
# %%
