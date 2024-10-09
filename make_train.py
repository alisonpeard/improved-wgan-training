# %%
import os
import numpy as np
import xarray as xr
from dotenv import load_dotenv
load_dotenv()

CHANNELS = ['u10', 'tp']
# datapath = '~/Documents/DPhil/paper1.nosync/training/18x22/data_pretrain.nc'
datapath = os.getenv('DATAPATH')
datapath = os.path.join(datapath, 'data_pretrain.nc')
print(f'Loading data from {datapath}...')
data = xr.open_dataset(datapath).sel(channel=CHANNELS)
arr_unif = data.uniform.values
arr_gumbel = -np.log(-np.log(arr_unif))
np.savez('train/gumbel.npz', data=arr_gumbel)
print('Saved Gumbel data...')
np.savez('train/uniform.npz', data=arr_unif)
print('Saved uniform data...')
print('Done.')

# %%
