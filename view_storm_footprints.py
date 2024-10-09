# %%
import numpy as np
import glob

samplepath = "arrs/latest_sample.npz"

# %%
print("Loading most recent sample: ", samplepath)
x = np.load(samplepath)
x = x['samples']
x = x.reshape([x.shape[0], 28, 28])
# x = np.flip(x, axis=1)

# %%
import xarray as xr
import cartopy.crs as ccrs
from cartopy import feature
import matplotlib.pyplot as plt

contour = True
i = np.random.randint(0, x.shape[0])
# i = 27
lats = np.linspace(10, 25, 28)
lons = np.linspace(80, 95, 28)

da = xr.DataArray(x, dims=['time', 'lat', 'lon'], coords={'lat': lats, 'lon': lons})
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
if contour:
    da.isel(time=i).plot.contourf(cmap='Spectral_r', levels=50, ax=ax)
    # da.isel(time=i).plot.contour(colors='black', levels=12, ax=ax, linewidths=0.1)
else:
    da.isel(time=i).plot(colors='Spectral_r', levels=15, ax=ax, linewidths=0.1)
ax.add_feature(feature.COASTLINE, edgecolor='black', linewidth=0.1)
ax.add_feature(feature.BORDERS, linestyle=':', edgecolor='black', linewidth=0.1)
ax.set_title('Sample from GAN over Bay of Bengal')

# %% copy the plots png style from
vmin = 0
vmax = 1

fig, axs = plt.subplots(8, 16,
                        sharex=True, sharey=True,
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        gridspec_kw={'hspace': 0., 'wspace': 0.},
                        figsize=(16, 8))

for i, ax in enumerate(axs.ravel()):
    da.isel(time=i).plot.contourf(cmap='Spectral_r',
                                  levels=20, ax=ax,
                                  vmin=vmin, vmax=vmax,
                                  add_colorbar=False)
    ax.add_feature(feature.COASTLINE, edgecolor='k', linewidth=0.1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.label_outer()
    ax.axis('off')

plt.tight_layout()
# %%
