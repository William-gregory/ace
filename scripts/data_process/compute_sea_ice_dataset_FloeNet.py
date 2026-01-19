import os
import glob
import numpy as np
import argparse
import xarray as xr
import xesmf as xe
from xgcm import Grid
import warnings
from dask.distributed import Client
warnings.filterwarnings('ignore')

def stitch(ds, ocean, ice, threshold=0.15):
    return xr.where(ds['siconc']>threshold, ds[ice], ds[ocean])

def _pick_first_element_of_missing_dims(mask: xr.DataArray, data: xr.DataArray):
    missing_dims = [di for di in mask.dims if di not in data.dims]
    if len(missing_dims) == 0:
        return mask
    else:
        return mask.isel({di: 0 for di in missing_dims})

def apply_mask(ds: xr.Dataset, mask: xr.DataArray):
    """Applies mask to same and lower dimensional data"""
    ds_out = xr.Dataset(attrs=ds.attrs)
    for var in ds.data_vars:
        data = ds[var]
        mask_pruned = _pick_first_element_of_missing_dims(mask, data)
        ds_out[var] = data.where(mask_pruned)
    return ds_out

def interpolate_to_cell_centers(
    ds: xr.Dataset,
    like: xr.DataArray,
    grid: Grid,
):
    """Interplate variables defined on cell boundaries to cell centers.

    Args:
        ds: Input dataset.
        like: The data array to use as a template for the interpolation.
        grid: The grid object which will perform the interplation.

    """
    xh, yh = grid.axes["X"].coords["center"], grid.axes["Y"].coords["center"]
    xq, yq = grid.axes["X"].coords["right"], grid.axes["Y"].coords["right"]
    ds_interpolated = xr.Dataset()
    for var in ds.data_vars:
        da = ds[var]
        if set([xh, yh]).issubset(da.dims):
            ds_interpolated[var] = da
        elif xq in da.dims or yq in da.dims:
            # fill the velocities with 0 before interpolation to avoid mismatches in nans
            ds_interpolated[var] = grid.interp_like(da.fillna(0), like)
        if var in ds_interpolated:
            ds_interpolated[var].attrs = da.attrs

    return ds_interpolated

def rotate_vectors(u, v, angle):
    """Rotates vector components u and v using `angle`
    (assumed to be defined in deg, and in the CCW direction)
    Currently only works when all components are on the same grid position
    """
    # angle should be a 2d array
    if not len(angle.dims) == 2:
        raise ValueError(f"Expected only two dimensions on `angle`. Got {angle.dims}")
    # assert that all components are on the same position
    if not (
        set(angle.dims).issubset(set(u.dims)) and set(angle.dims).issubset(set(v.dims))
    ):
        raise ValueError("`u` and `v` need to be on the same grid position as `angle`.")

    # rotate velocities
    theta = np.deg2rad(angle)
    vec = xr.concat([u, v], dim="dim_in")
    # construct rotation matrix
    rot = xr.concat(
        [
            xr.concat([np.cos(theta), np.sin(theta)], dim="dim_out"),
            xr.concat([-np.sin(theta), np.cos(theta)], dim="dim_out"),
        ],
        dim="dim_in",
    )
    rotated_vector = xr.dot(vec, rot, dims="dim_in")
    u_rotated = rotated_vector.isel(dim_out=0)
    v_rotated = rotated_vector.isel(dim_out=1)
    return u_rotated, v_rotated

def horizontal_regrid(ds, ds_target):
    """Regrid `ds` horizontally, and conserve the integral in space"""
    regridder_kwargs = dict(ignore_degenerate=True, periodic=True, unmapped_to_nan=True)

    # try to run this with higher precision (TODO: Test if this actually makes a difference).
    s = xr.Dataset(
        coords={
            co: ds[co].astype("float128") for co in ["lon", "lat", "lon_b", "lat_b"]
        }
    )
    t = xr.Dataset(
        coords={
            co: ds_target[co].astype("float128")
            for co in ["lon", "lat", "lon_b", "lat_b"]
        }
    )

    regridder_nearest = xe.Regridder(s, t, "nearest_s2d", **regridder_kwargs)
    sithick_nearest = regridder_nearest(ds['sithick'], skipna=True, na_thres=1)
    t['mask'] = ~np.isnan(sithick_nearest.isel(time=0).drop_vars("time")).load()
    s["mask"] = ~np.isnan(ds['sithick'].isel(time=0).drop_vars("time")).load()

    regridder = xe.Regridder(s, t, "conservative", **regridder_kwargs)
    ds_regridded = regridder(ds, skipna=True, na_thres=1)

    # get lon/lats from the target grid
    lon = ds_target.lon
    lat = ds_target.lat

    lon_b = ds_target.lon_b
    lat_b = ds_target.lat_b

    # get x and y values
    x = lon.isel(y=0)
    y = lat.isel(x=0)

    # calculate new area
    r_earth = 6356  # in km
    new_area = xe.util.cell_area(ds_target, r_earth) * 1e6

    ## calculate the wetmask afterwards...
    wetmask = ~np.isnan(ds_regridded.sithick.isel(time=0).drop_vars("time")).load()
    sfrac = regridder(ds.wetmask.astype("float64")).fillna(0.0)
    sfrac = sfrac.where(sfrac <= 1.0, 1.0) #sea_surface_fraction
    #ofrac = (1 - ds_regridded['siconc']) * sfrac #ocean_fraction

    ds_regridded = ds_regridded.drop_vars(["lon_b", "lat_b"])
    ds_regridded = ds_regridded.assign_coords(
        lon=lon,
        lat=lat,
        lon_b=lon_b,
        lat_b=lat_b,
        areacello=new_area,
        x=x,
        y=y,
        mask_2d=wetmask,
        sea_surface_fraction=sfrac,
        land_fraction=1-sfrac,
        #ocean_fraction=ofrac,
    )
    ds_regridded.attrs = ds.attrs | ds_regridded.attrs

    return ds_regridded

# load supergrid and extract the angles
# Some awesome material to understand the 'supergrid' (is that the same as the mosaic?) https://gist.github.com/adcroft/c1e207024fe1189b43dddc5f1fe7dd6c
def convert_super_grid(ds_super_grid: xr.Dataset):
    h_rename = {"nyp": "yh", "nxp": "xh"}
    b_rename = {"nyp": "yh_b", "nxp": "xh_b"}

    h_indicies = dict(nyp=slice(1, None, 2), nxp=slice(1, None, 2))
    b_indicies = dict(
        nyp=slice(0, None, 2), nxp=slice(0, None, 2)
    )  # locations of 'bound variables required by xesmf

    angle_h = ds_super_grid.angle_dx.isel(**h_indicies).rename(h_rename)
    lon_h = ds_super_grid.x.isel(**h_indicies).rename(h_rename)
    lat_h = ds_super_grid.y.isel(**h_indicies).rename(h_rename)

    lon_b = ds_super_grid.x.isel(**b_indicies).rename(b_rename)
    lat_b = ds_super_grid.y.isel(**b_indicies).rename(b_rename)
    return angle_h, lon_h, lat_h, lon_b, lat_b

def sis2_preprocessing(date, ds_grid, ds_super_grid, snapshots=True):
    """SIS2 specific preprocessing"""
    pth = '/archive/William.Gregory/fre/FMS2023.04_mom6_20231130/OM4p25_JRA55do1.5_r6_4Emulator_noleap/gfdl.ncrc5-intel23-prod/history/'    
    mean_vars = ['SWDN','LWDN','SH','LH','FA_X','FA_Y','SNOWFL','RAIN','BHEAT','BMELT','TMELT','SSH',\
                 'SALTF','LSRCi','LSNKi','XPRTi','LSRCc','LSNKc','XPRTc','LSRCs','LSNKs','XPRTs','UI','VI']
    snap_vars = ['siconc','sisnthick','sithick','simass','sisnmass','TS']
        
    if snapshots:
        ds_mean = xr.open_zarr(pth+date+'.ice_6hourly.zarr')[mean_vars]
        ds_snap = xr.open_zarr(pth+date+'.ice_6hourly_snap.zarr')[snap_vars]
        ds_mean['time'] = ds_snap['time']
        ds = xr.merge([ds_mean,ds_snap])
    else:
        ds = xr.open_zarr(pth+date+'.ice_6hourly.zarr')[mean_vars+snap_vars]

    ds_ocn = xr.open_zarr(pth+date+'.ocean_6hourly.zarr')[['tauuo','tauvo']]
    ds_ocn = ds_ocn.rename({'xh':'xT','yh':'yT','xq':'xB','yq':'yB'})
    ds_ocn['time'] = ds['time']
    ds = xr.merge([ds,ds_ocn])
            
    # trim excess padding
    if ds["xB"].size == ds["xT"].size + 1:
        ds = ds.isel(xB=slice(1, None))
    if ds["yB"].size == ds["yT"].size + 1:
        ds = ds.isel(yB=slice(1, None))

    grid = Grid(
        ds,
        coords={
            "X": {"center": "xT", "right": "xB"},
            "Y": {"center": "yT", "right": "yB"},
        },
        boundary="extend",
        periodic=["xT", "xT"],
    )
    ds_interpolated = interpolate_to_cell_centers(ds, ds.sithick, grid)
    ds_interpolated['wind_stress_x'] = stitch(ds_interpolated, 'tauuo', 'FA_X')
    ds_interpolated['wind_stress_y'] = stitch(ds_interpolated, 'tauvo', 'FA_Y')

    # remove the same areas as for the tracers again
    tracer_wetmask = ~np.isnan(ds_interpolated.sithick.isel(time=0)).drop_vars("time")
    ds = apply_mask(ds_interpolated, tracer_wetmask)
    ds = ds.assign_coords(wetmask=tracer_wetmask)

    ds_grid = ds_grid.drop_vars("time")
    ds_grid = ds_grid.set_coords([v for v in ds_grid.data_vars])
    ds = ds.assign_coords(
        lon=ds_grid.geolon, lat=ds_grid.geolat, areacello=ds_grid.areacello
    )

    # drop (for now) all the coords on non-tracer position
    required_coords = [
        "lon",
        "lat",
        "time",
        "xT",
        "yT",
        "wetmask",
    ]
    drop_coords = [co for co in ds.coords.keys() if co not in required_coords]
    ds = ds.drop(drop_coords)

    a, lon, lat, lon_b, lat_b = convert_super_grid(ds_super_grid)
    lon_expected = ds_grid.load().geolon.reset_coords(drop=True).drop(["xh", "yh"])
    lat_expected = ds_grid.load().geolat.reset_coords(drop=True).drop(["xh", "yh"])

    # asser that the grid positions extracted are correct (this should maybe live in a test for an upstream function?)
    xr.testing.assert_allclose(lon, lon_expected)
    xr.testing.assert_allclose(lat, lat_expected)

    ds = ds.assign_coords(lon_b=lon_b, lat_b=lat_b, angle=a, lon=lon, lat=lat)
    ds = ds.rename({"xT": "x", "yT": "y", "xh": "x", "yh": "y", "xh_b": "x_b", "yh_b": "y_b"})
    if "time_bnds" in ds.data_vars:
        ds = ds.drop_vars(["time_bnds"])

    return ds

def main():
    parser = argparse.ArgumentParser(description="Process snapshots argument.")
    parser.add_argument("--snapshots", action="store_true", help="Enable snapshots processing")
    args = parser.parse_args()
    snapshots = args.snapshots
    pth = '/archive/William.Gregory/fre/FMS2023.04_mom6_20231130/OM4p25_JRA55do1.5_r6_4Emulator_noleap/gfdl.ncrc5-intel23-prod/history/'
    if snapshots:
        out_name = 'snapshots'
        print('Processing snapshots data...',flush=True)
    else:
        out_name = 'means'
        print('Processing means data...',flush=True)
    files = sorted(glob.glob(pth+'*.nc.tar'))
    ds_grid = xr.open_dataset(pth+'ocean.static.nc').load()
    ds_super_grid = xr.open_dataset(pth+'ocean_hgrid.nc').load()
    ds_target_grid = xr.open_dataset(pth+'1deg_Gaussian_grid_180x360.nc').rename({'lon':'x','lat':'y','lon_b':'x_b','lat_b':'y_b'}).load()
    lon,lat = np.meshgrid(ds_target_grid['x'].to_numpy(),ds_target_grid['y'].to_numpy())
    lonb,latb = np.meshgrid(ds_target_grid['x_b'].to_numpy(),ds_target_grid['y_b'].to_numpy())
    ds_target_grid['lon'] = (('y','x'),lon)
    ds_target_grid['lat'] = (('y','x'),lat)
    ds_target_grid['lon_b'] = (('y_b','x_b'),lonb)
    ds_target_grid['lat_b'] = (('y_b','x_b'),latb)

    outer_chunks = {
        "time": 360, 
        "lat": -1, 
        "lon": -1,
    }
    inner_chunks = {
        "time": 1, 
        "lat": -1, 
        "lon": -1,
    }

    ds = []
    client = Client(n_workers=2, memory_limit='225GB')
    for file in files:
        date = file.split('/')[-1].split('.')[0]
        savepath = 'FloeNet_6hourly_'+out_name+'/'+date+'.FloeNet_6hourly_'+out_name+'.zarr'
        if not os.path.exists(savepath):
            print(date,flush=True)
            ds_sis = sis2_preprocessing(date,ds_grid,ds_super_grid,snapshots=snapshots)

            attrs = {}
            for var in ds_sis.data_vars:
                attrs[var] = ds_sis[var].attrs

            for varname_x, varname_y in [['FA_X','FA_Y'],['tauuo','tauvo'],['wind_stress_x','wind_stress_y'],['UI','VI']]:
                x_rotated, y_rotated = rotate_vectors(ds_sis[varname_x], ds_sis[varname_y], ds_sis['angle'])
                ds_sis[varname_x] = x_rotated.astype(np.float64)
                ds_sis[varname_y] = y_rotated.astype(np.float64)


            ds_sis = ds_sis.chunk({"time":1,"x":-1,"y":-1})
            ds_sis_regridded = horizontal_regrid(ds_sis, ds_target_grid).astype(np.float32)
            ds_sis_regridded = ds_sis_regridded.drop_vars(['lon_b','lat_b','x_b','y_b']).rename({'lon':'longitude','lat':'latitude','x':'lon','y':'lat'})

            for var, attrs in attrs.items():
                ds_sis_regridded[var].attrs = attrs

            ds_sis_regridded.attrs['history'] = (
                'Dataset computed on ppan in /archive/William.Gregory/fre/FMS2023.04_mom6_20231130/OM4p25_JRA55do1.5_r6_4Emulator_noleap/gfdl.ncrc5-intel23-prod/history'
            )

            drop_dims = [x for x in list(ds_sis_regridded.dims) if x not in ['lon','lat','time']]
            ds_sis_regridded = ds_sis_regridded.drop_dims(drop_dims)
            ds_sis_regridded.to_zarr(savepath,zarr_version=3,mode='w')
        else:
            ds_sis_regridded = xr.open_zarr(savepath)
        ds.append(ds_sis_regridded)

    ds = xr.concat(ds,dim='time')
    ds = ds.reset_coords(names=['longitude','latitude','areacello','mask_2d','land_fraction','sea_surface_fraction'])
    #ds['sea_ice_fraction'] = ds['siconc']*ds['sea_surface_fraction']
    ### the sum of all surface fractions (ds['sea_ice_fraction'] + ds['ocean_fraction'] + ds['land_fraction']) should equal 1 to within roundoff
    ds['UI'] = xr.where(ds['siconc']>0, ds['UI'], 0)
    ds['VI'] = xr.where(ds['siconc']>0, ds['VI'], 0)

    ### set up variable masks
    ocn = ['BHEAT','SWDN','LWDN','SH','LH','wind_stress_x','wind_stress_y','RAIN','SNOWFL','tauuo','tauvo','SSH'] #mask with mask_2d
    ice = ['FA_X','FA_Y','XPRTi','LSRCi','LSNKi','XPRTs','LSRCs','LSNKs','XPRTc','LSRCc','LSNKc','SALTF',\
           'sithick','sisnthick','siconc','TS','BMELT','TMELT','simass','sisnmass','UI','VI'] #mask at sea ice locations

    variables = ocn+ice
    ice_present = ds['siconc'].sum('time')
    mask = (ice_present > 0).astype(float)
    for variable in variables:
        if variable in ice:
            ds[variable] = ds[variable].where(mask==1)
            ds['mask_'+variable] = mask
        elif variable in ocn:
            ds[variable] = ds[variable].where(ds['mask_2d']==1)
            ds['mask_'+variable] = ds['mask_2d']

    ds = ds.chunk(outer_chunks)
    ds = ds.reset_coords()
    for v in ds.variables.values():
        v.attrs.pop("coordinates", None)
        v.encoding.pop("coordinates", None)

    ds_sub = ds[['SSH','BHEAT','tauuo','tauvo']].resample(time='5D').mean()
    ds_sub = ds_sub.reindex(time=ds['time'],method='ffill').chunk({'time':360,'lon':-1,'lat':-1})
    ds_sub = ds_sub.rename({'SSH':'SSH_5d','BHEAT':'BHEAT_5d','tauuo':'tauuo_5d','tauvo':'tauvo_5d'})
    ds = xr.merge([ds,ds_sub])
    ds.to_zarr('1958-2022.FloeNet_6hourly_'+out_name+'_NoLeap.zarr',zarr_version=3,mode='w',encoding={var: {"compressor": None} for var in ds.data_vars})
    client.close()
    
if __name__ == "__main__":
    main()