import os
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
import rioxarray as rxr
from models import WeightedLogisticModels

def stack_grids_xy(
    ds,
    feature_bands,
    label_bands,
    target_nx=500,
    target_ny=500,
    skip_grid_id=0,
    include_coords=False,
    normalize=True,
):
    """
    Stack features and labels together from an xarray dataset into a NumPy array (N, C, H, W).

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing 'grid_id', spatial dims ('x', 'y'), and variables.
    feature_bands : list[str]
        Names of variables to use as input features.
    label_bands : list[str]
        Names of variables to use as labels (e.g. 'presence').
    target_nx, target_ny : int
        Target width/height for each grid.
    skip_grid_id : int
        Skip any grid with this ID (e.g., background grid).
    include_coords : bool
        If True, include 'x_local' and 'y_local' as extra input channels.
    normalize : bool
        If True, normalize feature bands to [0, 1] (labels are NOT normalized).

    Returns
    -------
    Xy : np.ndarray
        Array of shape (N, C_total, H, W), where C_total = len(feature_bands) + len(label_bands)
        (+2 if include_coords=True).
    channel_names : list[str]
        Names of the stacked channels (for reference).
    grid_ids : np.ndarray
        List of grid IDs used.
    """
    # Identify valid grid IDs
    grid_ids = np.unique(ds["grid_id"].values)
    grid_ids = grid_ids[grid_ids != skip_grid_id]
    n_grids = len(grid_ids)

    feature_bands = [b for b in feature_bands if b != "grid_id"]
    label_bands = [b for b in label_bands if b != "grid_id"]

    band_names = feature_bands.copy()
    if include_coords:
        band_names.extend(["x_local", "y_local"])
    band_names.extend(label_bands)  # labels last

    Xy_list = []

    for gid in grid_ids:
        gds = ds.where(ds["grid_id"] == gid, drop=True)
        ny = min(target_ny, gds.sizes["y"])
        nx = min(target_nx, gds.sizes["x"])

        arr = np.full((len(band_names), target_ny, target_nx), np.nan, dtype=np.float32)

        # --- Features ---
        for i, v in enumerate(feature_bands):
            if v not in gds:
                continue
            val = gds[v].values[:ny, :nx]
            arr[i, :ny, :nx] = val

        # --- Coordinates (optional) ---
        offset = len(feature_bands)
        if include_coords:
            x2d, y2d = np.meshgrid(gds["x"].values[:nx], gds["y"].values[:ny], indexing="xy")
            arr[offset, :ny, :nx] = x2d
            arr[offset + 1, :ny, :nx] = y2d
            offset += 2

        # --- Labels ---
        for j, v in enumerate(label_bands):
            if v not in gds:
                continue
            val = gds[v].values[:ny, :nx]
            arr[offset + j, :ny, :nx] = val

        # --- Normalize features only ---
        if normalize:
            for i in range(len(feature_bands)):
                a = arr[i]
                valid = np.isfinite(a)
                if valid.sum() == 0:
                    continue
                amin, amax = np.nanpercentile(a[valid], (1, 99))
                if amax > amin:
                    arr[i, valid] = np.clip((a[valid] - amin) / (amax - amin), 0, 1)

        Xy_list.append(arr)

    Xy = np.stack(Xy_list, axis=0)
    return Xy, band_names, grid_ids


def xarray_pred_to_tif(ds, model, bands, out_fp, batch_size=100000, mask_band=None):
    assert out_fp.endswith('.tif'), "Invalid output file. Must be a .tif GeoTIFF file."
    
    ny, nx = ds.sizes['y'], ds.sizes['x']
    total = ny * nx

    # flatten data
    flat_data = {b: ds[b].values.ravel() for b in bands}

    # build mask
    valid_mask = np.ones(total, dtype=bool)
    for b in bands:
        valid_mask &= ~np.isnan(flat_data[b])

    # apply mask_band if provided
    if mask_band is not None and mask_band in ds:
        flat_mask = ds[mask_band].values.ravel()
        valid_mask &= (flat_mask == 1)
    
    # prepare prediction array
    preds = np.full(total, np.nan, dtype=np.float32)

    # predict only on valid indices
    valid_idx = np.where(valid_mask)[0]

    # batch loop
    for start in range(0, len(valid_idx), batch_size):
        end = min(start + batch_size, len(valid_idx))
        idx_batch = valid_idx[start:end]
        df = pd.DataFrame({b: flat_data[b][idx_batch] for b in bands})

        if isinstance(model, WeightedLogisticModels):
            preds[idx_batch] = model.predict(df)
        elif hasattr(model, "predict_proba"):
            # sklearn-style
            preds[idx_batch] = model.predict_proba(df)[:, 1]
        else:
            # Booster
            dmat = xgb.DMatrix(df)
            preds[idx_batch] = model.predict(dmat)
    
    # reshape to 2D DataArray
    pred_da = xr.DataArray(
        preds.reshape((ny, nx)),
        dims = ('y', 'x'),
        coords = {'y': ds['y'], 'x': ds['x']},
        name = 'pred'
    )

    nodata_val = -9999.0
    pred_da = pred_da.fillna(nodata_val)
    pred_da.rio.write_nodata(nodata_val, inplace=True)
    pred_da.rio.write_crs(ds.rio.crs, inplace=True)
    pred_da.rio.write_transform(ds.rio.transform(), inplace=True)

    pred_da.rio.to_raster(out_fp, compress='LZW', dtype='float32')
    print(f"Saved GeoTIFF: {out_fp}")

    return


def xarray_pred_all_to_tif(data_tiles, model, bands, out_dir, batch_size=100000, mask_band=None):
    """
    Run predictions for all tiles and save each as a GeoTIFF in its native CRS.
    """
    os.makedirs(out_dir, exist_ok=True)
    tile_to_file = {}

    for tileId, ds in data_tiles.items():
        out_fp = os.path.join(out_dir, f"{tileId}_pred.tif")
        if os.path.exists(out_fp):
            print(f"Skipping {tileId}: already exists.")
            continue

        print(f"Predicting for {tileId}...")
        try:
            xarray_pred_to_tif(ds, model, bands, out_fp, batch_size=batch_size, mask_band=mask_band)
            tile_to_file[tileId] = out_fp
            print(f"Prediction completed for {tileId}")
        except Exception as e:
            print(f"Prediction failed for {tileId}: {e}")
            continue

    return tile_to_file
