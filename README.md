# comfort-db

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](python_badge)

Python library for working with the [COMFORT](https://www.doi.org/10.11582/2022.00039)[1] oceanographic SQLite database. 

`comfort-db` wraps database access, preprocessing and basic analysis utilities into one library. 
This includes filtering for quality and space-time ranges, unit conversions, spatiotemporal gridding, profile and water-mass analysis, 
geospatial utilities, scaling, plotting and visualisation using a dashboard.

The data structure broadly follows oceanographic ship measurement practice: 
At a latitude, longitude and time, a station defines where measurements were taken.
At a station, profiles are measured, i.e. a parameter like temperature at different depths with a specific instrument.
There can be multiple profiles at the same station, though in the COMFORT database, it is mostly one profile per station. 


[1] Korablev, A., Olsen, A., Geophysical Institute, University of Bergen, Bjerknes Centre for Climate Research (2022). COMFORT Dataset [Data set]. NIRD RDA. https://doi.org/10.11582/2022.00039

[2] McDougall, T., & Barker, P. (2011). Getting started with TEOS-10 and the Gibbs Seawater (GSW) Oceanographic Toolbox. SCOR/IAPSO WG127

---

## Installation

```bash
pip install comfort-db
```

Optional extras add more specialised functionality:

| Extra | Adds | Enables                                               |
|---|---|-------------------------------------------------------|
| `plot` | matplotlib, seaborn | `comfort.plot`                                        |
| `geo` | shapely, geopandas, cartopy | Water masses, Longhurst provinces, coastline distance |
| `grid` | xarray, netcdf4 | Spatiotemporal gridding, bathymetry masking           |
| `scale` | scikit-learn | `ParamScaler`                                         |
| `dash` | dash, plotly | Interactive Dash explorer app                         |
| `all` | everything above |                                                       |

Install via e.g.

```bash
pip install "comfort-db[geo,plot]"
```

**From source (editable, for development):**

```bash
git clone https://github.com/y-jenniges/comfort_lib.git
cd comfort_lib
pip install -e ".[dev]"
```

Requires Python $\ge$ 3.10.

---

## Quick start

```python
import comfort

# Overview of the database (no data loaded, takes some minutes)
comfort.info("/path/to/comfort.sqlite")

# Filter at SQL level, get back an xarray Dataset
ds = comfort.load_comfort(
    "/path/to/comfort.sqlite",
    parameters=["NITRATE", "DIC"],
    quality_flags=comfort.QC_GOOD,  # quality filter
    lat_min=30, lat_max=70,  # latitude filter
    date_min="2000-01-01",  # date filter
    target_depths=[0, 50, 100, 200, 500, 1000],  # depth interpolation
    limit=500_000,  # safeguard for large tables
)
# ds.NITRATE -> xr.DataArray with dims (profile, depth)

# Table exploration
with comfort.connect("/path/to/comfort.sqlite") as conn:
    summary = comfort.describe_variables(conn, parameters=["NITRATE", "DIC"], quality_flags=comfort.QC_GOOD)
```

---

## Documentation and examples

Full documentation (built with Sphinx):

```bash
pip install -e ".[docs]"
sphinx-build documentation/source documentation/build/html
```

Runnable scripts (partly self-contained, partly requiring the database) covering every module live in [`examples/`](examples/) - start with `01_quickstart.py`.

---

## Testing

```bash
pip install -e ".[dev]"
pytest

# Include the bathymetry round-trip test (needs a GEBCO NetCDF file)
export BATHYMETRY_PATH=/path/to/gebco_bathymetry.nc
pytest
```

---

## Citing

If you use `comfort-db` in published work, please cite the COMFORT dataset [1] and this library.
TEOS-10 physical oceanography (`comfort.physics`) is computed via the [`gsw`](https://github.com/TEOS-10/GSW-Python) 
library - please also cite [2]. 

## License

MIT
