.. COMFORT Library documentation master file, created by
   sphinx-quickstart on Thu Dec 23 15:22:05 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to comfort-db documentation!
===========================================

Python library for working with the [1] oceanographic SQLite database.

`comfort-db` wraps database access, preprocessing and basic analysis utilities into one library.
This includes filtering for quality and space-time ranges, unit conversions, spatiotemporal gridding, profile and water-mass analysis,
geospatial utilities, scaling, plotting and visualisation using a dashboard.

The data structure broadly follows oceanographic ship measurement practice:
At a latitude, longitude and time, a station defines where measurements were taken.
At a station, profiles are measured, i.e. a parameter like temperature at different depths with a specific instrument.
There can be multiple profiles at the same station, though in the COMFORT database, it is mostly one profile per station.


[1] Korablev, A., Olsen, A., Geophysical Institute, University of Bergen, Bjerknes Centre for Climate Research (2022). COMFORT Dataset [Data set]. NIRD RDA. https://doi.org/10.11582/2022.00039

.. McDougall, T., & Barker, P. (2011). Getting started with TEOS-10 and the Gibbs Seawater (GSW) Oceanographic Toolbox. SCOR/IAPSO WG127
.. report on comfort ?

.. toctree::
   :maxdepth: 6
   :caption: Contents:

   io
   qc
   profile_analysis
   physics
   geo
   sections
   units
   scaling
   gridding
   database
   plot
   app



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
