# Known data issues
This file describes known data issues in the COMFORT database. 

## 1. Salinity-like oxygen values
There are 9,507,497 oxygen values (14.7%) that resemble their co-located salinity values with a Pearson correlation of 1.0. 
After filtering with the `comfort.qc.QC_GOOD` preset, this number reduces to 329,731 samples (0.8%). 
The values can be flagged by the `comfort.qc.flag_salinity_like_oxygen` function. 

There are 54,758 oxygen values (0.135%) below 40 umol/kg that do not have co-located salinity values, so 
`flag_salinity_like_oxygen` cannot check them. Most fall within known oxygen minimum zones or are close by
(open-ocean OMZs, Black Sea, Baltic Sea, etc.), though a small fraction of unphysical values could not be ruled out.
