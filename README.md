# First Flash
## The first lightning flash (i.e. first flash) from the GOES-R Series Geostationary Lightning Mapper (GLM)

### Premise
This repository is desinged to identify first flashes from the GOES-East/-West GLMs, save the lightning flash locations and attributes, and compare them to coincident...
- ABI imagery
- Radar (MRMS)
- Ground based lightning networks (Earth Networks and Lightning Mapping Arrays)
- RAP-based objective analysis from the SPC (GEMPAK-based SPC MesoAnalysis archive)

This project focused on data from 2022 featuring the GOES-16 and GOES-17 GLMs, and would need to be modified for the GOES-18 and GOES-19 GLMs. 
Many of the files are output as 'RAW' csv files in two hour chunks using my desktops 12 CPUs and multithreading to increase processing speed, and then combined into a 'compiled' csv file later.
GLM first flashes can be output as netCDFs on a per GLM basis for the entire field of view or land-only points. 
ABI imagery, MRMS data, and ENI data from the perspective of each GLM can all be combined with the GLM first flash files into a 'jumbo' netCDF file.

### Processing steps
Coming soon...
