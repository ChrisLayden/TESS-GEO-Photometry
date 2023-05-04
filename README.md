Provides an interface to estimate the photometric performance of TESS-GEO.

Requires pysynphot, matplotlib, numpy, and scipy.

CPL 05/04/23

To open the interface, do

```
cd scripts
python TESS-GEO_Photometry.py
```

There you can set the parameters of your observing system and predict its
response to various spectra. You may also choose to use the current best
estimates for the TESS-GEO sensors and telescopes. Currently the values
for the UV sensor and telescope are not known too well, so don't trust
the photometric results you get with those options yet.