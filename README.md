Provides an interface to estimate the photometric performance of TESS-GEO.

Requires tkinter, pysynphot, matplotlib, astropy, numpy, and scipy.

CPL 05/04/23

Steps to make pysynphot operational:
1) Install pysynphot: if you have a conda environment set up, use

```
conda install pysynphot
```

Otherwise, if you have python and pip installed, use

```
pip install pysynphot
```

If neither of these work, see more information at https://pysynphot.readthedocs.io/en/latest/

2) Download just the first two sets of data files
(http://ssb.stsci.edu/trds/tarfiles/synphot1.tar.gz and
http://ssb.stsci.edu/trds/tarfiles/synphot2.tar.gz); unpack these
to some directory /my/dir

3) set
```
export PYSYN_CDBS=/my/dir/grp/redcat/trds
```

To open the interface, do

```
cd scripts
python TESS-GEO_Photometry.py
```

There you can set the parameters of your observing system and predict its
response to various spectra. You may also choose to use the current best
estimates for the TESS-GEO sensors and telescopes.
