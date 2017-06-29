# STH-Multigrid-Reconstruction
This repository provides necessary functions and examples on how to do multi-grid reconstruction on projection image collected in open CT format. For numerical shepp-logan examples, see [odl-multigrid](https://github.com/kohr-h/odl-multigrid). 

## Requirements

The script rely on several python modules, paritculatrly 

```
odl
numpy
matplotlib
pickle
os
glob
dicom
```

The ```dicom``` module can be most easily installed by ```easy_install pydicom```. Note that the conversion here intends to generate DICOM-CT-PD-format, instead of standard DICOM. For this, the default ```_dicom_dict.py``` dictionary in ```pydicom``` has to be replaced by the ```_dicom_dict.py``` file given in this repository. Generally, this file (to be replaced) can be found in ```/path/to/lib/python2.7/site-packages/pydicom-0.9.8-py2.7.egg/dicom/```.

For the reconstruction, the python library [```odl```](https://github.com/odlgroup/odl/) is used. 


## Files

### functions/data_storage.py
Function to save reconstructed data file as pickled .txt-files

### functions/display_function.py
Function to display multigrid reconstructions, overlaying fine and coarse grid reconstructions. The function is created for one ROI, and several ROI display remains to be implemented. Note that for the dual-display, the coarse grid is expanded (using ```ExpandMatrix```) to have the same artificial discretization as the fine grid.

### functions/sinogram_generation.py
Generate sinogram from input openCT dicom-files. Option input in the form of light field and potential min/max definitions for truncation.

### functions/odl_modification/*
Modified odl-scripts, basically created to enable output of time/iteration and the L2-norm between Radon transform forward operator and projection data. For more information see ```functions/odl_modification/README.md```

### examples/FBP_multigrid.py
FBP example.

### examples/CG_multigrid.py
CG example.

### examples/TV_multigrid.py
TV example.

### examples/CG_multigrid_2ROIs.py
CG example with 2 fine-discretization ROIs. Note that the display function cannot handle this case for now.

