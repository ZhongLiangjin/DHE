# Source code for DHE model

* [Overview](#overview)
* [How to Use](#How-to-Use)

## Overview

The code includes the distributed hybrid ecohydrological (DHE) model presented in paper titled "***Advancing streamflow prediction in data-scarce regions through vegetation-constrained distributed hybrid ecohydrological models***"  published in *Journal of Hydrology*.

If you have any questions or suggestions with the code or find a bug, please let us know. You are welcome to contact Liangjin Zhong at _zhonglj21@mails.tsinghua.edu.cn_

## How to Use

The code was built on Pytorch. To use this code, please do:

1. Install the dependencies use the following command:

   ```none
   pip install -r requirements.txt
   ```

2. Prepare your dataset 

   Please follow the instructions to prepare data for your own study area.

   - Meteorological data (including temperature, precipitation, and pontential evaportranspiration) and static attributes (more details please refer to the paper) are needed to force the models, please organize the data by sub-basins (such as the format shown in '*data/data.pkl*').
   - Streamflow and LAI data are required for parameterization. Please sort data in the format shown in the '*data/streamflow.csv*' and '*data/LAI.csv*' .
   - The shapefile of sub-basins should be included in the folder '*data/sub-basins*'.

3. Modify the model configuration in the '*main.py*' if needed and run the DHE model. 

4. More details about hyperparameter tuning could be found in the paper and codes.
