# Setup

1. Extract the pnr data from *link here*

2. Add the path to the data you extracted to the `lineup/data/constant.py` file.
```py
import os
if os.environ['HOME'] == '/home/neil':
    data_dir = '/home/neil/projects/pnr-labels/pnr'
    # insert data path as data_dir
else:
    raise Exception("Unspecified data_dir, unknown environment")
```

3. Extract the data provided from [link](https://drive.google.com/open?id=0B3s12MhYb3jOZFM4V1U2emdwQ0k)

4. Install the packages required from the repo dir.
```
python setup.py build
python setup.py install
```
