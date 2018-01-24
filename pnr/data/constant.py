import os
if os.environ['HOME'] == '/home/neil':
    data_dir = '/home/neil/projects/pnr-labels/pnr'
else:
    raise Exception("Unspecified data_dir, unknown environment")
