from _config_section import ConfigSection
from pnr.data.constant import data_dir
import os
REAL_PATH = data_dir

data = ConfigSection("data")
data.dir = "%s/%s" % (REAL_PATH, "data")

data.config = ConfigSection("config")
data.config.dir = "%s/%s" % (data.dir, "config")
