"""
train & test entity linking for Web Q.
"""

import sys
import numpy as np
import tensorflow as tf

from xusheng.task.qa.linking.model import EntityLinker
from xusheng.util.config import ConfigDict
from xusheng.util.log_util import LogInfo

if __name__ == '__main__':
    # config
    setting_dir = sys.argv[1]
    try_dir = sys.argv[2]
    root_path = 'runnings/%s/%s' % (setting_dir, try_dir)
    config_path = '%s/param_config' % root_path
    config = ConfigDict(config_path)

    # model
    model = EntityLinker(config=config)

