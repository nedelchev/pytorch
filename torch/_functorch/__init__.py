# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import sys
from ..config_utils import make_config_dataclass
from . import config as config_module

FunctorchConfig = make_config_dataclass('FunctorchConfig', config_module)
config = FunctorchConfig()
sys.modules[f'{__name__}.config'] = config
__all__ = ['config', 'FunctorchConfig']
