# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Simplified logging utilities to replace nemo.utils.logging
"""

import logging as _logging
import sys
from typing import Optional

import torch


def is_global_rank_zero() -> bool:
    """
    Check if current process is global rank 0.
    Useful for DDP training to only print/log from rank 0.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True  # If not distributed, always return True


def get_logger(name: Optional[str] = None):
    """Get a logger instance."""
    return _logging.getLogger(name or __name__)


# Create a logging module-like object for compatibility
class LoggingModule:
    """Module-like object for logging compatibility."""
    
    def info(self, msg, *args, **kwargs):
        _logging.info(msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        _logging.warning(msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        _logging.error(msg, *args, **kwargs)
    
    def debug(self, msg, *args, **kwargs):
        _logging.debug(msg, *args, **kwargs)


# Configure logging
_logging.basicConfig(
    level=_logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Create module-like object
logging = LoggingModule()

# Export for compatibility with nemo.utils.logging
__all__ = ['get_logger', 'logging']

