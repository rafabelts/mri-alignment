"""
Compatibility with the 'voxelmorph' library (v0.2), which was written for a version of Python earlier than
3.11 and uses 'inspect.getargspec', that was removed from Python 3.11+


IMPORTANT: This module must be imported BEFORE `voxelmorph`/`neurite` in any file that uses them, since 
environment variables only take effect if they are defined before the import.
"""

import os
import inspect
from collections import namedtuple

# vxm/neurite supports a pythorch or tensorflow backend, here we use pytorch.
# Note: "VXM_BACKEND" and "NEURITE_BACKEND" are the exact names required
# the Voxelmorph library itself — they cannot be configured by this project,
# unlike the MRI_* variables used in config.py.

os.environ.setdefault("VXM_BACKEND", "pytorch")
os.environ.setdefault("NEURITE_BACKEND", "pytorch")

if not hasattr(inspect, "getargspec"):
    _ArgSpec = namedtuple("ArgSpec", ["args", "varargs", "varkw", "defaults"])

    def _getargspec(func):
        spec = inspect.getfullargspec(func)
        return _ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = _getargspec
