#  Copyright (c) 2026 Parallel Squared Technology Institute
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


class RunState:
    """Values specific to one run.

    ``file_name`` is the run's data file, set when the run starts, and
    ``results_folder`` is where its output goes, set once that exists.  The
    ``opt_`` fields are fitted from the run's data by the first search,
    starting from the defaults in config (e.g. ``opt_rt_tol`` from
    ``config.rt_tol``).  ``target_decoy_ratio`` is counted when the library is
    calibrated to the run.

    Every field starts unset.  Reading one before it has been set raises, and
    so does assigning a name that is not a field, so a path that forgets to set
    a value fails immediately instead of falling back to a default or to the
    previous run's value.
    """

    __slots__ = (
        "file_name",
        "results_folder",
        "opt_rt_tol",
        "opt_ms1_tol",
        "opt_im_precision",
        "opt_im_accuracy",
        "target_decoy_ratio",
    )

    def __getattr__(self, name):
        # Only reached when normal lookup fails: an unset field or a non-field
        if name in RunState.__slots__:
            raise AttributeError(f"RunState.{name} was read before it was set")
        raise AttributeError(f"{name!r} is not a RunState field")

    def as_dict(self):
        """Every field, with None for those not set yet (for writing out)."""
        return {name: getattr(self, name, None) for name in RunState.__slots__}
