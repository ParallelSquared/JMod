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

import os

# All environment variables jmod sets live here, and only here. This module
# is the first `src` import in every entry path (CLI wrapper, GUI wrapper,
# pytest), which is early enough for libraries that read their config at
# first load (polars' allocator, TensorFlow). All are setdefault: anything
# exported in the shell wins.

# TODO: revisit if TF C++ INFO logs are ever needed; hides the benign oneDNN
# round-off notice
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

# Make polars' allocator return freed pages to the OS immediately instead of
# pooling them. The streaming engine allocates thousands of small pipe buffers
# whose freed pages the allocator otherwise retains indefinitely: on a 23M-row
# library parse this held ~9 GB of RSS with <1 GB of live data, and the
# fragmented pool makes later large contiguous allocations (e.g. the initial
# search's multi-GB buffers) likelier to fail. Measured on macOS: retained RSS
# 9.9 -> 3.3 GB, peak 14.5 -> 11.4 GB, parse time unchanged.
#
# Must be set before polars is first imported (the allocator reads its config
# at library load). setdefault, so an explicitly exported value always wins.
# polars links jemalloc on macOS/Linux and mimalloc on Windows; each allocator
# reads its own variable and ignores the other.
os.environ.setdefault("_RJEM_MALLOC_CONF", "dirty_decay_ms:0,muzzy_decay_ms:0")
os.environ.setdefault("MIMALLOC_PURGE_DELAY", "0")
