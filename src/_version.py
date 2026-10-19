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

# git fills these in when it builds an archive of JMod (a zip or tarball
# download from GitHub; see .gitattributes).  In a git checkout they stay as
# written, and run_jmod asks git instead.
COMMIT = "$Format:%H$"
DESCRIBE = "$Format:%(describe:tags)$"
