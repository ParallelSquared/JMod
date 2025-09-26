#!/usr/bin/env python3
"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""

"""
Wrapper script to run JMod from the project root directory.
This allows you to keep the main script in src/ while still running from root.
"""

import sys
import os

from pyinstrument import Profiler
from pyinstrument.renderers import JSONRenderer, SpeedscopeRenderer, ConsoleRenderer

# Add src directory to Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)




# if __name__ == "__main__":
#     from src import logger
#     from src.run_jmod import main
#     jsons = [
#         r"C:\Users\zcohe\Jmod\JMod_Profiling\Output\Changed_merging\Faster_fit_mtraq\config.json"
#     ]


#     for json in jsons:
#         if not os.path.exists(json):
#             raise FileNotFoundError(f"Config file not found: {json}")

#     for i, jfile in enumerate(jsons, start=1):
#         profiler = Profiler()
#         profiler.start()

#         main(jfile)

#         profiler.stop()

#         run_dir = os.path.dirname(jfile)
#         os.makedirs(run_dir, exist_ok=True)

#         renderer = JSONRenderer(show_all=False)
#         renderer.min_percentage = 0.01
#         with open(os.path.join(run_dir, "profile.json"), "w") as f:
#             f.write(renderer.render(profiler.last_session))

#         renderer = SpeedscopeRenderer(show_all=False)
#         renderer.min_percentage = 0.01
#         with open(os.path.join(run_dir, "profile.speedscope.json"), "w") as f:
#             f.write(renderer.render(profiler.last_session))



# if __name__ == "__main__":
#     # profiler = Profiler()
#     # profiler.start()

#     from src import logger
#     # Import and run the main module
#     from src.run_jmod import main
#     main()

    # profiler.stop()

    # base_path = r"C:\Users\zcohe\Jmod\JMod_Profiling\Output\mess around 2"

#     # Text
#     # with open(os.path.join(base_path, "profile.txt"), "w") as f:
#     #     f.write(ConsoleRenderer().render(profiler.last_session))

    # # JSON
    # with open(os.path.join(base_path, "profile.json"), "w") as f:
    #     f.write(JSONRenderer(show_all=True).render(profiler.last_session))

    # # Speedscope
    # with open(os.path.join(base_path, "profile.speedscope.json"), "w") as f:
    #     f.write(SpeedscopeRenderer(show_all=True).render(profiler.last_session))

if __name__ == "__main__":
    from src import logger
    # Import and run the main module
    from src.run_jmod import main
    main()

