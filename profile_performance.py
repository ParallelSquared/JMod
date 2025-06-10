#!/usr/bin/env python3
"""
Quick performance profiling script to identify bottlenecks in JMod.

This script can be run to get timing information about which functions
are taking the most time during the initial search.
"""

import sys
import time
import cProfile
import pstats
from io import StringIO

def profile_jmod():
    """Run JMod with profiling enabled."""
    
    # Add src to path
    sys.path.insert(0, 'src')
    
    # Import and run the main function
    from src.run_jmod import main
    
    print("Starting JMod with profiling...")
    print("This will run for a short time then show performance stats.")
    
    # Create a profiler
    profiler = cProfile.Profile()
    
    try:
        # Start profiling
        profiler.enable()
        
        # Run for just a short time to get initial timing
        start_time = time.time()
        main()
        
    except KeyboardInterrupt:
        print("\nProfiling interrupted by user")
    except Exception as e:
        print(f"\nProfiling stopped due to: {e}")
    finally:
        # Stop profiling
        profiler.disable()
        
        # Print timing summary from our custom timing system
        try:
            from src.utils.timing_debug import get_timing_summary
            print("\n" + "="*80)
            print("CUSTOM TIMING SUMMARY")
            print("="*80)
            print(get_timing_summary())
        except:
            print("Custom timing not available")
        
        # Generate profiling report
        print("\n" + "="*80)
        print("PYTHON PROFILER RESULTS (Top 20 slowest functions)")
        print("="*80)
        
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)
        
        profile_output = s.getvalue()
        print(profile_output)
        
        # Also save to file
        with open('profile_results.txt', 'w') as f:
            f.write("CUSTOM TIMING SUMMARY\n")
            f.write("="*80 + "\n")
            try:
                f.write(get_timing_summary())
            except:
                f.write("Custom timing not available\n")
            f.write("\n\nPYTHON PROFILER RESULTS\n")
            f.write("="*80 + "\n")
            f.write(profile_output)
        
        print(f"\nDetailed results saved to: profile_results.txt")


if __name__ == "__main__":
    profile_jmod()