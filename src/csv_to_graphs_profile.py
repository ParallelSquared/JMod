import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import re
import shutil

#TO Run:
# - Change output folder
# - Go to RT TOlerance experiment
#    - Change the csv path dict
#    - ms1 spectra, ms2 spectra, and lib precursors will be the same for RT TOl experiment, but if doing another experiment and want to graph these you can add them
#    - change peptide IDs and average candidates per spectrum. Get peptide IDs from log file. Get average candidates per spectrum from output of json_list_to_hist.py

def main():
    graphing_dict = {}
    main_times = {}

    output_folder = r"C:\Users\zcohe\Jmod\JMod_Profiling\Output\Changed_merging\PyInstrument Graphs"
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    colors = ["steelblue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]

    """FIRST EXPERIMENT"""

    # file_list = [
    #     'LF Small Library 40 min gradient', 
    #     'LF Largish Library 40 min gradient', 
    #     'PSMtag d0 40 min gradient', 
    #     'PSMtag 5plex 40 min gradient', 
    #     'PSMtag 9plex 40 min gradient',
    #     'PSMtag 5plex 10 min gradient'
    #     ]
    
    # csv_path_dict = {
    #     'LF Small Library 40 min gradient': r"C:\Users\zcohe\Jmod\JMod_Profiling\Output\LF Small Library 40 min gradient\profile_paths.csv",
    #     'LF Largish Library 40 min gradient': r"C:\Users\zcohe\Jmod\JMod_Profiling\Output\LF Largish Library 40 min gradient\profile_paths.csv",
    #     'PSMtag d0 40 min gradient': r"C:\Users\zcohe\Jmod\JMod_Profiling\Output\PSMtag d0 40 min gradient\profile_paths.csv",
    #     'PSMtag 5plex 40 min gradient': r"C:\Users\zcohe\Jmod\JMod_Profiling\Output\PSMtag 5plex 40 min gradient\profile_paths.csv",
    #     'PSMtag 9plex 40 min gradient': r"C:\Users\zcohe\Jmod\JMod_Profiling\Output\PSMtag 9plex 40 min gradient\profile_paths.csv",
    #     'PSMtag 5plex 10 min gradient': r"C:\Users\zcohe\Jmod\JMod_Profiling\Output\PSMtag 5plex 10 min gradient\profile_paths.csv"
    #     }

    # ms2_spectra_dict = {
    #     'LF Small Library 40 min gradient': 16095,
    #     'LF Largish Library 40 min gradient': 16095,
    #     'PSMtag d0 40 min gradient': 32680,
    #     'PSMtag 5plex 40 min gradient': 35557,
    #     'PSMtag 9plex 40 min gradient': 37054,
    #     'PSMtag 5plex 10 min gradient': 47523
    #     }

    
    # ms1_spectra_dict = {
    #     'LF Small Library 40 min gradient': 1074,
    #     'LF Largish Library 40 min gradient': 1074,
    #     'PSMtag d0 40 min gradient': 3391,
    #     'PSMtag 5plex 40 min gradient': 3642,
    #     'PSMtag 9plex 40 min gradient': 3686,
    #     'PSMtag 5plex 10 min gradient': 977
    #     }

    
    # lib_precursors_dict = {
    #     'LF Small Library 40 min gradient': 21743,
    #     'LF Largish Library 40 min gradient': 209712,
    #     'PSMtag d0 40 min gradient': 46972,
    #     'PSMtag 5plex 40 min gradient': 234860,
    #     'PSMtag 9plex 40 min gradient': 422748,
    #     'PSMtag 5plex 10 min gradient': 234860
    # }


    # peptide_IDs_dict = {
    #     'LF Small Library 40 min gradient': 7321,
    #     'LF Largish Library 40 min gradient': 11934,
    #     'PSMtag d0 40 min gradient': 30613,
    #     'PSMtag 5plex 40 min gradient': 84394,
    #     'PSMtag 9plex 40 min gradient': 130473,
    #     'PSMtag 5plex 10 min gradient': 27475
    # }


    # average_frags_per_spectrum = ["0.0125", "0.025", "0.05", "0.1", "0.2", "0.4", "0.8", "1.6", "3.2", "6.4"]
    # average_frags_per_spectrum_dict = {}
    # for i, x in enumerate(rt_tol_numbers):
    #     average_frags_per_spectrum_dict[f'5plex_40min_RTtol_{x}'] = average_frags_per_spectrum[i]

    """RT TOLERANCE EXPERIMENT"""

    rt_tol_numbers = ["Pre", "Faster_fit_mtraq"]
    file_list = []
    for x in rt_tol_numbers:
        file_list.append(f'{x}_Merging_Change')
    
    file_to_color = {file: colors[i] for i, file in enumerate(file_list)}

    csv_path_dict = {}
    for x in rt_tol_numbers:
        csv_path_dict[f'{x}_Merging_Change'] = rf"C:\Users\zcohe\Jmod\JMod_Profiling\Output\Changed_merging\{x}_Merge\profile_paths.csv"
    
    # ms2_spectra_dict = {}
    # for x in rt_tol_numbers:
    #     ms2_spectra_dict[f'5plex_40min_RTtol_{x}'] = 35557
        
    # ms1_spectra_dict = {}
    # for x in rt_tol_numbers:
    #     ms1_spectra_dict[f'5plex_40min_RTtol_{x}'] = 3642
        
    # lib_precursors_dict = {}
    # for x in rt_tol_numbers:
    #     lib_precursors_dict[f'5plex_40min_RTtol_{x}'] = 234860
        
    # peptide_IDs_initial_list = [80685, 81211]
    # peptide_IDs_dict = {}
    # for i, x in enumerate(rt_tol_numbers):
    #     peptide_IDs_dict[f'{x}_Merging_Change'] = peptide_IDs_initial_list[i]

    # average_candidates_per_spectrum = [2.095, 3.950, 6.623, 8.808, 9.809, 10.622, 12.106, 14.860, 19.564, 26.654]
    # average_candidates_per_spectrum_dict = {}
    # for i, x in enumerate(rt_tol_numbers):
    #     average_candidates_per_spectrum_dict[f'5plex_40min_RTtol_{x}'] = average_candidates_per_spectrum[i]

    # rt_tol_dict = {}
    # for i, x in enumerate(rt_tol_numbers):
    #     rt_tol_dict[f'5plex_40min_RTtol_{x}'] = float(x)





    ######## START CODE HERE #######

    for file, csv_path in csv_path_dict.items():
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                function = " - ".join(row[1:-2])
                time = row[-1]
                if function not in graphing_dict.keys():
                    graphing_dict[function] = {}
                graphing_dict[function][file] = float(time)
        main_times[file] = graphing_dict["<module> - main -  -  -  -  -  -  -  -  -  -  -  - "][file]
    

    new_graphing_dict = graphing_dict.copy()
    for function, file_time_dict in graphing_dict.items():
        self_child_name = function + " - z_not_otherwise_profiled"
        new_graphing_dict[self_child_name] = {}
        for file in file_time_dict.keys():
            children = get_children(function, graphing_dict)
            child_sum = sum(graphing_dict[ch].get(file, 0) for ch in children)
            total_time = file_time_dict.get(file, 0)
            self_time = total_time - child_sum
            new_graphing_dict[self_child_name][file] = self_time
    graphing_dict = new_graphing_dict


    for function, file_time_dict in graphing_dict.items():

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

        files = list(file_time_dict.keys())
        raw_times = [r/60 for r in list(file_time_dict.values())]
        percents = [time / main_times[file] * 100 for file, time in file_time_dict.items()]

        # --- Raw times ---
        ax1.bar(files, raw_times, color=[file_to_color[f] for f in files])
        ax1.set_title("Raw time")
        ax1.set_ylabel("Time (min)")

        # --- Percentages ---
        ax2.bar(files, percents, color=[file_to_color[f] for f in files])
        ax2.set_title("Percent of main time")
        ax2.set_ylabel("% of main time")
        ax2.set_ylim(0, 100)


        parent = get_parent(function)
        if parent and parent in graphing_dict:
            parent_times = graphing_dict[parent]
            percents_parent = [
                (time / parent_times[file] * 100) if file in parent_times else 0
                for file, time in file_time_dict.items()
            ]
        else:
            percents_parent = [0 for _ in files]

        # --- Percentages of parent ---
        parts = [p for p in function.split(" - ") if p]
        parent_name = parts[-2] if len(parts) >= 2 else None
        ax3.bar(files, percents_parent, color=[file_to_color[f] for f in files])
        ax3.set_title(f"Percent of parent ({parent_name}) time")
        ax3.set_ylabel("% of parent time")
        ax3.set_ylim(0, 100)

        # # --- Spectra normalized time ---
        #time_in_sec = [r * 60 for r in raw_times]
        # spectra_normalized_times = [(time_in_sec[i]*1000) / ms2_spectra[i] for i in range(len(files))]
        # ax4.bar(files, spectra_normalized_times, color=[file_to_color[f] for f in files])
        # ax4.set_title("Spectra Normalized Time")
        # ax4.set_ylabel("Time/ 1000 spectra(sec)")


        # # --- Libsize normalized time ---
        # libsize_normalized_times = [(time_in_sec[i]*1000) / lib_precursors[i] for i in range(len(files))]
        # ax5.bar(files, libsize_normalized_times, color=[file_to_color[f] for f in files])
        # ax5.set_title("LibSize Normalized time")
        # ax5.set_ylabel("Time/ 1000 precursors (sec)")

        # # --- Spectra vs Time ---
        from scipy.stats import linregress
        import numpy as np
        time_in_sec = [r * 60 for r in raw_times]

        # # --- MS2 Spectra vs Time ---
        # x_vals = [ms2_spectra_dict[f] for f in files]
        # y_vals = time_in_sec
        # if len(set(x_vals)) > 1:
        #     ax4.scatter(x_vals, y_vals, color=[file_to_color[f] for f in files])
        #     for i, f in enumerate(files):
        #         ax4.annotate(f, (x_vals[i], y_vals[i]), fontsize=8, ha="right")
        #     ax4.set_title("Time vs MS2 Spectra")
        #     ax4.set_xlabel("# MS2 spectra")
        #     ax4.set_ylabel("Time (sec)")
        #     ax4.set_ylim(0, max(y_vals)*1.1)

        #     slope, intercept, r_value, p_value, std_err = linregress(x_vals, y_vals)
        #     line_x = np.linspace(min(x_vals), max(x_vals), 100)
        #     line_y = slope * line_x + intercept
        #     ax4.plot(line_x, line_y, color="black", linestyle="--", label=f"R² = {r_value**2:.3f}")
        #     ax4.legend(loc="lower right")


        # # --- Library size vs Time ---
        # x_vals = [lib_precursors_dict[f] for f in files]
        # y_vals = time_in_sec
        # if len(set(x_vals)) > 1:
        #     ax5.scatter(x_vals, y_vals, color=[file_to_color[f] for f in files])
        #     for i, f in enumerate(files):
        #         ax5.annotate(f, (x_vals[i], y_vals[i]), fontsize=8, ha="right")
        #     ax5.set_title("Time vs Library Size")
        #     ax5.set_xlabel("# Lib Precursors")
        #     ax5.set_ylabel("Time (sec)")
        #     ax5.set_ylim(0, max(y_vals)*1.1)

        #     slope, intercept, r_value, p_value, std_err = linregress(x_vals, y_vals)
        #     line_x = np.linspace(min(x_vals), max(x_vals), 100)
        #     line_y = slope * line_x + intercept
        #     ax5.plot(line_x, line_y, color="black", linestyle="--", label=f"R² = {r_value**2:.3f}")
        #     ax5.legend(loc="lower right")


        # --- Peptide IDs vs Time ---
        # x_vals = [peptide_IDs_dict[f] for f in files]
        # y_vals = time_in_sec
        # if len(set(x_vals)) > 1:
        #     ax4.scatter(x_vals, y_vals, color=[file_to_color[f] for f in files])
        #     # for i, f in enumerate(files):
        #     #     ax4.annotate(f, (x_vals[i], y_vals[i]), fontsize=8, ha="right")
        #     ax4.set_title("Time vs Peptide IDs")
        #     ax4.set_xlabel("# Peptide IDs")
        #     ax4.set_ylabel("Time (sec)")
        #     ax4.set_ylim(0, max(y_vals)*1.1)

        #     slope, intercept, r_value, p_value, std_err = linregress(x_vals, y_vals)
        #     line_x = np.linspace(min(x_vals), max(x_vals), 100)
        #     line_y = slope * line_x + intercept
        #     ax4.plot(line_x, line_y, color="black", linestyle="--", label=f"R² = {r_value**2:.3f}")
        #     ax4.legend(loc="lower right")

        # # --- MS1 Spectra vs Time ---
        # x_vals = [ms1_spectra_dict[f] for f in files]
        # y_vals = time_in_sec
        # if len(set(x_vals)) > 1:
        #     ax7.scatter(x_vals, y_vals, color=[file_to_color[f] for f in files])
        #     for i, f in enumerate(files):
        #         ax7.annotate(f, (x_vals[i], y_vals[i]), fontsize=8, ha="right")
        #     ax7.set_title("Time vs MS1 Spectra")
        #     ax7.set_xlabel("# MS1 Spectra")
        #     ax7.set_ylabel("Time (sec)")
        #     ax7.set_ylim(0, max(y_vals)*1.1)

        #     slope, intercept, r_value, p_value, std_err = linregress(x_vals, y_vals)
        #     line_x = np.linspace(min(x_vals), max(x_vals), 100)
        #     line_y = slope * line_x + intercept
        #     ax7.plot(line_x, line_y, color="black", linestyle="--", label=f"R² = {r_value**2:.3f}")
        #     ax7.legend(loc="lower right")

        # # --- Average Fragments per Spectrum vs. Time ---
        # x_vals = [average_frags_per_spectrum_dict[f] for f in files]
        # y_vals = time_in_sec
        # if len(set(x_vals)) > 1:
        #     ax8.scatter(x_vals, y_vals, color=[file_to_color[f] for f in files])
        #     for i, f in enumerate(files):
        #         ax8.annotate(f, (x_vals[i], y_vals[i]), fontsize=8, ha="right")
        #     ax8.set_title("Time vs Average Fragments per Spectrum")
        #     ax8.set_xlabel("# Average Fragments per Spectrum")
        #     ax8.set_ylabel("Time (sec)")
        #     ax8.set_ylim(0, max(y_vals)*1.1)

        #     slope, intercept, r_value, p_value, std_err = linregress(x_vals, y_vals)
        #     line_x = np.linspace(min(x_vals), max(x_vals), 100)
        #     line_y = slope * line_x + intercept
        #     ax8.plot(line_x, line_y, color="black", linestyle="--", label=f"R² = {r_value**2:.3f}")
        #     ax8.legend(loc="lower right")

        # --- Average Candidates per Spectrum vs. Time ---
        # x_vals = [average_candidates_per_spectrum_dict[f] for f in files]
        # y_vals = time_in_sec
        # if len(set(x_vals)) > 1:
        #     ax5.scatter(x_vals, y_vals, color=[file_to_color[f] for f in files])
        #     # for i, f in enumerate(files):
        #     #     ax5.annotate(f, (x_vals[i], y_vals[i]), fontsize=8, ha="right")
        #     ax5.set_title("Time vs Average Candidates per Spectrum")
        #     ax5.set_xlabel("Average Candidates per Spectrum")
        #     ax5.set_ylabel("Time (sec)")
        #     ax5.set_ylim(0, max(y_vals)*1.1)

        #     slope, intercept, r_value, p_value, std_err = linregress(x_vals, y_vals)
        #     line_x = np.linspace(min(x_vals), max(x_vals), 100)
        #     line_y = slope * line_x + intercept
        #     ax5.plot(line_x, line_y, color="black", linestyle="--", label=f"R² = {r_value**2:.3f}")
        #     ax5.legend(loc="lower right")

        # --- RT_tol vs. Time ---
        # x_vals = [rt_tol_dict[f] for f in files]
        # y_vals = time_in_sec
        # if len(set(x_vals)) > 1:
        #     ax6.scatter(x_vals, y_vals, color=[file_to_color[f] for f in files])
        #     # for i, f in enumerate(files):
        #     #     ax5.annotate(f, (x_vals[i], y_vals[i]), fontsize=8, ha="right")
        #     ax6.set_title("Time vs RT_tol")
        #     ax6.set_xlabel("RT_tol")
        #     ax6.set_ylabel("Time (sec)")
        #     ax6.set_ylim(0, max(y_vals)*1.1)

        #     slope, intercept, r_value, p_value, std_err = linregress(x_vals, y_vals)
        #     line_x = np.linspace(min(x_vals), max(x_vals), 100)
        #     line_y = slope * line_x + intercept
        #     ax6.plot(line_x, line_y, color="black", linestyle="--", label=f"R² = {r_value**2:.3f}")
        #     ax6.legend(loc="lower right")

        # # --- RT_Tol vs. Time and Average Candidates per Spectrum ---
        # x_vals = [rt_tol_dict[f] for f in files]
        # y_vals_1 = [average_candidates_per_spectrum_dict[f] for f in files]
        # y_vals_2 = [x/60 for x in time_in_sec]

        # sorted_idx = np.argsort(x_vals)
        # x_sorted = np.array(x_vals)[sorted_idx]
        # y1_sorted = np.array(y_vals_1)[sorted_idx]
        # y2_sorted = np.array(y_vals_2)[sorted_idx]


        # if len(set(x_sorted)) > 1:
        #     ax6.plot(x_sorted, y1_sorted, color="blue", label="Average Candidates per Spectrum")
        #     ax6.plot(x_sorted, y2_sorted, color="red", label="Time")
        #     ax6.set_title("RT_Tol vs. Time and Average Candidates per Spectrum")
        #     ax6.set_xlabel("RT_Tol")
        #     ax6.set_ylabel("Time (min)")
        #     ax6.legend(loc="upper left")

        #     from scipy.interpolate import make_interp_spline
        #     xnew = np.linspace(min(x_sorted), max(x_sorted), 300)
        #     spline1 = make_interp_spline(x_sorted, y1_sorted, k=3)
        #     spline2 = make_interp_spline(x_sorted, y2_sorted, k=3)
        #     y1_smooth = spline1(xnew)
        #     y2_smooth = spline2(xnew)
        #     ax6.plot(xnew, y1_smooth, color="blue")
        #     ax6.plot(xnew, y2_smooth, color="red")



        for ax in [ax1, ax2, ax3]:
            ax.tick_params(axis="x", rotation=45)
            for label in ax.get_xticklabels():
                label.set_horizontalalignment("right")
                label.set_rotation_mode("anchor")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plot_title = " - ".join(parts)
        fig.suptitle(plot_title, fontsize=16)

        parts = [re.sub(r'[<>:"/\\|?*]', "_", p) for p in function.split(" - ") if p]
        if parts:
            os.makedirs(os.path.join(output_folder, *parts[:-1]), exist_ok=True)
            save_path = os.path.join(output_folder, *parts[:-1], parts[-1] + ".png")
            plt.savefig(save_path)
        plt.close()

def get_parent(function: str) -> str | None:
    parts = function.split(" - ")
    if len(parts) <= 1:
        return None
    # remove the last non-empty part
    for i in range(len(parts) - 1, -1, -1):
        if parts[i]:
            parts[i] = ""
            break
    return " - ".join(parts)

def get_children(function: str, graphing_dict: dict) -> list[str]:
    """Return list of direct children of a function."""
    return [cand for cand in graphing_dict if get_parent(cand) == function]

if __name__ == "__main__":
    main()