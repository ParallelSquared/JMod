
"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""

# Update imports to relative imports
from ast import arg
import tkinter as tk
from tkinter import ttk
from sklearn import tree
from ttkthemes import ThemedTk
import sv_ttk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox
import json
from idlelib.tooltip import Hovertip
import os
from config import parser
from default_dict import default_dict
import shlex
from pathlib import Path
from pyteomics import mass
import re
from src.logger import logger, ElapsedFormatter
import logging
import threading
import tempfile



def make_GUI():
    """
    Function to initialize and run the JMod GUI.
    This function is called when the JMod is ran from GUI
    """

    class JModGUI(ThemedTk):
        def __init__(self):
            super().__init__()  #theme='vista' is classic
            self.title("JMod")
            self.state('zoomed')

            sv_ttk.set_theme("light")
            caption_font = tkfont.nametofont("SunValleyCaptionFont")
            caption_font.configure(size=10)
            #style = ttk.Style()
            #style.configure("Accent.TButton")
            #style.configure("Red.TButton", foreground="black", background="#cc0202")

            #frames:  1) Logging Frame 2) input frame,  3) MS frame  4)multiplex frame  5)additional frame   6) Output frame
            #functions 1) Logging Funcs 2) input funcs,  3) MS funcs  4) multiplex funcs 5) additional funcs  6) output funcs


            ####         Logging Frame      #######

            self.logging_frame = ttk.LabelFrame(self, text="Logging")
            self.logging_frame.grid(row=0, column=11, columnspan=10, rowspan=5, padx=10, pady=10, sticky="ew")

            self.text_widget = tk.Text(self.logging_frame, height=52, width=100)
            self.text_widget.pack(fill="both", expand=True)

            class TkinterHandler(logging.Handler):
                def __init__(self, text_widget):
                    super().__init__()
                    self.text_widget = text_widget

                def emit(self, record):
                    msg = self.format(record)
                    self.text_widget.after(0, self.append, msg)

                def append(self, msg):
                    self.text_widget.insert(tk.END, msg + "\n")
                    self.text_widget.see(tk.END)

            tk_handler = TkinterHandler(self.text_widget)
            tk_handler.setFormatter(ElapsedFormatter("%(asctime)s - %(levelname)s - %(message)s"))
            tk_handler.setLevel(logging.INFO)

            logger.addHandler(tk_handler)

            logger.info("GUI logger started.\n")


            #######     input frame      #######
            self.input_frame = ttk.LabelFrame(self, text="Input Files")
            self.input_frame.grid(row=0, column=0, columnspan=10, padx=10, pady=10, sticky="ew")

            # mzML and .d file input
            # Not saving this in default dict because we are only running one at a time
            self.mzml_label = ttk.Label(self.input_frame, text=".mzML:")
            self.mzml_label.grid(row=0, column=0, padx=10, pady=10, sticky="e")
            Hovertip(self.mzml_label, "Add any number of .mzML files to be ran sequentially ")
            self.file_dropdown = FileListDropdown(self.input_frame)
            self.file_dropdown.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
            self.mzml_button = ttk.Button(self.input_frame, text="Browse", style="Accent.TButton", command=self.select_mzml)
            self.mzml_button.grid(row=0, column=2, padx=10, pady=10)

            # Speclib file input
            self.tsv_label = ttk.Label(self.input_frame, text="Spectral Library:")
            self.tsv_label.grid(row=1, column=0, padx=10, pady=10, sticky="e")
            Hovertip(self.tsv_label, "Add a spectral library in .tsv format to be used for all runs ")
            self.tsv_entry = ttk.Entry(self.input_frame, width=50)
            self.tsv_entry.grid(row=1, column=1, padx=10, pady=10)
            default_dict["speclib"]["tk_handle"] = self.tsv_entry
            self.tsv_button = ttk.Button(self.input_frame, text="Browse", style="Accent.TButton", command=self.select_tsv)
            self.tsv_button.grid(row=1, column=2, padx=10, pady=10)

            #JSON Config input
            # Not saving this in default dict because once we have loaded in the JSON we don't care about it anymore
            self.json_label = ttk.Label(self.input_frame, text="Config JSON:")
            self.json_label.grid(row=2, column=0, padx=10, pady=10, sticky="e")
            Hovertip(self.json_label, "OPTIONAL: Add a JSON configuration file to automatically load parameters\n\n This file can be generated from 'Save Configuration' or found in the output folder \n of a previous run as 'config.json'")
            self.json_entry = ttk.Entry(self.input_frame, width=50)
            self.json_entry.grid(row=2, column=1, padx=10, pady=10)
            self.json_button = ttk.Button(self.input_frame, text="Browse", style="Accent.TButton", command=lambda: self.select_json())
            self.json_button.grid(row=2, column=2, padx=10, pady=10)

            ####         MS Frame      #######
            self.ms_frame = ttk.LabelFrame(self, text="MS Settings")
            self.ms_frame.grid(row=1, column=0, columnspan=10, padx=10, pady=10, sticky="ew")

            def on_combobox_select(event):
                event.widget.icursor("end")
                event.widget.selection_clear()

            # MS1 ppm
            self.ms1_lab = ttk.Label(self.ms_frame, text="MS1 ppm:")
            self.ms1_lab.grid(row=0, column=0, padx=5, pady=5, sticky="e")
            Hovertip(self.ms1_lab, "MS1 ppm error tolerance ")
            self.ms1_ppm_entry = ttk.Entry(self.ms_frame, width=4)
            self.ms1_ppm_entry.insert(0, "0")
            self.ms1_ppm_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
            default_dict["ms1_ppm"]["tk_handle"] = self.ms1_ppm_entry

            # MS2 ppm
            self.ms2_ppm_lab = ttk.Label(self.ms_frame, text="MS2 ppm: ")
            self.ms2_ppm_lab.grid(row=1, column=0, padx=5, pady=5, sticky="e")
            Hovertip(self.ms2_ppm_lab, "MS2 ppm error tolerance ")
            self.ms2_ppm_entry = ttk.Entry(self.ms_frame, width=4)
            self.ms2_ppm_entry.insert(0, "10")
            self.ms2_ppm_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
            default_dict["ppm"]["tk_handle"] = self.ms2_ppm_entry

            # Min Fragments (combobox)
            self.min_frag_lab = ttk.Label(self.ms_frame, text="Min Fragments:")
            self.min_frag_lab.grid(row=0, column=2, padx=20, pady=5, sticky="e")
            Hovertip(self.min_frag_lab, "Required number of fragments matched from top N fragments (N=10) ")
            self.min_frag_var = tk.IntVar(value=3)
            default_dict["atleast_m"]["tk_handle"] = self.min_frag_var
            self.min_frag_dropdown = ttk.Combobox(self.ms_frame, textvariable=self.min_frag_var, values=list(range(1, 11)), width=3, state="readonly")
            self.min_frag_dropdown.grid(row=0, column=3, padx=5, pady=5, sticky="w")
            self.min_frag_dropdown.bind("<<ComboboxSelected>>", on_combobox_select)

            # Empirical RT (label + checkbox)
            self.empirical_rt_lab = ttk.Label(self.ms_frame, text="Force lib RT:")
            self.empirical_rt_lab.grid(row=1, column=2, padx=20, pady=5, sticky="e")
            Hovertip(self.empirical_rt_lab, "Force use of library retention time for alignment ")
            self.empirical_rt_var = tk.BooleanVar(value=False)
            default_dict["use_emp_rt"]["tk_handle"] = self.empirical_rt_var
            self.empirical_rt_cb = ttk.Checkbutton(self.ms_frame, variable=self.empirical_rt_var)
            self.empirical_rt_cb.grid(row=1, column=3, padx=5, pady=5, sticky="w")

            # RT Tolerance (label + checkbox + entry)
            self.rt_tolerance_lab = ttk.Label(self.ms_frame, text="RT Tolerance:")
            self.rt_tolerance_lab.grid(row=0, column=5, padx=20, pady=5, sticky="e")
            Hovertip(self.rt_tolerance_lab, "Force use of specified retention time tolerance for alignment ")
            self.rt_tolerance_var = tk.BooleanVar(value=False)
            default_dict["user_rt_tol"]["tk_handle"] = self.rt_tolerance_var
            self.rt_tolerance_cb = ttk.Checkbutton(self.ms_frame, variable=self.rt_tolerance_var)
            self.rt_tolerance_cb.grid(row=0, column=6, padx=5, pady=10, sticky="w")
            self.rt_tolerance_entry = ttk.Entry(self.ms_frame, width=7)
            self.rt_tolerance_entry.grid(row=0, column=7, padx=5, pady=5, sticky="w")
            default_dict["rt_tol"]["tk_handle"] = self.rt_tolerance_entry

            # Isotopes (label + checkbox + combobox)
            self.isotopes_lab = ttk.Label(self.ms_frame, text="MS2 Isotopes:")
            self.isotopes_lab.grid(row=1, column=5, padx=20, pady=5, sticky="e")
            Hovertip(self.isotopes_lab, "Select if and the number of MS2 isotopes used in search ")
            self.isotopes_var = tk.BooleanVar(value=False)
            default_dict["iso"]["tk_handle"] = self.isotopes_var
            self.isotopes_cb = ttk.Checkbutton(self.ms_frame, variable=self.isotopes_var)
            self.isotopes_cb.grid(row=1, column=6, padx=5, pady=5, sticky="w")
            self.isotopes_count_var = tk.IntVar(value=2)
            default_dict["num_iso"]["tk_handle"] = self.isotopes_count_var
            self.isotopes_dropdown = ttk.Combobox(self.ms_frame, textvariable=self.isotopes_count_var, values=list(range(2, 11)), width=3, state="readonly")
            self.isotopes_dropdown.grid(row=1, column=7, padx=5, pady=5, sticky="w")
            self.isotopes_dropdown.bind("<<ComboboxSelected>>", on_combobox_select)

            ####         Multiplex Frame      #######
            self.multiplex_frame = ttk.LabelFrame(self, text="Multiplexing")
            self.multiplex_frame.grid(row=2, column=0, columnspan=10, padx=10, pady=10, sticky="ew")

            def combined_handler(event):
                on_combobox_select(event)
                self.update_masstag()
            
            # Tag label + dropdown
            self.tag_list = self.get_tag_list()
            self.tag_label = ttk.Label(self.multiplex_frame, text="Tag:")
            self.tag_label.grid(row=0, column=0, padx=10, pady=10, sticky="e")
            Hovertip(self.tag_label, "Select tag used for multiplexing ")
            self.tag_var = tk.StringVar(value='None')
            default_dict["tag"]["tk_handle"] = self.tag_var
            self.tag_dropdown = ttk.Combobox(self.multiplex_frame, textvariable=self.tag_var, values=self.tag_list, width=15, state="readonly")
            self.tag_dropdown.grid(row=0, column=1, padx=10, pady=10, sticky="w")
            self.tag_dropdown.bind("<<ComboboxSelected>>", combined_handler)
            self.channel_labels = []
            self.mass_labels = []
            self.tag_checkboxes = []
            self.tag_checkbox_list = []

            # Create New Tag button
            self.create_tag_button = ttk.Button(self.multiplex_frame, text="Create New Tag", style="Accent.TButton", command=self.create_new_tag, width=15)
            self.create_tag_button.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
            Hovertip(self.create_tag_button, "Create a custom tag for use in search ")


            # Timeplex label + dropdown
            self.timeplex_label = ttk.Label(self.multiplex_frame, text="Timeplex:")
            self.timeplex_label.grid(row=2, column=0, padx=10, pady=10, sticky="e")
            Hovertip(self.timeplex_label, "Select the number of timeplexes (1 for no timeplexing) ")
            self.timeplex_count_var = tk.IntVar(value=0)
            default_dict["num_timeplex"]["tk_handle"] = self.timeplex_count_var
            self.timeplex_dropdown = ttk.Combobox(self.multiplex_frame, textvariable=self.timeplex_count_var, values=[0, 2, 3, 4, 5], width=15)
            self.timeplex_dropdown.grid(row=2, column=1, padx=10, pady=10, sticky="w")
            self.timeplex_dropdown.bind("<<ComboboxSelected>>", on_combobox_select)

            self.tagInfo_frame = ttk.LabelFrame(self.multiplex_frame, text="Mass Tag Info")
            self.tagInfo_frame.grid(row=0, column=2, columnspan=10, rowspan=6, padx=10, pady=10, sticky="ew")


            self.update_idletasks()
            self.tagInfo_canvas_width_coE = 0.5
            self.tagInfo_canvas = tk.Canvas(self.tagInfo_frame, width=self.ms_frame.winfo_width()*self.tagInfo_canvas_width_coE, height=self.ms_frame.winfo_height()*1.64)
            self.tagInfo_canvas.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

            self.inner_frame = tk.Frame(self.tagInfo_canvas)
            self.tagInfo_canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")

            def on_frame_configure(event):
                self.tagInfo_canvas.configure(scrollregion=self.tagInfo_canvas.bbox("all"))

            self.inner_frame.bind("<Configure>", on_frame_configure)

            self.inner_frame.grid_columnconfigure(0, weight=1)
            self.inner_frame.grid_columnconfigure(1, weight=1)

            self.massTag_font = ("Arial", 12)

            self.channel_Label = tk.Label(self.inner_frame, text="   Channel   ", borderwidth=1, relief="solid", font=self.massTag_font)
            self.channel_Label.grid(row=1, column=0, sticky='nsew')
            self.mass_Label = tk.Label(self.inner_frame, text="  Mass Shift  ", borderwidth=1, relief="solid", font=self.massTag_font)
            self.mass_Label.grid(row=1, column=1, sticky='nsew')
            self.rules_Label = tk.Label(self.inner_frame, text="Rules: " + "N/A")
            self.rules_Label.grid(row=0, column=0, sticky='nsew')
            self.base_mass_Label = tk.Label(self.inner_frame, text='  Base Mass: ' + "N/A")
            self.base_mass_Label.grid(row=0, column=1, sticky='nsew')
            channel_L = tk.Label(self.inner_frame, text=f"\u0394 N/A", borderwidth=1, relief="solid", font=self.massTag_font)
            channel_L.grid(row=2, column=0, sticky='nsew')
            mass_L = tk.Label(self.inner_frame, text="N/A", borderwidth=1, relief="solid", font=self.massTag_font)
            mass_L.grid(row=2, column=1, sticky='nsew')
            self.channel_labels.append(channel_L)
            self.mass_labels.append(mass_L)


            ####         Additional Frame      #######
            self.additional_frame = ttk.LabelFrame(self, text="Additional Commands")
            self.additional_frame.grid(row=3, column=0, columnspan=10, padx=10, pady=10, sticky="ew")

            # Large multi-line text box
            self.additional_text = tk.Text(self.additional_frame, wrap="word", height=5, width=50, font=("Courier New", 11))
            self.additional_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

            # Make the frame resize nicely with window
            self.additional_frame.columnconfigure(0, weight=1)
            self.additional_frame.rowconfigure(0, weight=1)



            ####         Output Frame      #######
            self.output_frame = ttk.LabelFrame(self, text="Output")
            self.output_frame.grid(row=4, column=0, columnspan=10, padx=10, pady=10, sticky="ew")

            # Output Folder Label + Entry + Select Button
            self.output_folder_label = ttk.Label(self.output_frame, text="Output Folder:")
            self.output_folder_label.grid(row=0, column=0, padx=10, pady=10, sticky="e")
            Hovertip(self.output_folder_label, "Select the folder where output folders will be saved ")

            self.output_folder_var = tk.StringVar()
            self.output_folder_entry = ttk.Entry(self.output_frame, textvariable=self.output_folder_var, width=50)
            self.output_folder_entry.grid(row=0, column=1, columnspan=5, padx=10, pady=10, sticky="ew")
            default_dict["output_folder"]["tk_handle"] = self.output_folder_entry

            self.select_output_button = ttk.Button(self.output_frame, text="Select", style="Accent.TButton", command=self.select_output_folder)
            self.select_output_button.grid(row=0, column=6, padx=10, pady=10)

            # Save Configuration Label + Save Button
            self.save_config_button = ttk.Button(self.output_frame, text="Save Configuration", style="Accent.TButton", width=20, command=self.save_configuration)
            self.save_config_button.grid(row=1, column=1, padx=20, pady=10, sticky="e")
            Hovertip(self.save_config_button, "Save a configuration JSON file for the current settings ")

            # Run button
            self.run_button = ttk.Button(self.output_frame, text="Run JMod", style="Accent.TButton", width=20, command=self.run_process)
            self.run_button.grid(row=1, column=2, padx=10, pady=10)


        ####         Logging Funcs      #######

        
        #######     input funcs      #######

        def select_mzml(self):
            files = filedialog.askopenfilenames(title="Select .mzML or .d Files", filetypes=[("Mass Spec Files", "*.mzML *.d")])
            if files:
                for file in files:
                    if file not in self.file_dropdown.files:
                        self.file_dropdown.add_files([file])

        def select_tsv(self):
            file_path = filedialog.askopenfilename(
                title="Select .tsv File", filetypes=[("TSV files", "*.tsv")]
            )
            if file_path:
                self.tsv_entry.delete(0, tk.END)
                self.tsv_entry.insert(0, file_path)

        def select_json(self):
            file_path = filedialog.askopenfilename(
                title="Select JSON Config File", filetypes=[("JSON files", "*.json")]
            )
            if file_path:
                self.json_entry.delete(0, tk.END)
                self.json_entry.insert(0, file_path)
                self.import_json(file_path)
            else:
                tk.messagebox.showerror("No File Selected", "No JSON file was selected")


        def import_json(self, file_path): 
            """Load configuration parameters from a JSON file."""
            
            try:
                args_list = []
                with open(file_path, 'r') as f:
                    config_data = json.load(f)
                
                # Update args with values from JSON
                for key, value in config_data.items():
                    if key in default_dict.keys():
                        if key in ["mzml", "i", "plexDIA", "timeplex", "diaPASEF"]: ##does not allow file input from JSON. plexDIA, timeplex, and diaPASEF are not explicit lines in the GUI because we are inferring them from other objects
                            continue
                        if default_dict[key]['in_GUI'] is True:  ##if key is part of GUI
                            if default_dict[key]['widget'] == 'checkbutton':
                                try:
                                    default_dict[key]['tk_handle'].set(value)
                                except Exception as e:
                                    tk.messagebox.showerror("JSON Read Error", f"Error reading '{key}':\n\n {e}\n\n Ensure '{value}' can be converted to boolean")
                            elif default_dict[key]['widget'] == 'dropdown':
                                if default_dict[key]['type'] == 'int':
                                    try:
                                        default_dict[key]['tk_handle'].set(int(value))
                                    except Exception as e:
                                        tk.messagebox.showerror("JSON Read Error", f"Error reading '{key}':\n\n {e}\n\nEnsure '{value}' can be converted to integer")
                                elif default_dict[key]['type'] == 'str':
                                    try:
                                        default_dict[key]['tk_handle'].set(str(value))
                                    except Exception as e:
                                        tk.messagebox.showerror("JSON Read Error", f"Error reading '{key}':\n\n {e}\n\nEnsure '{value}' can be converted to string")
                                elif default_dict[key]['type'] == 'float':
                                    try:
                                        default_dict[key]['tk_handle'].set(float(value))
                                    except Exception as e:
                                        tk.messagebox.showerror("JSON Read Error", f"Error reading '{key}':\n\n {e}\n\nEnsure '{value}' can be converted to float")
                            elif default_dict[key]['widget'] == 'entry':
                                try:
                                    default_dict[key]['tk_handle'].delete(0, tk.END)
                                    default_dict[key]['tk_handle'].insert(0, str(value))
                                except Exception as e:
                                    tk.messagebox.showerror("JSON Read Error", f"Error reading '{key}':\n\n {e}\n\nEnsure '{value}' can be converted to string")
                        elif default_dict[key]['in_GUI'] is False:  ##if key is only a command line key
                            if value != default_dict[key]['default']:  ##only add to command line interface if different from default
                                if default_dict[key]['takes_value'] is False:
                                    args_list.append(default_dict[key]['flags'][0])
                                else:
                                    args_list.append(f'{default_dict[key]["flags"][0]} {value}')
                    else:
                        tk.messagebox.showwarning("JSON Read Warning", f"Unknown configuration key in JSON: {key}")

                self.additional_text.delete("1.0", tk.END)
                for arg in args_list:
                    self.additional_text.insert(tk.END, f"{arg} ")

            except json.JSONDecodeError as e:
                tk.messagebox.showerror("Failed to Decode JSON File", f"Failed to decode JSON file:\n\nCheck JSON file for formatting errors\n\n {e}")
            except Exception as e:
                tk.messagebox.showerror("JSON Read Error", f"Unknown Error loading JSON configuration: \n\n {e}")
            self.update_masstag()


        ####         MS Funcs      #######



        ####         Multiplex Funcs      #######

        def get_tag_list(self):
            from src.mass_tags import available_tags
            tag_list = ['None'] + [tag for tag in available_tags.keys() if "#" not in tag]  ##filter out subsetted tags from dropdown might be cleaner?
            #tag_list = ['None'] + list(available_tags.keys())
            default_dict["tag"]["values"] = tag_list
            return tag_list

        def create_new_tag(self): 
            if hasattr(self, "new_tag_window") and self.new_tag_window.winfo_exists():
                self.new_tag_window.destroy()
            self.new_tag_window = tk.Toplevel(self)
            self.new_tag_window.title("Create New Mass Tag")
            self.new_tag_window.geometry("260x220")

            #Num Channels
            tk.Label(self.new_tag_window, text="Number of Channels:").grid(row=0, column=0, padx=5, pady=5)
            self.num_channels_var = tk.StringVar()
            self.num_channels_entry = tk.Entry(self.new_tag_window, textvariable=self.num_channels_var, width=5)
            self.num_channels_entry.grid(row=0, column=1, padx=5, pady=5)

            #Go Button
            self.go_button = ttk.Button(self.new_tag_window, text="Go", style="Accent.TButton", command=self.make_second_tag_window)
            self.go_button.grid(row=0, column=2, padx=5, pady=5)

            #Atomic Composition Label
            tk.Label(self.new_tag_window, text="Base Atomic Composition:").grid(row=1, column=0, padx=5, pady=5)

            #C
            tk.Label(self.new_tag_window, text="C:").grid(row=2, column=0, padx=5, pady=5)
            self.atomic_composition_c = tk.StringVar(value="")
            self.atomic_composition_c_entry = tk.Entry(self.new_tag_window, textvariable=self.atomic_composition_c, width=5)
            self.atomic_composition_c_entry.grid(row=2, column=1, padx=5, pady=5)

            #H
            tk.Label(self.new_tag_window, text="H:").grid(row=3, column=0, padx=5, pady=5)
            self.atomic_composition_h = tk.StringVar(value="")
            self.atomic_composition_h_entry = tk.Entry(self.new_tag_window, textvariable=self.atomic_composition_h, width=5)
            self.atomic_composition_h_entry.grid(row=3, column=1, padx=5, pady=5)

            #N
            tk.Label(self.new_tag_window, text="N:").grid(row=4, column=0, padx=5, pady=5)
            self.atomic_composition_n = tk.StringVar(value="")
            self.atomic_composition_n_entry = tk.Entry(self.new_tag_window, textvariable=self.atomic_composition_n, width=5)
            self.atomic_composition_n_entry.grid(row=4, column=1, padx=5, pady=5)

            #O
            tk.Label(self.new_tag_window, text="O:").grid(row=5, column=0, padx=5, pady=5)
            self.atomic_composition_o = tk.StringVar(value="")
            self.atomic_composition_o_entry = tk.Entry(self.new_tag_window, textvariable=self.atomic_composition_o, width=5)
            self.atomic_composition_o_entry.grid(row=5, column=1, padx=5, pady=5)

        def make_second_tag_window(self):
            if hasattr(self, "second_window") and self.second_window.winfo_exists():
                self.second_window.destroy()
            try:
                self.n_channels = int(self.num_channels_var.get())
            except Exception as e:
                tk.messagebox.showerror("Channel Error", f"Channel Error: Please enter a valid integer number of channels.\n\n{e}")
                self.new_tag_window.focus()
                return
            try:
                totals = {
                    "C": int(self.atomic_composition_c.get()),
                    "H": int(self.atomic_composition_h.get()),
                    "N": int(self.atomic_composition_n.get()),
                    "O": int(self.atomic_composition_o.get()),
                }
            except Exception as e:
                tk.messagebox.showerror("Atomic Composition Error", f"Atomic Composition Error: Please enter valid numbers for atomic composition.\n\n{e}")
                self.new_tag_window.focus()
                return

            self.new_tag_window.destroy()

            self.second_window = tk.Toplevel()
            self.second_window.title("Define Channels")
            self.second_window.geometry("600x400")


            self.iso_pairs = {
                "C": "C[13]",
                "H": "H[2]",
                "N": "N[15]",
                "O": "O[18]"
            }

            self.rev_iso_pairs = {v: k for k, v in self.iso_pairs.items()}

            def on_entry_event(varname, index, mode):  ##when isotope entrybox is updated to x, set the other equal to total - x
                for tab, entry_vars in self.frame_dict_entryvars.items():
                    for name, sv in entry_vars.items():
                        if str(sv) == varname:
                            if sv.get() == "":
                                return

                            if name in self.iso_pairs.keys():  #atom
                                isotope_name = self.iso_pairs[name]
                                if int(entry_vars[isotope_name].get()) != totals[name]-int(sv.get()):
                                    entry_vars[isotope_name].set(totals[name]-int(sv.get()))
                            elif name in self.rev_iso_pairs.keys(): #isotope
                                atom_name = self.rev_iso_pairs[name]
                                if int(entry_vars[atom_name].get()) != totals[atom_name]-int(sv.get()):
                                    entry_vars[atom_name].set(totals[atom_name]-int(sv.get()))

            def validate_number(text):  #make sure isotope entryboxes can only take numbers
                return text.isdigit() or text == ""
            vcmd = (self.register(validate_number), "%P")

            def on_focus_out(entry_var): #when moving away from an isotope entrybox, if it has nothing put a 0 in it, this will force on_entry_event and make sure totals add up
                if entry_var.get() == "":
                    entry_var.set("0")
                if entry_var.get().strip() == "-":
                    entry_var.set("0")
                try:
                    int(entry_var.get())
                except:
                    entry_var.set("0")

            notebook = ListboxNotebook(self.second_window) #custom widget
            notebook.pack(fill="both", expand=True)

            self.frame_dict_entryvars = {}
            self.frame_dict_channels = {}
            for i in range(1, self.n_channels+1):
                frame = tk.Frame(notebook.content_frame, bg="white")
                notebook.add(f"Channel {i}", frame) 

                tk.Label(frame, text=f"Channel Name:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
                channel_entry = tk.Entry(frame)
                channel_entry.grid(row=0, column=1, padx=5, pady=5)
                self.frame_dict_channels[i] = channel_entry

                if i == 1:
                    tk.Label(frame, text="This channel will be used to calculate base mass").grid(row=len(self.iso_pairs)+1, column=0, columnspan=5, padx=5, pady=5, sticky="w")

                entry_vars = {}
                for j, (atom, isotope) in enumerate(self.iso_pairs.items(), start=1):
                    total = totals[atom]
                    entry_vars[atom] = tk.StringVar(value=str(total))
                    tk.Label(frame, text=f"{atom}:").grid(row=j, column=0, padx=5, pady=5, sticky="e")
                    atom_entry = tk.Entry(frame, textvariable=entry_vars[atom], validate="key", validatecommand=vcmd)
                    atom_entry.grid(row=j, column=1, padx=5, pady=5)
                    atom_entry.bind("<FocusOut>", lambda e, sv=entry_vars[atom]: on_focus_out(sv))
                    entry_vars[atom].trace_add("write", on_entry_event)
                    entry_vars[isotope] = tk.StringVar(value="0")
                    tk.Label(frame, text=f"{isotope}:").grid(row=j, column=2, padx=5, pady=5, sticky="e")
                    isotope_entry = tk.Entry(frame, textvariable=entry_vars[isotope], validate="key", validatecommand=vcmd)
                    isotope_entry.grid(row=j, column=3, padx=5, pady=5)
                    isotope_entry.bind("<FocusOut>", lambda e, sv=entry_vars[isotope]: on_focus_out(sv))
                    entry_vars[isotope].trace_add("write", on_entry_event)
                self.frame_dict_entryvars[i] = entry_vars

            self.tag_saveas_frame = tk.Frame(notebook.content_frame, bg="white")
            notebook.add(f"Save As", self.tag_saveas_frame) 

            tk.Label(self.tag_saveas_frame, text="Mass Tag Name:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
            self.tag_saveas_entry = tk.Entry(self.tag_saveas_frame)
            self.tag_saveas_entry.grid(row=0, column=1, padx=5, pady=5)
            tk.Label(self.tag_saveas_frame, text="Rules:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
            self.tag_rules_dropdown = ttk.Combobox(self.tag_saveas_frame, values=["nK", "R"], state="readonly", width=10)
            self.tag_rules_dropdown.grid(row=1, column=1, padx=5, pady=5)
            self.tag_rules_dropdown.set("nK")
            massTag_save_button = ttk.Button(self.tag_saveas_frame, text="Save Mass Tag", command=self.save_masstag, style="Accent.TButton")
            massTag_save_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

        def save_masstag(self):
            from src.mass_tags import refresh_tags
            base_composition = {}
            for element in list(self.iso_pairs.keys()) + list(self.rev_iso_pairs.keys()):
                base_composition[element] = int(self.frame_dict_entryvars[1][element].get())
            base_mass = mass.calculate_mass(base_composition)


            compositions = {}
            for i in range(1, self.n_channels+1):
                channel_name = self.frame_dict_channels[i].get()
                if channel_name == "":
                    tk.messagebox.showerror("Channel Name Error", f"Channel Name Error: In tab {i} the channel is unnamed.")
                    self.second_window.focus()
                    return
                if channel_name in compositions.keys():
                    tk.messagebox.showerror("Duplicate Channel Error", f"Duplicate Channel Error: Tab {i} has the same channel name as another channel")
                    self.second_window.focus()
                    return
                compositions[channel_name] = {}
                for element in list(self.iso_pairs.keys()) + list(self.rev_iso_pairs.keys()):
                    compositions[channel_name][element] = int(self.frame_dict_entryvars[i][element].get())

            name = self.tag_saveas_entry.get().strip()
            name = re.sub(r'[<>:"/\\|?*\n\r\t]', '_', name)

            if name == "" or name == "None" or name is None:
                tk.messagebox.showerror("Tag Name Error", f"Tag Name Error. Tag must have a name, and the name cannot be 'None'")
                self.second_window.focus()
                return
            
            delta = [mass.calculate_mass(compositions[channel_name]) - base_mass for channel_name in compositions]
            delta_zero_count = delta.count(0)
            if delta_zero_count > 1:
                tk.messagebox.showerror("Mass Shift Error", f"Mass Shift Error. {delta_zero_count} channels have a mass shift of 0. This must be 1 or less. Check compositions and try again.")
                self.second_window.focus()
                return
            
            if len(delta) != len(set(delta)):
                continue_on = tk.messagebox.askyesno("Mass Shift Warning", f"Some channels have identical mass shifts. Continue anyways?", default="no")
                if not continue_on:
                    self.second_window.focus()
                    return

            mass_tag_dict = {
                "rules": self.tag_rules_dropdown.get(),
                "base_mass": base_mass,
                "delta": delta,
                "channel_names": list(compositions.keys()),
                "name": name,
                "compositions": compositions
            }

            mass_tags_dir = Path(__file__).parent / "MassTags"
            filename = mass_tags_dir / (name + ".json")
            if os.path.exists(filename):
                overwrite_file = tk.messagebox.askyesno("Tag already exists", f"{name} already exists. Are you sure you want to overwrite it?", default='no')
                if not overwrite_file:
                    self.second_window.focus()
                    return
            with open(filename, 'w') as f:
                json.dump(mass_tag_dict, f, indent=4)
            tk.messagebox.showinfo("Tag Saved", f"{name} has been saved")
            default_dict['tag']['tk_handle'].set(name)
            self.update_masstag()
            refresh_tags()
            self.tag_list = self.get_tag_list()
            self.tag_dropdown['values'] = self.tag_list


        def update_masstag(self):
            filename = default_dict['tag']['tk_handle'].get()
            if filename == "None" or filename == "":
                self.update_mass_tag_viewer([], [], None, None, None)
                return
            if not filename.endswith(".json"):
                filename += ".json"
            mass_tags_dir = Path(__file__).parent / "MassTags"
            mass_tag_files = set(os.listdir(mass_tags_dir))
            if filename in mass_tag_files or filename.split("#")[0] + ".json" in mass_tag_files:
                mass_tag_subset = False
                if filename not in mass_tag_files:
                    mass_tag_subset = True
                    subset = os.path.splitext((filename.split("#")[1]))[0]
                    filename = filename.split("#")[0] + ".json"
                    #default_dict['tag']['tk_handle'].set(os.path.splitext(filename)[0])
                    if filename not in mass_tag_files:
                        tk.messagebox.showerror("Mass tag Error", f"Mass Tag {filename}/{filename}#{subset}.json not in mass tag dir")
                        default_dict['tag']['tk_handle'].set("None")
                        self.update_mass_tag_viewer([], [], None, None, None)
                        return
                
                mass_tag_json = (os.path.join(mass_tags_dir,filename))
                if mass_tag_json:
                    with open(mass_tag_json, 'r') as f:
                        mass_tag_data = json.load(f)
                    if mass_tag_data['rules'] not in ["nK", "R"]:
                        tk.messagebox.showerror("Mass Tag Error", f"Invalid mass tag rules in file '{filename}'.\n\n {mass_tag_data['rules']} not in {['nK', 'R']}")
                        default_dict['tag']['tk_handle'].set("None")
                        self.update_mass_tag_viewer([], [], None, None, None)
                        return
                    try:
                        float(mass_tag_data["base_mass"])
                    except Exception as e:
                        tk.messagebox.showerror("Mass Tag Error", f"Invalid base mass in file '{filename}'.\n\n {mass_tag_data['base_mass']} could not be converted to float")
                        default_dict['tag']['tk_handle'].set("None")
                        self.update_mass_tag_viewer([], [], None, None, None)
                        return
                    try:
                        for delta in mass_tag_data["delta"]:
                            float(delta)
                    except Exception as e:
                        tk.messagebox.showerror("Mass Tag Error", f"Invalid mass shift in file '{filename}'.\n\n {delta} could not be converted to float")
                        default_dict['tag']['tk_handle'].set("None")
                        self.update_mass_tag_viewer([], [], None, None, None)
                        return
                    if len(mass_tag_data["channel_names"]) != len(mass_tag_data["delta"]):
                        tk.messagebox.showerror("Mass Tag Error", f"Channel names and delta values length mismatch in file '{filename}'.\n\n {len(mass_tag_data['channel_names'])} != {len(mass_tag_data['delta'])}")
                        default_dict['tag']['tk_handle'].set("None")
                        self.update_mass_tag_viewer([], [], None, None, None)
                        return
                    if mass_tag_subset is False:
                        self.update_mass_tag_viewer(mass_tag_data["channel_names"], mass_tag_data["delta"], mass_tag_data["base_mass"], mass_tag_data["rules"], None)
                    else:
                        default_dict['tag']['tk_handle'].set(mass_tag_data['name'])
                        self.update_mass_tag_viewer(mass_tag_data["channel_names"], mass_tag_data["delta"], mass_tag_data["base_mass"], mass_tag_data["rules"], subset)
                else:
                    tk.messagebox.showerror("Mass Tag Error", f"Mass tag file '{filename}' is empty or not found.")
                    default_dict['tag']['tk_handle'].set("None")
                    self.update_mass_tag_viewer([], [], None, None, None)
            else:
                tk.messagebox.showerror("Mass Tag Error", f"Mass tag file '{filename}' not found in MassTags directory.")
                default_dict['tag']['tk_handle'].set("None")
                self.update_mass_tag_viewer([], [], None, None, None)


        def update_mass_tag_viewer(self, channels, deltas, base_mass, rules, subset):
            for label in self.channel_labels:
                label.destroy()
            for label in self.mass_labels:
                label.destroy()
            for cbutton in self.tag_checkboxes:
                cbutton.destroy()
            self.tag_checkbox_list.clear()
            self.channel_labels.clear()
            self.mass_labels.clear()
            if hasattr(self, "tagInfo_canvas"):
                self.tagInfo_canvas.destroy()
            if hasattr(self, 'inner_frame'):
                self.inner_frame.destroy()
            if hasattr(self, 'massTag_scrollbar'):
                self.massTag_scrollbar.destroy()
            if hasattr(self, 'rules_Label'):
                self.rules_Label.destroy()
            if hasattr(self, 'base_mass_Label'):
                self.base_mass_Label.destroy()

            self.update_idletasks()
            self.tagInfo_canvas = tk.Canvas(self.tagInfo_frame, width=self.ms_frame.winfo_width()*self.tagInfo_canvas_width_coE, height=self.ms_frame.winfo_height()*1.64)
            self.tagInfo_canvas.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

            self.inner_frame = tk.Frame(self.tagInfo_canvas)
            self.tagInfo_canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")

            def on_frame_configure(event):
                self.tagInfo_canvas.configure(scrollregion=self.tagInfo_canvas.bbox("all"))

            self.inner_frame.bind("<Configure>", on_frame_configure)
            self.inner_frame.grid_columnconfigure(0, weight=1)
            self.inner_frame.grid_columnconfigure(1, weight=1)

            self.channel_Label = tk.Label(self.inner_frame, text="   Channel   ", borderwidth=1, relief="solid", font=self.massTag_font)
            self.channel_Label.grid(row=1, column=0, sticky='nsew')
            self.mass_Label = tk.Label(self.inner_frame, text="  Mass Shift  ", borderwidth=1, relief="solid", font=self.massTag_font)
            self.mass_Label.grid(row=1, column=1, sticky='nsew')

            if rules:
                self.rules_Label = tk.Label(self.inner_frame, text="Rules: " + rules)
                self.rules_Label.grid(row=0, column=0, sticky='nsew')
                if rules != "":
                    if rules == "nK":
                        rule_tip = "Tag on N-term and Lys"
                    elif rules == "R":
                        rule_tip = "Tag on Arg"
                    else:
                        rule_tip = "Unknown Tagging Rule"
                Hovertip(self.rules_Label, rule_tip)
            if base_mass:
                self.base_mass_Label = tk.Label(self.inner_frame, text='  Base Mass: ' + format(round(base_mass, 8), ".8f") if base_mass else '')
                self.base_mass_Label.grid(row=0, column=1, sticky='nsew')
                Hovertip(self.base_mass_Label, base_mass)

            if default_dict['tag']['tk_handle'].get() == "None":
                self.rules_Label = tk.Label(self.inner_frame, text="Rules: " + "N/A")
                self.rules_Label.grid(row=0, column=0, sticky='nsew')
                self.base_mass_Label = tk.Label(self.inner_frame, text='  Base Mass: ' + "N/A")
                self.base_mass_Label.grid(row=0, column=1, sticky='nsew')
                channel_L = tk.Label(self.inner_frame, text=f"\u0394 N/A", borderwidth=1, relief="solid", font=self.massTag_font)
                channel_L.grid(row=2, column=0, sticky='nsew')
                mass_L = tk.Label(self.inner_frame, text="N/A", borderwidth=1, relief="solid", font=self.massTag_font)
                mass_L.grid(row=2, column=1, sticky='nsew')
                self.channel_labels.append(channel_L)
                self.mass_labels.append(mass_L)
            

            if len(channels) > 5:
                self.massTag_scrollbar = tk.Scrollbar(self.tagInfo_frame, orient="vertical", command=self.tagInfo_canvas.yview)
                self.massTag_scrollbar.grid(row=0, column=2, sticky="ns")
                self.tagInfo_canvas.configure(yscrollcommand=self.massTag_scrollbar.set)

            for i, (channel, delta) in enumerate(zip(channels, deltas), start=2):
                channel_L = tk.Label(self.inner_frame, text=f"\u0394{channel}", borderwidth=1, relief="solid", font=self.massTag_font)
                channel_L.grid(row=i, column=0, sticky='nsew')
                mass_L = tk.Label(self.inner_frame, text=format(round(float(delta), 8), ".8f"), borderwidth=1, relief="solid", font=self.massTag_font)
                mass_L.grid(row=i, column=1, sticky='nsew')
                checkbutton_var = tk.BooleanVar(value=True)
                channel_checkbox = ttk.Checkbutton(self.inner_frame, variable=checkbutton_var, padding=(5, 0,), takefocus=0)
                channel_checkbox.grid(row=i, column=2, sticky='nsew')
                if subset:
                    channel_subsets = [x for x in subset.split("d") if x]
                    if str(channel) not in channel_subsets:
                        checkbutton_var.set(False)
                self.channel_labels.append(channel_L)
                self.mass_labels.append(mass_L)
                self.tag_checkboxes.append(channel_checkbox)
                self.tag_checkbox_list.append(checkbutton_var)
                Hovertip(mass_L, delta)
                
        
        def generate_tag_subset(self, base_tag_name):
            from src.mass_tags import available_tags, refresh_tags
            base_tag_instance = available_tags[base_tag_name]
            channels = base_tag_instance.channel_names
            using_channels_idxs = []
            channel_string = ''
            for i, channel in enumerate(channels):
                if self.tag_checkbox_list[i].get():
                    using_channels_idxs.append(i)
                    channel_string += f"d{channel}"

            new_tag_name = f"{base_tag_name}#{channel_string}"

            base_mass_tag_JSON = os.path.join("src", "MassTags", base_tag_name + ".json")
            with open(base_mass_tag_JSON, 'r') as f:
                mass_tag_data = json.load(f)

            new_deltas = [delta for i, delta in enumerate(mass_tag_data['delta']) if i in using_channels_idxs]
            new_channels = [channel for i, channel in enumerate(mass_tag_data['channel_names']) if i in using_channels_idxs]
            new_composition_list = [comp for i, comp in enumerate(list(mass_tag_data['compositions'].values())) if i in using_channels_idxs]
            new_compositions = dict(zip(new_channels, new_composition_list))

            new_tag_dict = {
                'rules': mass_tag_data['rules'],
                'base_mass': mass_tag_data['base_mass'],
                'delta': new_deltas,
                'channel_names': new_channels,
                'name': new_tag_name,
                'compositions': new_compositions
            }

            file_path = os.path.join("src", "MassTags", "Subsets", new_tag_name + ".json")
            with open (file_path, 'w') as f:
                json.dump(new_tag_dict, f, indent=4)
            refresh_tags()

            return new_tag_name

        ####         Additional Funcs      #######


        def read_additional_funcs(self):
            try:
                user_input = self.additional_text.get("1.0", tk.END).strip()
                args_list = shlex.split(user_input)
                flag_to_key = {flag: key for key, data in default_dict.items() for flag in data['flags']}
                for arg in args_list:   ##catches error of unknown flags
                    if arg.startswith('-') and arg not in flag_to_key:
                        tk.messagebox.showerror("Additional Commands Error", f"Unknown flag: {arg}")
                        return False
                for i, arg in enumerate(args_list): #missing or incorrect value errors
                    if arg in [flag for _, data in default_dict.items() if data['takes_value'] for flag in data['flags']]: #catches error of missing value
                        if i + 1 >= len(args_list) or args_list[i+1].startswith('-'):
                            tk.messagebox.showerror("Additional Commands Error",f"Argument {arg} requires a value.")
                            return False
                        
                        val = args_list[i+1] #incorrect value errors
                        if default_dict[flag_to_key[arg]]['type'] == 'int':
                            try:
                                int(val)
                            except ValueError:
                                tk.messagebox.showerror("Error", f"Argument {arg} expects an integer, got '{val}'.")
                                return False
                        elif default_dict[flag_to_key[arg]]['type'] == 'float':
                            try:
                                float(val)
                            except ValueError:
                                tk.messagebox.showerror("Error", f"Argument {arg} expects a number, got '{val}'.")
                                return False
                            
                args, unknown = parser.parse_known_args(args_list)
                if unknown:  #catches errors like random strings in additional commands
                    tk.messagebox.showerror(
                        "Additional Commands Error", f"Failed to parse additional commands. Ensure they are formatted correctly.\n\nError: {' '.join(unknown)}"
                    )
                    return False
                args_dict = vars(args) #make into dictionary
                for key in args_dict.keys(): #tell user if they have specified something that is already set in the GUI
                    if default_dict[key]['in_GUI']:
                        if any(flag in args_list for flag in default_dict[key]['flags']):
                            tk.messagebox.showwarning("Additional Commands Warning", f"Argument {key}/{'/'.join(default_dict[key]['flags'])} is already set in the GUI and will be ignored.")
                for key in args_dict.keys():  ##No explicit setting of plexDIA, timeplex, diaPASEF
                    if key in ["plexDIA", "timeplex", "diaPASEF"]:
                        if any(flag in args_list for flag in default_dict[key]['flags']):
                            tk.messagebox.showwarning("Additional Commands Warning", f"Argument {key}/{'/'.join(default_dict[key]['flags'])} is not explicitly settable in GUI and will be ignored.")
                filtered_args = {k: v for k, v in args_dict.items() if default_dict[k]['in_GUI'] is False and default_dict[k]['default'] != v}
                return filtered_args
            except Exception as e:
                tk.messagebox.showerror("Unknown Additional Commands Error", f"Unknown Additional Commands Error.\n\nError: {e}")
                return False

        ####         Output Funcs      #######

        def save_configuration(self):
            if not self.mzml_and_lib_error_check():
                return
            config_args_dict = self.make_config_dict(None)
            if config_args_dict is None:
                return
            logger.info("Saving Configuration")
            filename = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")], title="Save Configuration As")
            if filename:
                try:
                    with open(filename, 'w') as json_file:
                        json.dump(config_args_dict, json_file)
                    tk.messagebox.showinfo("Configuration Saved", f"Configuration saved successfully to {filename}")
                except Exception as e:
                    tk.messagebox.showerror("JSON Error", f"Failed to write JSON file: {e}")

        def select_output_folder(self):
            folder_selected = filedialog.askdirectory()
            if folder_selected:
                self.output_folder_var.set(folder_selected)


        def run_process(self):
            if not self.mzml_and_lib_error_check():
                return
            
            #self.run_button.config(state="disabled")
            threading.Thread(target=self.run_main, daemon=True).start()

        def run_main(self):
            for handler in logger.handlers:
                if isinstance(handler.formatter, ElapsedFormatter):
                    handler.formatter.reset_start_time()

            mzml_files = self.file_dropdown.files
            for i, mzml_path in enumerate(mzml_files, start=1):
                config_args_dict = self.make_config_dict(mzml_path=mzml_path)
                if config_args_dict is None:
                    return 
                try:
                    with tempfile.NamedTemporaryFile(mode='w', suffix=".json", delete=False) as tmp_file:
                        json.dump(config_args_dict, tmp_file)
                        tmp_filename = tmp_file.name

                except Exception as e:
                    tk.messagebox.showerror("JSON Error", f"Failed to write JSON file: {e}")

                logger.info(f"\n\nRunning JMod: File {i} of {len(mzml_files)}\n")
                from src.run_jmod import main
                main(tmp_filename)
                os.remove(tmp_filename)
                logger.info(f"Finished File {i} of {len(mzml_files)}\n")
            logger.info("JMod Finished")
            #self.run_button.config(state="normal")

        def mzml_and_lib_error_check(self):
            mzml_files = self.file_dropdown.files
            tsv_path = self.tsv_entry.get().strip()

            if not mzml_files:
                messagebox.showerror("Missing mzML or .d Files", "Please select the mzML or .d files before running.")
                return False
            if not tsv_path:
                messagebox.showerror("Missing Library File", "Please select the Library file before running.")
                return False
            for mzml_path in mzml_files:
                if os.path.exists(mzml_path) is False:
                    tk.messagebox.showerror("Data File Not Found", f"Data File not found: {mzml_path}")
                    return False
                if os.path.splitext(mzml_path)[1].lower() not in [".mzml", ".d"]:
                    tk.messagebox.showerror("Data File Type Error", f"Data File type not supported: {mzml_path}")
                    return False
            if os.path.exists(tsv_path) is False:
                tk.messagebox.showerror("Speclib Not Found", f"Speclib not found: {tsv_path}")
                return False
            if os.path.splitext(tsv_path)[1].lower() != ".tsv":
                tk.messagebox.showerror("Speclib Type Error", f"Speclib filetype type ({os.path.splitext(tsv_path)[1].lower()}) not supported: {tsv_path}")
                return False
            if self.output_folder_var.get() == "":
                tk.messagebox.showerror("Output Folder Error", "Output Folder Error: Please Specify an Output Folder")
                return False
            if not os.path.isdir(self.output_folder_var.get()):
                tk.messagebox.showerror("Output Folder Error", "Output Folder Error: Please make sure output folder exists")
                return False
            return True
            
        def make_config_dict(self, mzml_path=None):
            handles_map = {k: v['tk_handle'] for k, v in default_dict.items()}
            filtered_args = self.read_additional_funcs()  ##get additional args from text box
            if filtered_args is False:
                return None
            config_args_dict = {}
            for key, data in default_dict.items():
                if key == 'mzml':  #hardcoded mzml
                    config_args_dict[key] = str(mzml_path)
                elif key == 'diaPASEF':  ##Hardcoded diaPASEF because it relies on mzml_path
                    config_args_dict[key] = True if os.path.splitext(mzml_path)[1].lower() == ".d" else False
                elif callable(data['special_upload']):  ##if special upload function is defined
                    try:
                        config_args_dict[key] = data['special_upload'](data['tk_handle'], handles_map)
                    except Exception as e:
                        tk.messagebox.showerror("Read Error", f"Failed to read data: {key}:\n\nError: {e}")
                        return None
                elif data['in_GUI'] is False: #Non GUI element from command line
                    if key in filtered_args.keys():
                        try:
                            config_args_dict[key] = filtered_args[key]
                        except Exception as e:
                            tk.messagebox.showerror("Read Error", f"Failed to read data from additional commands: {key}:\n\nError: {e}")
                            return None
                    else:
                        try:
                            config_args_dict[key] = data['default']
                        except Exception as e:
                            tk.messagebox.showerror("Read Error", f"Failed to read data from additional commands: {key}:\n\nError: {e}")
                            return None
                else:  ## In GUI special upload not required
                    if data['widget'] == 'checkbutton':
                        try:
                            config_args_dict[key] = data['tk_handle'].get()
                        except Exception as e:
                            tk.messagebox.showerror("Checkbutton Error", f"Failed to get value from {key} checkbutton:\n\n'{data['tk_handle'].get()}\n\nError: {e}")
                            return None
                    elif data['widget'] == 'dropdown':
                        if data['type'] == 'int':
                            try:
                                config_args_dict[key] = int(data['tk_handle'].get())
                            except Exception as e:
                                tk.messagebox.showerror("Dropdown Error", f"Failed to convert {key} dropdown value to integer:\n\n'{data['tk_handle'].get()}\n\nError: {e}")
                                return None
                        elif data['type'] == 'float':
                            try:
                                config_args_dict[key] = float(data['tk_handle'].get())
                            except Exception as e:
                                tk.messagebox.showerror("Dropdown Error", f"Failed to convert {key} dropdown value to float:\n\n'{data['tk_handle'].get()}\n\nError: {e}")
                                return None
                        elif data['type'] == 'str':
                            try:
                                config_args_dict[key] = str(data['tk_handle'].get())
                            except Exception as e:
                                tk.messagebox.showerror("Dropdown Error", f"Failed to convert {key} dropdown value to string:\n\n'{data['tk_handle'].get()}\n\nError: {e}")
                                return None
                    elif data['widget'] == 'entry':
                        if data['type'] == 'int':
                            try:
                                config_args_dict[key] = int(data['tk_handle'].get())
                            except Exception as e:
                                tk.messagebox.showerror("Entry Error", f"Failed to convert {key} entry value to integer:\n\n'{data['tk_handle'].get()}\n\nError: {e}")
                                return None
                        elif data['type'] == 'float':
                            try:
                                config_args_dict[key] = float(data['tk_handle'].get())
                            except Exception as e:
                                tk.messagebox.showerror("Entry Error", f"Failed to convert {key} entry value to float:\n\n'{data['tk_handle'].get()}\n\nError: {e}")
                                return None
                        elif data['type'] == 'str':
                            try:
                                config_args_dict[key] = str(data['tk_handle'].get())
                            except Exception as e:
                                tk.messagebox.showerror("Entry Error", f"Failed to convert {key} entry value to string:\n\n'{data['tk_handle'].get()}\n\nError: {e}")
                                return None

                if data['values'] == 'any':  ##while we know data are the right type, now make sure they are valid given specific parameters for each command
                    continue
                elif type(data['values']) == list:
                    if config_args_dict[key] not in data['values']:
                        tk.messagebox.showerror("Invalid Value", f"Value for {key} is not in the allowed list:\n\n{data['values']}\n\nValue: '{config_args_dict[key]}'")
                        return None
                elif data['values'].startswith('+<'):
                    if config_args_dict[key] > float(data['values'].lstrip('+<')):
                        tk.messagebox.showerror("Invalid Value", f"Value for {key} must be less than or equal to {data['values'].lstrip('+<')}\n\nValue: {config_args_dict[key]}")
                        return None
                elif data['values'] == '+':
                    if config_args_dict[key] < 0:
                        tk.messagebox.showerror("Invalid Value", f"Value for {key} must be positive: \n\nValue: {config_args_dict[key]}")
                        return None
            base_tag_name = config_args_dict['tag']
            if base_tag_name != "None":  
                if not all([x.get() for x in self.tag_checkbox_list]): ##if the user has specified only specific tag channels
                    new_tag_name = self.generate_tag_subset(base_tag_name)
                    config_args_dict['tag'] = new_tag_name
            return config_args_dict
        

            

    
    app = JModGUI()
    app.mainloop()


def send_raise_to_TK(error_text):
    import sys, importlib
    if "src.config" in sys.modules:    ##was some weird issue with from . import config here. This seems to fix it and keep the file id the same
        config = sys.modules["src.config"]
    else:
        config = importlib.import_module("src.config")

    logger.error(error_text)
    logger.error("JMod Exited\n")
    if config.ran_from_GUI is True:
        tk.messagebox.showerror("Error", error_text)

###### Custom Widgets

## file list dropdown for entering multiple and removing files (mzml)
class FileListDropdown(tk.Frame):
    def __init__(self, master, max_visible=5):
        super().__init__(master)
        self.files = []
        self.expanded = False
        self.max_visible = max_visible

        self.toggle_btn = ttk.Button(self, text="No files selected ▼", command=self.toggle_dropdown)
        self.toggle_btn.pack(fill="x")

        # Frame for scrollable area
        self.container_frame = ttk.Frame(self)
        self.canvas = tk.Canvas(self.container_frame, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.container_frame, orient="vertical", command=self.canvas.yview)
        self.scroll_frame = ttk.Frame(self.canvas)

        self.scroll_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

    def toggle_dropdown(self):
        if self.expanded:
            self.container_frame.pack_forget()
            self.expanded = False
        else:
            self.container_frame.pack(fill="x", pady=2)

            # Calculate height: min(number of files, max_visible) * 25 pixels per row
            visible_count = min(len(self.files), self.max_visible)
            height = visible_count * 25 if visible_count > 0 else 50
            self.canvas.config(height=height)

            self.canvas.pack(side="left", fill="x", expand=True)
            self.scrollbar.pack(side="right", fill="y")

            self.refresh_list()
            self.expanded = True
        self.update_button_label()


    def update_button_label(self):
        count = len(self.files)
        arrow = "▲" if self.expanded else "▼"
        self.toggle_btn.config(text=f"{count} file{'s' if count != 1 else ''} selected {arrow}")

    def refresh_list(self):
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

        for i, f in enumerate(self.files):
            row = ttk.Frame(self.scroll_frame)
            row.pack(fill="x", pady=1)

            # Truncate filename if too long
            max_chars = 55
            display_name = self.truncate_middle(f, max_chars)

            label = ttk.Label(row, text=display_name, anchor="w")
            label.pack(side="left", fill="x", expand=True)

            remove_btn = ttk.Button(row, text="X", width=2, style="Red.TButton", command=lambda idx=i: self.remove_file(idx))
            remove_btn.pack(side="right")

    def add_files(self, filepaths):
        self.files.extend(filepaths)
        self.update_button_label()
        if self.expanded:
            self.refresh_list()

    def remove_file(self, idx):
        del self.files[idx]
        self.update_button_label()
        if self.expanded:
            self.refresh_list()


    def truncate_middle(self, text, max_len):
        if len(text) <= max_len:
            return text
        half = (max_len - 3) // 2
        return text[:half] + "..." + text[-half:]


class ListboxNotebook(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        # Left: List of "tabs"
        self.listbox = tk.Listbox(self, height=20, exportselection=False)
        self.listbox.pack(side="left", fill="y")

        # Right: Content area
        self.content_frame = tk.Frame(self, bg="white")
        self.content_frame.pack(side="right", fill="both", expand=True)

        # Storage for frames
        self.frames = {}

        # When listbox selection changes
        self.listbox.bind("<<ListboxSelect>>", self.show_frame)

    def add(self, name, frame):
        """Add a new 'tab' (frame)"""
        index = self.listbox.size()
        self.listbox.insert(index, name)
        self.frames[name] = frame
        frame.place(in_=self.content_frame, x=0, y=0, relwidth=1, relheight=1)
        frame.lower()  # keep hidden initially

        if index == 0:  # first one gets shown
            self.listbox.selection_set(0)
            self.show_frame()

    def show_frame(self, event=None):
        """Show the selected frame"""
        selection = self.listbox.curselection()
        if selection:
            name = self.listbox.get(selection[0])
            frame = self.frames[name]
            frame.lift()
        
