import json
import csv
import os

#To Run:
# - Change paramaters in def main()

def main():
    for x in ["Faster_fit_mtraq"]:
        pyinstrument_json_to_paths(
            rf"C:\Users\zcohe\Jmod\JMod_Profiling\Output\Changed_merging\{x}_Merge\profile.json",     #path to profile json  
            rf"C:\Users\zcohe\Jmod\JMod_Profiling\Output\Changed_merging\{x}_Merge\profile_paths.csv", #output csv path
            15,  #maximum depth read from json
            f'{x}_Merging_Change'  #give it a name
        )



def walk_tree(node, path, results, filename, max_depth):
    """Recursive walker that records top-level call paths + time."""
    current_func = node.get("function", "")
    new_path = path + [current_func]
    time = node.get("time", 0.0)

    if len(new_path) > max_depth:
        return

    # Record this path and the time spent in the current node
    row = [filename] + new_path + [time]
    results.append(row)

    for child in node.get("children", []):
        walk_tree(child, new_path, results, filename, max_depth)


def pyinstrument_json_to_paths(json_file, csv_file, max_depth, filename):
    with open(json_file, "r") as f:
        data = json.load(f)

    results = []

    walk_tree(data["root_frame"], [], results, filename, max_depth)

    # Pad rows so all have same number of columns (filename + levels + time)
    max_cols = max_depth + 2
    for row in results:
        while len(row) < max_cols:
            row.insert(-1, "")

    headers = ["filename"] + [f"level{i}" for i in range(1, max_depth+1)] + ["time"]

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(results)


if __name__ == "__main__":
    main()
