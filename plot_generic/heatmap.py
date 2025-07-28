#!/usr/bin/env python3

# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License

import os, sys, argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import subprocess
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.rc('pdf', fonttype=42) # To avoid issues with camera-ready submission
rcParams['figure.figsize'] = 6.75,6.75
big_font_size = 18
small_font_size = 15
fmt=".2f"
sbrn_palette = sns.color_palette("deep")
sota_palette = [sbrn_palette[i] for i in range(len(sbrn_palette)) if sbrn_palette[i] != sns.xkcd_rgb['red']]


metrics = ["mean", "median", "percentile_90"]


def human_readable_size(num_bytes):
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB']:
        if num_bytes < 1024:
            return f"{int(num_bytes)} {unit}"
        num_bytes /= 1024
    return f"{int(num_bytes)} PiB"

def get_summaries(args):
    # Read metadata file
    metadata_file = f"results/" + args.system + "_metadata.csv"
    if not os.path.exists(metadata_file):
        print(f"Metadata file {metadata_file} not found. Exiting.", file=sys.stderr)
        sys.exit(1)
    metadata = pd.read_csv(metadata_file)
    nnodes = [n for n in str(args.nnodes).split(",")]
    summaries = {} # Contain the folder for each node count
    # Search all the entries we might need
    for nodes in nnodes:
        if "tasks_per_node" in metadata.columns:
            filtered_metadata = metadata[(metadata["collective_type"].str.lower() == args.collective.lower()) & \
                                        (metadata["nnodes"].astype(str) == str(nodes)) & \
                                        (metadata["tasks_per_node"].astype(int) == args.tasks_per_node)]
        else:
            filtered_metadata = metadata[(metadata["collective_type"].str.lower() == args.collective.lower()) & \
                                        (metadata["nnodes"].astype(str) == str(nodes))]            
        if args.notes:
            filtered_metadata = filtered_metadata[(filtered_metadata["notes"].str.strip() == args.notes.strip())]
        else:
            # Keep only those without notes
            filtered_metadata = filtered_metadata[filtered_metadata["notes"].isnull()]
            
        if filtered_metadata.empty:
            print(f"Metadata file {metadata_file} does not contain the requested data. Exiting.", file=sys.stderr)
            sys.exit(1)
    
        # Among the remaining ones, keep only tha last one
        filtered_metadata = filtered_metadata.iloc[-1]
        #summaries[nodes] = "results/" + args.system + "/" + filtered_metadata["timestamp"] + "/" + str(filtered_metadata["test_id"]) + "/aggregated_result_summary.csv"
        summaries[nodes] = "results/" + args.system + "/" + filtered_metadata["timestamp"] + "/"
    return summaries

def get_summaries_df(args):
    summaries = get_summaries(args)
    df = pd.DataFrame()
    # Loop over the summaries
    for nodes, summary in summaries.items():
        # Create the summary, by calling the summarize_data.py script
        # Check if the summary already exists
        if not os.path.exists(summary + "/aggregated_results_summary.csv") or True:        
            subprocess.run([
                "python3",
                "./plot/summarize_data.py",
                "--result-dir",
                summary
            ],
            stdout=subprocess.DEVNULL)

        s = pd.read_csv(summary + "/aggregated_results_summary.csv")        
        # Filter by collective type
        s = s[s["collective_type"].str.lower() == args.collective.lower()]      
        # Drop the rows where buffer_size is equal to 4 (we do not have them for all results :( )  
        s = s[s["buffer_size"] != 4]
        s["Nodes"] = nodes
        # Append s to df
        df = pd.concat([df, s], ignore_index=True)        
    return df

def augment_df(df, metric, target_algo):
    # Step 1: Create an empty list to hold the new rows
    new_data = []
    
    # For each (buffer_size, nodes) group the data so that for each algo_name we only keep the entry with the highest bandwidth_mean
    df = df.loc[df.groupby(['buffer_size', 'Nodes', 'algo_name'])['bandwidth_' + metric].idxmax()]
    
    # Step 2: Group by 'buffer_size' and 'Nodes'
    for (buffer_size, nodes), group in df.groupby(['buffer_size', 'Nodes']):        
        # Step 3: Get the best algorithm        
        best_algo_row = group.loc[group['bandwidth_' + metric].idxmax()]
        best_algo = best_algo_row['algo_name']
        
        # Step 4: Get the second best algorithm (excluding the best one)
        tmp = group[group['algo_name'] != best_algo]['bandwidth_' + metric]
        if tmp.empty:
            print(f"Warning: No second best algorithm found for buffer_size {buffer_size} and nodes {nodes}. Skipping.", file=sys.stderr)
            continue
        second_best_algo_row = group.loc[tmp.idxmax()]
        second_best_algo = second_best_algo_row['algo_name']

        # Get target algo bandwidth_mean for this group
        target_algo_row = group.loc[group['algo_name'] == target_algo]
        if target_algo_row.empty:
            print(f"Warning: No {target_algo} algorithm found for buffer_size {buffer_size} and nodes {nodes}. Skipping.", file=sys.stderr)
            continue
        
        target_algo_bandwidth_mean = target_algo_row['bandwidth_' + metric].values[0]

        print(f"Buffer size: {buffer_size}, Nodes: {nodes}, Best algo: {best_algo}, Second best algo: {second_best_algo}")
        #print(group)

        ratio = target_algo_bandwidth_mean / best_algo_row['bandwidth_' + metric]
        # Truncate to 1 decimal place
        ratio = round(ratio, 1)
        
        if best_algo == target_algo:
            cell = best_algo_row['bandwidth_' + metric] / second_best_algo_row['bandwidth_' + metric]  
        else:
            cell = ratio         
        
        # Step 6: Append the data for this group (including old columns)
        new_data.append({
            'buffer_size': buffer_size,
            'Nodes': nodes,
            #'algo_name': best_algo,
            #'bandwidth_' + metric: best_algo_row['bandwidth_' + metric],
            'cell': cell,
        })

    # Step 7: Create a new DataFrame
    return pd.DataFrame(new_data)

def main():
    parser = argparse.ArgumentParser(description="Generate graphs")
    parser.add_argument("--system", type=str, help="System name", required=True)
    parser.add_argument("--collective", type=str, help="Collective", required=True)        
    parser.add_argument("--nnodes", type=str, help="Number of nodes (comma separated)", required=True)
    parser.add_argument("--target_algo", type=str, help="Target algorithm to compare against", default="ring_ompi")
    parser.add_argument("--tasks_per_node", type=int, help="Tasks per node", default=1)    
    parser.add_argument("--notes", type=str, help="Notes")   
    parser.add_argument("--exclude", type=str, help="Algos to exclude", default=None)   
    parser.add_argument("--metric", type=str, help="Metric to consider [mean|median|percentile_90]", default="mean")   
    args = parser.parse_args()

    #print("Called with args:")
    #print(args)

    df = get_summaries_df(args)    
          
    # Drop the columns I do not need
    df = df[["buffer_size", "Nodes", "algo_name", "mean", "median", "percentile_90"]]

    if args.exclude:
        for e in args.exclude.split(","):
            # Remove the algo_name that contains e
            df = df[~df["algo_name"].str.contains(e, case=False)]
        
    # Compute the bandwidth for each metric
    for m in metrics:
        if m == args.metric:
            df["bandwidth_" + m] = ((df["buffer_size"]*8.0)/(1000.0*1000*1000)) / (df[m].astype(float) / (1000.0*1000*1000))
    
    # drop all the metrics
    for m in metrics:
        df = df.drop(columns=[m])

    # print full df, no limts on cols or rows
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)

    df = augment_df(df, args.metric, args.target_algo)
    #print(df)

    # We need to separate numerical and string cells
    # Step 1: Create the 'numeric' version of the DataFrame, where strings are NaN
    df_numeric = df.copy()
    df_numeric['cell'] = pd.to_numeric(df['cell'], errors='coerce')

    # Step 2: Pivot the numeric data for heatmap plotting
    heatmap_data_numeric = df_numeric.pivot(index='buffer_size', columns='Nodes', values='cell')
    heatmap_data_numeric = heatmap_data_numeric[args.nnodes.split(",")]    

    # Step 3: Pivot the original data for string annotation
    heatmap_data_string = df.pivot(index='buffer_size', columns='Nodes', values='cell')
    heatmap_data_string = heatmap_data_string[args.nnodes.split(",")]

    # Set up the figure and axes
    plt.figure()

    # Create a matrix of colors for the heatmap cells based on the content of the dataframe
    # Create an empty matrix of the same shape as df for background colors
    cell_colors = np.full(heatmap_data_string.shape, 'white', dtype=object)  # Default white for numbers

    # Create the heatmap with numerical values for color
    red_green = LinearSegmentedColormap.from_list("RedGreen", ["darkred", "white", "darkgreen"])
    ax = sns.heatmap(heatmap_data_numeric, 
                    annot=True, 
                    cmap=red_green, 
                    fmt=fmt,
                    center=1, 
                    cbar=True, 
                    #square=True,
                    annot_kws={'size': big_font_size, 'weight': 'bold'},
                    cbar_kws={"orientation": "horizontal", "location" : "top", "aspect": 40},
                    )

    # Get the colorbar and set the font size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=small_font_size)  # Adjust font size of ticks

    # For each ror use the corresponding buffer_size_hr rather than buffer_size as labels
    # Get all the row names, sort them (numerically), and the apply to each of them the human_readable_size function
    # to get the human-readable size
    # Then set the x-ticks labels to these human-readable sizes
    # Get heatmap_data.rows and convert to a list of int
    buffer_sizes = heatmap_data_string.index.astype(int).tolist()
    buffer_sizes.sort()
    buffer_sizes = [human_readable_size(int(x)) for x in buffer_sizes]
    # Use buffer_sizes as labels
    plt.yticks(ticks=np.arange(len(buffer_sizes)) + 0.5, labels=buffer_sizes)

    plt.xlabel("# Nodes", fontsize=big_font_size)
    plt.ylabel("Vector Size", fontsize=big_font_size)
    plt.xticks(fontsize=small_font_size)
    plt.yticks(fontsize=small_font_size)
    # Do not rotate xticklabels
    plt.xticks(rotation=0)   

    # Make dir if it does not exist
    outdir = "plot_generic/" + args.system + "/heatmaps/" + args.collective + "/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # in outfile name we should save all the infos in args
    # Convert args to a string with param name and param value
    args_str = "_".join([f"{k}_{v}" for k, v in vars(args).items() \
                        if k != "nnodes" and k != "system" and k != "collective" and (k != "notes" or v != None) and (k != "exclude" or v != None)])
    args_str = args_str.replace("|", "_")
    outfile = outdir + "/" + args_str + ".pdf"
    # Save as PDF
    plt.savefig(outfile, bbox_inches="tight")

if __name__ == "__main__":
    main()

