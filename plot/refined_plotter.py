import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import sys


# Style
sns.set_style("whitegrid")
sns.set_context("paper")

def DrawLinePlot(data, name):
    """
    Draw mean bandwidth with a variability band per line using Seaborn.
    Uses standard deviation ('sd') bands so they're visible with per-iteration data.
    """
    print(f"Plotting data collective: {name}")


    f, ax1 = plt.subplots(figsize=(12, 8))

    # DataFrame
    df = pd.DataFrame(data)
    df['cluster_collective'] = df['Cluster'].astype(str) + '_' + df['collective'].astype(str)

    # Order x by true byte size (avoid alphabetical order of labels)
    order = (
        df.sort_values('message_bytes')
          .drop_duplicates('Message')['Message']
          .tolist()
    )
    df['Message'] = pd.Categorical(df['Message'], categories=order, ordered=True)

    # Palette
    palette = sns.color_palette("tab10", n_colors=df['cluster_collective'].nunique())

    # Lineplot with SD band (more visible than CI with your data)
    try:
        # seaborn >= 0.12
        sns.lineplot(
            data=df,
            x='Message',
            y='bandwidth',
            hue='cluster_collective',
            style='cluster_collective',
            markers=True,
            markersize=9,
            linewidth=2,           # thinner so the band isn't hidden
            estimator='mean',
            errorbar=('sd', 0.5),         # <-- show standard deviation band
            ax=ax1,
            palette=palette,
            # sort=False             # we've already ordered via categorical dtype
        )
    except TypeError:
        # seaborn < 0.12 fallback
        sns.lineplot(
            data=df,
            x='Message',
            y='bandwidth',
            hue='cluster_collective',
            style='cluster_collective',
            markers=True,
            markersize=9,
            linewidth=2,
            # estimator=np.mean,
            # ci='sd',               # <-- SD band in older seaborn
            ax=ax1,
            palette=palette,
            # sort=False
        )

    # Labels & legend
    ax1.set_ylabel('Bandwidth (Gb/s)', fontsize=15)
    ax1.set_xlabel('Message Size', fontsize=15)
    ax1.set_title(f'{name}', fontsize=18)
    ax1.tick_params(axis='both', which='major')

    # Legend cleanup
    handles, labels = ax1.get_legend_handles_labels()
    new_labels = [' '.join(w.capitalize() for w in lbl.replace('_', ' ').split())
                  for lbl in labels]
    ax1.legend(handles, new_labels, fontsize=20, loc='upper left', ncol=1, frameon=True)

    plt.tight_layout()

    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{name}_sd_line.png', dpi=300, bbox_inches='tight')
    plt.close()


def LoadData(data, cluster, nodes, path, messages, coll=None, cong=False):

    print (f"Loading data from {path}")

    for msg in messages:
        msg_mult = msg.strip().split(' ')[1]
        msg_value = msg.strip().split(' ')[0]
        if msg_mult == 'B':
            multiplier = 1
        elif msg_mult == 'KiB':
            multiplier = 1024
        elif msg_mult == 'MiB':
            multiplier = 1024**2
        elif msg_mult == 'GiB':
            multiplier = 1024**3
        else:
            raise ValueError(f"Unknown message size unit in {msg}")

        message_bytes = int(msg_value) * multiplier

        for file_name in os.listdir(path):

            run_id = file_name  # treat each file as one replicate
            file_path = os.path.join(path, file_name)

            if cong == False and len(file_name.strip().split("_")) == 4:
                continue

            if cong == True and len(file_name.strip().split("_")) == 3:
                continue

            found_message_bytes = int(file_name.strip().split("_")[0])
            if found_message_bytes != message_bytes:
                continue

            collective = file_name.strip().split("_")[1].split(".")[0]
            if coll is not None and collective != coll:
                continue

            print(f"Processing file: {file_name}")

            i = 1
            iterations = []
            latencies = []
            with open(file_path, 'r') as file:
                lines = file.readlines()[2:]  # Skip the first line
                for line in lines:
                    latency = float(line.strip())
                    latencies.append(latency)
                    iterations.append(i)
                    i += 1

            gb_sent = 0
            if collective == 'all2all':
                gb_sent = ((message_bytes/1e9)*(nodes-1))*8
            elif collective == 'allgather' or collective == 'reducescatter':
                gb_sent = (message_bytes/1e9)*((nodes-1)/nodes)*8
            elif collective == 'allreduce':
                gb_sent = 2*(message_bytes/1e9)*((nodes-1)/nodes)*8
            elif collective == "pointpoint":
                gb_sent = (message_bytes/1e9)*8

            bandwidth = [gb_sent / x for x in latencies]

            data['latency'].extend(latencies)
            data['iteration'].extend(iterations)
            data['bandwidth'].extend(bandwidth)
            data['Message'].extend([msg]*len(latencies))
            data['message_bytes'].extend([message_bytes]*len(latencies))
            data['Cluster'].extend([cluster]*len(latencies))
            data['collective'].extend([collective]*len(latencies))
            data['run'].extend([run_id]*len(latencies))

    return data


def DrawStackedLatencyBars(data, name,
                           sizes=('32 MiB', '2 KiB', '128 KiB', '8 MiB', '512 MiB')):
    """
    Build stacked bars (percent of baseline) for latency decomposition:
    Memcpy, Reduction, Other. Uses mean latency per condition.
    Only draws sizes that actually exist in the data.
    """
    print(f"Building stacked latency bars for: {name}")

    df = pd.DataFrame(data)

    # Keep only allreduce rows (your four conditions are for allreduce)
    df = df[df['collective'] == 'allreduce'].copy()

    # Mean latency per (Message, Cluster)
    mean_lat = (df.groupby(['Message', 'message_bytes', 'Cluster'], observed=True)
                  .agg(latency_mean=('latency', 'mean'))
                  .reset_index())

    # We’ll need the four clusters by name
    # (these are the labels you used when loading)
    want_clusters = ['baseline', 'op_null', 'no_memcpy', 'no_memcpy_op_null']

    # Pivot to columns = clusters
    piv = (mean_lat[mean_lat['Cluster'].isin(want_clusters)]
               .pivot_table(index=['Message', 'message_bytes'],
                            columns='Cluster',
                            values='latency_mean',
                            aggfunc='mean')
               .reset_index())

    # Keep only requested sizes that exist
    sizes_present = [s for s in sizes if s in piv['Message'].values]

    if not sizes_present:
        print("No requested sizes found in data; nothing to plot.")
        return

    # Order x by true bytes
    piv = piv[piv['Message'].isin(sizes_present)].copy()
    piv = piv.sort_values('message_bytes')
    x_order = piv['Message'].tolist()

    # Decompose baseline (all in absolute time first)
    # Guard against missing columns
    for col in ['baseline', 'op_null', 'no_memcpy']:
        if col not in piv:
            piv[col] = np.nan

    memcpy = (piv['baseline'] - piv['no_memcpy']).clip(lower=0)
    reduction = (piv['baseline'] - piv['op_null']).clip(lower=0)
    other = (piv['baseline'] - (memcpy + reduction)).clip(lower=0)

    # Normalize to percent of baseline
    base = piv['baseline']
    pct_memcpy = (memcpy / base) * 100.0
    pct_reduction = (reduction / base) * 100.0
    pct_other = (other / base) * 100.0

    # Assemble plotting frame
    bars = pd.DataFrame({
        'Message': piv['Message'],
        'pct_memcpy': pct_memcpy,
        'pct_reduction': pct_reduction,
        'pct_other': pct_other
    }).set_index('Message').loc[x_order]

    # Plot (matplotlib stacked, using a seaborn palette)
    fig, ax = plt.subplots(figsize=(12, 8))
    pal = sns.color_palette("tab10", 3)
    ax.bar(bars.index, bars['pct_other'].values, label='Network Communications', width=0.7, color=pal[0])
    ax.bar(bars.index, bars['pct_reduction'].values, bottom=bars['pct_other'].values,
           label='Reduction Computation', width=0.7, color=pal[1])
    ax.bar(
        bars.index,
        bars['pct_memcpy'].values,
        bottom=(bars['pct_other'].values + bars['pct_reduction'].values),
        label='Data Movements', width=0.7, color=pal[2]
    )

    # Cosmetics
    ax.set_ylabel('Latency share (% of baseline)', fontsize=15)
    ax.set_xlabel('Message Size', fontsize=15)
    ax.set_title(f'{name} – Per step breakdown', fontsize=18)
    ax.set_ylim(0, 100)
    ax.tick_params(axis='x', rotation=0)
    ax.legend(loc='lower left', frameon=True, fontsize=20)

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{name}_stacked_latency.png', dpi=300, bbox_inches='tight')
    plt.close()

def CleanData(data):
    for key in data.keys():
        data[key] = []
    return data

if __name__ == "__main__":

    data = {
        'Message': [],
        'message_bytes': [],
        'latency': [],
        'bandwidth': [],
        'Cluster': [],
        'collective': [],
        'iteration': [],
        'run': []               # NEW
    }

    if len(sys.argv) > 1:
        nodes = int(sys.argv[1])
    else:
        nodes = 8

    messages = ['32 B', '256 B' , '2 KiB', '16 KiB', '128 KiB', '1 MiB', '8 MiB', '64 MiB', '512 MiB']
    collectives = ['allreduce']

    folder_1 = f"leonardo/with_mem_max/{nodes}"
    folder_2 = f"leonardo/without_mem_max/{nodes}"
    folder_3 = f"leonardo/with_mem_no_red/{nodes}"
    folder_4 = f"leonardo/without_mem_no_red/{nodes}"

    for coll in collectives:
        data = LoadData(data, "baseline", nodes , folder_1, messages=messages, cong=False, coll=coll)
        data = LoadData(data, "op_null", nodes , folder_3, messages=messages, cong=False, coll=coll)
        data = LoadData(data, "no_memcpy", nodes , folder_2, messages=messages, cong=False, coll=coll)
        data = LoadData(data, "no_memcpy_op_null", nodes , folder_4, messages=messages, cong=False, coll=coll)

        DrawLinePlot(data, f"Leonardo, Allreduce, 8 nodes")
        DrawStackedLatencyBars(data, f"Leonardo, Allreduce, 8 nodes",
                       sizes=messages)
        CleanData(data)
