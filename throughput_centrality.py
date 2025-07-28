import os
import re
import json
import matplotlib.pyplot as plt
import numpy as np
from src.env.network import Network
import networkx as nx

# Path to your log files
log_dir = "/rds/general/user/mss124/home/thesis/marl_graph_exploration/20250711_133739_shortest_path_throughput_evals/"  # update as needed

# Dictionary to store throughput values
results = {}

# Regex patterns
file_pattern = re.compile(r"shortest(-nocong)?-paths-eval-topo-(\d+)-run-(\d+)")
throughput_pattern = re.compile(r'"throughput_mean":\s*([0-9.]+)')

# Parse logs and extract throughputs
for filename in os.listdir(log_dir):
    if not filename.endswith(".log"):
        continue

    match = file_pattern.search(filename)
    if not match:
        continue

    nocong_flag = bool(match.group(1))
    seed = match.group(2)
    run = match.group(3)

    filepath = os.path.join(log_dir, filename)
    with open(filepath, "r") as f:
        content = f.read()

    throughput_match = throughput_pattern.search(content)
    if throughput_match:
        throughput = float(throughput_match.group(1))
        key = (seed, run, "nocong" if nocong_flag else "cong")
        results[key] = throughput

# Organize throughput values
data_points = {}  # (seed, run) → {'cong': val, 'nocong': val}
for (seed, run, label), throughput in results.items():
    key = (seed, run)
    if key not in data_points:
        data_points[key] = {}
    data_points[key][label] = throughput

# Build centrality mapping
centrality_map = {}  # seed → avg betweenness centrality

for (seed, run) in data_points:
    if seed in centrality_map:
        continue  # Already computed

    # Reconstruct network
    net = Network(n_nodes=30, provided_seeds=[int(seed)])
    net.reset()
    G = net.G

    # Compute betweenness centrality
    centrality = nx.betweenness_centrality(G, weight='weight')
    avg_centrality = np.max(list(centrality.values()))
    centrality_map[seed] = avg_centrality

# Prepare scatter data
x_vals, y_vals, colors = [], [], []

for (seed, run), vals in data_points.items():
    if 'cong' in vals and 'nocong' in vals:
        x_vals.append(vals['nocong'])
        y_vals.append(vals['cong'])
        colors.append(centrality_map[seed])  # color by centrality

# Normalize colors
colors = np.array(colors)
norm = plt.Normalize(vmin=min(colors), vmax=max(colors))
cmap = plt.cm.viridis

# Plot
plt.figure(figsize=(8, 6))
sc = plt.scatter(x_vals, y_vals, c=colors, cmap='viridis', alpha=0.8)

# Labels and colorbar

plt.xlabel("Throughput without bandwidth limitation", fontsize=14)
plt.ylabel("Throughput with bandwidth limitation", fontsize=14)
cbar = plt.colorbar(sc)
cbar.set_label("Max betweenness centrality", fontsize=14)

plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("throughput_vs_centrality_colored.png", dpi=300)
plt.show()


# Assuming results is a dict like: results[(seed, run, 'cong' or 'nocong')] = throughput

matches = []

for (seed, run), vals in data_points.items():
    if 'cong' in vals and 'nocong' in vals:
        cong_tp = vals['cong']
        nocong_tp = vals['nocong']
        
        if 5.5 <= nocong_tp <= 6.0 and 2.0 <= cong_tp <= 2.5:
            matches.append((seed, run, nocong_tp, cong_tp))

# Print results
for seed, run, no_bl, bl in matches:
    print(f"Seed: {seed}, Run: {run}, Without BL: {no_bl:.2f}, With BL: {bl:.2f}")

print(f"\nTotal matches found: {len(matches)}")

# === 2. Pick the first match (you can loop if needed) ===
seed, run, no_bl, bl = matches[0]
print(f"\nVisualizing seed: {seed}, run: {run}")

# === 3. Load the graph ===
net = Network(n_nodes=30, provided_seeds=[int(seed)])
net.reset()
G = net.G  # NetworkX graph

# === 4. Draw and save ===
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)  # consistent layout

# Optional: node color based on betweenness centrality
centrality = nx.betweenness_centrality(G)
node_colors = [centrality[n] for n in G.nodes()]

nx.draw_networkx_nodes(G, pos, node_size=300, node_color=node_colors, cmap='viridis')
nx.draw_networkx_edges(G, pos, alpha=0.3)
nx.draw_networkx_labels(G, pos, font_size=8)

plt.title(f"Seed {seed} – Run {run}\nThroughput: nocong={no_bl:.2f}, cong={bl:.2f}")
plt.axis('off')
plt.tight_layout()
plt.savefig(f"graph_seed_{seed}_run_{run}.png", dpi=300)
plt.show()

import random

# === 5. Plot graphs for low, medium, high max betweenness centrality ===

# Convert centrality_map to list of (seed, centrality)
centrality_list = [(str(seed), val) for seed, val in centrality_map.items()]
centrality_list.sort(key=lambda x: x[1])  # sort by max centrality

# Bin into thirds
n = len(centrality_list)
if n < 3:
    print("Not enough data to create 3 bins.")
else:
    bins = {
        "low": centrality_list[:n // 3],
        "mid": centrality_list[n // 3: 2 * n // 3],
        "high": centrality_list[2 * n // 3:]
    }

    selected = {}

    for label, seed_bin in bins.items():
        random.shuffle(seed_bin)  # add randomness
        for str_seed, _ in seed_bin:
            for run in range(100):
                str_run = str(run)
                key = (str_seed, str_run)
                if key in data_points and 'cong' in data_points[key] and 'nocong' in data_points[key]:
                    selected[label] = (str_seed, str_run)
                    break
            if label in selected:
                break

    if len(selected) < 3:
        print("Couldn't find valid seeds in all bins with both throughput values.")
    else:
        for label, (str_seed, str_run) in selected.items():
            cong_val = data_points[(str_seed, str_run)]['cong']
            nocong_val = data_points[(str_seed, str_run)]['nocong']

            net = Network(n_nodes=30, provided_seeds=[int(str_seed)])
            net.reset()
            G = net.G

            centrality = nx.betweenness_centrality(G)
            node_colors = [centrality[n] for n in G.nodes()]

            plt.figure(figsize=(10, 8))
            pos = nx.spring_layout(G, seed=42)

            nx.draw_networkx_nodes(G, pos, node_size=300, node_color=node_colors, cmap='viridis')
            nx.draw_networkx_edges(G, pos, alpha=0.3)
            nx.draw_networkx_labels(G, pos, font_size=8)

            plt.title(
                f"{label.upper()} Centrality\n"
                f"Seed {str_seed} – Run {str_run}\n"
                f"Throughput: nocong={nocong_val:.2f}, cong={cong_val:.2f}\n"
                f"Max Centrality: {centrality_map[str_seed]:.4f}"
            )
            plt.axis('off')
            plt.tight_layout()
            #plt.savefig(f"graph_{label}_centrality_seed_{str_seed}_run_{str_run}.png", dpi=300)
            plt.show()

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# Provided seeds and runs (you can update run IDs if needed)
selected = {
    "high": ("415127002", "0"),
    "low": ("1493043848", "0"),
    "mid": ("256224875", "0")
}

fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

all_centralities = []

graphs_data = {}
for label, (str_seed, str_run) in selected.items():
    net = Network(n_nodes=30, provided_seeds=[int(str_seed)])
    net.reset()
    G = net.G

    centrality = nx.betweenness_centrality(G)
    all_centralities.extend(centrality.values())

    graphs_data[label] = {
        "G": G,
        "centrality": centrality,
        "str_seed": str_seed,
        "str_run": str_run,
        "min_c": min(centrality.values()),
        "max_c": max(centrality.values()),
        "mean_c": np.mean(list(centrality.values()))
    }

# Normalize color scale across all graphs
norm = Normalize(vmin=min(all_centralities), vmax=max(all_centralities))
cmap = plt.cm.viridis

for ax, label in zip(axes, ["low", "mid", "high"]):
    data = graphs_data[label]
    G = data["G"]
    centrality = data["centrality"]
    node_colors = [centrality[n] for n in G.nodes()]
    pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_nodes(G, pos, node_size=300, node_color=node_colors, cmap=cmap, vmin=min(all_centralities), vmax=max(all_centralities), ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='black')
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    # Title with (min, max, mean) betweenness centrality
    ax.set_title(
        f"$G_{{{label.upper()}}}$ "
        f"({data['min_c']:.2f}, {data['max_c']:.2f}, {data['mean_c']:.2f})",
        fontsize=14
    )
    ax.axis("off")

# Shared colorbar
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.025, pad=0.02)
cbar.set_label("Betweenness centrality", fontsize=12)

plt.savefig("combined_centrality_graphs_with_stats.png", dpi=300)
plt.show()
