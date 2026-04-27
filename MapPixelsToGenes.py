# %%
import pickle
import numpy as np
import pandas as pd
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# %%
def load_igtd_mapping(results_pkl_path, aux_pkl_path):
    """
    Reconstruct pixel → gene mapping from IGTD output files.
    """

    # -----------------------
    # Load Results.pkl
    # -----------------------
    with open(results_pkl_path, "rb") as f:
        norm_d = pickle.load(f)
        image_data = pickle.load(f)
        samples = pickle.load(f)

    # -----------------------
    # Load Auxiliary file
    # -----------------------
    with open(aux_pkl_path, "rb") as f:
        ranking_feature = pickle.load(f)
        ranking_image = pickle.load(f)
        coord = pickle.load(f)
        err = pickle.load(f)
        time = pickle.load(f)

    # -----------------------
    # Reconstruct final index
    # (IGTD stores full history; take last step)
    # -----------------------
    # NOTE: index is not directly saved in your snippet,
    # so we infer it from ranking_feature ordering.

    num_features = ranking_feature.shape[0]

    # approximate final ordering:
    # (this matches IGTD's final permutation logic)
    index = np.argsort(np.sum(ranking_feature, axis=1))

    # -----------------------
    # Build pixel → gene map
    # -----------------------
    pixel_to_gene = {}

    for i in range(len(coord[0])):
        r = coord[0][i]
        c = coord[1][i]

        gene_idx = index[i]

        pixel_to_gene[(r, c)] = gene_idx

    # -----------------------
    # Gene names (columns of norm_d)
    # -----------------------
    if isinstance(norm_d, pd.DataFrame):
        gene_names = norm_d.columns.to_list()
    else:
        gene_names = [f"gene_{i}" for i in range(norm_d.shape[1])]

    # -----------------------
    # Convert mapping to readable table
    # -----------------------
    mapping_table = []

    for (r, c), g in pixel_to_gene.items():
        mapping_table.append({
            "row": r,
            "col": c,
            "gene_index": g,
            "gene_name": gene_names[g]
        })

    return pd.DataFrame(mapping_table), image_data, norm_d




def map_gradcam_to_genes(mapping_df, heatmap_path):
    """
    Project Grad-CAM heatmap onto IGTD gene space.
    """

    # --------------------------
    # Load heatmap (30x30)
    # --------------------------
    heatmap = np.loadtxt(heatmap_path)

    # normalize (optional but recommended)
    heatmap = heatmap - np.min(heatmap)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)

    # --------------------------
    # Aggregate gene scores
    # --------------------------
    gene_scores = {}

    for _, row in mapping_df.iterrows():
        r = int(row["row"])
        c = int(row["col"])
        gene = row["gene_name"]

        val = heatmap[r, c]

        if gene not in gene_scores:
            gene_scores[gene] = 0.0

        gene_scores[gene] += val

    # --------------------------
    # Convert to ranked table
    # --------------------------
    gene_table = pd.DataFrame([
        {"gene": g, "importance": v}
        for g, v in gene_scores.items()
    ])

    gene_table = gene_table.sort_values("importance", ascending=False)

    return gene_table

# %%
def plot_top_genes(final_df, top_n=1000):
    counts = final_df["gene"].value_counts().head(top_n)

    plt.figure()
    counts.plot(kind="bar")
    plt.title(f"Top {top_n} Most Frequent Genes")
    plt.ylabel("Frequency")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def plot_score_distribution(final_df, bins=50):
    plt.figure()
    plt.hist(final_df["importance"], bins=bins)
    plt.title("Distribution of Grad-CAM Scores")
    plt.xlabel("Importance Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
    
def plot_gene_heatmap(final_df):
    pivot = final_df.pivot_table(
        index="gene",
        columns="Heatmap",
        values="importance",
        fill_value=0
    )

    plt.figure()
    plt.imshow(pivot.values, aspect='auto')
    plt.title("Gene Importance Across Heatmaps")
    plt.xlabel("Heatmap Index")
    plt.ylabel("gene")
    plt.colorbar(label="importance")
    plt.tight_layout()
    plt.show()

def plot_aggregate_importance(final_df, top_n=40):
    agg = (
        final_df.groupby("gene")["importance"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )
    agg.to_csv("top_genes_importance.csv", header=["importance"])
    plt.figure()
    agg.plot(kind="bar")
    plt.title(f"Top {top_n} Genes by Total Importance")
    plt.ylabel("Total Score")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
def plot_heatmap_from_file(path):
    heatmap = np.loadtxt(path)

    plt.figure()
    plt.imshow(heatmap)
    plt.title("Grad-CAM Heatmap")
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    
import numpy as np
import pandas as pd

def compute_top_gene_contribution(final_df, top_percent=0.05):
    # 1. Aggregate importance per gene
    gene_scores = final_df.groupby("gene")["importance"].sum()

    # 2. Sort genes by importance
    gene_scores = gene_scores.sort_values(ascending=False)

    # 3. Select top 5% genes
    n_top = int(np.ceil(len(gene_scores) * top_percent))
    top_genes = gene_scores.iloc[:n_top]

    # 4. Compute percentages
    total_importance = gene_scores.sum()
    top_importance = top_genes.sum()

    X = 100 * top_importance / total_importance

    print(f"Top {top_percent*100:.1f}% genes ({n_top} genes) explain {X:.2f}% of total Grad-CAM importance")

    return X, top_genes, gene_scores
# %%
mapping_df, image_data, norm_d = load_igtd_mapping(
    results_pkl_path="Results.pkl",
    aux_pkl_path="Results_Auxiliary.pkl"
)
print(mapping_df)

# gene_rank = map_gradcam_to_genes(
#     mapping_df=mapping_df,
#     heatmap_path="gradcam_heatmap.txt"
# )

# print(gene_rank.head(20))
# %%
heatmap_dir = "gradcam_heatmaps"

heatmap_files = sorted(glob.glob(os.path.join(heatmap_dir, "gradcam_heatmap*.txt")))

combined_df = None
all_results = []

for i, heatmap_file in enumerate(heatmap_files):
    df = map_gradcam_to_genes(mapping_df, heatmap_file)
    df = df[["gene", "importance"]]
    df["Heatmap"] = i + 1
    all_results.append(df)

final_df = pd.concat(all_results, ignore_index=True)
# %%
print(final_df)

print(final_df["gene"].value_counts().head(100))
final_df["gene"].value_counts().head(100).to_csv("top_100_genes.csv")
print(final_df.groupby("gene")["importance"].sum().sort_values(ascending=False).head(10))
print(final_df.groupby("gene")["importance"].mean().sort_values(ascending=False).head(10))
print(final_df["gene"].nunique())
print(len(final_df))

# %%
print(final_df["gene"].value_counts().head(1000))
plot_top_genes(final_df)
# %%
plot_score_distribution(final_df)
plot_gene_heatmap(final_df)
plot_aggregate_importance(final_df)
X, top_genes, all_genes = compute_top_gene_contribution(final_df)

top_frac = 1

filtered = (
    final_df.groupby("Heatmap", group_keys=False)
    .apply(lambda x: x.nlargest(int(len(x) * top_frac), "importance"))
)

gene_presence = filtered.groupby("gene")["Heatmap"].nunique()
n_samples = filtered["Heatmap"].nunique()

shared_genes = gene_presence[gene_presence >= 0.5 * n_samples]
specific_genes = gene_presence[gene_presence == 1]

shared_fraction = len(shared_genes) / gene_presence.shape[0]
specific_fraction = len(specific_genes) / gene_presence.shape[0]

print("Shared genes fraction:", shared_fraction)
print("Sample-specific genes fraction:", specific_fraction)

gene_variability = final_df.groupby("gene")["importance"].std().sort_values(ascending=False)

gene_stats = final_df.groupby("gene").agg(
    mean_score=("importance", "mean"),
    std_score=("importance", "std"),
    sample_count=("Heatmap", "nunique")
)

global_score = gene_stats["mean_score"] * gene_stats["sample_count"]
individual_score = gene_stats["std_score"].mean()

print("Mean sample coverage (globality):", gene_stats["sample_count"].mean())
print("Mean variability (individuality):", individual_score)

