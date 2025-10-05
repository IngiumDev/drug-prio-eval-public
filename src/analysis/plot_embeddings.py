#!/usr/bin/env python3
"""
plot_embeddings.py

Load a combined TSV (with `parents` and `children` as Python list literals), one-hot-encode the parents,
build a numeric feature matrix, run PCA and UMAP, and save three plots:
  - PCA (PC1 vs PC2)
  - PCA faceted at p_value_DCG > 0.5
  - UMAP (on top PCs)
"""
import argparse
import ast
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, MinMaxScaler

_BASE_MD = {"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None}
# Keys are case-sensitive; use exactly these names. :contentReference[oaicite:2]{index=2}

# Patch plt.savefig
_orig_plt_savefig = plt.savefig


def _plt_savefig(fname, *a, **k):
    meta = {**_BASE_MD, **(k.get("metadata") or {})}
    k["metadata"] = meta
    return _orig_plt_savefig(fname, *a, **k)


plt.savefig = _plt_savefig

# Patch Figure.savefig (covers fig.savefig)
_orig_fig_savefig = Figure.savefig


def _fig_savefig(self, fname, *a, **k):
    meta = {**_BASE_MD, **(k.get("metadata") or {})}
    k["metadata"] = meta
    return _orig_fig_savefig(self, fname, *a, **k)


Figure.savefig = _fig_savefig


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def combine_duplicate_columns_equals(df: pd.DataFrame, sep: str = "|") -> pd.DataFrame:
    """
    Collapse columns that are identical by content, even if their names differ.
    - Keeps the first column in each identical group
    - Renames it to the original names joined with `sep`
    - Uses Series.equals to avoid hashing issues with unhashable types (lists, arrays)
    """
    cols = list(df.columns)
    used = set()
    groups = []

    for i, c in enumerate(cols):
        if c in used:
            continue
        same = [c]
        for j in range(i + 1, len(cols)):
            c2 = cols[j]
            if c2 in used:
                continue
            # True if element-wise all equal, NaNs match, lists compare by value
            if df[c].equals(df[c2]):
                same.append(c2)
                used.add(c2)
        used.add(c)
        groups.append(same)

    # Build output: keep the first col of each group, rename to joined names
    kept_cols = [g[0] for g in groups]
    out = df[kept_cols].copy()
    out.columns = [sep.join(g) for g in groups]

    merged_count = sum(len(g) - 1 for g in groups)
    logging.info(f"Merged {merged_count} duplicate columns into {len(groups)} unique columns")
    return out


def main():
    args = parse_arguments()
    setup_logging()

    # Ensure output directory exists
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Loading combined data from {args.input_tsv}")
    combined = pd.read_csv(args.input_tsv, sep='\t')

    # Parse 'parents' and 'children' from Python list literals
    logging.info("Evaluating parent/children list literals")
    for col in ['parents', 'children']:
        combined[col] = combined[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])

    # One-hot encode all possible parents
    logging.info("One-hot encoding parents")
    mlb = MultiLabelBinarizer(sparse_output=False)
    parents_ohe = pd.DataFrame(mlb.fit_transform(combined['parents']), columns=[f"parent_{p}" for p in mlb.classes_],
                               index=combined.index)
    combined = pd.concat([combined, parents_ohe], axis=1)
    combined = combine_duplicate_columns_equals(combined, sep="|").query("candidate_count !=0")

    # Identify numeric feature columns (exclude p-value & label cols)
    exclude_cols = ['PC1', 'PC2', 'seeds_not_in_network', 'p_value_DCG', 'p_value_without_ranks',
                    'percent_true_drugs_found', 'num_drugs', 'num_approved_drugs', 'num_targets',
                    'approved_drugs_with_targets', 'significant', 'observed_DCG', 'observed_overlap',
                    'dcg_exceed_count', 'overlap_exceed_count', ]
    all_numeric = combined.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [c for c in all_numeric if c not in exclude_cols]
    # Start time for benchmarking
    run_logistic_regression(combined, exclude_cols)
    # run_logistic_regression_statsmodels(combined)
    run_random_forest(combined, exclude_cols)
    # feature_cols = parents_ohe.select_dtypes(include=["number"]).columns.tolist()
    logging.info(f"Using {len(feature_cols)} numeric features for embeddings")

    # Build feature matrix X and scale
    X = combined[feature_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    run_pca(X_scaled, combined, output_dir)
    run_umap(X_scaled, combined, output_dir)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot PCA and UMAP from combined data TSV")
    parser.add_argument("--input_tsv", default="../../data/input/combined_data.tsv",
                        help="Path to the combined_data.tsv generated by your main script")
    parser.add_argument("--output_dir", default="../../plots/embeddings",
                        help="Directory to save the plots (default: ../../plots/embeddings)")
    args = parser.parse_args()
    return args


def run_pca(X_scaled, combined, output_dir):
    # PCA for visualization
    logging.info("Running PCA")
    pca = PCA(random_state=0)
    pcs = pca.fit_transform(X_scaled)
    combined['PC1'], combined['PC2'] = pcs[:, 0], pcs[:, 1]
    # Plot PCA
    logging.info("Plotting PCA (PC1 vs PC2)")
    plt.figure(figsize=(6, 5))
    plt.scatter(combined['PC1'], combined['PC2'], c=combined['p_value_DCG'], cmap='viridis', rasterized=True, s=30,
                edgecolor='k', linewidth=0.2)
    plt.colorbar(label='p_value_DCG')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA (PC1 vs PC2) colored by p_value_DCG')
    plt.tight_layout()
    pca_path = os.path.join(output_dir, "pca.pdf")
    plt.savefig(pca_path, dpi=300)
    plt.close()
    logging.info(f"Saved PCA plot to {pca_path}")
    # Facet by p_value_DCG threshold
    logging.info("Plotting faceted PCA by p_value_DCG > 0.5")
    combined['p_value_cat'] = np.where(combined['p_value_DCG'] > 0.5, '> 0.5', '≤ 0.5')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, (label, df_sub) in zip(axes, combined.groupby('p_value_cat')):
        sc = ax.scatter(df_sub['PC1'], df_sub['PC2'], c=df_sub['p_value_DCG'], cmap='viridis', rasterized=True, s=30,
                        edgecolor='k', linewidth=0.2)
        ax.set_title(f"p_value_DCG {label}")
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), label='p_value_DCG')
    plt.tight_layout()
    facet_path = os.path.join(output_dir, "pca_facet.pdf")
    plt.savefig(facet_path, dpi=300)
    plt.close()
    logging.info(f"Saved faceted PCA plot to {facet_path}")


def run_logistic_regression(combined, exclude_cols):
    # --- Logistic regression to predict significance (p_value_DCG < 0.05) ---
    # 1. Create binary target
    combined['significant'] = (combined['p_value_DCG'] < 0.05).astype(int)
    # 2. Select numeric feature columns (include binarized parents), excluding p-value columns and the new target
    numeric_cols = combined.select_dtypes(include=['number']).columns.tolist()
    feature_cols_lr = [c for c in numeric_cols if c not in exclude_cols]
    # 3. Prepare X and y
    X = combined[feature_cols_lr].fillna(0)
    y = combined['significant']
    # Scale all features and fit on full dataset for inference
    scaler_lr = MinMaxScaler()
    X_scaled = scaler_lr.fit_transform(X)
    lr = LogisticRegression(max_iter=1000, random_state=0)
    lr.fit(X_scaled, y)
    # 7. Top 20 features by absolute coefficient magnitude
    # After fitting on X_scaled and y:
    coeffs = pd.Series(lr.coef_[0], index=feature_cols_lr)

    # Top 10 positive coefficients (features that most increase log-odds)
    top10_pos = coeffs.sort_values(ascending=False).head(10)

    # Top 10 negative coefficients (features that most decrease log-odds)
    top10_neg = coeffs.sort_values(ascending=True).head(10)

    print("Top 10 positive coefficients:")
    print(top10_pos)

    print("\nTop 10 negative coefficients:")
    print(top10_neg)
    # In-sample performance metrics for logistic regression
    y_pred = lr.predict(X_scaled)
    y_prob = lr.predict_proba(X_scaled)[:, 1]

    # Overall metrics
    print("In-sample Accuracy:         ", accuracy_score(y, y_pred))
    print("In-sample Precision (pos):  ", precision_score(y, y_pred))
    print("In-sample Recall (pos):     ", recall_score(y, y_pred))
    print("In-sample ROC AUC:          ", roc_auc_score(y, y_prob))

    # Per-class precision & recall
    precision_per_class = precision_score(y, y_pred, average=None, labels=[0, 1])
    recall_per_class = recall_score(y, y_pred, average=None, labels=[0, 1])
    print("In-sample Precision by class [0,1]:", precision_per_class)
    print("In-sample Recall    by class [0,1]:", recall_per_class)

    # Full classification report
    print("\nClassification Report:\n", classification_report(y, y_pred))


def run_logistic_regression_statsmodels(combined):
    # 1. Define exclusions and feature list
    exclude_cols = ['dcg_exceed_count', 'observed_DCG', 'observed_overlap', 'overlap_exceed_count',
                    'p_value_without_ranks', 'p_value_DCG', 'significant', 'PC1', 'PC2', 'seeds_not_in_network',
                    'num_drugs',
                    'largest_component', 'isolated_nodes']
    numeric_cols = combined.select_dtypes(include=['number']).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    # 2. Build feature matrix and scale to [0,1]
    X = combined[feature_cols].fillna(0)
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    logging.info("Ranges after MinMax scaling: "
                 f"{X_scaled.min().to_dict()} -> {X_scaled.max().to_dict()}")

    # 3. Collapse perfectly identical columns
    sig_to_cols = {}
    for col in X_scaled.columns:
        key = tuple(X_scaled[col].values)
        sig_to_cols.setdefault(key, []).append(col)

    X_merged = pd.DataFrame(index=X_scaled.index)
    for group in sig_to_cols.values():
        merged_name = "|".join(group)
        X_merged[merged_name] = X_scaled[group[0]]

    if X_merged.shape[1] < X_scaled.shape[1]:
        dropped = X_scaled.shape[1] - X_merged.shape[1]
        logging.info(f"Merged {dropped} identical column group(s): "
                     f"{[g for g in sig_to_cols.values() if len(g) > 1]}")

    # 4. Compute correlation matrix and find near-perfect pairs
    corr = X_merged.corr().abs()  # Pearson by default
    near_pairs = []
    cols = corr.columns.tolist()
    for i, col_i in enumerate(cols):
        for col_j in cols[i + 1:]:
            if corr.loc[col_i, col_j] > 0.999 or corr.loc[col_i, col_j] < -0.999:
                near_pairs.append((col_i, col_j, corr.loc[col_i, col_j]))

    if near_pairs:
        logging.info("Perfect or near-perfectly correlated pairs (|corr|>0.999):")
        for a, b, val in near_pairs:
            logging.info(f"  {a} ↔ {b} : {val:.4f}")
    else:
        logging.info("No near-perfectly correlated feature pairs found.")
    # Calculate rank of the matrix vs columns
    rank = np.linalg.matrix_rank(X_merged.values)
    logging.info(f"Rank of the feature matrix: {rank} out of {X_merged.shape[1]} columns")
    dupes = X_merged.duplicated().sum()
    logging.info(f"Number of duplicate rows: {dupes}")
    # drop duplicate rows
    X_merged = X_merged.drop_duplicates()
    rank = np.linalg.matrix_rank(X_merged.values)
    logging.info(f"Rank of the feature matrix: {rank} out of {X_merged.shape[1]} columns")


def run_random_forest(combined, exclude_cols):
    # Random forest classification to predict significance (p_value_DCG < 0.05)
    combined['significant'] = (combined['p_value_DCG'] < 0.05).astype(int)
    # Select numeric features, excluding label and metadata columns
    numeric_cols = combined.select_dtypes(include=['number']).columns.tolist()
    feature_cols_rf = [c for c in numeric_cols if c not in exclude_cols]
    X = combined[feature_cols_rf].fillna(0)
    y = combined['significant']
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    # Instantiate and fit Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    rf.fit(X_train, y_train)
    # Predictions and probabilities
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    # Evaluation metrics
    print("Random Forest Classification Metrics:")
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:   ", recall_score(y_test, y_pred))
    print("ROC AUC:  ", roc_auc_score(y_test, y_prob))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    # Feature importances
    importances = pd.Series(rf.feature_importances_, index=feature_cols_rf)
    top20 = importances.sort_values(ascending=False).head(20)
    print("\nTop 20 features by importance:")
    print(top20)


def run_umap(X_scaled, combined, output_dir):
    logging.info("Running UMAP")
    umap_reducer = umap.UMAP(n_jobs=-1)
    umap_coords = umap_reducer.fit_transform(X_scaled)
    combined['UMAP1'], combined['UMAP2'] = umap_coords[:, 0], umap_coords[:, 1]
    # Plot UMAP
    logging.info("Plotting UMAP")
    plt.figure(figsize=(6, 5))
    plt.scatter(combined['UMAP1'], combined['UMAP2'], c=combined['p_value_DCG'], cmap='viridis', rasterized=True, s=30,
                edgecolor='k', linewidth=0.2)
    plt.colorbar(label='p_value_DCG')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.title('UMAP (on top PCs) colored by p_value_DCG')
    plt.tight_layout()
    umap_path = os.path.join(output_dir, "umap.pdf")
    plt.savefig(umap_path, dpi=300)
    plt.close()
    logging.info(f"Saved UMAP plot to {umap_path}")


if __name__ == "__main__":
    main()
