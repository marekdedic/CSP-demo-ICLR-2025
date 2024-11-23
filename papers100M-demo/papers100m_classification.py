"""
This script demonstrates how to use Polars to compute CSP algorithm on the Papers100M dataset.
Steps to run this script:
 * get the dataset https://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip
 * and extract it to the datasets folder and set the DATASET_PATH variable to the path of the extracted dataset
 * install polars with `pip install polars`
 * prepare the dataset with `python create_datasets.py`
 * run this script with `python papers100m_classification.py`
 * Bs controls the speed/memory tradeoff. A higher value will use more memory but will be faster.
"""

import datetime
import os

import polars as pl

DATASET_PATH = "data"  # Path to the dataset
Bs = 5  # batch size -- number of labels to process in parallel due to fitting in memory
start = datetime.datetime.now()
print("Starting CSP execution at: ", start)

# Load preprocessed data
df_graph = pl.scan_parquet(os.path.join(DATASET_PATH, "papers100M-bin/processed/graph.parquet"))
label_df = pl.read_parquet(os.path.join(DATASET_PATH, "papers100M-bin/processed/labels.parquet"))
edge_degree = pl.read_parquet(
    os.path.join(DATASET_PATH, "papers100M-bin/processed/edge_hist.parquet")
)
node_degree = pl.read_parquet(
    os.path.join(DATASET_PATH, "papers100M-bin/processed/node_hist.parquet")
)
print("Data loaded, time taken: ", datetime.datetime.now() - start)

# Split the labels into batches
inds = [[i + Bs * j for i in range(Bs)] for j in range((171 + Bs) // Bs)]

results = []
for i, act_inds in enumerate(inds):
    print(f"Processing batch {i + 1}/{len(inds)}, ellapsed time: ", datetime.datetime.now() - start)
    train_label = (
        label_df.filter(pl.col("split") == "train")
        .filter(pl.col("label").is_in(act_inds))
        .select(
            "nodeId", pl.lit(1).cast(pl.Float32).alias("score"), pl.col("label").alias("target")
        )
    )  # filter out labels not in training set - only these labels are propagated by CSP
    stage1_cnt = (
        pl.LazyFrame(train_label)
        .join(df_graph, on="nodeId")
        .group_by("edgeId", "target")
        .agg(pl.sum("score"))
        .collect()
    )  # Count the number of times each edge is connected to a target label
    stage1_output = stage1_cnt.join(edge_degree, on="edgeId").select(
        "edgeId", "target", (pl.col("score") / pl.col("edge_degree")).alias("edge_score")
    )  # Normalize the edge scores - calculating average using edge degree instead of windowing over all edges
    stage2_cnt = (
        pl.LazyFrame(stage1_output)
        .join(df_graph, on="edgeId")
        .group_by("nodeId", "target")
        .agg(pl.sum("edge_score"))
        .collect()
    )  # Count the number of times each node is connected to a target label
    stage2_output = node_degree.join(stage2_cnt, on="nodeId").select(
        "nodeId", "target", (pl.col("edge_score") / pl.col("node_degree")).alias("node_score")
    )  # Normalize the node scores - calculating average using node degree instead of windowing over all nodes
    predictions = stage2_output.select(
        "nodeId",
        "node_score",
        pl.col("node_score").rank(descending=True, method="random").over("nodeId").alias("rank"),
        "target",
    ).filter(
        pl.col("rank") == 1
    )  # Select label with highest score for each node
    results.append(predictions)  # append batch results
merged_results = (  # merge results from all batches
    pl.concat(results)
    .select(
        "nodeId",
        "node_score",
        pl.col("node_score").rank(descending=True, method="random").over("nodeId").alias("rank"),
        "target",
    )
    .filter(pl.col("rank") == 1)
)
result = label_df.join(merged_results, on="nodeId", how="left")
accuracy = (  # calculate accuracy for all splits
    result.select("split", (pl.col("target") == pl.col("label")).cast(pl.Float32).alias("hit"))
    .fill_null(0)
    .group_by("split")
    .agg(pl.mean("hit"))
)
print(accuracy)
print("CSP execution done, time taken: ", datetime.datetime.now() - start)
