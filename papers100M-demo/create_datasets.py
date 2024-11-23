import datetime
import gzip
import os

import numpy as np
import polars as pl

DATASET_PATH = "data"
os.makedirs(os.path.join(DATASET_PATH, "papers100M-bin/processed"), exist_ok=True)

start_time = datetime.datetime.now()
print("Preprocessing data for papers100M-bin at: ", start_time)


# Load information about nodes (splits and labels)
def load_split(file_path):
    with gzip.open(file_path) as f:
        return [int(i) for i in f.read().split()]


train_inds = load_split(os.path.join(DATASET_PATH, "papers100M-bin/split/time/train.csv.gz"))
valid_inds = load_split(os.path.join(DATASET_PATH, "papers100M-bin/split/time/valid.csv.gz"))
test_inds = load_split(os.path.join(DATASET_PATH, "papers100M-bin/split/time/test.csv.gz"))
split_df = pl.concat(
    [
        pl.DataFrame({"nodeId": train_inds}, schema={"nodeId": pl.UInt32}).with_columns(
            split=pl.lit("train")
        ),
        pl.DataFrame({"nodeId": valid_inds}, schema={"nodeId": pl.UInt32}).with_columns(
            split=pl.lit("valid")
        ),
        pl.DataFrame({"nodeId": test_inds}, schema={"nodeId": pl.UInt32}).with_columns(
            split=pl.lit("test")
        ),
    ]
)
npz_label_path = os.path.join(DATASET_PATH, "papers100M-bin/raw/node-label.npz")
with np.load(npz_label_path) as data:
    labels = data["node_label"].flatten()
label_df = (
    pl.DataFrame(data={"label": labels}, schema={"label": pl.Float32})
    .with_row_index("nodeId")
    .join(split_df, on="nodeId")
)
label_df.write_parquet(os.path.join(DATASET_PATH, "papers100M-bin/processed/labels.parquet"))
print(f"Writen split and labels to {DATASET_PATH}/papers100M-bin/processed/labels.parquet")

# Load the graph
npz_graph_path = os.path.join(DATASET_PATH, "papers100M-bin/raw/data.npz")
with np.load(npz_graph_path) as data:
    edge_list = data["edge_index"]

node_hist = np.unique(edge_list[0, :], return_counts=True)
pl.DataFrame(
    data={"nodeId": node_hist[0], "node_degree": node_hist[1]},
    schema={"nodeId": pl.UInt32, "node_degree": pl.UInt32},
).write_parquet(os.path.join(DATASET_PATH, "papers100M-bin/processed/node_hist.parquet"))
print(f"Writen node histogram to {DATASET_PATH}/papers100M-bin/processed/node_hist.parquet")

edge_hist = np.unique(edge_list[1, :], return_counts=True)
pl.DataFrame(
    data={"edgeId": edge_hist[0], "edge_degree": edge_hist[1]},
    schema={"edgeId": pl.UInt32, "edge_degree": pl.UInt32},
).write_parquet(os.path.join(DATASET_PATH, "papers100M-bin/processed/edge_hist.parquet"))
print(f"Writen edge histogram to {DATASET_PATH}/papers100M-bin/processed/edge_hist.parquet")

graph_df = pl.DataFrame(
    data={"nodeId": edge_list[0], "edgeId": edge_list[1]},
    schema={"nodeId": pl.UInt32, "edgeId": pl.UInt32},
)
graph_df.write_parquet(os.path.join(DATASET_PATH, "papers100M-bin/processed/graph.parquet"))
print(f"Writen graph to {DATASET_PATH}/papers100M-bin/processed/graph.parquet")

print("Preprocessing finished ellapsed time: ", datetime.datetime.now() - start_time)
