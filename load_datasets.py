"""Provides the datasets."""

import os
import pickle
from typing import Tuple

import numpy as np
import polars as pl
import sentencepiece as spm


def _split_labels(nodes: pl.DataFrame, preserve_label = True) -> pl.DataFrame:
    result = nodes.join(
        nodes
            .select("label")
            .unique(maintain_order = True)
            .with_row_index("label_index"),
        on = "label"
    )
    expand_res = result.pivot(
        values = "label_index",
        index = "nodeId",
        columns = "label_index",
        aggregate_function = "len",
        sort_columns = True,
    ).fill_null(
        pl.lit(0, dtype = pl.UInt32)
    )  # pola-rs/polars#13789
    expand_res = expand_res.rename(
        {c: f"y_{c}" for c in expand_res.columns if c.isnumeric()}
    )
    if preserve_label:
        return result.join(expand_res, on = "nodeId").select(
            "nodeId", pl.col("label_index").alias("label"), pl.col("^y_.*$")
        )
    else:
        return expand_res


def _load_cocitation_dataset(
    path: str,
    folder_name: str
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load a co-citation dataset.

    https://github.com/jianhao2016/AllSet
    """
    with open(
        os.path.join(path, "data/cocitation", folder_name, "labels.pickle"),
        "rb"
    ) as f:
        nodes_raw = pickle.load(f)

    nodes = pl.DataFrame(
        nodes_raw,
        schema = {"label": pl.Int64}
    ).with_row_index("nodeId")
    nodes = _split_labels(nodes)

    with open(
        os.path.join(path, "data/cocitation", folder_name, "hypergraph.pickle"),
        "rb"
    ) as f:
        edges_raw = pickle.load(f)

    edges = pl.DataFrame(
        [(s, list(t)) for (s, t) in list(edges_raw.items())],
        schema = {"source": pl.String, "targets": pl.List},
        orient = "row",
    )
    edges = (
        edges.join(
            edges
                .select("source")
                .unique(maintain_order = True)
                .with_row_index("source_id"),
            on = "source"
        )
        .drop("source")
        .explode("targets")
        .rename({"targets": "nodeId", "source_id": "edgeId"})
    )
    return nodes, edges


def _load_coauthorship_dataset(
    path: str, folder_name: str, with_features: bool = False
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load a co-authorship dataset.

    https://github.com/jianhao2016/AllSet
    """
    with open(
        os.path.join(path, "data/coauthorship", folder_name, "labels.pickle"),
        "rb"
    ) as f:
        nodes_raw = pickle.load(f)

    nodes = pl.DataFrame(
        nodes_raw,
        schema = {"label": pl.Int64}
    ).with_row_index("nodeId")
    nodes = _split_labels(nodes)

    with open(
        os.path.join(path, "data/coauthorship", folder_name, "hypergraph.pickle"),
        "rb"
    ) as f:
        edges_raw = pickle.load(f)

    edges = pl.DataFrame(
        list(edges_raw.items()),
        schema = {"author": pl.String, "papers": pl.List}
    )
    edges = (
        edges
        .join(
            edges
                .select("author")
                .unique(maintain_order = True)
                .with_row_index("author_id"),
            on = "author"
        )
        .drop("author")
        .explode("papers")
        .rename({"papers": "nodeId", "author_id": "edgeId"})
    )

    if not with_features:
        return nodes, edges

    with open(
        os.path.join(path, "data/coauthorship", folder_name, "features.pickle"),
        "rb"
    ) as f:
        features_raw = pickle.load(f)

    features = (
        pl.from_numpy(np.asarray(features_raw.todense()))
        .rename(lambda col: "x_" + col[7:])
        .with_row_index("nodeId")
    )
    nodes = nodes.join(features, on = "nodeId", how = "left")
    return nodes, edges


def _load_yelp_dataset(path: str) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load the Yelp dataset.

    https://github.com/jianhao2016/AllSet
    """
    nodes = (
        pl.read_csv(os.path.join(path, "data/yelp/yelp_restaurant_business_stars.csv"))
        .with_row_index("nodeId")
        .rename({"business_stars": "label"})
    )
    nodes = _split_labels(nodes)

    edges = (
        pl.read_csv(
            os.path.join(path, "data/yelp/yelp_restaurant_incidence_H.csv"),
            columns = ["he", "node"]
        )
        .select(pl.col("he").cast(pl.Int64), pl.col("node").cast(pl.Int64))
        .rename({"node": "nodeId", "he": "edgeId"})
    )
    return nodes, edges


def _load_movielens_dataset(
    path: str,
    edge_source: str
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load the movies dataset.

    https://grouplens.org/datasets/movielens/
    """
    nodes = (
        pl.read_csv(
            os.path.join(path, "data/ml-25m/movies.csv"),
            columns = ["movieId", "genres"]
        )
        .with_columns([pl.col("genres").str.split("|")])
        .rename({"movieId": "nodeId"})
    )
    node_genres = nodes.select("nodeId", "genres").explode("genres")
    node_genres = node_genres.join(
        node_genres
            .select("genres")
            .unique(maintain_order = True)
            .with_row_index("label"),
        on = "genres",
        how = "left"
    ).drop("genres")
    node_genres = _split_labels(node_genres)

    nodes = (
        nodes
        .drop("genres")
        .join(node_genres, on = "nodeId", how = "left")
        .drop("label")
        .unique(maintain_order = True)
    )
    edges = (
        pl.read_csv(os.path.join(path, "data/ml-25m/" + edge_source + ".csv"))
        .rename({"userId": "edgeId", "movieId": "nodeId"})
        .select(["nodeId", "edgeId"])
    )
    return nodes, edges


def _load_corona_dataset(
    path: str,
    num_edges = 1000
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load corona NLP dataset.

    https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification
    """
    df = pl.concat(
        [
            pl.read_csv(
                os.path.join(path, "data/Corona_NLP/Corona_NLP_train.csv"),
                separator = ",",
                encoding = "latin-1",
            ).with_columns(train_set = True),
            pl.read_csv(
                os.path.join(path, "data/Corona_NLP/Corona_NLP_test.csv"),
                separator = ",",
                encoding = "latin-1",
            ).with_columns(train_set = False),
        ]
    )
    corpus = "\n".join(
        df.filter(pl.col("train_set")).select(pl.col("OriginalTweet")).to_numpy().flatten()
    )
    with open("corona_corpus.txt", "w") as f:
        f.write(corpus)
    spm.SentencePieceTrainer.train(
        input = "corona_corpus.txt",
        model_prefix = f"corona_model_{num_edges}",
        vocab_size = num_edges,
        user_defined_symbols = [],
    )
    sp = spm.SentencePieceProcessor(model_file = f"corona_model_{num_edges}.model")
    edges = (
        df.with_row_index("nodeId")
        .select(
            "nodeId",
            pl.col("OriginalTweet")
            .map_elements(sp.encode, return_dtype = pl.List(pl.Int64))
            .alias("edgeId"),
        )
        .explode("edgeId")
    )
    nodes = (
        df.select("Sentiment")
        .unique(maintain_order = True)
        .with_row_index("label")
        .join(df.with_row_index("nodeId"), on = "Sentiment")
        .select("nodeId", "train_set", "label")
    )
    nodes = nodes.join(_split_labels(nodes, False), on = "nodeId")
    return nodes, edges


def load_dataset(
    name: str = "movielens-ratings", path: str = ".", with_features: bool = False
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Load a dataset."""
    if name == "Cora-CA":
        nodes, edges = _load_coauthorship_dataset(path, "cora", with_features)
    elif name == "DBLP-CA":
        nodes, edges = _load_coauthorship_dataset(path, "dblp", with_features)
    elif name == "Cora-CC":
        nodes, edges = _load_cocitation_dataset(path, "cora")
    elif name == "PubMed-CC":
        nodes, edges = _load_cocitation_dataset(path, "pubmed")
    elif name == "CiteSeer-CC":
        nodes, edges = _load_cocitation_dataset(path, "citeseer")
    elif name == "movielens-ratings":
        nodes, edges = _load_movielens_dataset(path, "ratings")
    elif name == "movielens-tags":
        nodes, edges = _load_movielens_dataset(path, "tags")
    elif name == "yelp":
        nodes, edges = _load_yelp_dataset(path)
    elif name == "corona":
        nodes, edges = _load_corona_dataset(path, num_edges = 1000)
    else:
        raise NotImplementedError
    nodes = nodes.with_columns(pl.col("nodeId").cast(pl.Int64))
    edges = edges.with_columns(
        pl.col("nodeId").cast(pl.Int64),
        pl.col("edgeId").cast(pl.Int64)
    )

    nodes = nodes.join(
        nodes.select("nodeId").unique(maintain_order = True).with_row_index("node_id"),
        on = "nodeId"
    )
    edges = edges.join(
        edges.select("edgeId").unique(maintain_order = True).with_row_index("edge_id"),
        on = "edgeId"
    ).join(
        nodes.select("node_id", "nodeId").unique(maintain_order = True),
        on = "nodeId"
    )
    nodes = (
        nodes
            .with_columns(nodeId = pl.col("node_id").cast(pl.Int64))
            .drop("node_id")
    )

    edges = (
        edges.with_columns(
            pl.col("node_id").cast(pl.Int64).alias("nodeId"),
            pl.col("edge_id").cast(pl.Int64).alias("edgeId"),
        )
        .drop("node_id")
        .drop("edge_id")
    )

    return nodes, edges
