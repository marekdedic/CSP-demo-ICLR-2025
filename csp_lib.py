"""
This file contains the implementation of the CSP algorithm.

Supporting implementation to ICLR 2025 submission.

Example usage:

import numpy as np

from csp_lib import SkLearnCSP

H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # binary valued incidence matrix
y = np.array([1, 0, 1]) # binary valued labels
csp = SkLearnCSP() # model
csp.fit(H, y) # fit the model
csp.predict(H) # predict the labels
"""

import numpy as np
import numpy.typing as npt
import polars as pl


def csp_prepare_data(edge_df: pl.DataFrame, node_df: pl.DataFrame) -> pl.DataFrame:
    """
    Prepare the data for the CSP algorithm.

    Parameters
    ----------
        edge_df : pl.DataFrame

        node_df : pl.DataFrame

    Returns
    -------
        pl.DataFrame

    """
    for col in ["nodeId", "edgeId"]:
        if col not in edge_df.columns:
            raise ValueError(f"The edge_df must have a '{col}' column")
    for col in ["nodeId", "nodeProperty"]:
        if col not in node_df.columns:
            raise ValueError(f"The node_df must have a '{col}' column")
    data_df = edge_df.join(node_df, on = "nodeId")
    if "training_set" in data_df.columns:
        data_df = data_df.with_columns(
            nodeProperty = data_df["training_set"] * pl.col("nodeProperty")
        )
    return data_df

def _csp_stage1(edge_df: pl.DataFrame) -> pl.DataFrame:
    for col in ["nodeId", "edgeId", "nodeProperty"]:
        if col not in edge_df.columns:
            raise ValueError(f"The edge_df must have a '{col}' column")
    return edge_df.with_columns(edgeProperty = pl.mean("nodeProperty").over("edgeId"))

def _csp_stage2(edge_df: pl.DataFrame) -> pl.DataFrame:
    for col in ["nodeId", "edgeId", "edgeProperty"]:
        if col not in edge_df.columns:
            raise ValueError(f"The edge_df must have a '{col}' column")
    return edge_df.with_columns(
        nodeProperty_updated = pl.mean("edgeProperty").over("nodeId")
    )

def csp_layer(edge_df: pl.DataFrame, alpha_prime: float = 1) -> pl.DataFrame:
    """
    Run a single layer of the CSP algorithm. Basic variant.

    Parameters
    ----------
        edge_df : pl.DataFrame
            A polars DataFrame with columns 'nodeId', 'edgeId', 'nodeProperty' defining
            the underlying hyper-graph (bipartite graph) and the labels of the nodes.
        alpha_prime : float
            default = 1
            An optional hyper-parameter of the CSP algorithm. It is a float between 0
            and 1. It controls the ratio between the current label and the updated
            label. In the default case, the current label is completely replaced by the
            updated label.

    Returns
    -------
        pl.DataFrame
            A polars DataFrame with columns 'nodeId', 'edgeId', 'edgeProperty' that
            contains the aggregated information for each edge and the 'nodeProperty'
            column contains the updated labels of the nodes.

    """
    for col in ["nodeId", "edgeId", "nodeProperty"]:
        if col not in edge_df.columns:
            raise ValueError(f"The edge_df must have a '{col}' column")
    edge_update = _csp_stage1(edge_df)
    node_update = _csp_stage2(edge_update)
    return node_update.select(
        "nodeId",
        "edgeId",
        "edgeProperty",
        (
            pl.col("nodeProperty") * (1 - alpha_prime)
            + (alpha_prime) * pl.col("nodeProperty_updated")
        ).alias("nodeProperty"),
    )

def _assert_is_binary(vector):
    """Check if the vector is binary."""
    if not np.array_equal(vector, vector.astype(bool)):
        raise ValueError("The vector must be binary")

class SkLearnCSP:
    """A scikit-learn-style interface for CSP."""

    def __init__(self, layers: int = 1, alpha_prime: float = 1):
        """Initialize the model hyperparameters."""
        self.layers = layers
        self.alpha_prime = alpha_prime

    def __get_dataframe_from_incidence_matrix(self, H: npt.ArrayLike) -> pl.DataFrame:
        edge_list = []
        zero_inds = np.flatnonzero(np.sum(H, axis = 1) == 0)
        incidence_matrix = np.delete(H, zero_inds, axis = 0)
        edge_list = [
            (i, j)
            for i in range(incidence_matrix.shape[0])
            for j in range(incidence_matrix.shape[1])
            if incidence_matrix[i, j] == 1
        ]
        return (
            pl.DataFrame(edge_list)
            .transpose()
            .select(
                pl.col("column_0").cast(pl.Int64).alias("nodeId"),
                pl.col("column_1").cast(pl.Int64).alias("edgeId"),
            )
        )

    def fit(self, H: npt.NDArray, y: npt.NDArray) -> None:
        """Train a model on a hypergraph with training labels."""
        _assert_is_binary(H)
        if H.shape[0] != len(y):
            raise ValueError("H and y must have the same number of rows")
        edge_df = self.__get_dataframe_from_incidence_matrix(H)
        node_df = pl.DataFrame({"nodeProperty": y}).with_columns(
            nodeId = pl.int_range(pl.len(), dtype = pl.Int64)
        )
        data_df = csp_prepare_data(edge_df, node_df)
        for _ in range(self.layers - 1):
            data_df = csp_layer(data_df, alpha_prime = self.alpha_prime)
        self.model = _csp_stage1(data_df).select("edgeId", "edgeProperty").unique()

    def predict(self, H: npt.ArrayLike) -> npt.ArrayLike:
        """Apply a trained CSP model to a hypergraph."""
        _assert_is_binary(H)
        return (
            self.__get_dataframe_from_incidence_matrix(H)
            .join(self.model, on = "edgeId")
            .group_by("nodeId")
            .agg(pl.mean("edgeProperty"))
            .sort(by = "nodeId")
            .to_numpy()[:, 1]
        )
