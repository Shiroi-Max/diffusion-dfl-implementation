"""
YAML Extraction Utilities

This module provides utility functions for loading and parsing YAML files
containing decentralized topology and label distribution information.

Functions
---------
- extract_neighbours_yaml(file_path): Extracts adjacency lists and returns a boolean matrix.
- extract_labels_yaml(file_path): Extracts the list of labels assigned to each node.
"""

from typing import List

import yaml


def extract_neighbours_yaml(file_path: str) -> List[List[bool]]:
    """
    Load the neighbourhood adjacency list from a YAML file and return a symmetric boolean matrix.

    Each node is assumed to be connected to itself and optionally to other nodes
    as specified in the YAML.

    Parameters
    ----------
    file_path : str
        Path to the YAML file defining neighbours for each node.

    Returns
    -------
    List[List[bool]]
        A square matrix where matrix[i][j] is True if node i is connected to node j.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    num_nodes = len(raw)
    neighbours = []

    for i in range(num_nodes):
        row = [False] * num_nodes
        row[i] = True  # self-loop
        for j in raw.get(i, []):
            row[j] = True
        neighbours.append(row)

    return neighbours


def extract_labels_yaml(file_path: str) -> List[List[int]]:
    """
    Load label distribution per node from a YAML file.

    Parameters
    ----------
    file_path : str
        Path to the YAML file containing labels assigned to each node.

    Returns
    -------
    List[List[int]]
        A list where each entry is a list of label indices assigned to a node.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    num_nodes = len(raw)
    return [raw.get(i, []) for i in range(num_nodes)]
