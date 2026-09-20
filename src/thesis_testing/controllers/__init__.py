"""Controller protocol and concrete static phenotype."""

from thesis_testing.controllers.base import Controller
from thesis_testing.controllers.genome import (
    decode_weight_genome,
    encode_weight_matrices,
    validate_weight_genome,
)
from thesis_testing.controllers.nchl import NcHLController, nchl_delta
from thesis_testing.controllers.static_mlp import StaticMLP

__all__ = [
    "Controller",
    "NcHLController",
    "StaticMLP",
    "decode_weight_genome",
    "encode_weight_matrices",
    "nchl_delta",
    "validate_weight_genome",
]
