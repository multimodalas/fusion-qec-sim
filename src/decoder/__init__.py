"""
Decoder algorithms for quantum LDPC codes.

Submodules
----------
gf2 : Dense GF(2) linear algebra (row-echelon, rank).
osd : Ordered Statistics Decoding (OSD-0, OSD-1, OSD-CS).
decimation : Deterministic threshold-based bit commitment.
"""

from .gf2 import binary_rank_dense, gf2_row_echelon
from .osd import osd0, osd1, osd_cs
from .decimation import decimate, decimation_round

__all__ = [
    "binary_rank_dense",
    "gf2_row_echelon",
    "osd0",
    "osd1",
    "osd_cs",
    "decimate",
    "decimation_round",
]
