# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add additional decode_heads

from .aspp_head import ASPPHead
from .daformer_head import DAFormerHead
from .isa_head import ISAHead
from .segformer_head import SegFormerHead
from .sep_aspp_head import DepthwiseSeparableASPPHead

__all__ = [
    'ASPPHead',
    'DepthwiseSeparableASPPHead',
    'SegFormerHead',
    'DAFormerHead',
    'ISAHead',
]
