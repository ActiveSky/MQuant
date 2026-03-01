# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
from .log2 import Log2Quantizer
from .non_uniform import NonUniformQuantizer
from .uniform import UniformQuantizer

str2quantizer = {
    "uniform": UniformQuantizer,
    "log2": Log2Quantizer,
    "non_uniform": NonUniformQuantizer,
}


def build_quantizer(quantizer_str, bit_type, observer, module_type):
    quantizer = str2quantizer[quantizer_str]
    return quantizer(bit_type, observer, module_type)
