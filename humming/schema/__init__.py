from humming.schema.awq import AWQWeightSchema
from humming.schema.base import BaseInputSchema, BaseWeightSchema
from humming.schema.bitnet import BitnetWeightSchema
from humming.schema.compressed_tensors import (
    CompressedTensorsWeightSchema,
    CompressedTensorsInputSchema,
)
from humming.schema.fp8 import Fp8WeightSchema, Fp8InputSchema
from humming.schema.gptq import GPTQWeightSchema
from humming.schema.humming import HummingInputSchema, HummingWeightSchema
from humming.schema.modelopt import ModeloptWeightSchema, ModeloptInputSchema
from humming.schema.mxfp4 import Mxfp4WeightSchema

WEIGHT_SCHEMA_MAP: dict[str, type[BaseWeightSchema]] = {
    "awq": AWQWeightSchema,
    "bitnet": BitnetWeightSchema,
    "compressed-tensors": CompressedTensorsWeightSchema,
    "fp8": Fp8WeightSchema,
    "gptq": GPTQWeightSchema,
    "humming": HummingWeightSchema,
    "modelopt": ModeloptWeightSchema,
    "mxfp4": Mxfp4WeightSchema,
}

INPUT_SCHEMA_MAP: dict[str, type[BaseInputSchema]] = {
    "compressed-tensors": CompressedTensorsInputSchema,
    "fp8": Fp8InputSchema,
    "humming": HummingInputSchema,
    "modelopt": ModeloptInputSchema,
}
