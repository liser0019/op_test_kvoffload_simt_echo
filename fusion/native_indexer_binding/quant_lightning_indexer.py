"""Benchmark adapter for the pinned upstream Indexer."""
from cann_ops_transformer.op_builder import OpBuilder


class NativeIndexerBuilder(OpBuilder):
    def __init__(self):
        super().__init__("qli_native_indexer_baseline", category="attention")

    def sources(self):
        return ["csrc/attention/quant_lightning_indexer.cpp"]

    def schema(self):
        return []

    def register_meta(self):
        pass


quant_lightning_indexer_op_builder = NativeIndexerBuilder()


def quant_lightning_indexer(**kwargs):
    return quant_lightning_indexer_op_builder.load().run(**kwargs)
