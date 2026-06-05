from .base import BenchmarkAdapter, set_constitution_mode
from .mssbench import MSSBenchAdapter
from .mssembodied import MSSEmbodiedAdapter
from .siuo import SIUOAdapter
from .beavertails import BeaverTailsAdapter
from .spavl import SPAVLAdapter
from .vlguard import VLGuardAdapter
from .vlsbench import VLSBenchAdapter

ADAPTER_REGISTRY = {
    "mssbench": MSSBenchAdapter,
    "mssembodied": MSSEmbodiedAdapter,
    "siuo": SIUOAdapter,
    "beavertails": BeaverTailsAdapter,
    "spavl": SPAVLAdapter,
    "vlguard": VLGuardAdapter,
    "vlsbench": VLSBenchAdapter,
}


def get_adapter(benchmark_name: str) -> BenchmarkAdapter:
    """根据 benchmark 名称获取对应的数据适配器实例"""
    if benchmark_name not in ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown benchmark: {benchmark_name}. "
            f"Available: {list(ADAPTER_REGISTRY.keys())}"
        )
    return ADAPTER_REGISTRY[benchmark_name]()
