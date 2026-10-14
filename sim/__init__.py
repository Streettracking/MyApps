"""FlyWire MB arena simulator.

Heavy modules stay lazy so the recognition trainer can be packaged without
loading the connectome runtime.
"""

__all__ = [
    "ACTIONS",
    "MushroomBodyRuntime",
    "SimConfig",
    "Simulator",
    "ArenaConfig",
    "ArenaWorld",
    "Zone",
]


def __getattr__(name):
    if name in ("ACTIONS", "MushroomBodyRuntime"):
        from . import mb_runtime

        return getattr(mb_runtime, name)
    if name in ("SimConfig", "Simulator"):
        from . import simulator

        return getattr(simulator, name)
    if name in ("ArenaConfig", "ArenaWorld", "Zone"):
        from . import world

        return getattr(world, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
