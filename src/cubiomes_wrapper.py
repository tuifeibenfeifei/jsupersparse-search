"""
cubiomes_wrapper.py
[Low level] Python call layer: wraps the ctypes interface and provides black-box environment querying
Exposes only the two pure functions get_temperature(x, z) and is_deep_ocean(x, z)
No business logic or state management is allowed in this file

Academic rationale: MC version is pinned to 1.17 (1170), because 1.18+ decoupled 3D terrain noise from surface deep-sea distribution,
and 1.17 is the last stable release dominated by 2D noise; its compute efficiency meets the large-scale ablation needs of 20 seeds x 20 repeats.

[Fix rationale] (<=100 words)
Added a query_count counter: the "total query budget" of Req. Sec. 1 must be billed by real black-box calls;
the original implementation had no counting, so the budget could only be estimated by the upper layer, making the query counts inconsistent with the reference implementation.
All other interface signatures and behaviors are unchanged.
"""

import os
import sys
import hashlib
import ctypes

# auto-locate the DLL path (relative to the directory of this file)
_DLL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cubiomes.dll")


class Pos(ctypes.Structure):
    """Pos struct: used for getBiomeAtPos calls (identical to the reference code)."""
    _fields_ = [("x", ctypes.c_int), ("z", ctypes.c_int)]


# cubiomes version constants (identical to the reference code)
_MC_VERSION = 0x12  # cubiomes internal version number, corresponds to MC 1.17

# coordinate-scale conversion factor for the temperature layer
# Based on the cubiomes source:
#   generator.h:57   L_OCEAN_TEMP_256 -- this layer works at a 256-block scale
#   layers.c:2565    mapOceanTemp:  samplePerlin(rnd, (i+x)/8.0, (j+z)/8.0)  x,z are "layer coordinates"
#   layers.c:3130    getTemperatureNoise: samplePerlin(rnd, x/8.0, z/8.0)     x,z are also "layer coordinates"
# So the world block coordinate bx must be converted to layer coordinate bx // 256 before being passed in, so that both point to the same perlin sample.
# Measured check: after conversion, the mean temperature of the 5 ocean variants falls exactly into mapOceanTemp's own decision bins
#   (warm +0.473∈[0.4,∞) / lukewarm +0.300∈[0.2,0.4] / normal +0.000∈[-0.2,0.2]
#    / cold -0.299 in [-0.4,-0.2] / frozen -0.475 in (-inf,-0.4]); Spearman=1.00 in descending variant order.
# If raw block coordinates are passed directly (scale=1), the temperature field degenerates to white noise (16-cell autocorrelation 0.0023),
# variant order Spearman=-0.20, all five variant means crowded within +/-0.009, and the feature completely fails.
_TEMP_LAYER_SCALE = 256

# deep-sea biome name list (identical to the reference code)
_DEEP_OCEAN_NAMES = frozenset({
    "deep_ocean",
    "deep_lukewarm_ocean",
    "deep_cold_ocean",
    "deep_frozen_ocean",
})


class CubiomesWrapper:
    """
    Black-box environment interface wrapper class
    Not thread-safe: each instance is bound to a single seed and is only called serially by main_algorithm.py
    Supports Mock mode: when the DLL is missing it automatically falls back to a pseudo-random generator so the pipeline stays debuggable
    Supports lazy initialization: can be constructed with no arguments, then the seed is set later via set_seed()

    [!] After no-arg construction _initialized is False, and any query raises RuntimeError.
       set_seed() must be called before use; otherwise the main loop's exception fallback silently swallows the failure.
    """

    def __init__(self, seed: int = None, use_mock: bool = False):
        self._seed = seed
        self._mock = use_mock or not os.path.exists(_DLL_PATH)
        self._lib = None
        self._gen = None
        self._initialized = False
        self.query_count = 0   # black-box query count (both is_deep_ocean and get_temperature count)

        if seed is not None:
            self._do_init(seed)

    def _do_init(self, seed: int):
        """Internal initialization logic, reused by __init__ and set_seed."""
        self._seed = seed
        self.query_count = 0

        if self._mock:
            print(f"[CubiomesWrapper] Mock mode enabled (seed={seed}); feedback is pseudo-random!", file=sys.stderr)
            self._lib = None
            self._gen = None
            self._initialized = True
            return

        # load the DLL (identical to the reference code: ctypes.CDLL without winmode)
        try:
            self._lib = ctypes.CDLL(_DLL_PATH)
        except OSError as e:
            raise RuntimeError(f"cubiomes.dll failed to load: {e}") from e

        # the generator uses an 8192-byte c_ubyte array (identical to the reference code)
        self._gen = (ctypes.c_ubyte * 8192)()

        # bind function signatures (exactly following the reference code's signatures)
        self._lib.setupGenerator.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.applySeed.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        self._lib.getBiomeAtPos.argtypes = [ctypes.c_void_p, Pos]
        self._lib.getBiomeAtPos.restype = ctypes.c_int
        self._lib.biome2str.argtypes = [ctypes.c_int]
        self._lib.biome2str.restype = ctypes.c_char_p
        self._lib.getTemperatureNoise.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self._lib.getTemperatureNoise.restype = ctypes.c_double

        # initialize the generator (MC version number 0x12, identical to the reference code)
        self._lib.setupGenerator(self._gen, _MC_VERSION)
        self._lib.applySeed(self._gen, ctypes.c_uint64(seed))

        self._initialized = True

    def set_seed(self, seed: int):
        """
        Switch the world seed at runtime (called dynamically by the ablation experiments)
        - Mock mode: only updates the internal seed value
        - DLL mode: if already initialized, call the DLL applySeed directly; otherwise perform full initialization
        """
        if not isinstance(seed, int):
            raise TypeError(f"seed must be of type int, got {type(seed)}")

        self._seed = seed
        self.query_count = 0

        if self._mock:
            self._initialized = True
            return

        if self._initialized and self._lib is not None and self._gen is not None:
            # already initialized -> only switch the seed (avoid reloading the DLL)
            self._lib.applySeed(self._gen, ctypes.c_uint64(seed))
        else:
            # not initialized (first call after no-arg construction) -> full initialization
            self._do_init(seed)

    def _ensure_ready(self):
        if not self._initialized:
            raise RuntimeError("CubiomesWrapper is not initialized; please call set_seed() first or construct with a seed argument")

    def get_temperature(self, x: int, z: int) -> float:
        """
        Black-box feedback: get the raw temperature value at (x,z)
        The return value is passed through the DLL's raw float output without any normalization/clipping

        Note: the input is world block coordinates, internally converted to temperature-layer coordinates by _TEMP_LAYER_SCALE.
        See the source-rationale note for that constant.
        """
        if not isinstance(x, int) or not isinstance(z, int):
            raise TypeError(f"coordinates must be of type int, got x={type(x)}, z={type(z)}")
        self._ensure_ready()
        self.query_count += 1

        if self._mock:
            h = hashlib.md5(f"temp_{self._seed}_{x}_{z}".encode()).hexdigest()
            return (int(h[:8], 16) / 0xFFFFFFFF) * 2.0 - 1.0

        try:
            lx = x // _TEMP_LAYER_SCALE
            lz = z // _TEMP_LAYER_SCALE
            return float(self._lib.getTemperatureNoise(self._gen, ctypes.c_int(lx), ctypes.c_int(lz)))
        except Exception as e:
            print(f"[CubiomesWrapper] get_temperature({x},{z}) error: {e}", file=sys.stderr)
            return float('nan')  # return NaN rather than a fabricated value, to avoid polluting Q-table learning

    def is_deep_ocean(self, x: int, z: int) -> bool:
        """
        Target verification: determine whether (x,z) is a deep-sea biome (including all 4 deep-sea variants)
        Uses biome2str to get the biome name and compares it against the deep-sea name list (identical to the reference code)
        """
        if not isinstance(x, int) or not isinstance(z, int):
            raise TypeError(f"coordinates must be of type int, got x={type(x)}, z={type(z)}")
        self._ensure_ready()
        self.query_count += 1

        if self._mock:
            temp = self.get_temperature(x, z)
            h = hashlib.md5(f"deep_{self._seed}_{x}_{z}".encode()).hexdigest()
            prob = int(h[:8], 16) / 0xFFFFFFFF
            return temp < 0.2 and prob < 0.12

        try:
            pos = Pos(x, z)
            biome_id = self._lib.getBiomeAtPos(self._gen, pos)
            biome_name_ptr = self._lib.biome2str(biome_id)
            if biome_name_ptr:
                return biome_name_ptr.decode("utf-8") in _DEEP_OCEAN_NAMES
            return False
        except Exception as e:
            print(f"[CubiomesWrapper] is_deep_ocean({x},{z}) error: {e}", file=sys.stderr)
            return False

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def is_mock(self) -> bool:
        return self._mock
