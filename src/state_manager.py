"""
state_manager.py
 [State] State discretization encoder: maps (chunk region coords + temperature) to a Q-table-ready hashable key
 Core duties: state encoding and new-state detection (for the curiosity reward)
 Must never contain reward computation, search logic, or database operations

 [Design constraint 1] Req.txt Sec. 1 "no prior knowledge":
     temperature enters the model only as a state feature; no hand-crafted temperature ranges/thresholds are allowed.
     therefore this file uses no temp_min / temp_max binning; instead it quantizes to a fixed number of decimal places:
        round(temperature, precision)
     precision is a resolution (how many decimals to keep), not a temperature range boundary,
     making no assumption about "which temperature is closer to deep sea"; that is learned autonomously.

 [Design constraint 2] spatial granularity must be coarser than the temperature resolution (the most common pitfall in this file):
     the temperature layer resolution is 16 chunks (256 blocks); temperature is constant within one chunk.
     if the state cell is finer than or equal to 16 chunks, temperature is derivable by integer division of the chunk coordinates,
     becoming a deterministic function with zero contribution to the state space; the temperature on/off ablation would then yield identical results.
     hence spatial_granularity (default 32) forces one state cell to span multiple temperature cells.
"""

import math
from typing import Tuple, Optional


 # independent marker for missing temperature (query failure / NaN).
 # -1 is avoided: it lies inside the real temperature range and could collide with a genuine quantized bin.
TEMP_MISSING = None

 # spatial resolution of the temperature layer (in chunks) = 256 blocks / 16 blocks per chunk = 16 chunks.
 # source basis: cubiomes generator.h:57 (L_OCEAN_TEMP_256) and layers.c:3130 (x/8.0 sampling).
 # measured: the temperature changes every 16 chunks on average along x; 2000 samples inside one chunk are
 # perfectly constant (max temperature difference 0.000000).
TEMP_RESOLUTION_CHUNKS = 16

 # default state-space granularity: twice the temperature resolution.
 # granularity must be coarser than the temperature resolution for temperature to add information; if finer or equal,
 # temperature becomes a deterministic function of chunk coordinates, contributing nothing to the state space.
DEFAULT_SPATIAL_GRANULARITY = 2 * TEMP_RESOLUTION_CHUNKS


class StateManager:
    """
     State manager
     WARNING: not thread-safe; for serial use within a single episode only
     when temperature is off, the state is determined by the chunk region coordinates only; when on, the quantized temperature joins the encoding

     Curiosity-reward ordering constraint:
         the reward module must call is_new_state() before mark_visited()
         reversing the order makes the curiosity reward always 0

     [Relation between spatial granularity and temperature resolution - the most critical design constraint of this class]
         the temperature layer resolution is fixed at TEMP_RESOLUTION_CHUNKS (16) chunks.
         - when spatial_granularity <= 16: temperature is constant within a state cell, so temperature is a
           deterministic function of the chunk coordinates; (region_x, region_z, temp) maps one-to-one to (region_x, region_z),
           temperature contributes nothing to the state space, and the on/off ablation yields identical results.
         - when spatial_granularity > 16: one state cell spans multiple temperature cells, and temperature distinguishes
           different temperature bands inside the same region, adding real information.
         default is therefore 32 (= 2x the temperature resolution), with each state cell covering about 2x2 temperature cells.
    """

    def __init__(
        self,
        temperature_enabled: bool = True,
        temperature_precision: int = 2,
        spatial_granularity: int = DEFAULT_SPATIAL_GRANULARITY,
    ):
        """
        Args:
             temperature_enabled: whether temperature enters the state encoding
                 True = black-box sensory mode (the innovation baseline)
                 False = pure blind-search control mode (for ablations)
             temperature_precision: temperature quantization precision, i.e. decimals kept (default 2 = 0.01 resolution)
                 used only as a resolution; no temperature range, no prior knowledge.
                 Stage 4 can ablate different granularities (1=coarse, 2=medium, 3=fine)
             spatial_granularity: state-cell spatial granularity, in chunks (default 32)
                 one state cell covers spatial_granularity x spatial_granularity chunks.
                 WARNING: must be > TEMP_RESOLUTION_CHUNKS (16) for temperature to add information; see the class doc.
                 this is a resolution hyperparameter of the state space, unrelated to "which temperature is closer to deep sea",
                 it is not the prior knowledge banned by Req. Sec. 1; Stage 4 can sweep it.
        """
        self._validate_config(temperature_precision, spatial_granularity)
        self._temp_enabled = bool(temperature_enabled)
        self._precision = int(temperature_precision)
        self._granularity = int(spatial_granularity)
        self._visited_states: set = set()

    @staticmethod
    def _validate_config(temperature_precision, spatial_granularity) -> None:
        """Validate the quantization precision and spatial granularity configuration"""
        if (not isinstance(temperature_precision, int)
                or isinstance(temperature_precision, bool)
                or temperature_precision < 0):
            raise ValueError(
                f"temperature_precision must be a non-negative integer, got: {temperature_precision} "
                f"(type={type(temperature_precision).__name__})"
            )
        if (not isinstance(spatial_granularity, int)
                or isinstance(spatial_granularity, bool)
                or spatial_granularity <= 0):
            raise ValueError(
                f"spatial_granularity must be a positive integer, got: {spatial_granularity} "
                f"(type={type(spatial_granularity).__name__})"
            )

    def reconfigure(
        self,
        temperature_enabled: bool = None,
        temperature_precision: int = None,
        spatial_granularity: int = None,
    ) -> None:
        """
         Dynamically reset the state manager configuration (for Stage 4 ablations)
         WARNING: visited records are cleared so old and new configs never mix
        Args:
             a None argument keeps the current value unchanged
        """
        new_enabled = temperature_enabled if temperature_enabled is not None else self._temp_enabled
        new_precision = temperature_precision if temperature_precision is not None else self._precision
        new_gran = spatial_granularity if spatial_granularity is not None else self._granularity

        self._validate_config(new_precision, new_gran)

        self._temp_enabled = bool(new_enabled)
        self._precision = int(new_precision)
        self._granularity = int(new_gran)
        self._visited_states.clear()  # visited records must be cleared when the config changes

    def get_state(self, chunk_x: int, chunk_z: int, temperature: Optional[float]) -> Tuple:
        """
         Encode continuous inputs into a discrete state key
        Returns:
             a hashable tuple, usable directly as a Q-table dict key
             missing temperature encodes to the distinct marker None, never confused with a real bin, never fabricating feedback
        """
        if not isinstance(chunk_x, int) or isinstance(chunk_x, bool):
            raise TypeError(f"chunk_x must be int, got {type(chunk_x).__name__}")
        if not isinstance(chunk_z, int) or isinstance(chunk_z, bool):
            raise TypeError(f"chunk_z must be int, got {type(chunk_z).__name__}")

         # coarsening by spatial granularity: Python's // floors, valid for negative coords, no wraparound misalignment
        region_x = chunk_x // self._granularity
        region_z = chunk_z // self._granularity

        if self._temp_enabled:
            return (region_x, region_z, self._quantize_temp(temperature))
        return (region_x, region_z)

    def is_new_state(self, state: Tuple) -> bool:
        """
         Whether the state has never been visited (called by the curiosity reward module)
         WARNING: query-only, does not record; must be called before mark_visited()
        """
        return state not in self._visited_states

    def mark_visited(self, state: Tuple) -> None:
        """
         Mark the state as visited
         WARNING: must be called after is_new_state() and when the agent completes the state transition
        """
        self._visited_states.add(state)

    def reset(self) -> None:
        """
         Clear visited records while keeping the current configuration
         WARNING: must be called before each new episode so no history leaks (Req. Sec. 1 #3)
        """
        self._visited_states.clear()

    @property
    def temperature_enabled(self) -> bool:
        return self._temp_enabled

    @property
    def temperature_precision(self) -> int:
        """Current temperature quantization precision (decimal places kept)"""
        return self._precision

    @property
    def spatial_granularity(self) -> int:
        """Current state-cell spatial granularity (chunks). Must be > TEMP_RESOLUTION_CHUNKS for temperature to add information"""
        return self._granularity

    @property
    def temperature_resolution_chunks(self) -> int:
        """Temperature layer resolution (chunks); constant, equals 16"""
        return TEMP_RESOLUTION_CHUNKS

    @property
    def granularity_is_coarse_enough(self) -> bool:
        """Whether granularity is coarser than the temperature resolution (True means temperature contributes to the state space)"""
        return self._granularity > TEMP_RESOLUTION_CHUNKS

    @property
    def visited_count(self) -> int:
        """Number of unique states visited in the current episode"""
        return len(self._visited_states)

    def _quantize_temp(self, temperature: Optional[float]) -> Optional[float]:
        """
         Quantize a continuous temperature value into a discrete key
         None / NaN / non-numeric -> TEMP_MISSING (distinct marker; no fabricated feedback)

         Note: quantization only rounds to a fixed number of decimals; no truncation, no shift, no range setting,
         so no prior like "some temperatures are more likely to be deep sea" is encoded into the state space.
        """
        if temperature is None:
            return TEMP_MISSING
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            return TEMP_MISSING
        if math.isnan(temperature):
            return TEMP_MISSING

        return round(float(temperature), self._precision)
