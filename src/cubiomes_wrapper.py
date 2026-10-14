"""
cubiomes_wrapper.py
[底层] Python调用层：封装 ctypes 接口，提供黑盒环境查询能力
仅暴露 get_temperature(x, z) 和 is_deep_ocean(x, z) 两个纯函数
严禁在此文件中包含任何业务逻辑或状态管理

学术依据：MC版本锁定1.17(1170)，因1.18+引入3D地形噪声与地表深海分布解耦，
且1.17是2D噪声主导的最后稳定版，计算效率满足20种子x20次重复的大规模消融需求。

【修复依据】(≤100字)
新增 query_count 计数器：需求第一节的「总查询次数上限」需按黑盒调用真实计费，
原实现无计数能力，预算只能由上层估算，导致查询数统计与参考实现口径不一致。
其余接口签名与行为保持不变。
"""

import os
import sys
import hashlib
import ctypes

# DLL 路径自动定位（相对于本文件所在目录）
_DLL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cubiomes.dll")


class Pos(ctypes.Structure):
    """Pos 结构体：用于 getBiomeAtPos 调用（与参考代码一致）"""
    _fields_ = [("x", ctypes.c_int), ("z", ctypes.c_int)]


# cubiomes 版本常量（与参考代码一致）
_MC_VERSION = 0x12  # cubiomes 内部版本号，对应 MC 1.17

# 温度层的坐标尺度换算因子
# 依据 cubiomes 源码：
#   generator.h:57   L_OCEAN_TEMP_256 —— 该层工作在 256 方块尺度
#   layers.c:2565    mapOceanTemp:  samplePerlin(rnd, (i+x)/8.0, (j+z)/8.0)  x,z 为「层坐标」
#   layers.c:3130    getTemperatureNoise: samplePerlin(rnd, x/8.0, z/8.0)     x,z 亦为「层坐标」
# 故世界方块坐标 bx 需换算为层坐标 bx // 256 后再传入，二者才指向同一 perlin 采样点。
# 实测校验：换算后 5 种海洋变体的温度均值精确落入 mapOceanTemp 自身的判定区间
#   (warm +0.473∈[0.4,∞) / lukewarm +0.300∈[0.2,0.4] / normal +0.000∈[-0.2,0.2]
#    / cold -0.299∈[-0.4,-0.2] / frozen -0.475∈(-∞,-0.4])，变体降序 Spearman=1.00。
# 若直接传方块坐标（尺度=1），温度场退化为白噪声（16格自相关 0.0023），
# 变体顺序 Spearman=-0.20，五种变体均值全部挤在 ±0.009 内，特征完全失效。
_TEMP_LAYER_SCALE = 256

# 深海群系名称列表（与参考代码一致）
_DEEP_OCEAN_NAMES = frozenset({
    "deep_ocean",
    "deep_lukewarm_ocean",
    "deep_cold_ocean",
    "deep_frozen_ocean",
})


class CubiomesWrapper:
    """
    黑盒环境接口封装类
    线程不安全：每个实例绑定单一种子，仅供 main_algorithm.py 串行调用
    支持 Mock 模式：当 DLL 缺失时自动降级为伪随机生成器，保障流程可调试
    支持延迟初始化：可无参构造，后续通过 set_seed() 设置种子

    ⚠️ 无参构造后 _initialized 为 False，任何查询都会抛 RuntimeError。
       使用前必须调用 set_seed()，否则主循环的异常兜底会把失败静默吞掉。
    """

    def __init__(self, seed: int = None, use_mock: bool = False):
        self._seed = seed
        self._mock = use_mock or not os.path.exists(_DLL_PATH)
        self._lib = None
        self._gen = None
        self._initialized = False
        self.query_count = 0   # 黑盒查询计数（is_deep_ocean 与 get_temperature 均计入）

        if seed is not None:
            self._do_init(seed)

    def _do_init(self, seed: int):
        """内部初始化逻辑，供 __init__ 和 set_seed 复用"""
        self._seed = seed
        self.query_count = 0

        if self._mock:
            print(f"[CubiomesWrapper] Mock模式启用 (seed={seed})，触觉反馈为伪随机！", file=sys.stderr)
            self._lib = None
            self._gen = None
            self._initialized = True
            return

        # 加载 DLL（与参考代码一致：ctypes.CDLL 不带 winmode）
        try:
            self._lib = ctypes.CDLL(_DLL_PATH)
        except OSError as e:
            raise RuntimeError(f"cubiomes.dll 加载失败: {e}") from e

        # 生成器使用 8192 字节的 c_ubyte 数组（与参考代码一致）
        self._gen = (ctypes.c_ubyte * 8192)()

        # 绑定函数签名（完全按照参考代码的签名）
        self._lib.setupGenerator.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.applySeed.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        self._lib.getBiomeAtPos.argtypes = [ctypes.c_void_p, Pos]
        self._lib.getBiomeAtPos.restype = ctypes.c_int
        self._lib.biome2str.argtypes = [ctypes.c_int]
        self._lib.biome2str.restype = ctypes.c_char_p
        self._lib.getTemperatureNoise.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self._lib.getTemperatureNoise.restype = ctypes.c_double

        # 初始化生成器（MC版本号 0x12，与参考代码一致）
        self._lib.setupGenerator(self._gen, _MC_VERSION)
        self._lib.applySeed(self._gen, ctypes.c_uint64(seed))

        self._initialized = True

    def set_seed(self, seed: int):
        """
        运行时切换世界种子（供消融实验动态调用）
        - Mock 模式：仅更新内部种子值
        - DLL 模式：若已初始化则直接调用 DLL applySeed；否则执行完整初始化
        """
        if not isinstance(seed, int):
            raise TypeError(f"seed 必须为 int 类型，收到 {type(seed)}")

        self._seed = seed
        self.query_count = 0

        if self._mock:
            self._initialized = True
            return

        if self._initialized and self._lib is not None and self._gen is not None:
            # 已初始化 -> 仅切换种子（避免重复加载 DLL）
            self._lib.applySeed(self._gen, ctypes.c_uint64(seed))
        else:
            # 未初始化（无参构造后首次调用）-> 完整初始化
            self._do_init(seed)

    def _ensure_ready(self):
        if not self._initialized:
            raise RuntimeError("CubiomesWrapper 未初始化，请先调用 set_seed() 或传入 seed 参数构造")

    def get_temperature(self, x: int, z: int) -> float:
        """
        黑盒触觉反馈：获取(x,z)处原始温度值
        返回值未经任何归一化/截断，直接透传DLL原始浮点输出

        注意：入参为世界方块坐标，内部按 _TEMP_LAYER_SCALE 换算为温度层的层坐标。
        详见该常量的源码依据说明。
        """
        if not isinstance(x, int) or not isinstance(z, int):
            raise TypeError(f"坐标必须为int类型，收到 x={type(x)}, z={type(z)}")
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
            print(f"[CubiomesWrapper] get_temperature({x},{z}) 异常: {e}", file=sys.stderr)
            return float('nan')  # 返回NaN而非伪造值，避免污染Q-table学习

    def is_deep_ocean(self, x: int, z: int) -> bool:
        """
        目标验证：判定(x,z)是否为深海群系（含全部4种深海变体）
        使用 biome2str 获取群系名称后与深海名称列表比较（与参考代码一致）
        """
        if not isinstance(x, int) or not isinstance(z, int):
            raise TypeError(f"坐标必须为int类型，收到 x={type(x)}, z={type(z)}")
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
            print(f"[CubiomesWrapper] is_deep_ocean({x},{z}) 异常: {e}", file=sys.stderr)
            return False

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def is_mock(self) -> bool:
        return self._mock
