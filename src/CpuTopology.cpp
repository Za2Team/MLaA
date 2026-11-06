#include <mlaa/CpuTopology.hpp>
#include <thread>
#include <cstdint>
#include <cstddef>

#if defined(_MSC_VER)
    #include <intrin.h>
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #include <cpuid.h>
#endif

#if defined(__linux__)
    #include <unistd.h>
    #include <fstream>
    #include <string>
#elif defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #define NOMINMAX
    #include <windows.h>
#endif

namespace mlaa {
namespace {

struct CpuidRegs {
    std::uint32_t eax{0};
    std::uint32_t ebx{0};
    std::uint32_t ecx{0};
    std::uint32_t edx{0};
};

[[nodiscard]] inline CpuidRegs query_cpuid(std::uint32_t leaf, std::uint32_t subleaf = 0) noexcept {
    CpuidRegs r;
#if defined(_MSC_VER)
    int regs[4];
    __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
    r.eax = static_cast<std::uint32_t>(regs[0]);
    r.ebx = static_cast<std::uint32_t>(regs[1]);
    r.ecx = static_cast<std::uint32_t>(regs[2]);
    r.edx = static_cast<std::uint32_t>(regs[3]);
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    __cpuid_count(leaf, subleaf, r.eax, r.ebx, r.ecx, r.edx);
#endif
    return r;
}

[[nodiscard]] inline std::uint64_t read_xcr0() noexcept {
#if defined(_MSC_VER)
    return _xgetbv(0);
#elif defined(__GNUC__) || defined(__clang__)
    std::uint32_t eax = 0, edx = 0;
    __asm__ volatile(".byte 0x0f, 0x01, 0xd0" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<std::uint64_t>(edx) << 32) | eax;
#else
    return 0;
#endif
}

inline void os_cache_fallback(CpuTopology& topo) noexcept {
#if defined(__linux__)
    if (topo.line_size == 0) {
        long lsize = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
        topo.line_size = (lsize > 0) ? static_cast<std::size_t>(lsize) : 64;
    }
    if (topo.l1d_size == 0) {
        long sz = sysconf(_SC_LEVEL1_DCACHE_SIZE);
        if (sz > 0) topo.l1d_size = static_cast<std::size_t>(sz);
    }
    if (topo.l2_size == 0) {
        long sz = sysconf(_SC_LEVEL2_CACHE_SIZE);
        if (sz > 0) topo.l2_size = static_cast<std::size_t>(sz);
    }
    if (topo.l3_size == 0) {
        long sz = sysconf(_SC_LEVEL3_CACHE_SIZE);
        if (sz > 0) topo.l3_size = static_cast<std::size_t>(sz);
    }
#elif defined(_WIN32)
    DWORD len = 0;
    GetLogicalProcessorInformation(nullptr, &len);
    if (GetLastError() == ERROR_INSUFFICIENT_BUFFER && len > 0) {
        std::size_t count = len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
        auto* buffer = static_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION*>(::operator new(len, std::nothrow));
        if (buffer) {
            if (GetLogicalProcessorInformation(buffer, &len)) {
                for (std::size_t i = 0; i < count; ++i) {
                    if (buffer[i].Relationship == RelationCache) {
                        const auto& cache = buffer[i].Cache;
                        if (topo.line_size == 0 && cache.LineSize > 0) topo.line_size = cache.LineSize;
                        if (cache.Level == 1 && (cache.Type == CacheData || cache.Type == CacheUnified)) {
                            if (topo.l1d_size == 0) topo.l1d_size = cache.Size;
                        } else if (cache.Level == 2) {
                            if (topo.l2_size == 0) topo.l2_size = cache.Size;
                        } else if (cache.Level == 3) {
                            if (topo.l3_size == 0) topo.l3_size = cache.Size;
                        }
                    }
                }
            }
            ::operator delete(buffer);
        }
    }
#endif
    if (topo.line_size == 0) topo.line_size = 64;
}

#if defined(_WIN32)
inline std::size_t query_win32_physical_cores() noexcept {
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (GetLastError() == ERROR_INSUFFICIENT_BUFFER && len > 0) {
        auto* buf = static_cast<char*>(::operator new(len, std::nothrow));
        if (buf) {
            std::size_t cores = 0;
            if (GetLogicalProcessorInformationEx(RelationProcessorCore, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf), &len)) {
                DWORD offset = 0;
                while (offset < len) {
                    auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf + offset);
                    if (info->Relationship == RelationProcessorCore) {
                        ++cores;
                    }
                    offset += info->Size;
                }
            }
            ::operator delete(buf);
            if (cores > 0) return cores;
        }
    }
    return 0;
}
#endif

} // namespace

const CpuTopology& CpuTopology::get() noexcept {
    static const CpuTopology instance = []() noexcept {
        CpuTopology topo;
        topo.line_size = 64;

        topo.logical_cores = std::thread::hardware_concurrency();
        if (topo.logical_cores == 0) topo.logical_cores = 1;

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        const auto leaf0 = query_cpuid(0, 0);
        const std::uint32_t max_basic = leaf0.eax;
        const bool is_amd = (leaf0.ebx == 0x68747541 && leaf0.edx == 0x69746e65 && leaf0.ecx == 0x444d4163);

        const auto leaf_ext0 = query_cpuid(0x80000000, 0);
        const std::uint32_t max_ext = leaf_ext0.eax;

        // (SIMD: AVX, AVX2, AVX-512) 
        if (max_basic >= 1) {
            const auto leaf1 = query_cpuid(1, 0);
            
            const bool osxsave_supported = (leaf1.ecx & (1U << 27)) != 0;
            const bool cpu_has_avx       = (leaf1.ecx & (1U << 28)) != 0;
            const bool cpu_has_fma       = (leaf1.ecx & (1U << 12)) != 0;

            bool os_avx_enabled = false;
            bool os_avx512_enabled = false;

            if (osxsave_supported) {
                const std::uint64_t xcr0 = read_xcr0();
                os_avx_enabled = (xcr0 & 0x6ULL) == 0x6ULL;
                os_avx512_enabled = os_avx_enabled && ((xcr0 & 0xE0ULL) == 0xE0ULL);
            }

            if (cpu_has_avx && os_avx_enabled) {
                topo.has_avx = true;
                topo.has_fma = cpu_has_fma;

                if (max_basic >= 7) {
                    const auto leaf7 = query_cpuid(7, 0);

                    // AVX2 EBX
                    if (leaf7.ebx & (1U << 5)) {
                        topo.has_avx2 = true;
                    }

                    if (os_avx512_enabled) {
                        topo.has_avx512f  = (leaf7.ebx & (1U << 16)) != 0; // Foundation
                        topo.has_avx512dq = (leaf7.ebx & (1U << 17)) != 0; // DQ
                        topo.has_avx512cd = (leaf7.ebx & (1U << 28)) != 0; // CD
                        topo.has_avx512bw = (leaf7.ebx & (1U << 30)) != 0; // BW
                        topo.has_avx512vl = (leaf7.ebx & (1U << 31)) != 0; // VL
                    }
                }
            }
        }

        if (topo.has_avx512f) {
            topo.highest_simd = SimdLevel::AVX512;
        } else if (topo.has_avx2) {
            topo.highest_simd = SimdLevel::AVX2;
        } else if (topo.has_avx) {
            topo.highest_simd = SimdLevel::AVX;
        } else {
#if defined(__x86_64__) || defined(_M_X64)
            topo.highest_simd = SimdLevel::SSE2; 
#else
            topo.highest_simd = SimdLevel::None;
#endif
        }

        // (Physical Cores) 
        std::uint32_t threads_per_core = 1;
        if (!is_amd && max_basic >= 0x0B) {
            // Intel Leaf 0x0B, subleaf 0 (SMT Level)
            const auto r = query_cpuid(0x0B, 0);
            threads_per_core = r.ebx & 0xFFFFU;
        } else if (is_amd && max_ext >= 0x8000001E) {
            // AMD Leaf 0x8000001E
            const auto r = query_cpuid(0x8000001E, 0);
            threads_per_core = ((r.ebx >> 8) & 0xFFU) + 1;
        }

        if (threads_per_core == 0) threads_per_core = 1;
        topo.physical_cores = topo.logical_cores / threads_per_core;
        if (topo.physical_cores == 0) topo.physical_cores = 1;

#if defined(_WIN32)
        std::size_t win_cores = query_win32_physical_cores();
        if (win_cores > 0) topo.physical_cores = win_cores;
#endif

        std::uint32_t cache_leaf = 0;
        if (is_amd && max_ext >= 0x8000001D) {
            cache_leaf = 0x8000001D;
        } else if (!is_amd && max_basic >= 4) {
            cache_leaf = 0x04;
        }

        if (cache_leaf != 0) {
            for (std::uint32_t i = 0;; ++i) {
                const auto r = query_cpuid(cache_leaf, i);
                const std::uint32_t cache_type = r.eax & 0x1FU;
                if (cache_type == 0) break;

                const std::uint32_t level = (r.eax >> 5) & 0x7U;
                const std::size_t line = (r.ebx & 0xFFFU) + 1;
                const std::size_t part = ((r.ebx >> 12) & 0x3FFU) + 1;
                const std::size_t ways = ((r.ebx >> 22) & 0x3FFU) + 1;
                const std::size_t sets = static_cast<std::size_t>(r.ecx) + 1;
                const std::size_t sz   = ways * part * line * sets;

                topo.line_size = line;
                if (level == 1 && (cache_type == 1 || cache_type == 3)) {
                    topo.l1d_size = sz;
                } else if (level == 2) {
                    topo.l2_size = sz;
                } else if (level == 3) {
                    topo.l3_size = sz;
                }
            }
        } else if (is_amd) {
            if (max_ext >= 0x80000005) {
                const auto r5 = query_cpuid(0x80000005, 0);
                topo.l1d_size = static_cast<std::size_t>(r5.ecx >> 24) * 1024;
                topo.line_size = static_cast<std::size_t>(r5.ecx & 0xFF);
            }
            if (max_ext >= 0x80000006) {
                const auto r6 = query_cpuid(0x80000006, 0);
                topo.l2_size = static_cast<std::size_t>(r6.ecx >> 16) * 1024;
                topo.l3_size = static_cast<std::size_t>((r6.edx >> 18) & 0x3FFFU) * 512 * 1024;
            }
        }
#endif

        os_cache_fallback(topo);

        return topo;
    }();

    return instance;
}

} // namespace mlaa