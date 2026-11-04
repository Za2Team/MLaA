#include <mlaa/CpuTopology.hpp>
#include <cstdint>
#include <cstddef>

#if defined(_MSC_VER)
    #include <intrin.h>
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #include <cpuid.h>
#endif

#if defined(__linux__)
    #include <unistd.h>
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

// Fallback به سیستم‌عامل در صورتی که CPUID در هایپروایزر مسدود یا صفر باشد
inline void os_fallback(CpuTopology& topo) noexcept {
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
                        if (topo.line_size == 0 && cache.LineSize > 0) {
                            topo.line_size = cache.LineSize;
                        }
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
    if (topo.line_size == 0) topo.line_size = 64; // حداقل استاندارد x86
}

} // namespace

const CpuTopology& CpuTopology::get() noexcept {
    static const CpuTopology instance = []() noexcept {
        CpuTopology topo;
        topo.line_size = 64;

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        // 1. بررسی حداکثر Basic Leaf و نام وندور
        const auto leaf0 = query_cpuid(0, 0);
        const std::uint32_t max_basic = leaf0.eax;
        // AMD: EBX="Auth" (0x68747541), EDX="enti" (0x69746e65), ECX="cAMD" (0x444d4163)
        const bool is_amd = (leaf0.ebx == 0x68747541 && leaf0.edx == 0x69746e65 && leaf0.ecx == 0x444d4163);

        // 2. بررسی حداکثر Extended Leaf
        const auto leaf_ext0 = query_cpuid(0x80000000, 0);
        const std::uint32_t max_ext = leaf_ext0.eax;

        std::uint32_t cache_leaf = 0;
        if (is_amd && max_ext >= 0x8000001D) {
            cache_leaf = 0x8000001D; // برگه توپولوژی کش پردازنده‌های AMD Zen
        } else if (!is_amd && max_basic >= 4) {
            cache_leaf = 0x04;         // برگه توپولوژی کش پردازنده‌های Intel
        }

        if (cache_leaf != 0) {
            for (std::uint32_t i = 0;; ++i) {
                const auto r = query_cpuid(cache_leaf, i);
                const std::uint32_t cache_type = r.eax & 0x1FU;
                if (cache_type == 0) break; // 0 = Null / پایان لیست کش‌ها

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
            // پشتیبانی برای پردازنده‌های خیلی قدیمی AMD (قبل از معماری Zen)
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

        os_fallback(topo);

        return topo;
    }();

    return instance;
}

} // namespace mlaa