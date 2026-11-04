#pragma once
#include <iostream>

namespace mlaa
{

struct CpuTopology {
    size_t l1d_size;
    size_t l2_size;
    size_t l3_size;
    size_t line_size;
    static const CpuTopology& get() noexcept;
};

} // namespace mlaa
