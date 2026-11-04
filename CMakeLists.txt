cmake_minimum_required(VERSION 3.20)

project(MLaA 
    VERSION 0.0.1 
    LANGUAGES CXX
)

# ۱. استاندارد مدرن C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ۲. اجبار به استفاده از Release در صورت خالی بودن تنظیمات کاربر
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type selected, defaulting to Release.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Release)" FORCE)
endif()

# ۳. تعریف کتابخانه
add_library(mlaa
    src/CpuTopology.cpp
    src/sample_sort.cpp
)

target_include_directories(mlaa
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ۴. فعال‌سازی بالاترین سطح بهینه‌سازی و سخت‌افزار میزبان (-march=native)
if(MSVC)
    target_compile_options(mlaa PUBLIC /O2 /Oi /Ot /arch:AVX2)
else()
    target_compile_options(mlaa PUBLIC 
        -O3 
        -march=native 
        -fno-math-errno 
        -fno-trapping-math
        -Wall 
        -Wextra
    )
endif()

# ۵. تعریف تست‌ها و بنچمارک
enable_testing()

add_executable(mlaa_tests
    tests/main.cpp
)

target_link_libraries(mlaa_tests
    PRIVATE
        mlaa
)

# اصلاح باگ نام تارگت در تست
add_test(NAME MLaATest COMMAND mlaa_tests)

# ۶. فعال‌سازی کامل LTO هم روی کتابخانه و هم روی باینری تست
include(CheckIPOSupported)
check_ipo_supported(RESULT lto_supported OUTPUT error)

if(lto_supported)
    message(STATUS "Link-Time Optimization (IPO/LTO) is ENABLED")
    set_target_properties(mlaa PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
    set_target_properties(mlaa_tests PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(WARNING "IPO/LTO not supported: ${error}")
endif()