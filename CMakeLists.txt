cmake_minimum_required (VERSION 2.8.11)
project (IsochoricThermodynamics)

# Add CoolProp dependency
set(COOLPROP_STATIC_LIBRARY true CACHE INTERNAL "Build CoolProp static library")
add_subdirectory ("${CMAKE_CURRENT_SOURCE_DIR}/externals/CoolProp")
set_property(TARGET CoolProp PROPERTY POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 14)

set(APP_SOURCES)
if (NOT ISOCHORIC_NO_REFPROP)
    file(GLOB APP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/wrappers/DLL/*.cpp")
endif()

find_package (Python COMPONENTS Interpreter Development)

# ----------------------------
# Build pybind11 python module
# ----------------------------

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/CoolProp/externals/REFPROP-headers")
add_subdirectory(externals/pybind11)
pybind11_add_module(VLEIsoTracer "${CMAKE_CURRENT_SOURCE_DIR}/src/tracer.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/externals/CoolProp/src/pybind11_interface.cxx" ${APP_SOURCES})
set_target_properties(VLEIsoTracer PROPERTIES COMPILE_FLAGS -DPYBIND11)
target_link_libraries (VLEIsoTracer PUBLIC CoolProp)
if (MSVC)
    target_sources( VLEIsoTracer PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/externals/CoolProp/externals/Eigen/debug/msvc/eigen.natvis")
endif()
if (NOT ISOCHORIC_NO_REFPROP)
    set_property(TARGET VLEIsoTracer APPEND_STRING PROPERTY COMPILE_FLAGS " -DREFPROP_SUPPORT")
endif()

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/CoolProp/externals/Eigen")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

if (ISOCHORIC_ALL)
    add_executable (tests "${CMAKE_CURRENT_SOURCE_DIR}/src/tests.cpp" )
    target_link_libraries (tests CoolProp)
    
    add_executable (check_cubics "${CMAKE_CURRENT_SOURCE_DIR}/src/check_cubics.cpp" )
    target_link_libraries (check_cubics CoolProp)

    add_executable (catch_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/catch_tests/catch_tests.cpp" )
    target_include_directories(catch_tests PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/externals/CoolProp/externals/Catch/single_include" "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries (catch_tests CoolProp)
    add_test(catch catch_tests)

    # ----------
    # Standalone
    # ----------
    set(SOURCES)
    if (MSVC)
        list(APPEND SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/externals/CoolProp/externals/Eigen/debug/msvc/eigen.natvis")
    endif()
    add_executable (tracer_cpp "${CMAKE_CURRENT_SOURCE_DIR}/src/tracer.cpp" ${SOURCES})
    target_link_libraries (tracer_cpp CoolProp)
endif()
