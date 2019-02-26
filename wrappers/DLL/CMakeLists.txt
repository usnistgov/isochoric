cmake_minimum_required (VERSION 2.8.11)
project (IsochoricThermodynamics)

set(CMAKE_CXX_STANDARD 14)
set (ROOT_DIR "../..")

# Add core dependency to get the list of files
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/${ROOT_DIR}" "_base")
# Add the directory with the core code to the include path
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/${ROOT_DIR}/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/${ROOT_DIR}/externals/CoolProp/externals/REFPROP-headers")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/${ROOT_DIR}/externals/CoolProp/include")

if (NOT REFPROP_PYTHON)
    # --------------
    #      DLL      
    # --------------
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        # 64 bits
        set(DLL_NAME ISOCHR64)
    elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
        # 32 bits
        set(DLL_NAME ISOCHR32)
    endif()
    add_library (${DLL_NAME} SHARED "${CMAKE_CURRENT_SOURCE_DIR}/refprop_funcs.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/refprop_tracer.cpp")
    set_target_properties(${DLL_NAME} PROPERTIES COMPILE_FLAGS -DISOCHORIC_DLL)
    target_link_libraries (${DLL_NAME} CoolProp)
else()

    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/CoolProp/externals/REFPROP-headers")
    
    pybind11_add_module(REFPROPIsoTracer "${CMAKE_CURRENT_SOURCE_DIR}/refprop_funcs.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/refprop_tracer.cpp")
    set_target_properties(REFPROPIsoTracer PROPERTIES COMPILE_FLAGS -DPYBIND11)
    target_link_libraries (REFPROPIsoTracer PUBLIC CoolProp)
endif()