cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

find_package(Python 3.8 
    REQUIRED COMPONENTS Interpreter Development.Module
    ${SKBUILD_SABI_COMPONENT})

find_package(nanobind CONFIG REQUIRED)

set(TARGET redistance)

execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c "import site; print(site.getsitepackages()[0])" OUTPUT_VARIABLE PYTHON_SITEPKG_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
set(drjit_DIR ${PYTHON_SITEPKG_DIR}/drjit/cmake/drjit)

find_package(drjit)
include_directories(${drjit_INCLUDE_DIR})

nanobind_add_module(${TARGET} STABLE_ABI NB_STATIC src/redistance.cpp)

target_link_libraries(${TARGET} PUBLIC nanothread drjit-core)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_compile_definitions(${TARGET} PRIVATE MODULE=${TARGET})

install(TARGETS ${TARGET} LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})