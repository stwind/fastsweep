cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

find_package(Python 3.8 
    REQUIRED COMPONENTS Interpreter Development.Module
    ${SKBUILD_SABI_COMPONENT})

find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(extension STABLE_ABI NB_STATIC src/extension.cpp)

install(TARGETS extension LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})