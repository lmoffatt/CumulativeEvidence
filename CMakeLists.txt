cmake_minimum_required(VERSION 3.5)

project(CumulativeEvidence LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(CumulativeEvidence main.cpp mcmc.h cuevi.h vector_field.h list.h range.h indexes.h
)

install(TARGETS CumulativeEvidence
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
