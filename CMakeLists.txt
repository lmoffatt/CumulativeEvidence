cmake_minimum_required(VERSION 3.5)

project(CumulativeEvidence LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_FLAGS "-ftemplate-backtrace-limit=0 -Werror=return-type")

find_package(BLAS)
find_package(LAPACK)
if(LAPACK_FOUND AND BLAS_FOUND)
   set(lapackblas_libraries ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()


add_executable(CumulativeEvidence main.cpp mcmc.h cuevi.h parallel_tempering.h multimodal_multivariate_model.h matrix.h matrix_random.h qr_factorization.h lapack_headers.h random_samplers.h maybe_error.h bayesian_linear_regression.h multivariate_normal_distribution.h
)
target_link_libraries(CumulativeEvidence ${lapackblas_libraries})

install(TARGETS CumulativeEvidence
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
