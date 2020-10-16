cmake_minimum_required(VERSION 3.14.5)
project(unstructured_interface LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/private")

set(CMAKE_CXX_STANDARD 17)

add_compile_options(-Wall)
#add_compile_options(-Wall -Wextra -pedantic -Wno-sign-compare)


include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    # set(CMAKE_CUDA_STANDARD 17)
endif()

include(FetchContent)
FetchContent_Declare(GridTools
    GIT_REPOSITORY https://github.com/GridTools/gridtools.git
    GIT_TAG        master
    # URL https://github.com/GridTools/gridtools/archive/???.tar.gz
)
FetchContent_MakeAvailable(GridTools)

add_library(gtnext INTERFACE)
target_include_directories(gtnext INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(eckit REQUIRED)
find_package(Atlas REQUIRED)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
    if(BUILD_TESTING)
        add_subdirectory(tests)
    endif()
endif()
