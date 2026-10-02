cmake_minimum_required(VERSION 3.24)
project(marketrank LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(FetchContent)
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_Declare(json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
set(HTTPLIB_REQUIRE_OPENSSL ON CACHE BOOL "" FORCE)
set(HTTPLIB_USE_BROTLI_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_ZLIB_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
FetchContent_Declare(httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG v0.18.1)
FetchContent_Declare(doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG v2.4.11)
FetchContent_Declare(duckdb_bin
  URL https://github.com/duckdb/duckdb/releases/download/v1.5.6/libduckdb-linux-amd64.zip
  URL_HASH SHA256=b845005f5132a7d8180057c35e14a7626632258782f871a90861b19c1c03841b
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(json httplib doctest duckdb_bin)
find_package(OpenSSL REQUIRED)
find_package(OpenMP REQUIRED)
add_library(duckdb SHARED IMPORTED GLOBAL)
set_target_properties(duckdb PROPERTIES
  IMPORTED_LOCATION ${duckdb_bin_SOURCE_DIR}/libduckdb.so
  INTERFACE_INCLUDE_DIRECTORIES ${duckdb_bin_SOURCE_DIR})

file(GLOB_RECURSE MR_CORE_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cpp)
list(REMOVE_ITEM MR_CORE_SOURCES ${CMAKE_SOURCE_DIR}/src/main.cpp)
add_library(marketrank_core STATIC ${MR_CORE_SOURCES})
target_include_directories(marketrank_core PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(marketrank_core PUBLIC CPPHTTPLIB_OPENSSL_SUPPORT)
target_compile_options(marketrank_core PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(marketrank_core PUBLIC nlohmann_json::nlohmann_json httplib::httplib
                      OpenSSL::SSL OpenSSL::Crypto OpenMP::OpenMP_CXX duckdb)

add_executable(marketrank src/main.cpp)
target_link_libraries(marketrank PRIVATE marketrank_core)
target_compile_definitions(marketrank PRIVATE MR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")  # --compare-13f records the git SHA

file(GLOB MR_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tests/*.cpp)
add_executable(marketrank_tests ${MR_TEST_SOURCES})
target_include_directories(marketrank_tests PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(marketrank_tests PRIVATE marketrank_core doctest::doctest)
target_compile_definitions(marketrank_tests PRIVATE MR_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
                           MR_BINARY="$<TARGET_FILE:marketrank>")
add_dependencies(marketrank_tests marketrank)  # the serve-lock test runs the real binary

enable_testing()
add_test(NAME marketrank_tests COMMAND marketrank_tests)
