cmake_minimum_required(VERSION 3.24)
project(fluxscape LANGUAGES CXX)

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
FetchContent_MakeAvailable(json httplib doctest)
find_package(OpenSSL REQUIRED)

file(GLOB_RECURSE FLUXCORE_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cpp)
list(REMOVE_ITEM FLUXCORE_SOURCES ${CMAKE_SOURCE_DIR}/src/main.cpp)
add_library(fluxcore STATIC ${FLUXCORE_SOURCES})
target_include_directories(fluxcore PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(fluxcore PUBLIC CPPHTTPLIB_OPENSSL_SUPPORT)
target_compile_options(fluxcore PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(fluxcore PUBLIC nlohmann_json::nlohmann_json httplib::httplib
                      OpenSSL::SSL OpenSSL::Crypto)

add_executable(fluxscape src/main.cpp)
target_link_libraries(fluxscape PRIVATE fluxcore)

file(GLOB FLUX_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tests/*.cpp)
add_executable(fluxtests ${FLUX_TEST_SOURCES})
target_include_directories(fluxtests PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(fluxtests PRIVATE fluxcore doctest::doctest)
target_compile_definitions(fluxtests PRIVATE FLUX_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

enable_testing()
add_test(NAME fluxtests COMMAND fluxtests)
