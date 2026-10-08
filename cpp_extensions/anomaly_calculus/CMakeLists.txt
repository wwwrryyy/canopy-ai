cmake_minimum_required(VERSION 3.14)
project(agri_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Pin Python to conda-forge 3.13 so pybind11 and Python3::Module agree on the same interpreter
set(Python3_ROOT_DIR "/opt/homebrew/Caskroom/miniforge/base" CACHE PATH "" FORCE)
set(Python3_EXECUTABLE "/opt/homebrew/Caskroom/miniforge/base/bin/python" CACHE FILEPATH "" FORCE)
set(Python3_FIND_STRATEGY LOCATION)

# Tell pybind11 to use the modern FindPython mechanism (avoids CMP0148 warnings)
set(PYBIND11_FINDPYTHON ON)

# Locate Python 3 Development.Module first so pybind11 inherits the same target
find_package(Python3 REQUIRED COMPONENTS Development.Module)

# Locate pybind11 via its own cmake config (conda-forge install)
find_package(pybind11 REQUIRED CONFIG
    PATHS "/opt/homebrew/Caskroom/miniforge/base/lib/python3.13/site-packages/pybind11/share/cmake/pybind11"
    NO_DEFAULT_PATH
)

# Locate OpenCV 5 (conda-forge installs as opencv5, not OpenCV)
find_package(OpenCV REQUIRED CONFIG
    PATHS "/opt/homebrew/Caskroom/miniforge/base/lib/cmake/opencv5"
    NO_DEFAULT_PATH
)

# Build the pybind11 extension module
pybind11_add_module(agri_core
    src/pybind_bridge.cpp
    src/video_ingestor.cpp
    src/anomaly_calculus.cpp
)

target_include_directories(agri_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(agri_core PRIVATE
    ${OpenCV_LIBS}
    Python3::Module
)

# Output directly into deployment/ so Python can import without sys.path manipulation
set_target_properties(agri_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../deployment"
)
