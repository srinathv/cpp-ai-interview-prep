if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH $ENV{ROCM_PATH})
    else()
        set(ROCM_PATH "/opt/rocm")
    endif()
endif()

list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/hip ${ROCM_PATH})

find_package(hip REQUIRED)

add_executable(dictionary_search_hip dictionary_search.cpp)

set_source_files_properties(dictionary_search.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

target_link_libraries(dictionary_search_hip PRIVATE hip::host)

install(TARGETS dictionary_search_hip DESTINATION bin)
