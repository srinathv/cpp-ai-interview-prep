enable_language(CUDA)

add_executable(dictionary_search_cuda dictionary_search.cu)

set_target_properties(dictionary_search_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
)

target_compile_options(dictionary_search_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

install(TARGETS dictionary_search_cuda DESTINATION bin)
