find_package(MPI REQUIRED)

add_executable(dictionary_search_mpi dictionary_search_mpi.cpp)
target_link_libraries(dictionary_search_mpi PRIVATE MPI::MPI_CXX)

install(TARGETS dictionary_search_mpi DESTINATION bin)
