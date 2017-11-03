cmake_minimum_required(VERSION 2.8.11)
project(okvis_timing)

  set(Boost_USE_STATIC_LIBS ON)
  set(Boost_USE_STATIC ON)
  set(Boost_USE_MULTITHREAD )
  FIND_PACKAGE(Boost REQUIRED COMPONENTS date_time filesystem system)
  include_directories(${Boost_INCLUDE_DIRS}) 

add_library(${PROJECT_NAME}
  src/Timer.cpp
  src/NsecTimeUtilities.cpp
)

target_link_libraries(${PROJECT_NAME} PUBLIC okvis_util)

# installation if required
install(TARGETS ${PROJECT_NAME}
  EXPORT okvisTargets 
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}" COMPONENT lib
)
install(DIRECTORY include/ DESTINATION ${INSTALL_INCLUDE_DIR} COMPONENT dev FILES_MATCHING PATTERN "*.hpp")
