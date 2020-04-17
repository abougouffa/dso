SET(PROJECT_NAME DSO)

PROJECT(${PROJECT_NAME})
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
#set(CMAKE_VERBOSE_MAKEFILE ON)


set(BUILD_TYPE Release)
#set(BUILD_TYPE RelWithDebInfo)

set(EXECUTABLE_OUTPUT_PATH bin)
set(LIBRARY_OUTPUT_PATH lib)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

if (BUILD_DOC)
  find_package(Doxygen)
  if (DOXYGEN_FOUND)
  	# set input and output files
  	set(DOXYGEN_IN ${PROJECT_SOURCE_DIR}/Doxyfile.in)
  	set(DOXYGEN_OUT ${PROJECT_SOURCE_DIR}/Doxyfile)  
  	# request to configure the file
  	configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
  	message("Doxygen build started")  
  	# note the option ALL which allows to build the docs together with the application
  	add_custom_target( doc ALL
  		COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
  		WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  		COMMENT "Generating API documentation with Doxygen"
  		VERBATIM )
  else (DOXYGEN_FOUND)
    message("Doxygen need to be installed to generate the doxygen documentation")
  endif (DOXYGEN_FOUND)
endif ()


# required libraries
find_package(SuiteParse REQUIRED)
find_package(Boost COMPONENTS system thread) 

# optional libraries
find_package(LibZip QUIET)
find_package(Pangolin 0.2 QUIET)
find_package(OpenCV QUIET)
find_package(Glog REQUIRED)

# flags
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS
    "${SSE_FLAGS} -O3 -std=c++11 -Wall -Werror"
#   "${SSE_FLAGS} -O3 -g -std=c++0x -fno-omit-frame-pointer"
)

if (MSVC)
     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
endif (MSVC)

# Sources files
set(dso_SOURCE_FILES
  ${PROJECT_SOURCE_DIR}/src/full_system/full_system.cc
  ${PROJECT_SOURCE_DIR}/src/full_system/full_system_optimize.cc
  ${PROJECT_SOURCE_DIR}/src/full_system/full_system_opt_point.cc
  ${PROJECT_SOURCE_DIR}/src/full_system/full_system_debug_stuff.cc
  ${PROJECT_SOURCE_DIR}/src/full_system/full_system_marginalize.cc
  ${PROJECT_SOURCE_DIR}/src/full_system/residuals.cc
  ${PROJECT_SOURCE_DIR}/src/full_system/tracker/coarse_tracker.cc
  ${PROJECT_SOURCE_DIR}/src/full_system/tracker/coarse_distance_map.cc
  ${PROJECT_SOURCE_DIR}/src/full_system/initializer/coarse_initializer.cc
  ${PROJECT_SOURCE_DIR}/src/full_system/immature_point.cc
  ${PROJECT_SOURCE_DIR}/src/full_system/pixel_selector2.cc
  ${PROJECT_SOURCE_DIR}/src/full_system/hessian_blocks/frame_frame_pre_calc.cc
  ${PROJECT_SOURCE_DIR}/src/full_system/hessian_blocks/frame_hessian.cc
  ${PROJECT_SOURCE_DIR}/src/full_system/hessian_blocks/point_hessian.cc
  ${PROJECT_SOURCE_DIR}/src/optimization_backend/energy_functional/ef_frame.cc 
  ${PROJECT_SOURCE_DIR}/src/optimization_backend/energy_functional/ef_point.cc 
  ${PROJECT_SOURCE_DIR}/src/optimization_backend/energy_functional/ef_residual.cc 
  ${PROJECT_SOURCE_DIR}/src/optimization_backend/energy_functional/energy_functional.cc
  ${PROJECT_SOURCE_DIR}/src/optimization_backend/accumulated_top_hessian_sse.cc
  ${PROJECT_SOURCE_DIR}/src/optimization_backend/accumulated_sc_hessian_sse.cc
  ${PROJECT_SOURCE_DIR}/src/optimization_backend/accumulators/accumulator_9.cc
  ${PROJECT_SOURCE_DIR}/src/optimization_backend/accumulators/accumulator_14.cc
  ${PROJECT_SOURCE_DIR}/src/optimization_backend/accumulators/accumulator_approx.cc
  ${PROJECT_SOURCE_DIR}/src/undistorter/undistorter.cc
  ${PROJECT_SOURCE_DIR}/src/undistorter/photometric_undistorter.cc
  ${PROJECT_SOURCE_DIR}/src/undistorter/undistorter_fov.cc
  ${PROJECT_SOURCE_DIR}/src/undistorter/undistorter_kb.cc 
  ${PROJECT_SOURCE_DIR}/src/undistorter/undistorter_pinhole.cc
  ${PROJECT_SOURCE_DIR}/src/undistorter/undistorter_rad_tan.cc
  ${PROJECT_SOURCE_DIR}/src/undistorter/undistorter_equidistant.cc
  ${PROJECT_SOURCE_DIR}/src/util/settings.cc
  ${PROJECT_SOURCE_DIR}/src/util/global_calib.cc
  ${PROJECT_SOURCE_DIR}/src/util/dataset_reader.cc
  ${PROJECT_SOURCE_DIR}/src/util/input_parser.cc
  ${PROJECT_SOURCE_DIR}/src/util/converter.cc
)



include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/thirdparty/Sophus
  ${PROJECT_SOURCE_DIR}/thirdparty/sse2neon
  "/usr/include/eigen3/"
) 


# decide if we have pangolin
if (Pangolin_FOUND)
	message("--- found PANGOLIN, compiling dso_pangolin library.")
	include_directories( ${Pangolin_INCLUDE_DIRS} ) 
	set(dso_pangolin_SOURCE_FILES 
	  ${PROJECT_SOURCE_DIR}/src/io_wrapper/pangolin/keyframe_display.cc
	  ${PROJECT_SOURCE_DIR}/src/io_wrapper/pangolin/pangolin_dso_viewer.cc)
	set(HAS_PANGOLIN 1)
else ()
	message("--- could not find PANGOLIN, not compiling dso_pangolin library.")
	message("    this means there will be no 3D display / GUI available for dso_dataset.")
	set(dso_pangolin_SOURCE_FILES )
	set(HAS_PANGOLIN 0)
endif ()

# decide if we have openCV
if (OpenCV_FOUND)
	message("--- found OpenCV, compiling dso_opencv library.")
	include_directories( ${OpenCV_INCLUDE_DIRS} )
	set(dso_opencv_SOURCE_FILES 
	  ${PROJECT_SOURCE_DIR}/src/io_wrapper/opencv/image_display.cc
	  ${PROJECT_SOURCE_DIR}/src/io_wrapper/opencv/image_rw.cc)
	set(HAS_OPENCV 1)
else ()
	message("--- could not find OpenCV, not compiling dso_opencv library.")
	message("    this means there will be no image display, and image read / load functionality.")
	set(dso_opencv_SOURCE_FILES 
	  ${PROJECT_SOURCE_DIR}/src/io_wrapper/image_display_dummy.cc
	  ${PROJECT_SOURCE_DIR}/src/io_wrapper/image_rw_dummy.cc)
	set(HAS_OPENCV 0)
endif ()

# decide if we have ziplib.
if (LIBZIP_LIBRARY)
	message("--- found ziplib (${LIBZIP_VERSION}), compiling with zip capability.")
	add_definitions(-DHAS_ZIPLIB=1)
	include_directories( ${LIBZIP_INCLUDE_DIR_ZIP} ${LIBZIP_INCLUDE_DIR_ZIPCONF} ) 
else()
	message("--- not found ziplib (${LIBZIP_LIBRARY}), compiling without zip capability.")
	set(LIBZIP_LIBRARY "")
endif()


# compile main library.
include_directories( ${CSPARSE_INCLUDE_DIR} ${CHOLMOD_INCLUDE_DIR}) 
add_library(dso ${dso_SOURCE_FILES} ${dso_opencv_SOURCE_FILES} ${dso_pangolin_SOURCE_FILES})

#set_property( TARGET dso APPEND_STRING PROPERTY COMPILE_FLAGS -Wall )


if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin") # OSX
    set(BOOST_THREAD_LIBRARY boost_thread-mt)
else()
    set(BOOST_THREAD_LIBRARY boost_thread)
endif()

# build main executable (only if we have both OpenCV and Pangolin)
if (OpenCV_FOUND AND Pangolin_FOUND)
	message("--- compiling dso_dataset.")
	add_executable(dso_dataset ${PROJECT_SOURCE_DIR}/app/main_dso_pangolin.cc )
	target_link_libraries(dso_dataset						  
						  dso 
						  boost_system
						  cxsparse 
						  glog
						  ${BOOST_THREAD_LIBRARY}
						  ${LIBZIP_LIBRARY} 
						  ${Pangolin_LIBRARIES} 
						  ${OpenCV_LIBS})

	add_executable(dso_new ${PROJECT_SOURCE_DIR}/app/main.cc )
	target_link_libraries(dso_new
						  dso 
						  boost_system
						  cxsparse 
						  glog
						  ${BOOST_THREAD_LIBRARY}
						  ${LIBZIP_LIBRARY} 
						  ${Pangolin_LIBRARIES} 
						  ${OpenCV_LIBS})
else()
	message("--- not building dso_dataset, since either don't have openCV or Pangolin.")
endif()

