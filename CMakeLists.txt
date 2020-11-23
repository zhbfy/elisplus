cmake_minimum_required(VERSION 3.6)

project(ELIS)

set(CMAKE_CXX_STANDARD 11)
include_directories(D:\\tool/ software\\eigen\\eigen)
## Install the executable to bin fodler


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -msse4")


#set(EXECUTABLE_OUTPUT_PATH debug)
#set(EXECUTABLE_OUTPUT_PATH D:\\code\\supervisedshapeletbasedtimeseriesclassification\\ELIS-master\\ELIS-master\\run)
#set(EXECUTABLE_OUTPUT_PATH D:\\code\\supervisedshapeletbasedtimeseriesclassification\\ELIS-master\\ELIS-master\\debug)





##==========================================================
#set(EXECUTABLE_OUTPUT_PATH D:\\code\\supervisedshapeletbasedtimeseriesclassification\\ELIS-master\\ELIS-master\\test)
#add_executable(discover tfidfinit.cpp)
#add_executable(adjust learn.cpp)

##==========================================================
#add_executable(adjust ts_fuzzy_learn.cpp)
#add_executable(learn learn.cpp)
#set(EXECUTABLE_OUTPUT_PATH D:\\code\\supervisedshapeletbasedtimeseriesclassification\\ELIS-master\\ELIS-master\\approximate)

##==========================================================
#set(EXECUTABLE_OUTPUT_PATH D:\\code\\supervisedshapeletbasedtimeseriesclassification\\ELIS-master\\ELIS-master\\eigen)
#add_executable(discover discover.cpp utils.cpp utils.h types.h)
#add_executable(adjust adjust.cpp utils.cpp utils.h elis.cpp elis.h types.h)

set(EXECUTABLE_OUTPUT_PATH D:\\code\\supervisedshapeletbasedtimeseriesclassification\\ELIS-master\\ELIS-master\\eigen_app)
add_executable(discover discover.cpp utils.cpp utils.h types.h)
add_executable(adjust adjust_fuzzy.cpp utils.cpp utils.h elis.cpp elis.h types.h)

#set(EXECUTABLE_OUTPUT_PATH D:\\code\\supervisedshapeletbasedtimeseriesclassification\\ELIS-master\\ELIS-master\\eigen_datagen)
#add_executable(discover discover.cpp utils.cpp utils.h types.h)
#add_executable(adjust adjust_fuzzy.cpp utils.cpp utils.h elis.cpp elis.h types.h)

#set(EXECUTABLE_OUTPUT_PATH D:\\code\\supervisedshapeletbasedtimeseriesclassification\\ELIS-master\\ELIS-master\\eigen_datagen)
#add_executable(discover discover.cpp utils.cpp utils.h types.h)
#add_executable(adjust utils.cpp utils.h elis_adam.cpp elis.h types.h adjust_fuzzy.cpp)

#set(EXECUTABLE_OUTPUT_PATH D:\\code\\supervisedshapeletbasedtimeseriesclassification\\ELIS-master\\ELIS-master\\eigen_smooth)
#add_executable(discovetime discover.cpp utils.cpp utils.h types.h)
#add_executable(adjust adjust_fuzzy.cpp elis.cpp elis.h types.h utils.cpp) 
#add_executable(adjusttest adjust_fuzzy_test.cpp elis.cpp elis.h types.h utils.cpp) 
#add_executable(adjust_real adjust_fuzzy_test.cpp elis_new.cpp elis.h types.h utils.cpp) 