# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.28.3/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.28.3/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/lucasimei/Documents/Uni-HPC/polimi/amsc23-24/Montecarlo/Montecarlo-Bellini-Simei-Tramacere

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/lucasimei/Documents/Uni-HPC/polimi/amsc23-24/Montecarlo/Montecarlo-Bellini-Simei-Tramacere/build

# Include any dependencies generated for this target.
include CMakeFiles/mainOmp.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/mainOmp.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/mainOmp.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mainOmp.dir/flags.make

CMakeFiles/mainOmp.dir/src/main.cpp.o: CMakeFiles/mainOmp.dir/flags.make
CMakeFiles/mainOmp.dir/src/main.cpp.o: /Users/lucasimei/Documents/Uni-HPC/polimi/amsc23-24/Montecarlo/Montecarlo-Bellini-Simei-Tramacere/src/main.cpp
CMakeFiles/mainOmp.dir/src/main.cpp.o: CMakeFiles/mainOmp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/lucasimei/Documents/Uni-HPC/polimi/amsc23-24/Montecarlo/Montecarlo-Bellini-Simei-Tramacere/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/mainOmp.dir/src/main.cpp.o"
	/usr/local/opt/llvm/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/mainOmp.dir/src/main.cpp.o -MF CMakeFiles/mainOmp.dir/src/main.cpp.o.d -o CMakeFiles/mainOmp.dir/src/main.cpp.o -c /Users/lucasimei/Documents/Uni-HPC/polimi/amsc23-24/Montecarlo/Montecarlo-Bellini-Simei-Tramacere/src/main.cpp

CMakeFiles/mainOmp.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/mainOmp.dir/src/main.cpp.i"
	/usr/local/opt/llvm/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/lucasimei/Documents/Uni-HPC/polimi/amsc23-24/Montecarlo/Montecarlo-Bellini-Simei-Tramacere/src/main.cpp > CMakeFiles/mainOmp.dir/src/main.cpp.i

CMakeFiles/mainOmp.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/mainOmp.dir/src/main.cpp.s"
	/usr/local/opt/llvm/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/lucasimei/Documents/Uni-HPC/polimi/amsc23-24/Montecarlo/Montecarlo-Bellini-Simei-Tramacere/src/main.cpp -o CMakeFiles/mainOmp.dir/src/main.cpp.s

# Object files for target mainOmp
mainOmp_OBJECTS = \
"CMakeFiles/mainOmp.dir/src/main.cpp.o"

# External object files for target mainOmp
mainOmp_EXTERNAL_OBJECTS =

mainOmp: CMakeFiles/mainOmp.dir/src/main.cpp.o
mainOmp: CMakeFiles/mainOmp.dir/build.make
mainOmp: libOptionPricing.a
mainOmp: libOptionPricing.a
mainOmp: CMakeFiles/mainOmp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/lucasimei/Documents/Uni-HPC/polimi/amsc23-24/Montecarlo/Montecarlo-Bellini-Simei-Tramacere/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable mainOmp"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mainOmp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mainOmp.dir/build: mainOmp
.PHONY : CMakeFiles/mainOmp.dir/build

CMakeFiles/mainOmp.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mainOmp.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mainOmp.dir/clean

CMakeFiles/mainOmp.dir/depend:
	cd /Users/lucasimei/Documents/Uni-HPC/polimi/amsc23-24/Montecarlo/Montecarlo-Bellini-Simei-Tramacere/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/lucasimei/Documents/Uni-HPC/polimi/amsc23-24/Montecarlo/Montecarlo-Bellini-Simei-Tramacere /Users/lucasimei/Documents/Uni-HPC/polimi/amsc23-24/Montecarlo/Montecarlo-Bellini-Simei-Tramacere /Users/lucasimei/Documents/Uni-HPC/polimi/amsc23-24/Montecarlo/Montecarlo-Bellini-Simei-Tramacere/build /Users/lucasimei/Documents/Uni-HPC/polimi/amsc23-24/Montecarlo/Montecarlo-Bellini-Simei-Tramacere/build /Users/lucasimei/Documents/Uni-HPC/polimi/amsc23-24/Montecarlo/Montecarlo-Bellini-Simei-Tramacere/build/CMakeFiles/mainOmp.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/mainOmp.dir/depend

