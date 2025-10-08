# Makefile for hw2_sift project
MPICXX   = mpicxx
# If mpicxx is Apple Clang, see note below for OpenMP flags.
MPIFLAGS = -std=c++17 -O3 -fopenmp
INCLUDES = -I.

TARGET   = hw2

# Source files (rename to mpi_sift.cpp to match your code)
SOURCES  = hw2.cpp sift.cpp image.cpp mpi_sift.cpp
OBJECTS  = $(SOURCES:.cpp=.o)

STB_HEADERS = stb_image.h stb_image_write.h
HEADERS  = sift.hpp image.hpp mpi_sift.hpp

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(MPICXX) $(MPIFLAGS) -o $@ $(OBJECTS)

# One pattern rule handles all .cpp -> .o compilations
%.o: %.cpp $(HEADERS) $(STB_HEADERS)
	$(MPICXX) $(MPIFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET) result.jpg

.PHONY: all clean

#export OMPI_CXX=/opt/homebrew/bin/g++-15