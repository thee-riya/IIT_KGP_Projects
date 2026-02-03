# Compiler
CXX = g++
# Compiler flags
CXXFLAGS = -std=c++20 -pthread -Wall -Wextra -O2
# Output executable name
TARGET = memfs
# Source file
SOURCES = 24CS60R71_Part1.cpp
# Object file
OBJECTS = $(SOURCES:.cpp=.o)

# Default target to build and run the program
all: $(TARGET)
	./$(TARGET)

# Rule to link the executable
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

# Rule to compile the .cpp file into .o file
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(TARGET) $(OBJECTS)

# PHONY targets (to prevent filename conflicts)
.PHONY: all clean