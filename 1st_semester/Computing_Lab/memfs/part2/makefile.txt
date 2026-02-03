# Compiler
CXX = g++

# Compiler flags (updated to use C++17)
CXXFLAGS = -std=c++17 -pthread -Wall

# Target executable
TARGET = 24CS60R71_Part2

# Source files
SRCS = 24CS60R71_Part2.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Default target: compile and run the program
all: $(TARGET)
	./$(TARGET)

# Link object files to create the executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Compile source files into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up generated files
clean:
	rm -f $(TARGET) $(OBJS)

# Phony targets
.PHONY: all clean
