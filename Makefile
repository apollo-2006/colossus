CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++20 -O3 -g -pthread -MMD -MP
INCLUDES = -Iinclude

OBJ_DIR = obj
LIB_SRCS = $(wildcard src/*.cpp)
LIB_OBJS = $(patsubst src/%.cpp,$(OBJ_DIR)/%.o,$(LIB_SRCS))

all: ngeo_build

$(OBJ_DIR)/%.o: src/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

ngeo_build: tools/ngeo_build.cpp $(LIB_OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

clean:
	rm -rf $(OBJ_DIR) ngeo_build

-include $(LIB_OBJS:.o=.d)

.PHONY: all clean
