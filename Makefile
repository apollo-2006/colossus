CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++20 -O3 -g -pthread -MMD -MP
INCLUDES = -Iinclude

OBJ_DIR = obj
LIB_SRCS = $(wildcard src/*.cpp)
LIB_OBJS = $(patsubst src/%.cpp,$(OBJ_DIR)/%.o,$(LIB_SRCS))

all: colossus_build colossus

$(OBJ_DIR)/%.o: src/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

colossus_build: tools/colossus_build.cpp $(LIB_OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

# the viewer: vulkan headers and loader, glfw, and glslc for the shaders, whose
# spir-v is built in.
SHADERS = $(wildcard viewer/shaders/*.comp viewer/shaders/*.task viewer/shaders/*.mesh viewer/shaders/*.frag)
SPIRV = $(patsubst viewer/shaders/%,$(OBJ_DIR)/shaders/%.inc,$(SHADERS))

SHADER_INCLUDES = $(wildcard viewer/shaders/*.glsl)

$(OBJ_DIR)/shaders/%.inc: viewer/shaders/% $(SHADER_INCLUDES)
	@mkdir -p $(OBJ_DIR)/shaders
	glslc --target-env=vulkan1.3 -O -mfmt=num -o $@ $<

# ao.comp again, with ray queries (the bounce's ground shadows under --shadows rt).
$(OBJ_DIR)/shaders/ao_rt.comp.inc: viewer/shaders/ao.comp $(SHADER_INCLUDES)
	@mkdir -p $(OBJ_DIR)/shaders
	glslc --target-env=vulkan1.3 -DRAY_QUERY -O -mfmt=num -o $@ $<

# shade.comp again, with ray queries.
$(OBJ_DIR)/shaders/shade_rt.comp.inc: viewer/shaders/shade.comp $(SHADER_INCLUDES)
	@mkdir -p $(OBJ_DIR)/shaders
	glslc --target-env=vulkan1.3 -DRAY_QUERY -O -mfmt=num -o $@ $<

colossus: viewer/main.cpp viewer/vk.hpp viewer/png.hpp $(SPIRV) $(OBJ_DIR)/shaders/shade_rt.comp.inc $(OBJ_DIR)/shaders/ao_rt.comp.inc $(LIB_OBJS)
	$(CXX) $(CXXFLAGS) -Wno-missing-field-initializers $(INCLUDES) -I$(OBJ_DIR)/shaders viewer/main.cpp $(LIB_OBJS) -o $@ -lvulkan -lglfw

tests/builder_test: tests/builder_test.cpp $(LIB_OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

tests/streamer_test: tests/streamer_test.cpp viewer/streamer.hpp $(LIB_OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) tests/streamer_test.cpp $(LIB_OBJS) -o $@

# builder and streamer tests on procedural meshes: no gpu, no downloads.
test: tests/builder_test tests/streamer_test
	./tests/builder_test
	./tests/streamer_test

clean:
	rm -rf $(OBJ_DIR) colossus_build colossus tests/builder_test tests/streamer_test

-include $(LIB_OBJS:.o=.d)

.PHONY: all clean test
