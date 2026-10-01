CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++20 -O3 -g -pthread -MMD -MP
INCLUDES = -Iinclude

OBJ_DIR = obj
LIB_SRCS = $(wildcard src/*.cpp)
LIB_OBJS = $(patsubst src/%.cpp,$(OBJ_DIR)/%.o,$(LIB_SRCS))

all: ngeo_build nexus_view

$(OBJ_DIR)/%.o: src/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

ngeo_build: tools/ngeo_build.cpp $(LIB_OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

# The viewer. Needs the Vulkan headers and loader, GLFW, and glslc for the
# shaders, whose SPIR-V is built into the program.
SHADERS = $(wildcard viewer/shaders/*.comp viewer/shaders/*.task viewer/shaders/*.mesh viewer/shaders/*.frag)
SPIRV = $(patsubst viewer/shaders/%,$(OBJ_DIR)/shaders/%.inc,$(SHADERS))

$(OBJ_DIR)/shaders/%.inc: viewer/shaders/% viewer/shaders/common.glsl
	@mkdir -p $(OBJ_DIR)/shaders
	glslc --target-env=vulkan1.3 -O -mfmt=num -o $@ $<

# shade.comp a second time, tracing shadows with ray queries.
$(OBJ_DIR)/shaders/shade_rt.comp.inc: viewer/shaders/shade.comp viewer/shaders/common.glsl
	@mkdir -p $(OBJ_DIR)/shaders
	glslc --target-env=vulkan1.3 -DRAY_QUERY -O -mfmt=num -o $@ $<

nexus_view: viewer/main.cpp viewer/vk.hpp viewer/png.hpp $(SPIRV) $(OBJ_DIR)/shaders/shade_rt.comp.inc $(LIB_OBJS)
	$(CXX) $(CXXFLAGS) -Wno-missing-field-initializers $(INCLUDES) -I$(OBJ_DIR)/shaders viewer/main.cpp $(LIB_OBJS) -o $@ -lvulkan -lglfw

clean:
	rm -rf $(OBJ_DIR) ngeo_build nexus_view

-include $(LIB_OBJS:.o=.d)

.PHONY: all clean
