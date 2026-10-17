// Host test target only. The App already links stb from FullNpuRuntime.cpp.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#include "stb_image.h"
