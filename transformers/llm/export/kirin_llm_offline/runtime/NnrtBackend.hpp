#pragma once
#include "FullNpuRuntime.hpp"
namespace KirinFullNpu {
std::unique_ptr<Backend> createNnrtBackend(Log log);
}
