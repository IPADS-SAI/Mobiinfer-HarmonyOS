#pragma once
#include "FullNpuRuntime.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace KirinFullNpu {
struct InputSync {
    const void* source = nullptr;
    uint64_t epoch = 0;
    int tokens = -1;
    bool ring = false;
};

// Copy logical rows into an NNRT input buffer. Only Cache::inputs may attach
// append-only metadata; ordinary tensors always take the full-copy path.
inline bool syncInput(const View& src, const Port& logical, DType targetType, void* target,
                      bool incremental, InputSync& state, size_t& written, std::string& error) {
    written = 0;
    const size_t sourceSize = elementSize(src.dtype), targetSize = elementSize(targetType);
    auto previous = state;
    state = {}; // A failed partial conversion must force a full retry.
    auto copy = [&](size_t offset, size_t elements) {
        auto* dst = static_cast<uint8_t*>(target) + offset * targetSize;
        auto* input = static_cast<const uint8_t*>(src.data) + offset * sourceSize;
        if (src.dtype == targetType) std::memcpy(dst, input, elements * targetSize);
        else if (src.dtype == DType::F32 && targetType == DType::F16) {
            auto* values = reinterpret_cast<const float*>(input);
            auto* half = reinterpret_cast<uint16_t*>(dst);
            for (size_t i = 0; i < elements; ++i) {
                half[i] = toHalf(values[i]);
                if (!std::isfinite(values[i]) || (half[i] & 0x7fff) == 0x7c00) {
                    error = "Device FP16 input overflow: " + logical.name; return false;
                }
            }
        } else if (src.dtype == DType::F16 && targetType == DType::F32) {
            auto* values = reinterpret_cast<const uint16_t*>(input);
            auto* floats = reinterpret_cast<float*>(dst);
            for (size_t i = 0; i < elements; ++i) floats[i] = fromHalf(values[i]);
        } else { error = "Integer device dtype differs from compiled interface"; return false; }
        written += elements * targetSize;
        return true;
    };
    const bool tagged = src.cacheEpoch && src.cacheTokens >= 0 && logical.shape.size() == 4 &&
                        logical.shape[0] == 1 && logical.elements() == src.elements;
    const bool delta = incremental && tagged && previous.source == src.data &&
                       previous.epoch == src.cacheEpoch && previous.tokens >= 0 &&
                       previous.tokens <= src.cacheTokens && previous.ring == src.cacheRing;
    if (!delta) {
        if (!copy(0, src.elements)) return false;
    } else {
        const int rows = logical.shape[2], channels = logical.shape[3];
        int begin = std::min(previous.tokens, rows), count = std::min(src.cacheTokens, rows) - begin;
        if (src.cacheRing) {
            count = std::min(src.cacheTokens - previous.tokens, rows);
            begin = (src.cacheTokens - count) % rows;
        }
        const int first = std::min(count, rows - begin);
        for (int head = 0; head < logical.shape[1]; ++head) {
            const size_t base = size_t(head) * rows * channels;
            if (first && !copy(base + size_t(begin) * channels, size_t(first) * channels)) return false;
            if (count > first && !copy(base, size_t(count - first) * channels)) return false;
        }
    }
    if (tagged) state = {src.data, src.cacheEpoch, src.cacheTokens, src.cacheRing};
    return true;
}
} // namespace KirinFullNpu
