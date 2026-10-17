#include "FullNpuRuntime.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>

namespace KirinFullNpu {
namespace {
std::atomic<uint64_t> cacheEpoch{0};
}

size_t elementSize(DType t) { return t == DType::I8 ? 1 : t == DType::F16 ? 2 : 4; }
size_t Port::elements() const {
    size_t result = 1;
    for (int d : shape) {
        if (d <= 0 || result > std::numeric_limits<size_t>::max() / size_t(d)) return 0;
        result *= d;
    }
    return result;
}

// IEEE754 round-to-nearest-even, including subnormal/tie/overflow handling.
uint16_t toHalf(float value) {
    uint32_t b;
    std::memcpy(&b, &value, 4);
    uint16_t sign = (b >> 16) & 0x8000;
    uint32_t mantissa = b & 0x7fffff;
    int exp = int((b >> 23) & 255) - 127 + 15;
    if (((b >> 23) & 255) == 255) return sign | 0x7c00 | (mantissa ? 0x200 : 0);
    if (exp >= 31) return sign | 0x7c00;
    if (exp < -10) return sign;
    if (exp <= 0) {
        mantissa |= 0x800000;
        int shift = 14 - exp;
        uint32_t result = mantissa >> shift;
        uint32_t tail = mantissa & ((uint32_t(1) << shift) - 1);
        uint32_t half = uint32_t(1) << (shift - 1);
        return sign | uint16_t(result + (tail > half || (tail == half && (result & 1))));
    }
    uint32_t result = (uint32_t(exp) << 10) | (mantissa >> 13);
    uint32_t tail = mantissa & 8191;
    return sign | uint16_t(result + (tail > 4096 || (tail == 4096 && (result & 1))));
}

float fromHalf(uint16_t value) {
    uint32_t sign = uint32_t(value & 0x8000) << 16;
    int exp = (value >> 10) & 31;
    uint32_t mantissa = value & 1023, bits;
    if (!exp) {
        if (!mantissa) bits = sign;
        else {
            exp = -14;
            while (!(mantissa & 1024)) { mantissa <<= 1; --exp; }
            bits = sign | uint32_t(exp + 127) << 23 | ((mantissa & 1023) << 13);
        }
    } else if (exp == 31) bits = sign | 0x7f800000 | (mantissa << 13);
    else bits = sign | uint32_t(exp - 15 + 127) << 23 | (mantissa << 13);
    float result;
    std::memcpy(&result, &bits, 4);
    return result;
}

Cache::Cache(const Profile& profile) : p(profile), layers(p.layers) {
    for (auto& l : layers) {
        if (p.kvType == DType::F16) {
            l.fullKey.resize(size_t(p.heads) * p.capacity * p.dim); l.fullValue.resize(l.fullKey.size());
            continue;
        }
        l.key.resize(size_t(p.heads) * p.capacity * p.dim); l.value.resize(l.key.size());
        l.keyScale.resize(l.key.size() / p.group); l.valueScale.resize(l.keyScale.size());
        l.sinkKey.resize(size_t(p.heads) * p.sink * p.dim); l.sinkValue.resize(l.sinkKey.size());
        l.recentKey.resize(size_t(p.heads) * p.recent * p.dim); l.recentValue.resize(l.recentKey.size());
    }
    reset();
}
void Cache::reset() {
    epoch = ++cacheEpoch; // Distinguish a new cache even if its allocation reuses an old address.
    length = 0;
    for (auto& l : layers) {
        std::fill(l.key.begin(), l.key.end(), 0); std::fill(l.value.begin(), l.value.end(), 0);
        std::fill(l.keyScale.begin(), l.keyScale.end(), 1); std::fill(l.valueScale.begin(), l.valueScale.end(), 1);
        for (auto* v : {&l.fullKey, &l.fullValue, &l.sinkKey, &l.sinkValue, &l.recentKey, &l.recentValue})
            std::fill(v->begin(), v->end(), 0);
    }
}
std::vector<View> Cache::inputs(int start, int count) const {
    std::vector<View> result;
    for (int i = start; i < start + count; ++i) {
        const auto& l = layers.at(i);
        if (p.kvType == DType::F16) {
            for (const auto* v : {&l.fullKey, &l.fullValue})
                result.push_back({DType::F16, v->data(), v->size(), epoch, length, false});
            continue;
        }
        result.push_back({DType::I8, l.key.data(), l.key.size()});
        result.push_back({DType::I8, l.value.data(), l.value.size()});
        result.push_back({DType::F32, l.keyScale.data(), l.keyScale.size()});
        result.push_back({DType::F32, l.valueScale.data(), l.valueScale.size()});
        if (p.sink) for (const auto* v : {&l.sinkKey, &l.sinkValue}) result.push_back({DType::F16, v->data(), v->size()});
        if (p.recent) for (const auto* v : {&l.recentKey, &l.recentValue}) result.push_back({DType::F16, v->data(), v->size()});
        const size_t stride = 4 + 2 * bool(p.sink) + 2 * bool(p.recent);
        for (size_t j = result.size() - stride; j < result.size(); ++j) {
            result[j].cacheEpoch = epoch;
            result[j].cacheTokens = length;
            result[j].cacheRing = p.recent && j >= result.size() - 2;
        }
    }
    return result;
}
bool Cache::append(int start, const std::vector<std::vector<float>>& out, int real, int width, std::string& error) {
    int count = int(out.size()) / 2;
    if (out.size() != size_t(2 * count) || start < 0 || start + count > p.layers ||
        real < 1 || real > width || length + real > p.capacity) { error = "Invalid KV update"; return false; }
    // Validate the complete chunk before committing any row; current-block padding is ignored.
    for (const auto& v : out) {
        if (v.size() != size_t(p.heads) * width * p.dim) { error = "KV delta shape mismatch"; return false; }
        for (int h = 0; h < p.heads; ++h) for (int t = 0; t < real; ++t) for (int d = 0; d < p.dim; ++d) {
            float x = v[(size_t(h) * width + t) * p.dim + d];
            if (!std::isfinite(x) || ((p.kvType == DType::F16 || p.recent || length + t < p.sink) && std::abs(x) > 65504)) {
                error = "Nonfinite or FP16-overflowing KV delta"; return false;
            }
        }
    }
    for (int i = 0; i < count; ++i) for (int k = 0; k < 2; ++k) {
        auto& l = layers[start + i];
        auto& q = k ? l.value : l.key;
        auto& scales = k ? l.valueScale : l.keyScale;
        auto& sink = k ? l.sinkValue : l.sinkKey;
        auto& recent = k ? l.recentValue : l.recentKey;
        const auto& values = out[2 * i + k];
        if (p.kvType == DType::F16) {
            auto& dst = k ? l.fullValue : l.fullKey;
            for (int h = 0; h < p.heads; ++h) for (int t = 0; t < real; ++t) for (int d = 0; d < p.dim; ++d)
                dst[(size_t(h) * p.capacity + length + t) * p.dim + d] =
                    toHalf(values[(size_t(h) * width + t) * p.dim + d]);
            continue;
        }
        for (int h = 0; h < p.heads; ++h) for (int t = 0; t < real; ++t) {
            int position = length + t;
            const float* src = values.data() + (size_t(h) * width + t) * p.dim;
            for (int g = 0; g < p.dim / p.group; ++g) {
                float maximum = 0;
                for (int d = 0; d < p.group; ++d) maximum = std::max(maximum, std::abs(src[g * p.group + d]));
                float scale = maximum == 0 ? 1 : std::max(maximum / 127.0f, std::numeric_limits<float>::min());
                size_t index = (size_t(h) * p.capacity + position) * (p.dim / p.group) + g;
                scales[index] = scale;
                for (int d = 0; d < p.group; ++d) {
                    float x = src[g * p.group + d] / scale;
                    // nearbyint follows nearest-even under the default IEEE rounding mode.
                    q[index * p.group + d] = int8_t(std::max(-127.0f, std::min(127.0f, std::nearbyint(x))));
                }
            }
            for (int d = 0; d < p.dim; ++d) {
                if (position < p.sink) sink[(size_t(h) * p.sink + position) * p.dim + d] = toHalf(src[d]);
                if (p.recent) recent[(size_t(h) * p.recent + position % p.recent) * p.dim + d] = toHalf(src[d]);
            }
        }
    }
    return true;
}
std::vector<float> Cache::mask(int real, int width) const {
    int stride = p.capacity + p.sink + p.recent + width;
    std::vector<float> result(size_t(width) * stride, -10000.0f);
    int middleEnd = std::max(p.sink, length - p.recent);
    for (int row = 0; row < width; ++row) {
        float* m = result.data() + size_t(row) * stride;
        for (int t = p.sink; t < std::min(length, middleEnd); ++t) m[t] = 0;
        for (int t = 0; t < std::min(length, p.sink); ++t) m[p.capacity + t] = 0;
        if (p.recent) for (int t = std::max(p.sink, length - p.recent); t < length; ++t)
            m[p.capacity + p.sink + t % p.recent] = 0;
        for (int t = 0; t < std::min(real, row + 1); ++t) m[p.capacity + p.sink + p.recent + t] = 0;
    }
    return result;
}
size_t Cache::bytes() const {
    size_t result = 0;
    for (const auto& l : layers) result += l.key.size() + l.value.size() +
        4 * (l.keyScale.size() + l.valueScale.size()) +
        2 * (l.fullKey.size() + l.fullValue.size() + l.sinkKey.size() + l.sinkValue.size() + l.recentKey.size() + l.recentValue.size());
    return result;
}
} // namespace KirinFullNpu
