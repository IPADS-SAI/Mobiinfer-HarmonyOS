#include "FullNpuRuntime.hpp"
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>

int main() {
    using namespace KirinFullNpu;
    // All finite binary16 encodings, including subnormals, must round-trip.
    for (uint32_t i = 0; i < 65536; ++i) {
        if ((i & 0x7c00) != 0x7c00) assert(toHalf(fromHalf(uint16_t(i))) == i);
    }
    assert(toHalf(1.00048828125f) == toHalf(1.0f));
    assert(toHalf(1.00146484375f) == 0x3c02);
    Profile p;
    p.layers = 2; p.heads = 2; p.dim = 32; p.group = 16; p.capacity = 256; p.recent = 16; p.sink = 4;
    Cache cache(p);
    for (int length : {0, 3, 4, 15, 16, 19, 20, 24, 100, 255}) {
        cache.length = length;
        auto mask = cache.mask(1, 4);
        int stride = p.capacity + p.sink + p.recent + 4;
        for (int row = 0; row < 4; ++row) {
            int visible = 0;
            for (int c = 0; c < stride; ++c) visible += mask[row * stride + c] == 0;
            assert(visible == length + 1);
            for (int position = 0; position < length; ++position) {
                int copies = mask[row * stride + position] == 0;
                if (position < p.sink) copies += mask[row * stride + p.capacity + position] == 0;
                if (position >= std::max(p.sink, length - p.recent))
                    copies += mask[row * stride + p.capacity + p.sink + position % p.recent] == 0;
                assert(copies == 1);
            }
        }
    }
    cache.reset();
    int width = 4;
    std::vector<std::vector<float>> delta(4, std::vector<float>(p.heads * width * p.dim));
    for (auto& v : delta) for (size_t i = 0; i < v.size(); ++i) v[i] = float(int(i % 31) - 15) / 7;
    std::string error;
    assert(cache.append(0, delta, 3, width, error));
    auto views = cache.inputs(0, 2);
    assert(views.size() == 16 && views[0].dtype == DType::I8 && views[4].dtype == DType::F16);
    const auto* q = static_cast<const int8_t*>(views[0].data);
    const auto* scales = static_cast<const float*>(views[2].data);
    for (int h = 0; h < p.heads; ++h) for (int t = 0; t < 3; ++t) for (int d = 0; d < p.dim; ++d) {
        size_t index = (h * p.capacity + t) * p.dim + d;
        float scale = scales[index / p.group];
        float decoded = float(q[index]) * scale;
        assert(std::abs(decoded - delta[0][(h * width + t) * p.dim + d]) <= scale * .501f);
    }
    std::vector<uint8_t> saved;
    for (auto view : views) {
        const auto* bytes = static_cast<const uint8_t*>(view.data);
        saved.insert(saved.end(), bytes, bytes + view.elements * elementSize(view.dtype));
    }
    delta[3][0] = std::numeric_limits<float>::infinity();
    assert(!cache.append(0, delta, 3, width, error));
    size_t offset = 0;
    for (auto view : views) {
        size_t bytes = view.elements * elementSize(view.dtype);
        assert(!std::memcmp(view.data, saved.data() + offset, bytes)); offset += bytes;
    }
    assert(cache.bytes() == saved.size());
    cache.reset();
    for (auto view : cache.inputs(0, 2)) {
        if (view.dtype == DType::I8) {
            auto* values = static_cast<const int8_t*>(view.data);
            for (size_t i = 0; i < view.elements; ++i) assert(values[i] == 0);
        }
    }
    std::cout << "PASS: binary16, mask uniqueness, codec bounds, typed ports, padding, atomic rejection and reset\n";
}
