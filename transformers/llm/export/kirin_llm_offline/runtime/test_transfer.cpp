#include "KvTransfer.hpp"
#include <cassert>
#include <cstring>
#include <iostream>

int main() {
    using namespace KirinFullNpu;
    Profile p; p.layers = 2; p.heads = 2; p.dim = 32; p.group = 16;
    p.capacity = 256; p.recent = 16; p.sink = 4;
    Cache cache(p);
    const char* names[] = {"past_key", "past_value", "past_key_scale", "past_value_scale",
                           "sink_key", "sink_value", "recent_key", "recent_value"};
    std::vector<Port> ports;
    auto views = cache.inputs(0, 2);
    for (size_t i = 0; i < views.size(); ++i) {
        int field = i % 8;
        int rows = field < 4 ? p.capacity : field < 6 ? p.sink : p.recent;
        int cols = field == 2 || field == 3 ? p.dim / p.group : p.dim;
        ports.push_back({names[field], views[i].dtype, {1, p.heads, rows, cols}});
    }
    std::vector<std::vector<uint32_t>> buffers(views.size());
    std::vector<InputSync> states(views.size());
    size_t copied = 0, full = 0;
    std::string error;
    auto check = [&](bool initial) {
        auto current = cache.inputs(0, 2);
        size_t writtenTotal = 0;
        for (size_t i = 0; i < current.size(); ++i) {
            size_t bytes = current[i].elements * elementSize(current[i].dtype), written = 0;
            buffers[i].resize((bytes + 3) / 4);
            assert(syncInput(current[i], ports[i], ports[i].dtype, buffers[i].data(), true, states[i], written, error));
            assert(!std::memcmp(buffers[i].data(), current[i].data, bytes));
            assert(written <= bytes);
            if (initial) assert(written == bytes);
            writtenTotal += written;
            full += bytes;
        }
        copied += writtenTotal;
        return writtenTotal;
    };
    assert(check(true) == cache.bytes());
    assert(check(false) == 0); // No history changed.
    for (int count : {3, 1, 12, 5, 40, 1, 80, 1}) {
        std::vector<std::vector<float>> delta(4, std::vector<float>(p.heads * count * p.dim));
        for (size_t field = 0; field < delta.size(); ++field)
            for (size_t i = 0; i < delta[field].size(); ++i)
                delta[field][i] = float(int((i + cache.length * 7 + field * 3) % 91) - 45) / 13;
        assert(cache.append(0, delta, count, count, error)); cache.length += count;
        check(false); // Compare every byte, including wrapped rings and unfilled rows.
        assert(check(false) == 0);
    }
    uint64_t epoch = cache.inputs(0, 1)[0].cacheEpoch;
    cache.reset(); assert(cache.inputs(0, 1)[0].cacheEpoch != epoch);
    check(true); // Reset at the same source addresses must clear stale device KV.
    Cache another(p);
    assert(another.inputs(0, 1)[0].cacheEpoch != cache.inputs(0, 1)[0].cacheEpoch);
    assert(copied < full / 2);

    // Boundary conversion uses the same row protocol, including FP16 rings.
    Port logical{"recent_key_0", DType::F32, {1, 2, 3, 4}};
    std::vector<float> input(logical.elements(), 1.25f);
    std::vector<uint16_t> output(input.size()); InputSync state;
    View src{DType::F32, input.data(), input.size(), 7, 0, true};
    size_t written = 0;
    assert(syncInput(src, logical, DType::F16, output.data(), true, state, written, error));
    input[0] = 2.5f; input[12] = -3.25f; src.cacheTokens = 1;
    assert(syncInput(src, logical, DType::F16, output.data(), true, state, written, error));
    assert(written == 2 * 4 * sizeof(uint16_t));
    for (size_t i = 0; i < input.size(); ++i) assert(output[i] == toHalf(input[i]));
    std::vector<float> replacement(input); replacement[5] = 8;
    src.data = replacement.data(); // A different owner at the same epoch/length needs a full copy.
    assert(syncInput(src, logical, DType::F16, output.data(), true, state, written, error));
    assert(written == input.size() * 2 && output[5] == toHalf(8));
    replacement[4] = 100000; ++src.cacheTokens;
    assert(!syncInput(src, logical, DType::F16, output.data(), true, state, written, error));
    assert(state.epoch == 0);
    {
        Profile halfProfile = p; halfProfile.kvType = DType::F16; halfProfile.sink = halfProfile.recent = 0;
        Cache halfCache(halfProfile);
        auto current = halfCache.inputs(0, halfProfile.layers);
        assert(current.size() == 4 && halfCache.bytes() == size_t(4 * p.heads * p.capacity * p.dim * 2));
        std::vector<std::vector<uint16_t>> device(current.size());
        std::vector<InputSync> sync(current.size());
        auto verify = [&](bool full) {
            auto inputs = halfCache.inputs(0, halfProfile.layers);
            size_t total = 0;
            for (size_t i = 0; i < inputs.size(); ++i) {
                Port port{"past", DType::F16, {1, p.heads, p.capacity, p.dim}};
                device[i].resize(inputs[i].elements);
                size_t bytes = 0;
                assert(syncInput(inputs[i], port, DType::F16, device[i].data(), true, sync[i], bytes, error));
                assert(!std::memcmp(device[i].data(), inputs[i].data, inputs[i].elements * 2));
                if (full) assert(bytes == inputs[i].elements * 2);
                total += bytes;
            }
            return total;
        };
        assert(verify(true) == halfCache.bytes());
        for (int count : {3, 128, 1}) {
            int width = count + 1; // The last padded row must not be committed.
            std::vector<std::vector<float>> delta(4, std::vector<float>(p.heads * width * p.dim));
            for (size_t i = 0; i < delta.size(); ++i) for (size_t j = 0; j < delta[i].size(); ++j)
                delta[i][j] = float(int((j + i * 7) % 67) - 33) / 11;
            assert(halfCache.append(0, delta, count, width, error)); halfCache.length += count;
            assert(verify(false) == size_t(4 * p.heads * count * p.dim * 2));
            auto mask = halfCache.mask(1, 1);
            for (int t = 0; t < p.capacity; ++t) assert((mask[t] == 0) == (t < halfCache.length));
            assert(mask.back() == 0);
        }
        assert(verify(false) == 0);
        halfCache.reset(); assert(verify(true) == halfCache.bytes());
        std::vector<std::vector<float>> overflow(2, std::vector<float>(p.heads * p.dim, 70000));
        assert(!halfCache.append(0, overflow, 1, 1, error));
    }
    std::cout << "PASS: incremental KV equals full copies across prefix, ring wraps, long gaps, resets and conversions\n";
}
