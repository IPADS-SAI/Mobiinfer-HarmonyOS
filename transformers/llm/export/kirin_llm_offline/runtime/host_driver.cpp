// Host-only driver. Python executes reference ONNX graphs over binary pipes;
// all image processing, tokenization, positions, masks and KV updates run here.
#include "FullNpuRuntime.hpp"
#include <cstdio>
#include <fstream>
#include <iostream>

namespace {
template<class T> bool transferRead(T& value) { return std::fread(&value, sizeof(T), 1, stdin) == 1; }
class PipeBackend : public KirinFullNpu::Backend {
public:
    bool probe(const KirinFullNpu::Graph&, std::string&) override { return true; }
    void release() override {}
    bool run(const KirinFullNpu::Graph& graph, const std::vector<KirinFullNpu::View>& inputs,
             std::vector<std::vector<float>>& outputs, std::string& error) override {
        uint32_t size = graph.name.size();
        std::fwrite(&size, 4, 1, stdout); std::fwrite(graph.name.data(), 1, size, stdout);
        for (const auto& view : inputs) {
            uint64_t bytes = view.elements * KirinFullNpu::elementSize(view.dtype);
            std::fwrite(&bytes, 8, 1, stdout); std::fwrite(view.data, 1, bytes, stdout);
        }
        std::fflush(stdout);
        uint32_t count = 0;
        if (!transferRead(count) || count > 128) { error = "Host backend disconnected"; return false; }
        outputs.clear();
        for (uint32_t i = 0; i < count; ++i) {
            uint64_t elements;
            if (!transferRead(elements) || elements > (uint64_t(1) << 29)) return false;
            outputs.emplace_back(elements);
            if (std::fread(outputs.back().data(), 4, elements, stdin) != elements) return false;
        }
        return true;
    }
};
}
int main(int argc, char** argv) {
    if (argc != 5) return 2;
    KirinFullNpu::Runtime runtime(std::unique_ptr<KirinFullNpu::Backend>(new PipeBackend), {});
    std::string error, output;
    if (!runtime.load(argv[1], error)) { std::cerr << error << '\n'; return 3; }
    std::ifstream input(argv[2]); std::string text(std::istreambuf_iterator<char>(input), {});
    bool ok = runtime.response(text, {}, {}, output, error, std::atoi(argv[4]));
    std::ofstream result(argv[3]); result << (ok ? output : "ERROR: " + error);
    return ok ? 0 : 4;
}
