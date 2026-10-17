#pragma once

#include "FullNpuRuntime.hpp"
#include "llm/llm.hpp"

namespace OfflineNpu2 {
// Separate from both the legacy chunk callback and the full-NPU scheduler.
struct State { std::string error; };
struct Features { std::vector<float> image, deepstack; };
struct Prompt {
    std::string text;
    std::vector<std::string> images;
    std::vector<size_t> imageOffsets;
    std::vector<int> positions; // [3, sequence], already includes history offset
    int nextPosition = 0;
};
bool parseImages(const std::string& input, Prompt& out, std::string& error);
bool makePositions(const std::vector<int>& ids, Prompt& prompt, int base, std::string& error);
bool injectFeatures(const Prompt&, const std::vector<Features>&, size_t tokens,
                    float* embeddings, std::vector<float>& deepstack, std::string& error);

class Vision {
public:
    Vision(std::unique_ptr<KirinFullNpu::Backend>, KirinFullNpu::Log, KirinFullNpu::Cancel = {});
    ~Vision();
    bool load(const std::string& config, std::string& error);
    bool run(const std::string& path, Features& features, std::string& error);
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

// Accepts a dedicated hybrid package: ordinary MNN CPU assets + vision/config.json.
// The backend is injectable for host reference tests. Production supplies NNRT.
std::unique_ptr<MNN::Transformer::Llm> create(
    const std::string& config, std::unique_ptr<KirinFullNpu::Backend> backend,
    KirinFullNpu::Log log, KirinFullNpu::Cancel cancel,
    const std::shared_ptr<State>& state);
// Retry after failure/cancellation, before response()'s initial status guard.
void prepareRequest(MNN::Transformer::Llm*, const std::shared_ptr<State>&);
} // namespace OfflineNpu2
