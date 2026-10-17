#pragma once
#include "FullNpuRuntime.hpp"

namespace KirinFullNpu {
inline bool embeddingSliceGraphs(const Graph& g, std::vector<Graph>& slices, std::string& error) {
    slices.clear();
    if ((g.name != "embedding_prefill" && g.name != "embedding_decode") ||
        g.inputs.size() != 1 || g.outputs.size() != 1 || !g.aliases.empty() ||
        g.inputs[0].name != "token_ids" || g.inputs[0].dtype != DType::I32 ||
        g.inputs[0].shape.size() != 2 || g.inputs[0].shape[0] != 1 || g.inputs[0].shape[1] < 1 ||
        g.outputs[0].name != "embeddings" || g.outputs[0].dtype != DType::F32 ||
        g.outputs[0].shape.size() != 3 || g.outputs[0].shape[0] != 1 ||
        g.outputs[0].shape[1] != g.inputs[0].shape[1] || g.outputs[0].shape[2] < 1 ||
        g.embeddingSlices.empty() || g.embeddingSlices.size() > 128) {
        error = g.name + ": invalid sliced embedding interface";
        return false;
    }
    int end = 0;
    for (const auto& part : g.embeddingSlices) {
        if (part.path.empty() || part.begin != end || part.end <= part.begin || part.end > g.outputs[0].shape[2]) {
            error = g.name + ": incomplete/overlapping embedding slices";
            return false;
        }
        Graph physical;
        physical.name = g.name + "_slice_" + std::to_string(slices.size());
        physical.path = part.path;
        physical.inputs = g.inputs; physical.outputs = g.outputs;
        physical.outputs[0].shape[2] = part.end - part.begin;
        slices.push_back(std::move(physical));
        end = part.end;
    }
    if (end != g.outputs[0].shape[2]) {
        error = g.name + ": incomplete embedding hidden dimension";
        return false;
    }
    return true;
}
} // namespace KirinFullNpu
