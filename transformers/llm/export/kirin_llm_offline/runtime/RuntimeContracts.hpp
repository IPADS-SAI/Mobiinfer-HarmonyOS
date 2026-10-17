#pragma once
#include "FullNpuRuntime.hpp"
#include "EmbeddingSlices.hpp"
#include <algorithm>
#include <set>

namespace KirinFullNpu {

// Logical ONNX interfaces are independent of singleton insertion and FP16
// boundary conversion performed by NNRT. Check them before copying features.
inline bool validateRuntimeGraphs(const Profile& p, const std::map<std::string, Graph>& graphs,
                                 const std::vector<std::string>& prefill,
                                 const std::vector<std::string>& decode,
                                 const std::vector<std::string>& vision, std::string& error) {
    auto check = [&](const Graph& g, const std::vector<Port>& inputs, const std::vector<Port>& outputs) {
        for (bool input : {true, false}) {
            const auto& actual = input ? g.inputs : g.outputs;
            const auto& expected = input ? inputs : outputs;
            if (actual.size() != expected.size()) {
                error = g.name + ": logical port count inconsistent with runtime profile";
                return false;
            }
            std::set<std::string> names;
            for (size_t i = 0; i < actual.size(); ++i) {
                if (actual[i].name != expected[i].name || actual[i].dtype != expected[i].dtype ||
                    actual[i].shape != expected[i].shape || !names.insert(actual[i].name).second) {
                    error = g.name + ": logical port inconsistent with runtime profile: " + expected[i].name;
                    return false;
                }
            }
        }
        for (const auto& alias : g.aliases) {
            auto a = std::find_if(outputs.begin(), outputs.end(), [&](const Port& port) { return port.name == alias.first; });
            auto b = std::find_if(outputs.begin(), outputs.end(), [&](const Port& port) { return port.name == alias.second; });
            if (a == outputs.end() || b == outputs.end() || a->shape != b->shape || a->dtype != b->dtype || a == b) {
                error = g.name + ": invalid logical output alias";
                return false;
            }
        }
        return true;
    };
    if (graphs.size() != prefill.size() + decode.size() + vision.size() + (p.fusedVision ? 4 : 5)) {
        error = "Unused, missing or nonconsecutive full-NPU graph";
        return false;
    }
    for (const auto& item : graphs) {
        if (item.second.embeddingSlices.empty()) continue;
        std::vector<Graph> slices;
        if (!embeddingSliceGraphs(item.second, slices, error)) return false;
    }
    const int n = (p.imageHeight / p.patch) * (p.imageWidth / p.patch);
    const int imageTokens = n / (p.merge * p.merge);
    const int visionDim = p.visionHidden / p.visionHeads;
    const std::vector<int> visualShape{1, n, p.visionHidden};
    for (bool decoding : {false, true}) {
        int width = decoding ? 1 : p.prefill;
        const auto& embedding = graphs.at(decoding ? "embedding_decode" : "embedding_prefill");
        if (!check(embedding, {{"token_ids", DType::I32, {1, width}}},
                   {{"embeddings", DType::F32, {1, width, p.hidden}}})) return false;
        for (const auto& name : decoding ? decode : prefill) {
            const auto& g = graphs.at(name);
            int count = g.end - g.start;
            std::vector<Port> inputs{{"hidden_states_in", DType::F32, {1, width, p.hidden}},
                {"rope_cos", DType::F32, {1, 1, width, p.dim}},
                {"rope_sin", DType::F32, {1, 1, width, p.dim}},
                {"attention_mask", DType::F32, {1, 1, width, p.capacity + p.sink + p.recent + width}},
                {"deepstack", DType::F32, {count, 1, width, p.hidden}}};
            std::vector<Port> outputs{{"hidden_states", DType::F32, {1, width, p.hidden}}};
            for (int layer = g.start; layer < g.end; ++layer) {
                auto id = std::to_string(layer);
                for (const char* name : {"past_key_", "past_value_"})
                    inputs.push_back({name + id, p.kvType, {1, p.heads, p.capacity, p.dim}});
                if (p.kvType == DType::I8) for (const char* name : {"past_key_scale_", "past_value_scale_"})
                    inputs.push_back({name + id, DType::F32, {1, p.heads, p.capacity, p.dim / p.group}});
                if (p.sink) for (const char* name : {"sink_key_", "sink_value_"})
                    inputs.push_back({name + id, DType::F16, {1, p.heads, p.sink, p.dim}});
                if (p.recent) for (const char* name : {"recent_key_", "recent_value_"})
                    inputs.push_back({name + id, DType::F16, {1, p.heads, p.recent, p.dim}});
                for (const char* name : {"key_delta_", "value_delta_"})
                    outputs.push_back({name + id, DType::F32, {1, p.heads, width, p.dim}});
            }
            if (!check(g, inputs, outputs)) return false;
        }
    }
    const auto& head = graphs.at("lm_head");
    if (head.outputs.size() != 1 || head.outputs[0].shape.size() != 3 ||
        head.outputs[0].shape[2] <= std::max(p.padToken, p.imageToken)) {
        error = "lm_head: invalid vocabulary shape";
        return false;
    }
    if (!check(head, {{"hidden_states", DType::F32, {1, 1, p.hidden}}},
               {{"logits", DType::F32, {1, 1, head.outputs[0].shape[2]}}})) return false;
    if (p.fusedVision) {
        const auto& g = graphs.at("vision_full");
        if (g.start != 0 || g.end != p.visionLayers || !vision.empty() || !g.aliases.empty()) {
            error = "Invalid fused visual graph coverage or aliases"; return false;
        }
        return check(g,
            {{"patches", DType::F32, {n, 3 * p.temporal * p.patch * p.patch}},
             {"position_indices", DType::I32, {n, 4}}, {"position_weights", DType::F32, {n, 4}},
             {"rope_cos", DType::F32, {1, 1, n, visionDim}},
             {"rope_sin", DType::F32, {1, 1, n, visionDim}},
             {"attention_mask", DType::F32, {1, 1, n, n}}},
            {{"image_embeddings", DType::F32, {1, imageTokens, p.hidden}},
             {"deepstack_features", DType::F32, {int(p.visionDeepstack.size()), 1, imageTokens, p.hidden}}});
    }
    if (!check(graphs.at("vision_pre"),
               {{"patches", DType::F32, {n, 3 * p.temporal * p.patch * p.patch}},
                {"position_indices", DType::I32, {n, 4}}, {"position_weights", DType::F32, {n, 4}}},
               {{"hidden_states", DType::F32, visualShape}})) return false;
    std::vector<Port> postInputs{{"hidden_states", DType::F32, visualShape}};
    for (const auto& name : vision) {
        const auto& g = graphs.at(name);
        if (g.outputs.empty() || g.end > 128) { error = name + ": invalid visual graph"; return false; }
        std::vector<Port> outputs{{"hidden_states", DType::F32, visualShape}};
        for (size_t i = 1; i < g.outputs.size(); ++i) {
            auto expected = "deepstack_";
            bool found = false;
            for (int layer = g.start; layer < g.end; ++layer)
                found = found || g.outputs[i].name == expected + std::to_string(layer);
            if (!found) { error = name + ": invalid DeepStack layer"; return false; }
            Port feature{g.outputs[i].name, DType::F32, visualShape};
            outputs.push_back(feature); postInputs.push_back(feature);
        }
        if (!check(g, {{"hidden_states_in", DType::F32, visualShape},
                       {"rope_cos", DType::F32, {1, 1, n, visionDim}},
                       {"rope_sin", DType::F32, {1, 1, n, visionDim}},
                       {"attention_mask", DType::F32, {1, 1, n, n}}}, outputs)) return false;
    }
    int depth = int(postInputs.size()) - 1;
    if (depth < 1 || depth > p.layers) { error = "Invalid visual DeepStack depth"; return false; }
    return check(graphs.at("vision_post"), postInputs,
                 {{"image_embeddings", DType::F32, {1, imageTokens, p.hidden}},
                  {"deepstack_features", DType::F32, {depth, 1, imageTokens, p.hidden}}});
}
} // namespace KirinFullNpu
