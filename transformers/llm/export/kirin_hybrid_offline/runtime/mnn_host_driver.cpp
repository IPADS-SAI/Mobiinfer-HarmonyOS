// Exercise the complete MNN visual path on host CPU. HiAI is not emulated.
#include "llm/llm.hpp"
#include "rapidjson/document.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>

namespace fs = std::filesystem;
template <class T> bool save(const fs::path& path, const T* values, size_t count) {
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(values), count * sizeof(T));
    return bool(stream);
}
int main(int argc, char** argv) {
    if (argc != 5) return 2; // config, image, output, image copies
    fs::path output = argv[3];
    fs::create_directories(output);
    std::ifstream stream(argv[1]);
    std::string text{std::istreambuf_iterator<char>(stream), {}};
    rapidjson::Document config;
    config.Parse(text.c_str());
    if (config.HasParseError() || !config.HasMember("visual_blocks_chunks") ||
        !config["visual_blocks_chunks"].IsArray()) return 3;
    std::string backends;
    for (size_t i = 0; i < config["visual_blocks_chunks"].Size(); ++i)
        backends += (i ? ",\"cpu\"" : "\"cpu\"");
    std::unique_ptr<MNN::Transformer::Llm> llm(MNN::Transformer::Llm::createLLM(argv[1]));
    std::string overrides = "{\"backend_type\":\"cpu\",\"thread_num\":4,\"mllm\":{\"backend_type\":\"cpu\","
        "\"thread_num\":4,\"precision\":\"normal\",\"memory\":\"low\"},\"visual_blocks_backend_type\":\"cpu\","
        "\"visual_blocks_chunk_backends\":[" + backends + "],\"sampler_type\":\"greedy\",\"prompt_cache\":false}";
    if (!llm || !llm->set_config(overrides) || !llm->load()) return 4;
    int copies = std::stoi(argv[4]);
    if (copies < 1 || copies > 2) return 5;
    std::string prompt = "<|im_start|>user\n请描述这张图片：";
    for (int i = 0; i < copies; ++i) prompt += "<img>" + fs::absolute(argv[2]).string() + "</img>";
    prompt += "<|im_end|>\n<|im_start|>assistant\n";
    auto ids = llm->tokenizer_encode(prompt);
    if (ids.empty() || !save(output / "tokens.bin", ids.data(), ids.size())) return 6;
    std::ostringstream answer;
    llm->response(ids, &answer, nullptr, 0);
    auto values = llm->getOutputs();
    if (values.empty() || values[0].get() == nullptr || !values[0]->getInfo()) return 7;
    auto logits = values[0]->readMap<float>();
    auto count = values[0]->getInfo()->size;
    if (!logits || count == 0) return 8;
    for (int i = 0; i < count; ++i) if (!std::isfinite(logits[i])) return 9;
    if (!save(output / "logits.bin", logits, count)) return 10;
    llm->generate(3);
    auto tokens = llm->getContext()->output_tokens;
    if (tokens.empty() || !save(output / "generated.bin", tokens.data(), tokens.size())) return 11;
    std::ofstream(output / "answer.txt") << answer.str();
    std::cerr << "MNN VISUAL HOST SUCCESS input_tokens=" << ids.size()
              << " generated=" << tokens.size() << " answer=" << answer.str() << '\n';
    return 0;
}
