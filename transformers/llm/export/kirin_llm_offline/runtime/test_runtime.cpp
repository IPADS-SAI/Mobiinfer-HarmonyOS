#include "FullNpuRuntime.hpp"
#include <cassert>
#include <iostream>
#include <limits>
#include <fstream>
#include <unistd.h>
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"

namespace {
class FakeBackend final : public KirinFullNpu::Backend {
public:
    bool nonfinite = false, fail = false, failProbe = false;
    int calls = 0, releases = 0, probes = 0;
    KirinFullNpu::BackendOptions configured;
    void configure(const KirinFullNpu::BackendOptions& options) override { configured = options; }
    bool probe(const KirinFullNpu::Graph&, std::string& error) override {
        ++probes;
        if (failProbe) { error = "injected probe failure"; return false; }
        return true;
    }
    void release() override { ++releases; }
    bool run(const KirinFullNpu::Graph& g, const std::vector<KirinFullNpu::View>&,
             std::vector<std::vector<float>>& outputs, std::string& error) override {
        ++calls;
        if (fail) { error = "injected backend failure"; return false; }
        outputs.clear();
        for (const auto& port : g.outputs) outputs.emplace_back(port.elements(), 0.0f);
        if (g.name == "lm_head") outputs[0][42] = 1;
        if (nonfinite && g.name == "llm_decode_0") outputs[0][0] = std::numeric_limits<float>::quiet_NaN();
        return true;
    }
};
}
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    {
        std::ifstream file(argv[1]);
        std::string original(std::istreambuf_iterator<char>(file), {});
        char* absolute = realpath(argv[1], nullptr); assert(absolute);
        std::string root(absolute); free(absolute); root.erase(root.find_last_of('/') + 1);
        char dir[] = "/tmp/kirin_runtime_contracts_XXXXXX"; assert(mkdtemp(dir));
        std::string temporary(dir);
        for (const char* name : {"tokenizer.mtok", "vision_constants.bin"})
            assert(symlink((root + name).c_str(), (temporary + "/" + name).c_str()) == 0);
        const bool fused = original.find("kirin_full_npu_runtime_v3") != std::string::npos;
        for (int mutation = 0; mutation < (fused ? 30 : 22); ++mutation) {
            rapidjson::Document doc; doc.Parse(original.c_str()); assert(!doc.HasParseError());
            if (mutation == 0) doc["runtime"]["hidden"].SetInt(doc["runtime"]["hidden"].GetInt() * 2);
            if (mutation == 1) doc["runtime"]["vision_hidden"].SetInt(-1024);
            if (mutation == 2) doc["runtime"]["capacity"].SetInt(1024);
            if (mutation == 3) doc["runtime"]["mrope_section"][0].SetString("invalid", doc.GetAllocator());
            // Mutate an existing option too: external v2 packages declare
            // these fields, whereas legacy v1 fixtures can omit them.
            const char* option = mutation == 10 ? "npu_max_resident_graphs" :
                mutation == 11 ? "npu_persistent_kv_mib" : mutation == 12 ? "npu_incremental_kv" :
                mutation == 13 ? "npu_resident_model_mib" : mutation >= 20 ? "kv_dtype" : nullptr;
            if (option && doc["runtime"].HasMember(option)) doc["runtime"].RemoveMember(option);
            if (mutation == 10) doc["runtime"].AddMember("npu_max_resident_graphs", 0, doc.GetAllocator());
            if (mutation == 11) doc["runtime"].AddMember("npu_persistent_kv_mib", -1, doc.GetAllocator());
            if (mutation == 12) doc["runtime"].AddMember("npu_incremental_kv", 1, doc.GetAllocator());
            if (mutation == 13) doc["runtime"].AddMember("npu_resident_model_mib", 0, doc.GetAllocator());
            if (mutation == 20) doc["runtime"].AddMember("kv_dtype", "fp32", doc.GetAllocator());
            if (mutation == 21) {
                doc["runtime"].AddMember("kv_dtype", "fp16", doc.GetAllocator());
                doc["runtime"]["kv_recent"].SetInt(1); // FP16 has no INT8 protected windows.
            }
            if (mutation == 22) doc["runtime"]["vision_deepstack_layers"][1].SetInt(0);
            if (mutation == 23) doc["runtime"]["vision_deepstack_layers"][0].SetInt(doc["runtime"]["vision_layers"].GetInt());
            if (mutation == 24) doc["runtime"]["vision_deepstack_layers"].Clear();
            if (mutation == 25) doc["runtime"]["vision_deepstack_layers"].PopBack();
            if (mutation == 29) doc["runtime"].RemoveMember("vision_layers");
            if (mutation == 28) {
                rapidjson::Value extra(doc["graphs"][0], doc.GetAllocator());
                extra["name"].SetString("vision_pre", doc.GetAllocator());
                doc["graphs"].PushBack(extra, doc.GetAllocator());
            }
            for (auto& graph : doc["graphs"].GetArray()) {
                const auto name = std::string(graph["name"].GetString());
                if (mutation == 4 && (name == "vision_post" || name == "vision_full"))
                    graph["output_info"][0]["shape"][1].SetInt(1);
                if (mutation == 26 && name == "vision_full") graph["layer_end"].SetInt(1);
                if (mutation == 27 && name == "vision_full") {
                    rapidjson::Value aliases(rapidjson::kObjectType);
                    aliases.AddMember("image_embeddings", "deepstack_features", doc.GetAllocator());
                    graph["output_aliases"].Swap(aliases);
                }
                if (mutation == 5 && std::string(graph["name"].GetString()) == "llm_prefill_0")
                    graph["inputs"][5].Swap(graph["inputs"][6]);
                if (mutation >= 6 && mutation <= 9 && std::string(graph["name"].GetString()) ==
                    (mutation == 9 ? "lm_head" : "embedding_decode")) {
                    rapidjson::Value parts(rapidjson::kArrayType);
                    if (mutation != 6) {
                        rapidjson::Value part(rapidjson::kObjectType);
                        part.AddMember("file", "om/slice.omc", doc.GetAllocator());
                        part.AddMember("hidden_start", mutation == 7 ? 1 : 0, doc.GetAllocator());
                        part.AddMember("hidden_end", doc["runtime"]["hidden"].GetInt() - (mutation == 8), doc.GetAllocator());
                        parts.PushBack(part, doc.GetAllocator());
                    }
                    if (graph.HasMember("embedding_slices")) graph.RemoveMember("embedding_slices");
                    graph.AddMember("embedding_slices", parts, doc.GetAllocator());
                }
                if (mutation >= 14 && mutation <= 19 && std::string(graph["name"].GetString()) == "embedding_decode") {
                    rapidjson::Value weights(rapidjson::kObjectType), files(rapidjson::kArrayType);
                    weights.AddMember("directory", rapidjson::Value(mutation == 14 ? "../weights" : "weights", doc.GetAllocator()), doc.GetAllocator());
                    if (mutation != 15) files.PushBack(rapidjson::Value(mutation == 16 ? "../bad.weight" : "SubGraph_0.weight", doc.GetAllocator()), doc.GetAllocator());
                    if (mutation == 17) files.PushBack("SubGraph_0.weight", doc.GetAllocator());
                    if (mutation == 18) files[0].SetInt(42);
                    weights.AddMember("files", files, doc.GetAllocator());
                    if (mutation == 19) weights.SetString("invalid", doc.GetAllocator());
                    if (graph.HasMember("external_weights")) graph.RemoveMember("external_weights");
                    graph.AddMember("external_weights", weights, doc.GetAllocator());
                }
            }
            rapidjson::StringBuffer buffer; rapidjson::Writer<rapidjson::StringBuffer> writer(buffer); doc.Accept(writer);
            auto path = temporary + "/config.json";
            { std::ofstream stream(path); stream << buffer.GetString(); }
            auto* backend = new FakeBackend;
            KirinFullNpu::Runtime invalid(std::unique_ptr<KirinFullNpu::Backend>(backend), {});
            std::string error;
            assert(!invalid.load(path, error) && !error.empty() && backend->probes == 0);
        }
        unlink((temporary + "/config.json").c_str());
        unlink((temporary + "/tokenizer.mtok").c_str());
        unlink((temporary + "/vision_constants.bin").c_str()); rmdir(dir);
    }
    {
        auto* backend = new FakeBackend;
        backend->failProbe = true;
        KirinFullNpu::Runtime failed(std::unique_ptr<KirinFullNpu::Backend>(backend), {});
        std::string error;
        assert(!failed.load(argv[1], error));
        assert(error.find("graph=embedding_decode injected probe failure") != std::string::npos);
        assert(backend->releases >= 2);
    }
    {
        // The App's explicit budget must override an older package's limits,
        // without replacing its KV pool budget or incremental-copy option.
        auto* backend = new FakeBackend;
        KirinFullNpu::Runtime expanded(std::unique_ptr<KirinFullNpu::Backend>(backend), {});
        std::string error;
        assert(expanded.load(argv[1], error));
        const auto original = backend->configured;
        assert(expanded.load(argv[1], error, {16, size_t(4096) << 20}));
        assert(backend->configured.maxResidentGraphs == 16);
        assert(backend->configured.residentModelBytes == (size_t(4096) << 20));
        assert(backend->configured.persistentKvBytes == original.persistentKvBytes);
        assert(backend->configured.incrementalKv == original.incrementalKv);
        assert(expanded.profile().backend.maxResidentGraphs == 16);
        const int probes = backend->probes;
        assert(!expanded.load(argv[1], error, {17, size_t(4096) << 20}));
        assert(error == "Invalid caller NNRT residency budget" && backend->probes == probes);
        assert(!expanded.load(argv[1], error, {16, (size_t(4096) << 20) + 1}));
        assert(error == "Invalid caller NNRT residency budget" && backend->probes == probes);
        assert(expanded.load(argv[1], error));
        assert(backend->configured.maxResidentGraphs == original.maxResidentGraphs);
        assert(backend->configured.residentModelBytes == original.residentModelBytes);
    }
    auto* fake = new FakeBackend;
    KirinFullNpu::Runtime runtime(std::unique_ptr<KirinFullNpu::Backend>(fake), {});
    std::string error, output, streamed;
    assert(runtime.load(argv[1], error));
    assert(runtime.load(argv[1], error)); // Reload must not retain old graph entries.
    auto input = KirinFullNpu::Runtime::chatTemplate({{"user", "Hello"}});
    assert(!runtime.response(input, [] { return true; }, {}, output, error, 2));
    assert(error == "cancelled" && fake->calls == 0);
    assert(!runtime.response("<img>missing</img>", {}, {}, output, error, 2));
    assert(error.find("decode image") != std::string::npos);
    assert(!runtime.response("<img><hw>bad</img>", {}, {}, output, error, 2));
    assert(error.find("Malformed") != std::string::npos);
    std::string longInput;
    for (int i = 0; i < 3000; ++i) longInput += " Hello";
    assert(!runtime.response(longInput, {}, {}, output, error, 2));
    assert(error.find("context limit") != std::string::npos);
    fake->fail = true;
    assert(!runtime.response(input, {}, {}, output, error, 2));
    assert(error.find("injected") != std::string::npos);
    fake->fail = false; fake->nonfinite = true;
    assert(!runtime.response(input, {}, {}, output, error, 2));
    assert(error.find("nonfinite") != std::string::npos);
    fake->nonfinite = false;
    assert(runtime.response(input, {}, [&](const std::string& s) { streamed += s; }, output, error, 2));
    assert(output == streamed && !output.empty());
    int at = fake->calls;
    assert(!runtime.response(input, [&] { return fake->calls >= at + 3; }, {}, output, error, 2));
    assert(error == "cancelled");
    assert(runtime.response(input, {}, {}, output, error, 1));
    std::cout << "PASS: cancellation, missing images, malformed tags, overflow, backend failure, nonfinite rejection, streaming and retry\n";
}
