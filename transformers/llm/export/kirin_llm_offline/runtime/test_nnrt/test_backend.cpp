#include "FakeNnrt.hpp"
#include "NnrtBackend.hpp"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

struct OH_NNCompilation { int slice = -1; bool kv = false; };
struct OH_NNExecutor { int slice = -1; bool kv = false; };
struct NN_TensorDesc { bool input; int slice = -1; bool kv = false; size_t index = 0; };
struct NN_Tensor { std::vector<float> data = std::vector<float>(2); };
namespace {
std::string failure;
std::vector<std::string> calls;
int compilationsDestroyed = 0, executorsDestroyed = 0, tensorsDestroyed = 0;
int liveExecutors = 0, peakExecutors = 0, liveTensors = 0;
int executorLimit = 1;
std::vector<int> executorsAtBuild;
std::string slicePaths[2];
std::string kvPath;
int kvFormat = 0;
bool failKvFormat = false;
std::vector<KirinFullNpu::Port> kvPorts;
std::vector<KirinFullNpu::View> expectedKv;
bool failSliceRun = false;
bool failTensorOnce = false;
OH_NN_ReturnCode step(const char* name) {
    calls.emplace_back(name);
    return failure == name ? OH_NN_FAILED : OH_NN_SUCCESS;
}
}
const char* HMS_HiAI_GetVersion() { return "host-test-substitute"; }
// A real C symbol: production resolves the optional device API with dlsym.
extern "C" OH_NN_ReturnCode KirinTest_InitWeights(OH_NNExecutor*, const char* directory) {
    std::ifstream weights(std::string(directory) + "/SubGraph_0.weight", std::ios::binary);
    assert(std::string(std::istreambuf_iterator<char>(weights), {}) == "fixture");
    return step("HMS_HiAIExecutor_InitWeights");
}
#ifndef KIRIN_TEST_LOCAL_HIAI
extern "C" OH_NN_ReturnCode HMS_HiAIExecutor_InitWeights(OH_NNExecutor* executor, const char* directory) {
    return KirinTest_InitWeights(executor, directory);
}
#endif
HiAI_Compatibility HMS_HiAICompatibility_CheckFromFile(const char*) { return HIAI_COMPATIBILITY_COMPATIBLE; }
OH_NN_ReturnCode OH_NNDevice_GetAllDevicesID(const size_t** ids, uint32_t* count) {
    static const size_t id = 7; *ids = &id; *count = 1; return OH_NN_SUCCESS;
}
OH_NN_ReturnCode OH_NNDevice_GetName(size_t, const char** name) { *name = "HIAI_F"; return OH_NN_SUCCESS; }
OH_NNCompilation* OH_NNCompilation_ConstructWithOfflineModelFile(const char* path) {
    auto* result = new OH_NNCompilation;
    for (int i = 0; i < 2; ++i) if (slicePaths[i] == path) result->slice = i;
    result->kv = kvPath == path;
    return result;
}
OH_NN_ReturnCode OH_NNCompilation_SetDevice(OH_NNCompilation*, size_t) { return step("OH_NNCompilation_SetDevice"); }
OH_NN_ReturnCode HMS_HiAIOptions_SetModelDeviceOrder(OH_NNCompilation*, HiAI_ExecuteDevice*, size_t) {
    return step("HMS_HiAIOptions_SetModelDeviceOrder");
}
OH_NN_ReturnCode HMS_HiAIOptions_SetFallbackMode(OH_NNCompilation*, HiAI_FallbackMode) {
    return step("HMS_HiAIOptions_SetFallbackMode");
}
OH_NN_ReturnCode HMS_HiAIOptions_SetBandMode(OH_NNCompilation*, HiAI_BandMode) {
    return step("HMS_HiAIOptions_SetBandMode_NORMAL");
}
OH_NN_ReturnCode OH_NNCompilation_Build(OH_NNCompilation*) {
    executorsAtBuild.push_back(liveExecutors);
    return step("OH_NNCompilation_Build");
}
void OH_NNCompilation_Destroy(OH_NNCompilation** p) { delete *p; *p = nullptr; ++compilationsDestroyed; }
OH_NNExecutor* OH_NNExecutor_Construct(OH_NNCompilation* c) {
    // Model a resource-constrained service: Build succeeds but another live
    // model prevents executor preparation, matching the device failure stage.
    if (step("OH_NNExecutor_Construct") != OH_NN_SUCCESS || liveExecutors >= executorLimit) return nullptr;
    peakExecutors = std::max(peakExecutors, ++liveExecutors);
    return new OH_NNExecutor{c->slice, c->kv};
}
void OH_NNExecutor_Destroy(OH_NNExecutor** p) {
    delete *p; *p = nullptr; ++executorsDestroyed; --liveExecutors;
}
OH_NN_ReturnCode OH_NNExecutor_GetInputCount(const OH_NNExecutor* e, size_t* count) {
    *count = e->kv ? kvPorts.size() : 1; return OH_NN_SUCCESS;
}
OH_NN_ReturnCode OH_NNExecutor_GetOutputCount(const OH_NNExecutor*, size_t* count) { *count = 1; return OH_NN_SUCCESS; }
NN_TensorDesc* OH_NNExecutor_CreateInputTensorDesc(const OH_NNExecutor* e, size_t i) {
    return new NN_TensorDesc{true, e->slice, e->kv, i};
}
NN_TensorDesc* OH_NNExecutor_CreateOutputTensorDesc(const OH_NNExecutor* e, size_t) { return new NN_TensorDesc{false, e->slice}; }
OH_NN_ReturnCode OH_NNTensorDesc_GetName(const NN_TensorDesc* d, const char** name) {
    if (d->kv && d->input) { *name = kvPorts[d->index].name.c_str(); return OH_NN_SUCCESS; }
    *name = d->slice >= 0 ? (d->input ? "token_ids" : "embeddings") :
                           (d->input ? "hidden_states_in" : "hidden_states");
    return OH_NN_SUCCESS;
}
OH_NN_ReturnCode OH_NNTensorDesc_GetDataType(const NN_TensorDesc* d, OH_NN_DataType* type) {
    if (d->kv && d->input) {
        auto t = kvPorts[d->index].dtype;
        *type = t == KirinFullNpu::DType::I8 ? OH_NN_INT8 : t == KirinFullNpu::DType::F16 ? OH_NN_FLOAT16 : OH_NN_FLOAT32;
        return OH_NN_SUCCESS;
    }
    *type = d->slice >= 0 && d->input ? OH_NN_INT32 : OH_NN_FLOAT32; return OH_NN_SUCCESS;
}
OH_NN_ReturnCode OH_NNTensorDesc_GetFormat(const NN_TensorDesc* d, OH_NN_Format* format) {
    if (failKvFormat && d->kv && d->input) return OH_NN_FAILED;
    *format = static_cast<OH_NN_Format>(d->kv && d->input ? kvFormat : 0); return OH_NN_SUCCESS;
}
OH_NN_ReturnCode OH_NNTensorDesc_GetShape(const NN_TensorDesc* d, int32_t** shape, size_t* rank) {
    if (d->kv && d->input) {
        *shape = kvPorts[d->index].shape.data(); *rank = kvPorts[d->index].shape.size(); return OH_NN_SUCCESS;
    }
    static int32_t sliceDims[][3] = {{1, 2, 2}, {1, 2, 1}};
    if (d->slice >= 0 && !d->input) { *shape = sliceDims[d->slice]; *rank = 3; return OH_NN_SUCCESS; }
    static int32_t dims[] = {1, 2}; *shape = dims; *rank = 2; return OH_NN_SUCCESS;
}
OH_NN_ReturnCode OH_NNTensorDesc_GetByteSize(const NN_TensorDesc* d, size_t* size) {
    if (d->kv && d->input) {
        const auto& p = kvPorts[d->index]; *size = p.elements() * KirinFullNpu::elementSize(p.dtype); return OH_NN_SUCCESS;
    }
    *size = d->slice == 0 && !d->input ? 16 : 8; return OH_NN_SUCCESS;
}
void OH_NNTensorDesc_Destroy(NN_TensorDesc** d) { delete *d; *d = nullptr; }
NN_Tensor* OH_NNTensor_Create(size_t, const NN_TensorDesc* d) {
    if (failTensorOnce) { failTensorOnce = false; return nullptr; }
    auto* t = new NN_Tensor; size_t bytes = 0; OH_NNTensorDesc_GetByteSize(d, &bytes); t->data.resize(bytes / 4);
    ++liveTensors; return t;
}
void OH_NNTensor_Destroy(NN_Tensor** p) { delete *p; *p = nullptr; ++tensorsDestroyed; --liveTensors; }
void* OH_NNTensor_GetDataBuffer(NN_Tensor* tensor) { return tensor->data.data(); }
OH_NN_ReturnCode OH_NNTensor_GetSize(NN_Tensor* t, size_t* size) { *size = t->data.size() * 4; return OH_NN_SUCCESS; }
OH_NN_ReturnCode OH_NNExecutor_RunSync(OH_NNExecutor* e, NN_Tensor* in[], size_t count, NN_Tensor* out[], size_t) {
    if (e->kv) {
        assert(count == expectedKv.size());
        for (size_t i = 0; i < count; ++i)
            assert(!std::memcmp(in[i]->data.data(), expectedKv[i].data,
                               expectedKv[i].elements * KirinFullNpu::elementSize(expectedKv[i].dtype)));
        out[0]->data[0] = float(expectedKv[0].cacheTokens); out[0]->data[1] = 1;
        return failure == "OH_NNExecutor_RunSync" ? OH_NN_FAILED : OH_NN_SUCCESS;
    }
    if (e->slice >= 0) {
        if (e->slice == 1 && failSliceRun) return OH_NN_FAILED;
        int width = e->slice == 0 ? 2 : 1, start = e->slice == 0 ? 0 : 2;
        int32_t ids[2]; std::memcpy(ids, in[0]->data.data(), sizeof(ids));
        for (int row = 0; row < 2; ++row) for (int col = 0; col < width; ++col)
            out[0]->data[row * width + col] = ids[row] * 10 + start + col;
        return OH_NN_SUCCESS;
    }
    for (int i = 0; i < 2; ++i) out[0]->data[i] = in[0]->data[i] * 2;
    return OH_NN_SUCCESS;
}

int main(int argc, char** argv) {
    using namespace KirinFullNpu;
    const std::string mode = argc > 1 ? argv[1] : "global";
    void* localLibrary = nullptr;
#ifdef KIRIN_TEST_LOCAL_HIAI
    assert(!dlsym(RTLD_DEFAULT, "HMS_HiAIExecutor_InitWeights"));
    if (mode == "local") {
        localLibrary = dlopen("libhiai_foundation.so", RTLD_LAZY | RTLD_LOCAL);
        assert(localLibrary);
        // Reproduces the original failure: the real API exists but is not global.
        assert(dlsym(localLibrary, "HMS_HiAIExecutor_InitWeights"));
        assert(!dlsym(RTLD_DEFAULT, "HMS_HiAIExecutor_InitWeights"));
    }
#endif
    char path[] = "/tmp/kirin_nnrt_loading_test_XXXXXX";
    int fd = mkstemp(path); assert(fd >= 0); assert(write(fd, "fixture", 7) == 7); close(fd);
    Graph graph; graph.name = "embedding_decode"; graph.path = path;
    graph.inputs.push_back({"hidden_states_in", DType::F32, {1, 2}});
    graph.outputs.push_back({"hidden_states", DType::F32, {1, 2}});
    {
        char directory[] = "/tmp/kirin_external_weights_XXXXXX"; assert(mkdtemp(directory));
        const auto weightPath = std::string(directory) + "/SubGraph_0.weight";
        { std::ofstream file(weightPath); file << "fixture"; }
        Graph external = graph; external.weightDir = directory; external.weightFiles = {"SubGraph_0.weight"};
        std::string error, logs;
        auto backend = createNnrtBackend([&](const std::string& s) { logs += s + "\n"; });
        calls.clear(); compilationsDestroyed = executorsDestroyed = tensorsDestroyed = 0;
        if (mode == "missing" || mode == "unavailable") {
            assert(!backend->probe(external, error));
            assert(calls.empty()); // No Build or executor on a truly unavailable API.
            assert(error.find("symbol lookup failed") != std::string::npos);
            assert(error.find("hiai_version=host-test-substitute") != std::string::npos);
            assert(error.find("global_symbol_error=") != std::string::npos);
            assert(error.find(mode == "missing" ? "library_symbol_error=" : "dlopen=failed") != std::string::npos);
            assert(backend->probe(graph, error)); // Embedded packages need no InitWeights.
            backend->release();
            unlink(weightPath.c_str()); rmdir(directory); unlink(path);
            std::puts("PASS: unavailable external API fails before Build with diagnostics; embedded models still load");
            return 0;
        }
        failure = "HMS_HiAIExecutor_InitWeights";
        assert(!backend->probe(external, error));
        if (localLibrary) {
            assert(logs.find("lookup=library_handle") != std::string::npos);
            assert(logs.find("provider=") != std::string::npos);
            dlclose(localLibrary); localLibrary = nullptr;
            // The backend's own reference must keep the function callable below.
            assert(!dlsym(RTLD_DEFAULT, "HMS_HiAIExecutor_InitWeights"));
        }
        assert(error.find("HMS_HiAIExecutor_InitWeights") != std::string::npos);
        assert(compilationsDestroyed == 1 && executorsDestroyed == 1 && tensorsDestroyed == 0);
        failure.clear(); calls.clear();
        BackendOptions options; options.residentModelBytes = 13; backend->configure(options);
        assert(!backend->probe(external, error) && calls.empty()); // OMC plus weights need 14 bytes.
        options.residentModelBytes = 100; options.maxResidentGraphs = 4;
        backend->configure(options); executorLimit = 4;
        calls.clear();
        float values[] = {1, 2}; std::vector<std::vector<float>> outputs;
        for (int token = 0; token < 5; ++token) {
            for (const char* name : {"embedding_decode", "llm_decode_0", "lm_head"}) {
                external.name = name;
                assert(backend->run(external, {{DType::F32, values, 2}}, outputs, error));
                assert(outputs == std::vector<std::vector<float>>({{2, 4}}));
            }
        }
        assert(std::count(calls.begin(), calls.end(), "OH_NNCompilation_Build") == 3);
        assert(std::count(calls.begin(), calls.end(), "HMS_HiAIExecutor_InitWeights") == 3);
        assert(logs.find("external_weight_bytes=7") != std::string::npos);
        backend->release(); assert(liveExecutors == 0 && liveTensors == 0);
        calls.clear(); external.weightFiles = {"missing.weight"};
        assert(!backend->probe(external, error) && calls.empty());
        assert(error.find("External weight file missing") != std::string::npos);
        unlink(weightPath.c_str()); rmdir(directory); executorLimit = 1;
        std::puts("SIMULATION: external OMC + InitWeights, three decode stages stay resident; zero builds after warmup");
    }
    for (const char* name : {"OH_NNCompilation_SetDevice", "HMS_HiAIOptions_SetModelDeviceOrder",
                             "HMS_HiAIOptions_SetFallbackMode", "OH_NNCompilation_Build"}) {
        failure = name; calls.clear(); compilationsDestroyed = 0;
        std::string logs, error;
        auto backend = createNnrtBackend([&](const std::string& s) { logs += s + "\n"; });
        assert(!backend->probe(graph, error));
        assert(error.find("step=" + failure + " ret=OH_NN_FAILED(1)") != std::string::npos);
        assert(error.find(graph.path) != std::string::npos);
        assert(logs.find("graph=embedding_decode level=load_begin") != std::string::npos);
        assert(logs.find("file_bytes=7") != std::string::npos);
        assert(calls.back() == name && compilationsDestroyed == 1);
    }
    // Match legacy behavior: an advisory band-mode failure does not prevent
    // Build, and its return code cannot mask a later Build result.
    for (const char* advisory : {"", "HMS_HiAIOptions_SetBandMode_NORMAL"}) {
        failure = advisory; calls.clear(); executorsDestroyed = tensorsDestroyed = 0;
        auto backend = createNnrtBackend({}); std::string error;
        assert(backend->probe(graph, error));
        assert(calls == std::vector<std::string>({"OH_NNCompilation_SetDevice", "HMS_HiAIOptions_SetModelDeviceOrder",
            "HMS_HiAIOptions_SetFallbackMode", "HMS_HiAIOptions_SetBandMode_NORMAL", "OH_NNCompilation_Build",
            "OH_NNExecutor_Construct"}));
        float values[] = {1, 2}; std::vector<std::vector<float>> outputs;
        assert(!backend->run(graph, {}, outputs, error));
        assert(!backend->run(graph, {{DType::F32, values, 1}}, outputs, error));
        assert(!backend->run(graph, {{DType::I32, values, 2}}, outputs, error));
        assert(!backend->run(graph, {{DType::F32, nullptr, 2}}, outputs, error));
        assert(backend->run(graph, {{DType::F32, values, 2}}, outputs, error));
        assert(outputs == std::vector<std::vector<float>>({{2, 4}}));
        assert(std::count(calls.begin(), calls.end(), "OH_NNCompilation_Build") == 1);
        backend->release(); assert(executorsDestroyed == 1 && tensorsDestroyed == 2);
    }
    {
        std::string logs, error;
        auto backend = createNnrtBackend([&](const std::string& s) { logs += s + "\n"; });
        failure = "OH_NNExecutor_Construct"; compilationsDestroyed = 0;
        assert(!backend->probe(graph, error));
        assert(error.find("step=OH_NNExecutor_Construct result=nullptr") != std::string::npos);
        assert(error.find(graph.path) != std::string::npos);
        assert(logs.find("resident_models=0") != std::string::npos);
        assert(compilationsDestroyed == 1 && liveExecutors == 0 && liveTensors == 0);
        failure.clear();
        assert(backend->probe(graph, error)); // A failed construction leaves no stale cache entry.
        backend->release();
    }
    {
        auto backend = createNnrtBackend({}); std::string error;
        executorsAtBuild.clear(); peakExecutors = 0;
        // Two prefill chunks followed by several decode tokens, with seven
        // model stages each. Host-owned results survive every model eviction.
        std::vector<std::vector<float>> saved;
        for (int block = 0; block < 6; ++block) {
            std::vector<float> hidden = {1, 2};
            for (int stage = 0; stage < 7; ++stage) {
                Graph part = graph;
                part.name = std::string(block < 2 ? "llm_prefill_" : "llm_decode_") + std::to_string(stage);
                std::vector<std::vector<float>> outputs;
                assert(backend->run(part, {{DType::F32, hidden.data(), hidden.size()}}, outputs, error));
                hidden = std::move(outputs[0]);
            }
            saved.push_back(hidden);
        }
        backend->release(); backend->release();
        assert(peakExecutors == 1 && liveExecutors == 0 && liveTensors == 0);
        assert(executorsAtBuild.size() == 42);
        assert(std::all_of(executorsAtBuild.begin(), executorsAtBuild.end(), [](int n) { return n == 0; }));
        for (const auto& output : saved) assert(output == std::vector<float>({128, 256}));
    }
    {
        for (int i = 0; i < 2; ++i) {
            slicePaths[i] = std::string(path) + "_" + std::to_string(i);
            auto* f = std::fopen(slicePaths[i].c_str(), "wb"); assert(f); std::fputs("slice", f); std::fclose(f);
        }
        Graph sliced; sliced.name = "embedding_prefill";
        sliced.inputs = {{"token_ids", DType::I32, {1, 2}}};
        sliced.outputs = {{"embeddings", DType::F32, {1, 2, 3}}};
        sliced.embeddingSlices = {{slicePaths[0], 0, 2}, {slicePaths[1], 2, 3}};
        auto backend = createNnrtBackend({}); std::string error;
        failure.clear(); executorsDestroyed = 0; calls.clear();
        assert(backend->probe(sliced, error));
        assert(executorsDestroyed == 2); // Probe never retains embedding weights.
        Graph decoder = graph; decoder.name = "resident_decoder";
        assert(backend->probe(decoder, error));
        int32_t ids[] = {7, 11}; std::vector<std::vector<float>> outputs;
        assert(backend->run(sliced, {{DType::I32, ids, 2}}, outputs, error));
        assert(outputs == std::vector<std::vector<float>>({{70, 71, 72, 110, 111, 112}}));
        assert(executorsDestroyed == 5); // Evict decoder before the first slice.
        size_t before = calls.size(); float values[] = {3, 5};
        assert(backend->run(decoder, {{DType::F32, values, 2}}, outputs, error));
        assert(calls.size() > before && outputs == std::vector<std::vector<float>>({{6, 10}}));
        failSliceRun = true;
        assert(!backend->run(sliced, {{DType::I32, ids, 2}}, outputs, error));
        assert(outputs.empty() && error.find("embedding_prefill_slice_1") != std::string::npos);
        failSliceRun = false;
        assert(backend->run(sliced, {{DType::I32, ids, 2}}, outputs, error));
        assert(outputs[0] == std::vector<float>({70, 71, 72, 110, 111, 112}));
        for (const auto& p : slicePaths) unlink(p.c_str());
    }
    {
        executorLimit = 4; peakExecutors = 0; calls.clear();
        auto backend = createNnrtBackend({}); BackendOptions options;
        options.maxResidentGraphs = 4; options.residentModelBytes = 1024; options.persistentKvBytes = 0;
        backend->configure(options);
        float values[] = {1, 2}; std::vector<std::vector<float>> outputs; std::string error;
        auto run = [&](const std::string& name, int start = 0) {
            Graph part = graph; part.name = name; part.start = start;
            assert(backend->run(part, {{DType::F32, values, 2}}, outputs, error));
            assert(outputs == std::vector<std::vector<float>>({{2, 4}}));
        };
        // A cyclic decode working set: 8 embedding models + 7 LLM + head.
        // Four resident sessions must retain three useful models rather than
        // behave like an LRU cache that rebuilds all sixteen on every cycle.
        for (int token = 0; token < 3; ++token) {
            run("lm_head");
            for (int i = 0; i < 8; ++i) run("embedding_decode_slice_" + std::to_string(i));
            for (int i = 0; i < 7; ++i) run("llm_decode_" + std::to_string(i), i * 4);
            assert(std::count(calls.begin(), calls.end(), "OH_NNCompilation_Build") == 16 + 13 * token);
        }
        assert(peakExecutors == 4);
        std::puts("SIMULATION: 4 resident models, steady decode builds 13 vs 16 per token; outputs unchanged");
        backend->release();

        // The model-file budget is enforced independently of model count.
        options.residentModelBytes = 14; backend->configure(options); peakExecutors = 0;
        for (int i = 0; i < 7; ++i) run("llm_decode_" + std::to_string(i), i * 4);
        assert(peakExecutors == 2);
        backend->release();
        options.residentModelBytes = 6; backend->configure(options); calls.clear();
        assert(!backend->probe(graph, error) && error.find("budget") != std::string::npos && calls.empty());
    }
    {
        // With a genuine two-executor limit the decoder remains resident;
        // only the cheaper embedding and head alternate around it.
        executorLimit = 2; calls.clear();
        std::string logs, error; std::vector<std::vector<float>> outputs;
        auto backend = createNnrtBackend([&](const std::string& s) { logs += s + "\n"; });
        BackendOptions options; backend->configure(options);
        float values[] = {1, 2};
        for (int token = 0; token < 3; ++token) for (const char* name : {"embedding_decode", "llm_decode_0", "lm_head"}) {
            Graph part = graph; part.name = name;
            assert(backend->run(part, {{DType::F32, values, 2}}, outputs, error));
        }
        assert(liveExecutors == 2);
        assert(logs.find("previous_graph=llm_decode_0") == std::string::npos);
        assert(std::count(calls.begin(), calls.end(), "OH_NNCompilation_Build") == 8);
        backend->release();
    }
    {
        executorLimit = 2; peakExecutors = 0;
        std::string logs, error; std::vector<std::vector<float>> outputs; float values[] = {1, 2};
        auto backend = createNnrtBackend([&](const std::string& s) { logs += s + "\n"; });
        BackendOptions options; backend->configure(options);
        for (int i = 0; i < 9; ++i) {
            Graph part = graph; part.name = "llm_decode_" + std::to_string(i); part.start = i;
            assert(backend->run(part, {{DType::F32, values, 2}}, outputs, error));
        }
        assert(peakExecutors == 2 && liveExecutors == 2);
        assert(logs.find("level=cache_backoff") != std::string::npos);
        assert(logs.find("max_graphs=2") != std::string::npos);
        assert(logs.find("level=error") == std::string::npos); // Recovered pressure is not a final failure.
        // Prefill pressure is not evidence that decode or the next request
        // cannot retain their three-model working set.
        for (bool newRequest : {false, true}) {
            if (newRequest) backend->release(); else backend->releaseGraphs();
            executorLimit = 4; calls.clear();
            for (int token = 0; token < 3; ++token) for (const char* name : {"embedding_decode", "llm_decode_0", "lm_head"}) {
                Graph part = graph; part.name = name;
                assert(backend->run(part, {{DType::F32, values, 2}}, outputs, error));
            }
            assert(liveExecutors == 3);
            assert(std::count(calls.begin(), calls.end(), "OH_NNCompilation_Build") == 3);
        }
        // A permanent construction error still surfaces; pressure recovery never loops.
        failure = "OH_NNExecutor_Construct";
        Graph bad = graph; bad.name = "broken_model";
        assert(!backend->run(bad, {{DType::F32, values, 2}}, outputs, error));
        assert(error.find("OH_NNExecutor_Construct") != std::string::npos && liveExecutors == 0);
        failure.clear(); backend->configure(options);
        assert(backend->run(graph, {{DType::F32, values, 2}}, outputs, error));
        failTensorOnce = true;
        bad.name = "allocation_pressure";
        assert(backend->run(bad, {{DType::F32, values, 2}}, outputs, error));
        backend->release();
    }
    {
        executorLimit = 4;
        kvPath = std::string(path) + "_kv";
        auto* f = std::fopen(kvPath.c_str(), "wb"); assert(f); std::fputs("kv", f); std::fclose(f);
        Profile p; p.layers = 1; p.heads = 8; p.dim = 128; p.group = 32;
        p.capacity = 2048; p.recent = 128; p.sink = 4;
        Cache cache(p); expectedKv = cache.inputs(0, 1);
        const char* names[] = {"past_key_0", "past_value_0", "past_key_scale_0", "past_value_scale_0",
                               "sink_key_0", "sink_value_0", "recent_key_0", "recent_value_0"};
        for (size_t i = 0; i < expectedKv.size(); ++i) {
            int rows = i < 4 ? p.capacity : i < 6 ? p.sink : p.recent;
            int cols = i == 2 || i == 3 ? p.dim / p.group : p.dim;
            kvPorts.push_back({names[i], expectedKv[i].dtype, {1, p.heads, rows, cols}});
        }
        Graph part; part.name = "llm_prefill_0"; part.path = kvPath; part.inputs = kvPorts;
        part.outputs = {{"hidden_states", DType::F32, {1, 2}}};
        std::vector<size_t> copied;
        auto backend = createNnrtBackend([&](const std::string& s) {
            auto at = s.find(" kv_copied_bytes=");
            if (at != std::string::npos) copied.push_back(std::strtoull(s.c_str() + at + 17, nullptr, 10));
        });
        BackendOptions options; options.maxResidentGraphs = 3; backend->configure(options);
        std::string error; std::vector<std::vector<float>> outputs;
        auto run = [&]() {
            expectedKv = cache.inputs(0, 1);
            return backend->run(part, expectedKv, outputs, error);
        };
        auto append = [&](int count) {
            std::vector<std::vector<float>> delta(2, std::vector<float>(p.heads * count * p.dim));
            for (size_t field = 0; field < delta.size(); ++field)
                for (size_t i = 0; i < delta[field].size(); ++i)
                    delta[field][i] = float(int((i + cache.length * 5 + field * 11) % 99) - 49) / 7;
            assert(cache.append(0, delta, count, count, error)); cache.length += count;
        };
        assert(run() && copied.back() == cache.bytes());
        append(7); assert(run() && copied.back() < cache.bytes());
        append(130); backend->releaseGraphs();
        assert(liveExecutors == 0 && liveTensors == 8);
        part.name = "llm_decode_0";
        assert(run() && copied.back() < cache.bytes() && liveTensors == 9);
        backend->releaseGraphs(); assert(run() && copied.back() == 0);
        append(1); assert(run() && copied.back() == 6400);
        assert(copied.back() * 28 == 179200); // Current 28-layer profile: 175 KiB, versus 140.44 MiB full.
        std::puts("SIMULATION: steady KV writes 6400 bytes/layer (179200 for 28 layers); all input bytes equal full copy");
        append(1); failure = "OH_NNExecutor_RunSync"; assert(!run());
        failure.clear(); assert(run() && copied.back() == cache.bytes());
        cache.reset(); assert(run() && copied.back() == cache.bytes());
        kvFormat = 1; backend->releaseGraphs();
        assert(run() && copied.back() == cache.bytes() && liveTensors == 17); // Never alias different layouts.
        kvFormat = 0;
        backend->release(); assert(liveTensors == 0);
        options.incrementalKv = false; backend->configure(options);
        assert(run() && copied.back() == cache.bytes());
        append(1); assert(run() && copied.back() == cache.bytes());
        backend->release(); unlink(kvPath.c_str());
        kvPath.clear(); expectedKv.clear(); kvPorts.clear();
    }
    assert(liveExecutors == 0 && liveTensors == 0);
    {
        Profile p; p.kvType = DType::F16; p.layers = 28; p.heads = 8; p.dim = 128; p.capacity = 2048;
        Cache cache(p); expectedKv = cache.inputs(0, p.layers); kvPath = path;
        for (int layer = 0; layer < p.layers; ++layer) for (const char* prefix : {"past_key_", "past_value_"})
            kvPorts.push_back({std::string(prefix) + std::to_string(layer), DType::F16, {1, p.heads, p.capacity, p.dim}});
        Graph fp; fp.name = "llm_prefill_0"; fp.path = path; fp.inputs = kvPorts;
        fp.outputs = {{"hidden_states", DType::F32, {1, 2}}};
        size_t copied = 0;
        auto backend = createNnrtBackend([&](const std::string& s) {
            auto at = s.find(" kv_copied_bytes=");
            if (at != std::string::npos) copied = std::strtoull(s.c_str() + at + 17, nullptr, 10);
        });
        BackendOptions options; options.persistentKvBytes = size_t(256) << 20; backend->configure(options);
        std::string error; std::vector<std::vector<float>> outputs;
        assert(backend->run(fp, expectedKv, outputs, error) && copied == cache.bytes());
        assert(cache.bytes() == 234881024 && expectedKv.size() == 56);
        std::vector<std::vector<float>> delta(56, std::vector<float>(p.heads * p.dim, 1.5f));
        assert(cache.append(0, delta, 1, 1, error)); ++cache.length; expectedKv = cache.inputs(0, p.layers);
        backend->releaseGraphs(); fp.name = "llm_decode_0";
        assert(backend->run(fp, expectedKv, outputs, error) && copied == 114688);
        // Evicting an executor under pressure does not also throw away KV.
        executorLimit = 1;
        float values[] = {1, 2}; Graph other = graph; other.name = "lm_head";
        kvPath.clear(); // The unrelated head has ordinary, non-KV ports.
        assert(backend->run(other, {{DType::F32, values, 2}}, outputs, error));
        kvPath = path;
        backend->releaseGraphs(); executorLimit = 4;
        assert(backend->run(fp, expectedKv, outputs, error) && copied == 0);
        backend->release(); assert(liveExecutors == 0 && liveTensors == 0);
        // Unknown formats cannot alias across executors. They can still use
        // append-only writes to the same executor's already allocated input.
        failKvFormat = true;
        assert(backend->run(fp, expectedKv, outputs, error) && copied == cache.bytes());
        assert(cache.append(0, delta, 1, 1, error)); ++cache.length; expectedKv = cache.inputs(0, p.layers);
        assert(backend->run(fp, expectedKv, outputs, error) && copied == 114688);
        backend->releaseGraphs();
        assert(backend->run(fp, expectedKv, outputs, error) && copied == cache.bytes());
        assert(backend->run(fp, expectedKv, outputs, error) && copied == 0);
        backend->release(); failKvFormat = false;
        kvPath.clear(); expectedKv.clear(); kvPorts.clear();
        std::puts("SIMULATION: all 28 FP16 KV layers persist across phases, 224 MiB pool and 112 KiB delta writes");
    }
    unlink(path);
    std::puts("PASS: NNRT call substitutes verify bounded residency, external weights, FP16/INT8 KV and cleanup");
}
