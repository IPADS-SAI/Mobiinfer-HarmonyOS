#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "NnrtBackend.hpp"
#include "PortBinding.hpp"
#include "EmbeddingSlices.hpp"
#include "KvTransfer.hpp"
#include "CANNKit/hiai_helper.h"
#include "CANNKit/hiai_options.h"
#include "neural_network_runtime/neural_network_core.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <sys/stat.h>

namespace KirinFullNpu {
namespace {
std::string returnCode(OH_NN_ReturnCode ret) {
    const char* name = ret == OH_NN_SUCCESS ? "OH_NN_SUCCESS" :
                       ret == OH_NN_FAILED ? "OH_NN_FAILED" :
                       ret == OH_NN_INVALID_PARAMETER ? "OH_NN_INVALID_PARAMETER" :
                       ret == OH_NN_MEMORY_ERROR ? "OH_NN_MEMORY_ERROR" :
                       ret == OH_NN_OPERATION_FORBIDDEN ? "OH_NN_OPERATION_FORBIDDEN" : "OH_NN_RETURN_CODE";
    return std::string(name) + "(" + std::to_string(int(ret)) + ")";
}

std::string resourceSnapshot() {
    std::string result;
    for (const char* path : {"/proc/self/status", "/proc/meminfo"}) {
        std::ifstream stream(path);
        std::string line;
        while (std::getline(stream, line)) {
            for (const char* key : {"VmRSS:", "VmSize:", "MemAvailable:", "CmaFree:"}) {
                if (line.compare(0, std::strlen(key), key) == 0) result += " " + line;
            }
        }
    }
    return result;
}

struct TensorStorage {
    NN_Tensor* handle = nullptr;
    InputSync sync;
    ~TensorStorage() { if (handle) OH_NNTensor_Destroy(&handle); }
};
struct Tensor {
    std::shared_ptr<TensorStorage> storage;
    Port port;
    size_t bytes = 0;
    // An executor's own input storage can receive deltas even when its format
    // is unavailable and it cannot safely enter the cross-executor pool.
    bool incremental = false;
};
struct DeviceGraph {
    OH_NNExecutor* executor = nullptr;
    OH_NNCompilation* compilation = nullptr;
    std::vector<Tensor> inputs, outputs;
    std::vector<size_t> inputMap, outputMap;
    size_t modelBytes = 0;
    uint64_t used = 0;
    int priority = 0;
    ~DeviceGraph() {
        inputs.clear(); outputs.clear();
        if (executor) OH_NNExecutor_Destroy(&executor);
        if (compilation) OH_NNCompilation_Destroy(&compilation);
    }
};

class NnrtBackend final : public Backend {
public:
    explicit NnrtBackend(Log log) : log(std::move(log)) {}
    ~NnrtBackend() override {
        release();
        if (hiaiLibrary) dlclose(hiaiLibrary);
    }
    void configure(const BackendOptions& value) override {
        resident.clear(); kvBuffers.clear(); kvBytes = 0; options = value; configured = true;
        restorePolicy();
        if (log) log("[FULL_NPU] level=cache_config max_graphs=" + std::to_string(effectiveGraphs) +
                     " model_budget_bytes=" + std::to_string(options.residentModelBytes) +
                     " kv_budget_bytes=" + std::to_string(effectiveKvBytes));
    }
    bool probe(const Graph& g, std::string& error) override {
        if (g.embeddingSlices.empty()) return load(g, error) != nullptr;
        std::vector<Graph> slices;
        if (!embeddingSliceGraphs(g, slices, error)) return false;
        for (const auto& part : slices) {
            bool ok = load(part, error) != nullptr;
            resident.erase(part.name);
            if (!ok) { error = part.name + ": " + error; return false; }
        }
        return true;
    }
    void releaseGraphs() override {
        resident.clear();
        restorePolicy();
    }
    void release() override { releaseGraphs(); kvBuffers.clear(); kvBytes = 0; }
    bool run(const Graph& g, const std::vector<View>& inputs,
             std::vector<std::vector<float>>& outputs, std::string& error) override {
        outputs.clear();
        if (inputs.size() != g.inputs.size()) { error = "NNRT input count mismatch"; return false; }
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (!inputs[i].data || inputs[i].dtype != g.inputs[i].dtype || inputs[i].elements != g.inputs[i].elements()) {
                error = "NNRT input view mismatch: " + g.inputs[i].name;
                return false;
            }
        }
        if (!g.embeddingSlices.empty()) {
            std::vector<Graph> slices;
            if (!embeddingSliceGraphs(g, slices, error)) return false;
            std::vector<float> assembled(g.outputs[0].elements());
            const int hidden = g.outputs[0].shape[2];
            const size_t rows = assembled.size() / hidden;
            for (size_t i = 0; i < slices.size(); ++i) {
                std::vector<std::vector<float>> partOutputs;
                bool ok = run(slices[i], inputs, partOutputs, error);
                // The host owns the copied output, so the Gather model and
                // its device buffers are no longer needed for concatenation.
                resident.erase(slices[i].name);
                if (!ok) { error = slices[i].name + ": " + error; return false; }
                const int width = slices[i].outputs[0].shape[2];
                if (partOutputs.size() != 1 || partOutputs[0].size() != rows * width) {
                    error = slices[i].name + ": embedding output size mismatch";
                    return false;
                }
                for (size_t row = 0; row < rows; ++row)
                    std::copy_n(partOutputs[0].data() + row * width, width,
                                assembled.data() + row * hidden + g.embeddingSlices[i].begin);
            }
            outputs.push_back(std::move(assembled));
            return true;
        }
        auto* device = load(g, error, true);
        if (!device) return false;
        size_t kvWritten = 0, kvFull = 0;
        auto copyBegin = std::chrono::steady_clock::now();
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto& dst = device->inputs[device->inputMap[i]];
            auto* buffer = OH_NNTensor_GetDataBuffer(dst.storage->handle);
            size_t allocated = 0;
            if (!buffer || OH_NNTensor_GetSize(dst.storage->handle, &allocated) != OH_NN_SUCCESS || allocated < dst.bytes) {
                error = "Missing or undersized device input buffer"; return false;
            }
            const auto& src = inputs[i];
            size_t written = 0;
            if (!syncInput(src, g.inputs[i], dst.port.dtype, buffer, dst.incremental && options.incrementalKv,
                           dst.storage->sync, written, error)) return false;
            if (src.cacheEpoch) { kvWritten += written; kvFull += dst.bytes; }
        }
        double copyMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - copyBegin).count();
        std::vector<NN_Tensor*> in, out;
        for (auto& t : device->inputs) in.push_back(t.storage->handle);
        for (auto& t : device->outputs) out.push_back(t.storage->handle);
        auto runBegin = std::chrono::steady_clock::now();
        auto ret = OH_NNExecutor_RunSync(device->executor, in.data(), in.size(), out.data(), out.size());
        double runMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - runBegin).count();
        if (ret != OH_NN_SUCCESS) {
            for (auto& t : device->inputs) t.storage->sync = {};
            error = "NNRT RunSync returned " + std::to_string(int(ret)); return false;
        }
        outputs.clear();
        for (size_t index : device->outputMap) {
            auto& src = device->outputs[index];
            const void* buffer = OH_NNTensor_GetDataBuffer(src.storage->handle);
            size_t allocated = 0;
            if (!buffer || OH_NNTensor_GetSize(src.storage->handle, &allocated) != OH_NN_SUCCESS || allocated < src.bytes) {
                error = "Missing or undersized device output buffer"; return false;
            }
            outputs.emplace_back(src.port.elements()); auto& values = outputs.back();
            if (src.port.dtype == DType::F32) std::memcpy(values.data(), buffer, src.bytes);
            else if (src.port.dtype == DType::F16) {
                auto* raw = static_cast<const uint16_t*>(buffer);
                for (size_t j = 0; j < values.size(); ++j) values[j] = fromHalf(raw[j]);
            } else { error = "Unsupported integer output boundary"; return false; }
        }
        if (log) log("[FULL_NPU] graph=" + g.name + " level=run_timing input_copy_ms=" + std::to_string(copyMs) +
                     " run_sync_ms=" + std::to_string(runMs) + " kv_copied_bytes=" + std::to_string(kvWritten) +
                     " kv_full_bytes=" + std::to_string(kvFull) + " persistent_kv_bytes=" + std::to_string(kvBytes) +
                     " effective_max_graphs=" + std::to_string(effectiveGraphs) +
                     " effective_kv_budget_bytes=" + std::to_string(effectiveKvBytes));
        return true;
    }
private:
    using InitWeights = OH_NN_ReturnCode (*)(OH_NNExecutor*, const char*);
    Log log;
    void* hiaiLibrary = nullptr;
    InitWeights weightInit = nullptr;
    std::string weightApiEvidence;
    BackendOptions options;
    int effectiveGraphs = 1; // A standalone backend remains conservative until configured.
    bool configured = false;
    size_t effectiveKvBytes = 0, kvBytes = 0;
    uint64_t clock = 0;
    std::map<std::string, std::unique_ptr<DeviceGraph>> resident;
    std::map<std::string, std::shared_ptr<TensorStorage>> kvBuffers;

    InitWeights externalWeightApi(std::string& evidence) {
        if (weightInit) {
            evidence = weightApiEvidence + " cached=1";
            return weightInit;
        }
        const char* version = HMS_HiAI_GetVersion();
        evidence = "hiai_version=" + std::string(version ? version : "unavailable");
        auto loaderError = []() {
            const char* value = dlerror();
            return std::string(value ? value : "symbol returned nullptr");
        };
        // HiAI may be local to the App's native module/loader namespace.
        // Match Lite's explicit library lookup; RTLD_DEFAULT alone misses
        // functions in RTLD_LOCAL libraries. Keep the handle until all
        // executors/compilations are destroyed, including on load failures.
        dlerror();
        if (!hiaiLibrary) hiaiLibrary = dlopen("libhiai_foundation.so", RTLD_LAZY | RTLD_LOCAL);
        if (hiaiLibrary) {
            evidence += " dlopen=success library=libhiai_foundation.so";
            dlerror();
            weightInit = reinterpret_cast<InitWeights>(dlsym(hiaiLibrary, "HMS_HiAIExecutor_InitWeights"));
            if (weightInit) evidence += " lookup=library_handle";
            else evidence += " library_symbol_error={" + loaderError() + "}";
        } else {
            evidence += " dlopen=failed loader_error={" + loaderError() + "}";
        }
        if (!weightInit) {
            // Supports runtimes exposing this API from the global scope and
            // the existing host call substitutes, without widening visibility.
            dlerror();
            weightInit = reinterpret_cast<InitWeights>(dlsym(RTLD_DEFAULT, "HMS_HiAIExecutor_InitWeights"));
            if (weightInit) evidence += " lookup=global_fallback";
            else evidence += " global_symbol_error={" + loaderError() + "}";
        }
        if (weightInit) {
            Dl_info provider {};
            if (dladdr(reinterpret_cast<void*>(weightInit), &provider) && provider.dli_fname)
                evidence += " provider=" + std::string(provider.dli_fname);
            evidence += " result=success";
            weightApiEvidence = evidence;
        } else {
            evidence += " result=unresolved";
        }
        return weightInit;
    }

    int priority(const Graph& g) const {
        // If only two models fit, retain the expensive decoder while cycling
        // embedding/head. Keeping the head instead reloads the whole decoder.
        if (g.name == "lm_head") return 10000;
        if (g.name.find("llm_decode_") == 0 || g.name.find("llm_prefill_") == 0) return 100000 - g.start;
        return 0;
    }
    void restorePolicy() {
        const int graphs = configured ? std::max(1, options.maxResidentGraphs) : 1;
        const size_t budget = configured && options.incrementalKv ? options.persistentKvBytes : 0;
        if (configured && log && (graphs != effectiveGraphs || budget != effectiveKvBytes))
            log("[FULL_NPU] level=cache_policy_restore max_graphs=" + std::to_string(graphs) +
                " kv_budget_bytes=" + std::to_string(budget));
        effectiveGraphs = graphs;
        effectiveKvBytes = budget;
    }
    void evictOne(const Graph& g, const std::string& policy) {
        auto victim = std::min_element(resident.begin(), resident.end(), [](const auto& a, const auto& b) {
            return a.second->priority != b.second->priority ? a.second->priority < b.second->priority :
                   a.second->used < b.second->used;
        });
        if (log) log("[FULL_NPU] graph=" + g.name + " level=resident_evict previous_graph=" +
                     victim->first + " policy=" + policy);
        resident.erase(victim);
    }
    size_t residentBytes() const {
        size_t bytes = 0;
        for (const auto& item : resident) bytes += item.second->modelBytes;
        return bytes;
    }
    bool backoff(const Graph& g, const std::string& reason) {
        // Every retry removes a model or the pool, bounding recovery attempts.
        // A failed larger working set must not also discard the KV optimization
        // or poison all subsequent requests/phases with single-model residency.
        if (!resident.empty()) {
            effectiveGraphs = std::max(1, std::min(effectiveGraphs, int(resident.size())));
            evictOne(g, "resource_pressure");
        } else if (!kvBuffers.empty()) {
            kvBuffers.clear(); kvBytes = 0; effectiveKvBytes = 0;
        } else return false;
        if (log) log("[FULL_NPU] graph=" + g.name + " level=cache_backoff reason=" + reason +
                     " max_graphs=" + std::to_string(effectiveGraphs) +
                     " persistent_kv_bytes=" + std::to_string(kvBytes) +
                     " kv_budget_bytes=" + std::to_string(effectiveKvBytes) + "; retrying_after_release");
        return true;
    }

    DeviceGraph* load(const Graph& g, std::string& error, bool cacheKv = false) {
        auto previous = resident.find(g.name);
        if (previous != resident.end()) {
            previous->second->used = ++clock;
            if (log) log("[FULL_NPU] graph=" + g.name + " level=cache_hit resident_models=" + std::to_string(resident.size()));
            return previous->second.get();
        }
        auto trace = [&](const std::string& text) { if (log) log("[FULL_NPU] graph=" + g.name + " " + text); };
        auto begin = std::chrono::steady_clock::now();
        struct stat fileInfo {};
        if (stat(g.path.c_str(), &fileInfo) != 0 || !S_ISREG(fileInfo.st_mode) || fileInfo.st_size <= 0) {
            error = "Offline graph file missing or empty: " + g.path;
            trace("level=error " + error);
            return nullptr;
        }
        InitWeights initWeights = nullptr;
        size_t modelBytes = size_t(fileInfo.st_size), weightBytes = 0;
        if (!g.weightDir.empty()) {
            if (g.weightFiles.empty()) { error = "External weight file list is empty"; return nullptr; }
            std::string evidence;
            initWeights = externalWeightApi(evidence);
            trace("level=load_step step=resolve_HMS_HiAIExecutor_InitWeights " + evidence);
            if (!initWeights) {
                error = "External-weight model requires HMS_HiAIExecutor_InitWeights; symbol lookup failed; " + evidence;
                trace("level=error " + error); return nullptr;
            }
            for (const auto& name : g.weightFiles) {
                const auto path = g.weightDir + "/" + name;
                struct stat weights {};
                if (stat(path.c_str(), &weights) != 0 || !S_ISREG(weights.st_mode) || weights.st_size <= 0 ||
                    uint64_t(weights.st_size) > std::numeric_limits<size_t>::max() - modelBytes) {
                    error = "External weight file missing, empty or oversized: " + path;
                    trace("level=error " + error); return nullptr;
                }
                modelBytes += size_t(weights.st_size); weightBytes += size_t(weights.st_size);
            }
        } else if (!g.weightFiles.empty()) { error = "External weight directory is missing"; return nullptr; }
        if (modelBytes > options.residentModelBytes) {
            error = "Offline graph exceeds the configured model-file budget: " + g.path;
            trace("level=error " + error); return nullptr;
        }
        // Reserve room before Build. Keep frequently reused LLM stages/head;
        // a pure LRU cache thrashes on a decoder cycle larger than its budget.
        while (!resident.empty() && (resident.size() >= size_t(effectiveGraphs) ||
               residentBytes() + modelBytes > options.residentModelBytes)) {
            evictOne(g, "bounded_priority");
        }
        const char* version = HMS_HiAI_GetVersion();
        trace("level=load_begin path=" + g.path + " file_bytes=" + std::to_string(fileInfo.st_size) +
              " external_weight_bytes=" + std::to_string(weightBytes) +
              " hiai_version=" + (version ? version : "unavailable") +
              " resident_models=" + std::to_string(resident.size()) + " resident_policy=bounded_priority" +
              " effective_max_graphs=" + std::to_string(effectiveGraphs) +
              " effective_kv_budget_bytes=" + std::to_string(effectiveKvBytes) + resourceSnapshot());
        auto compatibility = HMS_HiAICompatibility_CheckFromFile(g.path.c_str());
        trace("level=load_step step=HMS_HiAICompatibility_CheckFromFile result=" + std::to_string(int(compatibility)));
        if (compatibility != HIAI_COMPATIBILITY_COMPATIBLE) {
            error = "Offline graph incompatible with this device: " + g.name; return nullptr;
        }
        size_t deviceId = 0; bool selected = false; const size_t* ids = nullptr; uint32_t count = 0;
        if (OH_NNDevice_GetAllDevicesID(&ids, &count) == OH_NN_SUCCESS && ids)
            for (uint32_t i = 0; i < count; ++i) {
                const char* name = nullptr;
                if (OH_NNDevice_GetName(ids[i], &name) == OH_NN_SUCCESS && name && std::string(name) == "HIAI_F") {
                    deviceId = ids[i]; selected = true; break;
                }
            }
        if (!selected) { error = "HIAI_F NPU device not found"; return nullptr; }
        trace("level=load_step step=select_device name=HIAI_F id=" + std::to_string(deviceId));
        OH_NNCompilation* compilation = OH_NNCompilation_ConstructWithOfflineModelFile(g.path.c_str());
        if (!compilation) { error = "Cannot construct offline model: " + g.name; return nullptr; }
        const char* step = "OH_NNCompilation_SetDevice";
        auto ret = OH_NNCompilation_SetDevice(compilation, deviceId);
        auto traceStep = [&](const char* name, OH_NN_ReturnCode value) {
            trace("level=load_step step=" + std::string(name) + " ret=" + returnCode(value));
        };
        traceStep(step, ret);
        HiAI_ExecuteDevice order[] = {HiAI_ExecuteDevice::HIAI_EXECUTE_DEVICE_NPU};
        if (ret == OH_NN_SUCCESS) {
            step = "HMS_HiAIOptions_SetModelDeviceOrder";
            ret = HMS_HiAIOptions_SetModelDeviceOrder(compilation, order, 1);
            traceStep(step, ret);
        }
        if (ret == OH_NN_SUCCESS) {
            step = "HMS_HiAIOptions_SetFallbackMode";
            ret = HMS_HiAIOptions_SetFallbackMode(compilation, HIAI_FALLBACK_DISABLED);
            traceStep(step, ret);
        }
        if (ret == OH_NN_SUCCESS) {
            // Match the existing working ViT executor. Bandwidth preference is
            // advisory there too; it must not replace a Build failure/success.
            auto bandRet = HMS_HiAIOptions_SetBandMode(compilation, HIAI_BANDMODE_NORMAL);
            traceStep("HMS_HiAIOptions_SetBandMode_NORMAL", bandRet);
            step = "OH_NNCompilation_Build";
            trace("level=load_step step=" + std::string(step) + " state=begin" + resourceSnapshot());
            auto buildBegin = std::chrono::steady_clock::now();
            ret = OH_NNCompilation_Build(compilation);
            traceStep(step, ret);
            trace("level=load_timing step=" + std::string(step) + " elapsed_ms=" + std::to_string(
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - buildBegin).count()));
        }
        if (ret != OH_NN_SUCCESS) {
            error = "step=" + std::string(step) + " ret=" + returnCode(ret) + " path=" + g.path;
            OH_NNCompilation_Destroy(&compilation);
            if (std::string(step) == "OH_NNCompilation_Build" &&
                (ret == OH_NN_FAILED || ret == OH_NN_MEMORY_ERROR) && backoff(g, error)) {
                error.clear(); return load(g, error, cacheKv);
            }
            trace("level=error " + error + resourceSnapshot());
            return nullptr;
        }
        std::unique_ptr<DeviceGraph> device(new DeviceGraph);
        trace("level=load_step step=OH_NNExecutor_Construct state=begin resident_models=" +
              std::to_string(resident.size()) + resourceSnapshot());
        device->executor = OH_NNExecutor_Construct(compilation);
        if (initWeights) device->compilation = compilation;
        else OH_NNCompilation_Destroy(&compilation);
        trace(std::string("level=load_step step=OH_NNExecutor_Construct result=") +
              (device->executor ? "success" : "nullptr"));
        if (!device->executor) {
            error = "step=OH_NNExecutor_Construct result=nullptr path=" + g.path;
            device.reset();
            if (backoff(g, error)) { error.clear(); return load(g, error, cacheKv); }
            trace("level=error " + error + " resident_models=" + std::to_string(resident.size()) + resourceSnapshot());
            return nullptr;
        }
        if (initWeights) {
            auto weightBegin = std::chrono::steady_clock::now();
            ret = initWeights(device->executor, g.weightDir.c_str());
            traceStep("HMS_HiAIExecutor_InitWeights", ret);
            trace("level=load_timing step=HMS_HiAIExecutor_InitWeights elapsed_ms=" + std::to_string(
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - weightBegin).count()));
            if (ret != OH_NN_SUCCESS) {
                error = "step=HMS_HiAIExecutor_InitWeights ret=" + returnCode(ret) + " directory=" + g.weightDir;
                device.reset();
                if ((ret == OH_NN_MEMORY_ERROR || ret == OH_NN_FAILED) && backoff(g, error)) {
                    error.clear(); return load(g, error, cacheKv);
                }
                trace("level=error " + error); return nullptr;
            }
        }
        size_t incrementalKvBytes = 0;
        int unknownKvFormats = 0;
        for (bool input : {true, false}) {
            size_t count = 0;
            ret = input ? OH_NNExecutor_GetInputCount(device->executor, &count) : OH_NNExecutor_GetOutputCount(device->executor, &count);
            if (ret != OH_NN_SUCCESS || !count) { error = "Cannot query NNRT port count"; return nullptr; }
            auto& tensors = input ? device->inputs : device->outputs;
            for (size_t i = 0; i < count; ++i) {
                NN_TensorDesc* desc = input ? OH_NNExecutor_CreateInputTensorDesc(device->executor, i) : OH_NNExecutor_CreateOutputTensorDesc(device->executor, i);
                Tensor tensor; const char* name = nullptr; OH_NN_DataType type = OH_NN_UNKNOWN;
                int32_t* dims = nullptr; size_t rank = 0, bytes = 0;
                bool valid = desc && OH_NNTensorDesc_GetName(desc, &name) == OH_NN_SUCCESS && name &&
                    OH_NNTensorDesc_GetDataType(desc, &type) == OH_NN_SUCCESS &&
                    OH_NNTensorDesc_GetShape(desc, &dims, &rank) == OH_NN_SUCCESS && dims && rank &&
                    OH_NNTensorDesc_GetByteSize(desc, &bytes) == OH_NN_SUCCESS;
                if (valid) {
                    tensor.port.name = name; tensor.port.shape.assign(dims, dims + rank);
                    if (type == OH_NN_FLOAT32) tensor.port.dtype = DType::F32;
                    else if (type == OH_NN_FLOAT16) tensor.port.dtype = DType::F16;
                    else if (type == OH_NN_INT8) tensor.port.dtype = DType::I8;
                    else if (type == OH_NN_INT32) tensor.port.dtype = DType::I32;
                    else valid = false;
                    valid = valid && tensor.port.elements() && bytes == tensor.port.elements() * elementSize(tensor.port.dtype);
                }
                if (valid) {
                    std::string cacheKey;
                    if (input && cacheKv && options.incrementalKv) {
                        for (const auto& logical : g.inputs) {
                            const auto& name = logical.name;
                            bool kv = name.find("past_key_") == 0 || name.find("past_value_") == 0 ||
                                      name.find("sink_key_") == 0 || name.find("sink_value_") == 0 ||
                                      name.find("recent_key_") == 0 || name.find("recent_value_") == 0;
                            if (kv && logical.shape.size() == 4 && named(tensor.port.name, name)) {
                                tensor.incremental = true;
                                incrementalKvBytes += bytes;
                                OH_NN_Format format{};
                                auto formatRet = OH_NNTensorDesc_GetFormat(desc, &format);
                                if (formatRet == OH_NN_SUCCESS && effectiveKvBytes) {
                                    auto canonical = tensor.port; canonical.name = name;
                                    cacheKey = std::to_string(deviceId) + " format=" + std::to_string(int(format)) + " " + description(canonical);
                                } else if (formatRet != OH_NN_SUCCESS) {
                                    if (!unknownKvFormats) trace("level=kv_format_query ret=" + returnCode(formatRet) +
                                        " reuse=executor_local cross_executor_reuse=disabled");
                                    ++unknownKvFormats;
                                }
                                break;
                            }
                        }
                        auto found = kvBuffers.find(cacheKey);
                        if (found != kvBuffers.end()) tensor.storage = found->second;
                    }
                    if (!tensor.storage) {
                        tensor.storage = std::make_shared<TensorStorage>();
                        tensor.storage->handle = OH_NNTensor_Create(deviceId, desc);
                        if (tensor.storage->handle && !cacheKey.empty() && bytes <= effectiveKvBytes - kvBytes) {
                            kvBuffers[cacheKey] = tensor.storage; kvBytes += bytes;
                        }
                    }
                }
                if (desc) OH_NNTensorDesc_Destroy(&desc);
                if (!valid || !tensor.storage || !tensor.storage->handle) {
                    error = "Invalid device tensor descriptor/allocation";
                    device.reset();
                    if (valid && backoff(g, error)) { error.clear(); return load(g, error, cacheKv); }
                    return nullptr;
                }
                tensor.bytes = bytes; tensors.push_back(tensor);
                trace(std::string("level=tensor_desc direction=") + (input ? "input " : "output ") + description(tensor.port) + " bytes=" + std::to_string(bytes));
            }
        }
        std::vector<Port> inputPorts, outputPorts;
        for (const auto& t : device->inputs) inputPorts.push_back(t.port);
        for (const auto& t : device->outputs) outputPorts.push_back(t.port);
        if (!bindPorts(g.inputs, inputPorts, {}, device->inputMap, error) ||
            !bindPorts(g.outputs, outputPorts, g.aliases, device->outputMap, error)) return nullptr;
        if (incrementalKvBytes) trace("level=kv_storage incremental_input_bytes=" + std::to_string(incrementalKvBytes) +
            " persistent_pool_bytes=" + std::to_string(kvBytes) + " unknown_formats=" + std::to_string(unknownKvFormats));
        trace("level=graph_ready cpu_fallback=disabled load_ms=" + std::to_string(
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count()));
        device->modelBytes = modelBytes; device->used = ++clock; device->priority = priority(g);
        auto* result = device.get(); resident[g.name] = std::move(device); return result;
    }
};
} // namespace
std::unique_ptr<Backend> createNnrtBackend(Log log) { return std::unique_ptr<Backend>(new NnrtBackend(std::move(log))); }
} // namespace KirinFullNpu
