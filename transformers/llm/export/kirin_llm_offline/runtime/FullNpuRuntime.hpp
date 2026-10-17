#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace KirinFullNpu {

enum class DType { F32, F16, I8, I32 };
size_t elementSize(DType type);
uint16_t toHalf(float value);
float fromHalf(uint16_t value);

struct Port {
    std::string name;
    DType dtype = DType::F32;
    std::vector<int> shape;
    size_t elements() const;
};

// Views never own KV storage. A graph call consumes all views synchronously.
struct View {
    DType dtype;
    const void* data;
    size_t elements;
    // Optional append-only KV metadata; ordinary inputs leave epoch at zero.
    uint64_t cacheEpoch = 0;
    int cacheTokens = -1;
    bool cacheRing = false;
};

struct BackendOptions {
    int maxResidentGraphs = 4;
    size_t residentModelBytes = size_t(640) << 20;
    size_t persistentKvBytes = size_t(192) << 20;
    bool incrementalKv = true;
};

// Optional caller policy; zero keeps the model package's value. KV is separate.
struct ResidencyBudget {
    int maxResidentGraphs = 0;
    size_t residentModelBytes = 0;
};

struct Graph {
    std::string name, path;
    // OMG external-data mode; InitWeights takes the containing directory.
    std::string weightDir;
    std::vector<std::string> weightFiles;
    std::vector<Port> inputs, outputs;
    std::map<std::string, std::string> aliases;
    int start = 0, end = 0;
    struct EmbeddingSlice {
        std::string path;
        int begin = 0, end = 0;
    };
    // Optional physical Gather models partitioned along the hidden dimension.
    // Their outputs concatenate to this graph's original logical interface.
    std::vector<EmbeddingSlice> embeddingSlices;
};

using Log = std::function<void(const std::string&)>;
using Cancel = std::function<bool()>;
using Emit = std::function<void(const std::string&)>;

// Device implementation must honor names, real descriptor dtype and byte size.
// Floating boundary conversions are allowed; integer caches must remain INT8.
class Backend {
public:
    virtual ~Backend() = default;
    virtual void configure(const BackendOptions&) {}
    virtual bool probe(const Graph& graph, std::string& error) = 0;
    virtual bool run(const Graph& graph, const std::vector<View>& inputs,
                     std::vector<std::vector<float>>& outputs, std::string& error) = 0;
    virtual void release() = 0;
    // Phase boundary: discard model sessions, retaining reusable KV tensors.
    virtual void releaseGraphs() { release(); }
};

struct Profile {
    BackendOptions backend;
    DType kvType = DType::I8; // v1 packages default to the original INT8 protocol.
    int capacity = 0, prefill = 0, layers = 0, hidden = 0, heads = 0, dim = 0;
    int group = 0, recent = 0, sink = 0, imageWidth = 0, imageHeight = 0;
    int patch = 0, merge = 0, temporal = 0, visionHidden = 0, visionHeads = 0, positionSide = 0;
    int imageToken = 0, padToken = 0, maxNewTokens = 0;
    bool fusedVision = false;
    int visionLayers = 0;
    std::vector<int> visionDeepstack;
    float ropeTheta = 0, visionTheta = 0;
    std::vector<int> mrope;
};

// Shared with the host regression driver, including native INT8 and FP16 ports.
class Cache {
public:
    explicit Cache(const Profile& profile);
    void reset();
    std::vector<View> inputs(int start, int count) const;
    bool append(int start, const std::vector<std::vector<float>>& outputs,
                int real, int width, std::string& error);
    std::vector<float> mask(int real, int width) const;
    size_t bytes() const;
    int length = 0;
private:
    Profile p;
    uint64_t epoch = 0;
    struct Layer {
        std::vector<uint16_t> fullKey, fullValue;
        std::vector<int8_t> key, value;
        std::vector<float> keyScale, valueScale;
        std::vector<uint16_t> sinkKey, sinkValue, recentKey, recentValue;
    };
    std::vector<Layer> layers;
};

class Runtime {
public:
    Runtime(std::unique_ptr<Backend> backend, Log log, std::function<bool()> debug = {});
    ~Runtime();
    bool load(const std::string& configPath, std::string& error, ResidencyBudget residency = {});
    bool response(const std::string& modelInput, const Cancel& cancel, const Emit& emit,
                  std::string& output, std::string& error, int limit = -1);
    void reset();
    const Profile& profile() const;
    static std::string chatTemplate(const std::vector<std::pair<std::string, std::string>>& messages);
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace KirinFullNpu
