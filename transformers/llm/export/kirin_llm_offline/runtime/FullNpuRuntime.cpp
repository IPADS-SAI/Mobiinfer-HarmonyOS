#include "FullNpuRuntime.hpp"
#include "RuntimeContracts.hpp"
#include "Utf8Stream.hpp"
#include "tokenizer/tokenizer.hpp"
#include "rapidjson/document.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#include "stb_image.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

namespace KirinFullNpu {
namespace {
using Json = rapidjson::Value;
View floats(const std::vector<float>& v) { return {DType::F32, v.data(), v.size()}; }
std::string readText(const std::string& path) {
    std::ifstream f(path); return std::string(std::istreambuf_iterator<char>(f), {});
}
std::string stringField(const Json& v, const char* key) {
    return v.IsObject() && v.HasMember(key) && v[key].IsString() ? v[key].GetString() : "";
}
int number(const Json& v, const char* key) {
    return v.IsObject() && v.HasMember(key) && v[key].IsInt() ? v[key].GetInt() : 0;
}
bool safePath(const std::string& s) {
    if (s.empty() || s[0] == '/' || s.find('\\') != std::string::npos) return false;
    std::istringstream stream(s); std::string part;
    while (std::getline(stream, part, '/')) if (part.empty() || part == "." || part == "..") return false;
    return true;
}
bool port(const Json& v, Port& out) {
    out.name = stringField(v, "name");
    const auto type = stringField(v, "dtype");
    if (type == "float32") out.dtype = DType::F32;
    else if (type == "float16") out.dtype = DType::F16;
    else if (type == "int8") out.dtype = DType::I8;
    else if (type == "int32") out.dtype = DType::I32;
    else return false;
    if (out.name.empty() || !v.HasMember("shape") || !v["shape"].IsArray()) return false;
    for (const auto& d : v["shape"].GetArray()) {
        if (!d.IsInt() || d.GetInt() < 1 || d.GetInt() > 1000000) return false;
        out.shape.push_back(d.GetInt());
    }
    return !out.shape.empty() && out.elements() > 0 && out.elements() < (size_t(1) << 29);
}
double elapsed(std::chrono::steady_clock::time_point begin) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
}
// Pillow BICUBIC uses a=-0.5, widens the filter when downsampling and clips
// after each 8-bit separable pass. Decode/resize remains CPU preprocessing.
double cubic(double x) {
    x = std::abs(x);
    if (x < 1) return ((1.5 * x - 2.5) * x) * x + 1;
    if (x < 2) return ((-.5 * x + 2.5) * x - 4) * x + 2;
    return 0;
}
std::vector<uint8_t> resizeAxis(const std::vector<uint8_t>& src, int w, int h, int target, bool horizontal) {
    int old = horizontal ? w : h, nw = horizontal ? target : w, nh = horizontal ? h : target;
    if (old == target) return src;
    std::vector<uint8_t> out(size_t(nw) * nh * 3);
    double scale = double(old) / target, filter = std::max(1.0, scale);
    for (int i = 0; i < target; ++i) {
        double center = (i + .5) * scale;
        int first = std::max(0, int(center - 2 * filter + .5));
        int last = std::min(old, int(center + 2 * filter + .5));
        std::vector<double> weights; double total = 0;
        for (int j = first; j < last; ++j) { weights.push_back(cubic((j - center + .5) / filter)); total += weights.back(); }
        for (auto& x : weights) x /= total;
        for (int outer = 0; outer < (horizontal ? h : w); ++outer) for (int c = 0; c < 3; ++c) {
            double value = 0;
            for (int j = first; j < last; ++j) {
                size_t offset = horizontal ? (size_t(outer) * w + j) * 3 + c : (size_t(j) * w + outer) * 3 + c;
                value += src[offset] * weights[j - first];
            }
            size_t offset = horizontal ? (size_t(outer) * nw + i) * 3 + c : (size_t(i) * nw + outer) * 3 + c;
            out[offset] = uint8_t(std::max(0.0, std::min(255.0, std::floor(value + .5))));
        }
    }
    return out;
}
std::string decodePiece(MNN::Transformer::Tokenizer* tokenizer, int token) {
    auto text = tokenizer->decode(token);
    if (text.size() == 6 && text.substr(0, 3) == "<0x" && text[5] == '>')
        text = char(std::strtoul(text.substr(3, 2).c_str(), nullptr, 16));
    return text;
}
} // namespace

struct Runtime::Impl {
    std::unique_ptr<Backend> backend;
    Log log;
    std::function<bool()> debug;
    Profile p;
    std::map<std::string, Graph> graphs;
    std::unique_ptr<Cache> cache;
    std::unique_ptr<MNN::Transformer::Tokenizer> tokenizer;
    std::string root;
    std::vector<int32_t> indices;
    std::vector<float> weights, visionCos, visionSin;
    std::vector<std::string> llmPrefill, llmDecode, vision;

    void trace(const std::string& text) { if (log) log("[FULL_NPU] " + text); }
    bool run(const std::string& name, const std::vector<View>& inputs,
             std::vector<std::vector<float>>& outputs, const Cancel& cancel, std::string& error) {
        if (cancel && cancel()) { error = "cancelled"; return false; }
        auto it = graphs.find(name);
        if (it == graphs.end()) { error = "Missing graph " + name; return false; }
        const auto& g = it->second;
        if (inputs.size() != g.inputs.size()) { error = name + ": input count mismatch"; return false; }
        for (size_t i = 0; i < inputs.size(); ++i)
            if (inputs[i].dtype != g.inputs[i].dtype || inputs[i].elements != g.inputs[i].elements() || !inputs[i].data) {
                error = name + ": input contract mismatch for " + g.inputs[i].name; return false;
            }
        auto begin = std::chrono::steady_clock::now();
        bool ok = backend->run(g, inputs, outputs, error);
        if (!ok) { error = name + ": " + error; trace("level=error " + error); return false; }
        if (outputs.size() != g.outputs.size()) { error = name + ": output count mismatch"; return false; }
        for (size_t i = 0; i < outputs.size(); ++i) {
            if (outputs[i].size() != g.outputs[i].elements()) { error = name + ": output shape mismatch"; return false; }
            for (float x : outputs[i]) if (!std::isfinite(x)) { error = name + ": nonfinite output"; return false; }
            if (debug && debug()) {
                float lo = std::numeric_limits<float>::infinity(), hi = -lo;
                uint64_t hash = 1469598103934665603ULL;
                for (float x : outputs[i]) {
                    uint32_t bits; std::memcpy(&bits, &x, 4); hash = (hash ^ bits) * 1099511628211ULL;
                    lo = std::min(lo, x); hi = std::max(hi, x);
                }
                trace("level=tensor graph=" + name + " output=" + g.outputs[i].name +
                      " elements=" + std::to_string(outputs[i].size()) + " min=" + std::to_string(lo) +
                      " max=" + std::to_string(hi) + " fnv64=" + std::to_string(hash));
            }
        }
        trace("level=exec_ok graph=" + name + " elapsed_ms=" + std::to_string(elapsed(begin)) +
              " past=" + std::to_string(cache ? cache->length : 0));
        return true;
    }

    bool image(const std::string& path, std::vector<std::vector<float>>& features,
               const Cancel& cancel, std::string& error) {
        int w = 0, h = 0, channels = 0;
        if (!stbi_info(path.c_str(), &w, &h, &channels) || w < 1 || h < 1 || int64_t(w) * h > 64000000) {
            error = "Cannot decode image or image exceeds 64 MP: " + path; return false;
        }
        auto* raw = stbi_load(path.c_str(), &w, &h, &channels, 3);
        if (!raw) { error = "Cannot decode image: " + path; return false; }
        std::vector<uint8_t> pixels(raw, raw + size_t(w) * h * 3); stbi_image_free(raw);
        pixels = resizeAxis(pixels, w, h, p.imageWidth, true);
        pixels = resizeAxis(pixels, p.imageWidth, h, p.imageHeight, false);
        int gh = p.imageHeight / p.patch, gw = p.imageWidth / p.patch;
        std::vector<float> patches; patches.reserve(size_t(p.imageWidth) * p.imageHeight * 3 * p.temporal);
        // Qwen3-VL: merge-block-major, C,T,P,P order within each patch.
        for (int by = 0; by < gh / p.merge; ++by) for (int bx = 0; bx < gw / p.merge; ++bx)
        for (int dy = 0; dy < p.merge; ++dy) for (int dx = 0; dx < p.merge; ++dx)
        for (int c = 0; c < 3; ++c) for (int t = 0; t < p.temporal; ++t)
        for (int py = 0; py < p.patch; ++py) for (int px = 0; px < p.patch; ++px) {
            int y = (by * p.merge + dy) * p.patch + py, x = (bx * p.merge + dx) * p.patch + px;
            float v = float(pixels[(size_t(y) * p.imageWidth + x) * 3 + c]) * (1.0f / 255.0f);
            patches.push_back((v - .5f) / .5f);
        }
        std::vector<std::vector<float>> out, deep;
        std::vector<float> mask(size_t(gh) * gw * gh * gw, 0);
        if (p.fusedVision) {
            // DeepStack taps and their individual mergers are inside this graph.
            // Output order remains [image_embeddings, projected DeepStack stack].
            if (!run("vision_full", {floats(patches), {DType::I32, indices.data(), indices.size()},
                     floats(weights), floats(visionCos), floats(visionSin), floats(mask)}, features, cancel, error)) return false;
            backend->release();
            return true;
        }
        if (!run("vision_pre", {floats(patches), {DType::I32, indices.data(), indices.size()}, floats(weights)}, out, cancel, error)) return false;
        backend->release();
        auto hidden = std::move(out[0]);
        for (const auto& name : vision) {
            if (!run(name, {floats(hidden), floats(visionCos), floats(visionSin), floats(mask)}, out, cancel, error)) return false;
            hidden = std::move(out[0]);
            for (size_t i = 1; i < out.size(); ++i) deep.push_back(std::move(out[i]));
            backend->release();
        }
        std::vector<View> views{floats(hidden)};
        for (const auto& d : deep) views.push_back(floats(d));
        if (!run("vision_post", views, features, cancel, error)) return false;
        backend->release();
        return true;
    }

    void rope(const std::vector<int>& ids, const std::vector<int>& imageOffsets,
              std::vector<float>& cos, std::vector<float>& sin, int& nextPosition) const {
        int half = p.dim / 2, token = 0, position = 0;
        cos.resize(ids.size() * p.dim); sin.resize(cos.size());
        auto write = [&](int index, int temporal, int height, int width) {
            for (int d = 0; d < half; ++d) {
                int axis = d % 3 == 1 && d < p.mrope[1] * 3 ? height :
                           d % 3 == 2 && d < p.mrope[2] * 3 ? width : temporal;
                float inv = 1.0f / std::pow(p.ropeTheta, float(2 * d) / p.dim);
                float angle = float(axis) * inv;
                for (int j : {d, d + half}) { cos[size_t(index) * p.dim + j] = std::cos(angle); sin[size_t(index) * p.dim + j] = std::sin(angle); }
            }
        };
        int mh = p.imageHeight / p.patch / p.merge, mw = p.imageWidth / p.patch / p.merge;
        for (int start : imageOffsets) {
            while (token < start) { write(token++, position, position, position); ++position; }
            for (int y = 0; y < mh; ++y) for (int x = 0; x < mw; ++x) write(token++, position, position + y, position + x);
            position += std::max(mh, mw);
        }
        while (token < int(ids.size())) { write(token++, position, position, position); ++position; }
        nextPosition = position;
    }

    bool block(const std::vector<int>& ids, int offset, int real, bool decode,
               const std::vector<float>& cos, const std::vector<float>& sin,
               const std::vector<int>& imageOffsets, const std::vector<std::vector<std::vector<float>>>& images,
               std::vector<float>& last, const Cancel& cancel, std::string& error) {
        int width = decode ? 1 : p.prefill;
        std::vector<int32_t> tokens(width, p.padToken);
        for (int i = 0; i < real; ++i) tokens[i] = ids[offset + i];
        std::vector<std::vector<float>> out;
        if (!run(decode ? "embedding_decode" : "embedding_prefill", {{DType::I32, tokens.data(), tokens.size()}}, out, cancel, error)) return false;
        auto hidden = std::move(out[0]);
        int imageTokens = p.imageHeight / p.patch * (p.imageWidth / p.patch) / (p.merge * p.merge);
        for (size_t image = 0; image < images.size(); ++image) for (int row = 0; row < real; ++row) {
            int index = offset + row - imageOffsets[image];
            if (index >= 0 && index < imageTokens)
                std::copy_n(images[image][0].data() + size_t(index) * p.hidden, p.hidden, hidden.data() + size_t(row) * p.hidden);
        }
        std::vector<float> blockCos(size_t(width) * p.dim, 1), blockSin(blockCos.size(), 0);
        std::copy_n(cos.data() + size_t(offset) * p.dim, size_t(real) * p.dim, blockCos.data());
        std::copy_n(sin.data() + size_t(offset) * p.dim, size_t(real) * p.dim, blockSin.data());
        auto mask = cache->mask(real, width);
        for (const auto& name : decode ? llmDecode : llmPrefill) {
            const auto& g = graphs.at(name);
            int count = g.end - g.start;
            std::vector<float> deep(size_t(count) * width * p.hidden, 0);
            if (!decode) for (size_t image = 0; image < images.size(); ++image) for (int row = 0; row < real; ++row) {
                int index = offset + row - imageOffsets[image];
                if (index < 0 || index >= imageTokens) continue;
                int depth = int(images[image][1].size() / (size_t(imageTokens) * p.hidden));
                for (int local = 0; local < count && g.start + local < depth; ++local)
                    std::copy_n(images[image][1].data() + (size_t(g.start + local) * imageTokens + index) * p.hidden,
                                p.hidden, deep.data() + (size_t(local) * width + row) * p.hidden);
            }
            std::vector<View> views{floats(hidden), floats(blockCos), floats(blockSin), floats(mask), floats(deep)};
            auto past = cache->inputs(g.start, count); views.insert(views.end(), past.begin(), past.end());
            if (!run(name, views, out, cancel, error)) return false;
            hidden = std::move(out[0]);
            std::vector<std::vector<float>> deltas;
            for (size_t i = 1; i < out.size(); ++i) deltas.push_back(std::move(out[i]));
            if (!cache->append(g.start, deltas, real, width, error)) return false;
        }
        cache->length += real; // all layers succeeded; errors reset the whole request.
        last.assign(hidden.begin() + size_t(real - 1) * p.hidden, hidden.begin() + size_t(real) * p.hidden);
        return true;
    }
};

Runtime::Runtime(std::unique_ptr<Backend> backend, Log log, std::function<bool()> debug) : impl(new Impl) {
    impl->backend = std::move(backend); impl->log = std::move(log); impl->debug = std::move(debug);
}
Runtime::~Runtime() = default;
const Profile& Runtime::profile() const { return impl->p; }
void Runtime::reset() { if (impl->cache) impl->cache->reset(); impl->backend->release(); }
std::string Runtime::chatTemplate(const std::vector<std::pair<std::string, std::string>>& messages) {
    std::string out;
    for (const auto& m : messages) out += "<|im_start|>" + m.first + "\n" + m.second + "<|im_end|>\n";
    return out + "<|im_start|>assistant\n";
}

bool Runtime::load(const std::string& configPath, std::string& error, ResidencyBudget residency) {
    auto& s = *impl;
    reset();
    s.cache.reset(); s.tokenizer.reset(); s.graphs.clear(); s.p = Profile{};
    s.llmPrefill.clear(); s.llmDecode.clear(); s.vision.clear();
    s.indices.clear(); s.weights.clear(); s.visionCos.clear(); s.visionSin.clear();
    error.clear();
    rapidjson::Document doc;
    doc.Parse(readText(configPath).c_str());
    const auto format = stringField(doc, "format");
    if (doc.HasParseError() || !doc.IsObject() ||
        (format != "kirin_full_npu_runtime_v1" && format != "kirin_full_npu_runtime_v2" && format != "kirin_full_npu_runtime_v3")) {
        error = "Select a dedicated full-NPU model package (config format mismatch)"; return false;
    }
    s.root = configPath.substr(0, configPath.find_last_of('/') + 1);
    if (!doc.HasMember("runtime") || !doc["runtime"].IsObject()) { error = "Missing runtime profile"; return false; }
    const auto& r = doc["runtime"]; auto& p = s.p;
    p.capacity = number(r, "capacity"); p.prefill = number(r, "prefill"); p.layers = number(r, "layers");
    p.hidden = number(r, "hidden"); p.heads = number(r, "kv_heads"); p.dim = number(r, "head_dim");
    p.group = number(r, "kv_group"); p.recent = number(r, "kv_recent"); p.sink = number(r, "kv_sink");
    if (r.HasMember("kv_dtype")) {
        auto type = stringField(r, "kv_dtype");
        if (type != "int8" && type != "fp16") { error = "Invalid KV dtype"; return false; }
        p.kvType = type == "fp16" ? DType::F16 : DType::I8;
    } else if (format != "kirin_full_npu_runtime_v1") { error = "Missing KV dtype"; return false; }
    p.fusedVision = format == "kirin_full_npu_runtime_v3";
    if (p.fusedVision) {
        p.visionLayers = number(r, "vision_layers");
        if (p.visionLayers < 1 || p.visionLayers > 128 || !r.HasMember("vision_deepstack_layers") ||
            !r["vision_deepstack_layers"].IsArray() || r["vision_deepstack_layers"].Empty() ||
            r["vision_deepstack_layers"].Size() > unsigned(p.layers)) {
            error = "Invalid fused visual DeepStack configuration"; return false;
        }
        for (const auto& v : r["vision_deepstack_layers"].GetArray()) {
            if (!v.IsInt() || v.GetInt() < 0 || v.GetInt() >= p.visionLayers ||
                (!p.visionDeepstack.empty() && v.GetInt() <= p.visionDeepstack.back())) {
                error = "Invalid fused visual DeepStack layer order"; return false;
            }
            p.visionDeepstack.push_back(v.GetInt());
        }
    }
    if (p.kvType == DType::F16 && (p.sink || p.recent)) {
        error = "FP16 KV must not declare INT8 protected windows"; return false;
    }
    p.imageWidth = number(r, "image_width"); p.imageHeight = number(r, "image_height");
    p.patch = number(r, "patch"); p.merge = number(r, "merge"); p.temporal = number(r, "temporal");
    p.visionHidden = number(r, "vision_hidden"); p.visionHeads = number(r, "vision_heads");
    p.imageToken = number(r, "image_token"); p.padToken = number(r, "pad_token"); p.maxNewTokens = number(r, "max_new_tokens");
    for (const char* key : {"npu_max_resident_graphs", "npu_resident_model_mib", "npu_persistent_kv_mib"}) {
        if (r.HasMember(key) && (!r[key].IsInt() || r[key].GetInt() < 0 || r[key].GetInt() > 4096)) {
            error = "Invalid NNRT memory option: " + std::string(key); return false;
        }
    }
    if (r.HasMember("npu_max_resident_graphs")) p.backend.maxResidentGraphs = number(r, "npu_max_resident_graphs");
    if (r.HasMember("npu_resident_model_mib")) p.backend.residentModelBytes = size_t(number(r, "npu_resident_model_mib")) << 20;
    if (r.HasMember("npu_persistent_kv_mib")) p.backend.persistentKvBytes = size_t(number(r, "npu_persistent_kv_mib")) << 20;
    if (r.HasMember("npu_incremental_kv")) {
        if (!r["npu_incremental_kv"].IsBool()) { error = "Invalid npu_incremental_kv"; return false; }
        p.backend.incrementalKv = r["npu_incremental_kv"].GetBool();
    }
    if (p.backend.maxResidentGraphs < 1 || p.backend.maxResidentGraphs > 16 || !p.backend.residentModelBytes) {
        error = "Invalid NNRT residency budget"; return false;
    }
    if (residency.maxResidentGraphs < 0 || residency.maxResidentGraphs > 16 ||
        residency.residentModelBytes > (size_t(4096) << 20)) {
        error = "Invalid caller NNRT residency budget"; return false;
    }
    if (residency.maxResidentGraphs || residency.residentModelBytes) {
        const auto modelOptions = p.backend;
        if (residency.maxResidentGraphs) p.backend.maxResidentGraphs = residency.maxResidentGraphs;
        if (residency.residentModelBytes) p.backend.residentModelBytes = residency.residentModelBytes;
        s.trace("level=cache_budget_override source=caller from_max_graphs=" + std::to_string(modelOptions.maxResidentGraphs) +
                " from_model_budget_bytes=" + std::to_string(modelOptions.residentModelBytes) +
                " max_graphs=" + std::to_string(p.backend.maxResidentGraphs) +
                " model_budget_bytes=" + std::to_string(p.backend.residentModelBytes));
    }
    if (!r.HasMember("rope_theta") || !r["rope_theta"].IsNumber() || !r.HasMember("mrope_section") || !r["mrope_section"].IsArray()) {
        error = "Missing M-RoPE configuration"; return false;
    }
    p.ropeTheta = r["rope_theta"].GetFloat();
    for (const auto& v : r["mrope_section"].GetArray()) {
        if (!v.IsInt() || v.GetInt() < 0 || v.GetInt() > p.dim / 2) { error = "Invalid M-RoPE section"; return false; }
        p.mrope.push_back(v.GetInt());
    }
    if (p.capacity < 1 || p.capacity > 32768 || p.prefill < 1 || p.prefill > p.capacity || p.layers < 1 || p.layers > 128 ||
        p.hidden < 1 || p.hidden > 16384 || p.heads < 1 || p.heads > 128 || p.dim < 2 || p.dim > 512 || p.dim % 2 ||
        p.group < 1 || p.dim % p.group || p.sink < 0 || p.sink > p.capacity || p.recent < 0 || p.recent > p.capacity ||
        p.patch < 1 || p.patch > 128 || p.merge < 1 || p.merge > 16 || p.temporal != 2 || p.imageWidth < 1 || p.imageHeight < 1 ||
        p.imageWidth > 4096 || p.imageHeight > 4096 || p.imageWidth % (p.patch * p.merge) || p.imageHeight % (p.patch * p.merge) ||
        p.visionHidden < 1 || p.visionHidden > 16384 || p.visionHeads < 1 || p.visionHeads > 128 ||
        p.visionHidden % p.visionHeads || (p.imageWidth / p.patch) * (p.imageHeight / p.patch) > 4096 ||
        p.mrope.size() != 3 || !std::isfinite(p.ropeTheta) || p.ropeTheta <= 0 || p.maxNewTokens < 1 ||
        p.padToken < 0 || p.imageToken < 0 ||
        size_t(p.layers) * p.heads * p.capacity * p.dim > (size_t(1) << 29)) {
        error = "Invalid or excessive static runtime dimensions"; return false;
    }
    auto tokenizerPath = stringField(doc, "tokenizer_file"), constants = stringField(doc, "vision_constants");
    if (!safePath(tokenizerPath) || !safePath(constants)) { error = "Unsafe runtime file path"; return false; }
    s.tokenizer.reset(MNN::Transformer::Tokenizer::createTokenizer(s.root + tokenizerPath));
    if (!s.tokenizer) { error = "Tokenizer load failed"; return false; }
    size_t n = size_t(p.imageWidth / p.patch) * (p.imageHeight / p.patch);
    s.indices.resize(n * 4); s.weights.resize(n * 4);
    if (p.visionHeads < 1 || p.visionHidden % p.visionHeads) { error = "Invalid visual dimensions"; return false; }
    s.visionCos.resize(n * (p.visionHidden / p.visionHeads)); s.visionSin.resize(s.visionCos.size());
    std::ifstream f(s.root + constants, std::ios::binary);
    for (auto target : {std::make_pair(static_cast<void*>(s.indices.data()), s.indices.size()),
                        std::make_pair(static_cast<void*>(s.weights.data()), s.weights.size()),
                        std::make_pair(static_cast<void*>(s.visionCos.data()), s.visionCos.size()),
                        std::make_pair(static_cast<void*>(s.visionSin.data()), s.visionSin.size())})
        f.read(static_cast<char*>(target.first), target.second * 4);
    if (!f || f.peek() != std::char_traits<char>::eof()) { error = "Visual constants byte size mismatch"; return false; }
    if (!doc.HasMember("graphs") || !doc["graphs"].IsArray()) { error = "Missing graph contracts"; return false; }
    for (const auto& value : doc["graphs"].GetArray()) {
        Graph g; g.name = stringField(value, "name"); auto path = stringField(value, "file");
        if (g.name.empty() || !safePath(path) || s.graphs.count(g.name)) { error = "Invalid graph name/path"; return false; }
        g.path = s.root + path; g.start = number(value, "layer_start"); g.end = number(value, "layer_end");
        if (value.HasMember("external_weights")) {
            const auto& weights = value["external_weights"];
            auto directory = stringField(weights, "directory");
            if (!safePath(directory) || !weights.HasMember("files") || !weights["files"].IsArray() ||
                weights["files"].Empty() || weights["files"].Size() > 64 || value.HasMember("embedding_slices")) {
                error = "Invalid external weight contract"; return false;
            }
            g.weightDir = s.root + directory;
            for (const auto& file : weights["files"].GetArray()) {
                if (!file.IsString() || !safePath(file.GetString()) || std::string(file.GetString()).find('/') != std::string::npos ||
                    std::find(g.weightFiles.begin(), g.weightFiles.end(), file.GetString()) != g.weightFiles.end()) {
                    error = "Invalid external weight filename"; return false;
                }
                g.weightFiles.emplace_back(file.GetString());
            }
        }
        if (value.HasMember("embedding_slices")) {
            if (!value["embedding_slices"].IsArray()) { error = "Invalid embedding slices"; return false; }
            for (const auto& entry : value["embedding_slices"].GetArray()) {
                auto relative = stringField(entry, "file");
                if (!safePath(relative)) { error = "Unsafe embedding slice path"; return false; }
                g.embeddingSlices.push_back({s.root + relative, number(entry, "hidden_start"), number(entry, "hidden_end")});
            }
            if (g.embeddingSlices.empty()) { error = "Empty embedding slices"; return false; }
        }
        for (const char* key : {"inputs", "output_info"}) {
            if (!value.HasMember(key) || !value[key].IsArray()) { error = "Missing graph ports"; return false; }
            for (const auto& entry : value[key].GetArray()) {
                Port parsed;
                if (!port(entry, parsed)) { error = "Invalid graph port"; return false; }
                (std::string(key) == "inputs" ? g.inputs : g.outputs).push_back(std::move(parsed));
            }
        }
        if (value.HasMember("output_aliases") && value["output_aliases"].IsObject())
            for (auto it = value["output_aliases"].MemberBegin(); it != value["output_aliases"].MemberEnd(); ++it) {
                if (!it->value.IsString()) { error = "Invalid output alias"; return false; }
                g.aliases[it->name.GetString()] = it->value.GetString();
            }
        s.graphs.emplace(g.name, std::move(g));
    }
    for (const char* prefix : {"llm_prefill_", "llm_decode_", "vision_blocks_"}) {
        if (p.fusedVision && std::string(prefix) == "vision_blocks_") continue;
        auto& names = std::string(prefix) == "llm_prefill_" ? s.llmPrefill : std::string(prefix) == "llm_decode_" ? s.llmDecode : s.vision;
        int end = 0;
        for (int i = 0; s.graphs.count(std::string(prefix) + std::to_string(i)); ++i) {
            auto name = std::string(prefix) + std::to_string(i); auto& g = s.graphs.at(name);
            if (g.start != end || g.end <= g.start || (std::string(prefix) != "vision_blocks_" && g.end > p.layers)) {
                error = "Incomplete/overlapping decoder graph coverage"; return false;
            }
            end = g.end; names.push_back(name);
        }
        if (names.empty() || (std::string(prefix) != "vision_blocks_" && end != p.layers)) { error = "Incomplete graph coverage"; return false; }
    }
    for (const char* name : {"embedding_prefill", "embedding_decode", "lm_head"})
        if (!s.graphs.count(name)) { error = "Missing full model component"; return false; }
    for (const auto& name : p.fusedVision ? std::vector<std::string>{"vision_full"} :
                                           std::vector<std::string>{"vision_pre", "vision_post"})
        if (!s.graphs.count(name)) { error = "Missing visual model component"; return false; }
    if (!validateRuntimeGraphs(p, s.graphs, s.llmPrefill, s.llmDecode, s.vision, error)) return false;
    s.backend->configure(p.backend);
    // Probe sequentially, freeing each graph. Do not retain both LLM phases.
    // Allocate the persistent KV only after all model probes finish, leaving
    // more memory available for the driver's initial model-loading peak.
    for (const auto& item : s.graphs) {
        s.trace("level=probe_begin graph=" + item.first);
        if (!s.backend->probe(item.second, error)) {
            error = "graph=" + item.first + " " + error;
            s.trace("level=error " + error);
            s.backend->release();
            return false;
        }
        s.backend->release();
    }
    s.cache.reset(new Cache(p));
    s.trace("level=ready graphs=" + std::to_string(s.graphs.size()) + " context=" + std::to_string(p.capacity) +
            " kv_bytes=" + std::to_string(s.cache->bytes()) +
            (p.kvType == DType::F16 ? " kv_dtype=fp16" : " kv_dtype=int8") + " cpu_fallback=disabled");
    return true;
}

bool Runtime::response(const std::string& input, const Cancel& cancel, const Emit& emit,
                       std::string& output, std::string& error, int limit) {
    auto& s = *impl; output.clear(); error.clear(); reset();
    if (!s.tokenizer || !s.cache) { error = "Full NPU model not loaded"; return false; }
    auto fail = [&]() { reset(); return false; };
    std::string text = input;
    std::vector<std::string> paths;
    const int imageTokens = s.p.imageHeight / s.p.patch * (s.p.imageWidth / s.p.patch) / (s.p.merge * s.p.merge);
    size_t at = 0;
    while ((at = text.find("<img>", at)) != std::string::npos) {
        size_t end = text.find("</img>", at);
        if (end == std::string::npos) { error = "Malformed image tag"; return fail(); }
        auto path = text.substr(at + 5, end - at - 5);
        size_t hwStart = path.find("<hw>");
        if (hwStart != std::string::npos) {
            size_t hwEnd = path.find("</hw>");
            if (hwEnd == std::string::npos || hwEnd < hwStart) { error = "Malformed image size tag"; return fail(); }
            path.erase(hwStart, hwEnd + 5 - hwStart);
        }
        if (path.empty() || paths.size() >= size_t(s.p.capacity / imageTokens)) {
            error = "Missing image or images exceed the context capacity"; return fail();
        }
        paths.push_back(path);
        std::string placeholder = "<|vision_start|>";
        for (int i = 0; i < imageTokens; ++i) placeholder += "<|image_pad|>";
        placeholder += "<|vision_end|>";
        text.replace(at, end + 6 - at, placeholder); at += placeholder.size();
    }
    if (text.find("<audio>") != std::string::npos || text.find("<video>") != std::string::npos) {
        error = "This static full VLM package supports text and still images only"; return fail();
    }
    auto ids = s.tokenizer->encode(text);
    if (ids.empty() || ids.size() >= size_t(s.p.capacity)) {
        error = "Full NPU context limit exceeded (includes image and response tokens): " + std::to_string(s.p.capacity); return fail();
    }
    int vocabulary = s.graphs.at("lm_head").outputs[0].shape[2];
    for (int token : ids) if (token < 0 || token >= vocabulary) {
        error = "Tokenizer emitted an ID outside the compiled embedding vocabulary";
        return fail();
    }
    std::vector<int> offsets;
    for (size_t i = 0; i < ids.size();) {
        if (ids[i] != s.p.imageToken) { ++i; continue; }
        size_t begin = i; while (i < ids.size() && ids[i] == s.p.imageToken) ++i;
        if (i - begin != size_t(imageTokens)) { error = "Unexpected image token count"; return fail(); }
        offsets.push_back(int(begin));
    }
    if (offsets.size() != paths.size()) { error = "Image placeholder/path mismatch"; return fail(); }
    std::vector<std::vector<std::vector<float>>> images;
    auto begin = std::chrono::steady_clock::now();
    for (const auto& path : paths) {
        images.emplace_back(); if (!s.image(path, images.back(), cancel, error)) return fail();
    }
    double visionMs = elapsed(begin);
    std::vector<float> cos, sin, hidden; int nextPosition;
    s.rope(ids, offsets, cos, sin, nextPosition);
    begin = std::chrono::steady_clock::now();
    for (int offset = 0; offset < int(ids.size()); offset += s.p.prefill)
        if (!s.block(ids, offset, std::min(s.p.prefill, int(ids.size()) - offset), false,
                     cos, sin, offsets, images, hidden, cancel, error)) return fail();
    double prefillMs = elapsed(begin);
    s.backend->releaseGraphs(); // Drop prefill models; reuse their KV tensors in decode.
    int maximum = std::min(limit < 0 ? s.p.maxNewTokens : limit, s.p.capacity - int(ids.size()));
    int generated = 0; std::string pending;
    begin = std::chrono::steady_clock::now();
    for (int step = 0; step < maximum; ++step) {
        std::vector<std::vector<float>> logits;
        if (!s.run("lm_head", {floats(hidden)}, logits, cancel, error)) return fail();
        int token = int(std::max_element(logits[0].begin(), logits[0].end()) - logits[0].begin());
        if (s.debug && s.debug()) s.trace("level=sample step=" + std::to_string(step) +
            " token_id=" + std::to_string(token) + " logit=" + std::to_string(logits[0][token]) +
            " past=" + std::to_string(s.cache->length));
        if (s.tokenizer->is_stop(token)) break;
        pending += decodePiece(s.tokenizer.get(), token); ++generated;
        auto piece = takeUtf8(pending);
        output += piece; if (emit && !piece.empty()) emit(piece);
        if (cancel && cancel()) { error = "cancelled"; return fail(); }
        if (step + 1 >= maximum) break;
        std::vector<int> single{token};
        cos.resize(s.p.dim); sin.resize(s.p.dim);
        int position = nextPosition + step;
        for (int d = 0; d < s.p.dim / 2; ++d) {
            float angle = float(position) / std::pow(s.p.ropeTheta, float(2 * d) / s.p.dim);
            for (int j : {d, d + s.p.dim / 2}) { cos[j] = std::cos(angle); sin[j] = std::sin(angle); }
        }
        if (!s.block(single, 0, 1, true, cos, sin, {}, {}, hidden, cancel, error)) return fail();
    }
    auto tail = takeUtf8(pending, true);
    output += tail; if (emit && !tail.empty()) emit(tail);
    s.trace("level=request_complete prompt_tokens=" + std::to_string(ids.size()) + " decode_tokens=" + std::to_string(generated) +
            " vision_ms=" + std::to_string(visionMs) + " prefill_ms=" + std::to_string(prefillMs) +
            " decode_ms=" + std::to_string(elapsed(begin)) + " kv_length=" + std::to_string(s.cache->length));
    s.backend->release();
    return true;
}
} // namespace KirinFullNpu
