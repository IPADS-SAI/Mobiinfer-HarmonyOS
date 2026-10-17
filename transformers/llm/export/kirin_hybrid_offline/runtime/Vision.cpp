#include "OfflineNpu2.hpp"
#include "rapidjson/document.h"
#include "stb_image.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>

namespace OfflineNpu2 {
namespace {
using namespace KirinFullNpu;
using Json = rapidjson::Value;
std::string text(const std::string& path) { std::ifstream f(path); return {std::istreambuf_iterator<char>(f), {}}; }
std::string field(const Json& j, const char* k) {
    return j.IsObject() && j.HasMember(k) && j[k].IsString() ? j[k].GetString() : "";
}
bool relative(const std::string& p) {
    if (p.empty() || p[0]=='/' || p.find('\\')!=std::string::npos) return false;
    std::istringstream s(p); std::string part;
    while (std::getline(s, part, '/')) if (part.empty() || part=="." || part=="..") return false;
    return true;
}
View floats(const std::vector<float>& v) { return {DType::F32, v.data(), v.size()}; }
// Same RGB bicubic preprocessing and merge-block-major patch order as Full NPU.
// Keep the actual implementation identical to its numerically validated path.
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
}
struct Vision::Impl {
    std::unique_ptr<Backend> backend;
    Log log; Cancel cancel; Graph graph; bool ready = false;
    std::vector<int32_t> indices;
    std::vector<float> weights, cosine, sine, mask;
    void trace(const std::string& s) { if (log) log("[OFFLINE_NPU2] " + s); }
};
Vision::Vision(std::unique_ptr<Backend> backend, Log log, Cancel cancel) : impl(new Impl) {
    impl->backend = std::move(backend); impl->log = std::move(log); impl->cancel = std::move(cancel);
}
Vision::~Vision() { if (impl->backend) impl->backend->release(); }
bool Vision::load(const std::string& path, std::string& error) {
    auto& s = *impl; s.ready = false; error.clear();
    if (!s.backend) { error = "No NNRT vision backend"; return false; }
    rapidjson::Document d; auto content = text(path); d.Parse(content.c_str());
    if (d.HasParseError() || !d.IsObject() || field(d,"format") != "kirin_offline_npu2_vision_v1" ||
        !d.HasMember("graph") || !d["graph"].IsObject()) { error = "Invalid dedicated Offline NPU vision config"; return false; }
    const auto root = path.substr(0, path.find_last_of('/')+1);
    const auto& g = d["graph"]; s.graph = {}; s.graph.name = "vision_full";
    auto file = field(g,"file"), constants = field(d,"vision_constants");
    if (field(g,"name") != "vision_full" || !relative(file) || !relative(constants) ||
        !g.HasMember("external_weights") || !g["external_weights"].IsObject()) { error = "Expected fused vision graph with external weights"; return false; }
    s.graph.path = root+file;
    const auto& ew = g["external_weights"]; auto dir = field(ew,"directory");
    if (!relative(dir) || !ew.HasMember("files") || !ew["files"].IsArray() || ew["files"].Empty()) { error = "Invalid external visual weight paths"; return false; }
    s.graph.weightDir = root+dir;
    for (auto& f : ew["files"].GetArray()) {
        if (!f.IsString() || !relative(f.GetString())) { error = "Invalid weight file"; return false; }
        s.graph.weightFiles.push_back(f.GetString());
    }
    const std::vector<Port> inputs = {
        {"patches",DType::F32,{608,1536}}, {"position_indices",DType::I32,{608,4}},
        {"position_weights",DType::F32,{608,4}}, {"rope_cos",DType::F32,{1,1,608,64}},
        {"rope_sin",DType::F32,{1,1,608,64}}, {"attention_mask",DType::F32,{1,1,608,608}}};
    const std::vector<Port> outputs = {{"image_embeddings",DType::F32,{1,152,2048}},
                                      {"deepstack_features",DType::F32,{3,1,152,2048}}};
    auto contracts = [&](const char* key, const std::vector<Port>& ports) {
        if (!g.HasMember(key) || !g[key].IsArray() || g[key].Size()!=ports.size()) return false;
        for (size_t i=0; i<ports.size(); ++i) {
            const auto& p = g[key][static_cast<rapidjson::SizeType>(i)];
            if (!p.IsObject() || field(p,"name")!=ports[i].name ||
                field(p,"dtype")!=(ports[i].dtype==DType::I32?"int32":"float32") ||
                !p.HasMember("shape") || !p["shape"].IsArray() || p["shape"].Size()!=ports[i].shape.size()) return false;
            for (size_t j=0;j<ports[i].shape.size();++j)
                if (!p["shape"][j].IsInt() || p["shape"][j].GetInt()!=ports[i].shape[j]) return false;
        }
        return true;
    };
    if (!contracts("inputs",inputs) || !contracts("output_info",outputs)) { error = "Unsupported vision graph contract/grid (requires 608x256, DeepStack 3x152x2048)"; return false; }
    s.graph.inputs=inputs; s.graph.outputs=outputs;
    if (g.HasMember("output_aliases") && g["output_aliases"].IsObject())
        for (auto it=g["output_aliases"].MemberBegin();it!=g["output_aliases"].MemberEnd();++it)
            if (it->value.IsString()) s.graph.aliases[it->name.GetString()]=it->value.GetString();
    s.indices.resize(608*4); s.weights.resize(608*4); s.cosine.resize(608*64); s.sine.resize(608*64);
    std::ifstream f(root+constants,std::ios::binary);
    f.read(reinterpret_cast<char*>(s.indices.data()),s.indices.size()*4);
    for (auto* v : {&s.weights,&s.cosine,&s.sine}) f.read(reinterpret_cast<char*>(v->data()),v->size()*4);
    if (!f || f.peek()!=std::char_traits<char>::eof()) { error = "Vision constants size mismatch"; return false; }
    for (int i:s.indices) if (i<0 || i>=48*48) { error="Vision position index outside source table"; return false; }
    for (auto* v : {&s.weights,&s.cosine,&s.sine}) for (float x:*v)
        if (!std::isfinite(x)) { error="Nonfinite vision constants"; return false; }
    s.mask.assign(608*608,0);
    BackendOptions options; options.maxResidentGraphs=1; options.residentModelBytes=size_t(512)<<20;
    options.persistentKvBytes=0; options.incrementalKv=false; s.backend->configure(options);
    if (!s.backend->probe(s.graph,error)) { s.backend->release(); return false; }
    s.ready=true;
    s.trace("level=ready vision=single_external_weight_graph llm=cpu kv=cpu deepstack=3 grid=608x256");
    return true;
}
bool Vision::run(const std::string& path, Features& features, std::string& error) {
    auto& s=*impl;
    if (!s.ready) { error="Vision not loaded"; return false; }
    if (s.cancel && s.cancel()) { error="cancelled"; return false; }
    const auto begin=std::chrono::steady_clock::now();
    int w=0,h=0,c=0;
    if (!stbi_info(path.c_str(),&w,&h,&c) || w<1 || h<1 || int64_t(w)*h>64000000) { error="Invalid image or image exceeds 64 MP: "+path; return false; }
    auto* raw=stbi_load(path.c_str(),&w,&h,&c,3);
    if (!raw) { error="Cannot decode image: "+path; return false; }
    std::vector<uint8_t> pixels(raw,raw+size_t(w)*h*3); stbi_image_free(raw);
    pixels=resizeAxis(pixels,w,h,256,true); pixels=resizeAxis(pixels,256,h,608,false);
    std::vector<float> patches; patches.reserve(608*1536);
    for (int by=0;by<19;++by) for (int bx=0;bx<8;++bx)
    for (int dy=0;dy<2;++dy) for (int dx=0;dx<2;++dx)
    for (int ch=0;ch<3;++ch) for (int t=0;t<2;++t)
    for (int py=0;py<16;++py) for (int px=0;px<16;++px) {
        int y=(by*2+dy)*16+py,x=(bx*2+dx)*16+px;
        float v=float(pixels[(size_t(y)*256+x)*3+ch])*(1.0f/255.0f);
        patches.push_back((v-.5f)/.5f);
    }
    if (s.cancel && s.cancel()) { error="cancelled"; return false; }
    std::vector<std::vector<float>> outputs;
    if (!s.backend->run(s.graph,{floats(patches),{DType::I32,s.indices.data(),s.indices.size()},
            floats(s.weights),floats(s.cosine),floats(s.sine),floats(s.mask)},outputs,error)) return false;
    if (s.cancel && s.cancel()) { error="cancelled"; return false; }
    if (outputs.size()!=2 || outputs[0].size()!=152*2048 || outputs[1].size()!=3*152*2048) { error="Invalid vision output sizes"; return false; }
    for (const auto& v:outputs) for (float x:v) if (!std::isfinite(x)) { error="Nonfinite vision output"; return false; }
    features.image=std::move(outputs[0]); features.deepstack=std::move(outputs[1]);
    s.trace("level=exec_ok graph=vision_full total_ms="+std::to_string(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()));
    // Retain the only NPU graph across CPU prefill/decode and subsequent images.
    return true;
}
} // namespace OfflineNpu2
