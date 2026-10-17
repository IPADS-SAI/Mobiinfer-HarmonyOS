// Executes the production bridge and actual MNN CPU LLM, with CPU-reference
// visual outputs supplied by the Python parent. Never executes OMC on host.
#include "OfflineNpu2.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <MNN/expr/ExprCreator.hpp>
namespace fs=std::filesystem;
template<class T> void save(const fs::path& p,const T* v,size_t n) {
    std::ofstream f(p,std::ios::binary); f.write(reinterpret_cast<const char*>(v),n*sizeof(T));
}
std::string read(const fs::path& p) { std::ifstream f(p); return {std::istreambuf_iterator<char>(f),{}}; }
struct Reference final:KirinFullNpu::Backend {
    fs::path root; int runs=0,probes=0; bool fail=false;
    explicit Reference(fs::path root):root(std::move(root)){}
    bool probe(const KirinFullNpu::Graph& g,std::string&) override { ++probes; return g.name=="vision_full"; }
    void release() override {}
    bool run(const KirinFullNpu::Graph& g,const std::vector<KirinFullNpu::View>& in,
             std::vector<std::vector<float>>& out,std::string& error) override {
        if(fail) { error="injected vision failure"; return false; }
        for(size_t i=0;i<in.size();++i) save(root/ ("vision_input_"+std::to_string(i)+".bin"),
            static_cast<const char*>(in[i].data),in[i].elements*4);
        out.resize(2);
        for(int i=0;i<2;++i) {
            out[i].resize(g.outputs[i].elements());
            std::ifstream f(root/("feature_"+std::to_string(i)+".bin"),std::ios::binary);
            f.read(reinterpret_cast<char*>(out[i].data()),out[i].size()*4);
            if(!f) { error="missing host reference"; return false; }
        }
        ++runs; return true;
    }
};
int main(int argc,char** argv) {
    if(argc!=4) return 2;
    fs::path root=argv[2]; auto backend=new Reference(root);
    auto state=std::make_shared<OfflineNpu2::State>();
    auto llm=OfflineNpu2::create(argv[1],std::unique_ptr<KirinFullNpu::Backend>(backend),
        [](const std::string& s){std::cerr<<s<<'\n';},{},state);
    if(!llm || !llm->set_config("{\"sampler_type\":\"greedy\",\"max_new_tokens\":3,\"thread_num\":4}") || !llm->load()) {
        std::cerr<<"LOAD ERROR: "<<state->error<<'\n'; return 3;
    }
    std::string prompt=read(root/"prompt.txt");
    auto ids=llm->tokenizer_encode(prompt);
    if(ids.empty()) {std::cerr<<state->error; return 4;}
    save(root/"tokens.bin",ids.data(),ids.size());
    // Save pure text embeddings separately for the independent Python injector.
    auto pure=llm->MNN::Transformer::Llm::embedding(ids);
    save(root/"text_embeddings.bin",pure->readMap<float>(),pure->getInfo()->size);
    auto emb=llm->embedding(ids);
    auto positions=llm->gen_position_ids(ids.size());
    save(root/"embeddings.bin",emb->readMap<float>(),emb->getInfo()->size);
    save(root/"positions.bin",positions->readMap<int>(),positions->getInfo()->size);
    if(std::string(argv[3])=="prepare") return 0;
    if(std::string(argv[3])=="reference") {
        // Independent Python/HF tensors enter the unmodified base CPU language
        // module directly, bypassing every Hybrid injection/position override.
        using namespace MNN::Express;
        auto load=[&](const char* file,std::vector<int> shape,bool integer=false) {
            auto v=_Input(shape,NCHW,integer?halide_type_of<int>():halide_type_of<float>());
            std::ifstream f(root/file,std::ios::binary);
            f.read(integer?reinterpret_cast<char*>(v->writeMap<int>()):reinterpret_cast<char*>(v->writeMap<float>()),v->getInfo()->size*4);
            return v;
        };
        int n=ids.size();
        auto expected=load("expected_embeddings.bin",{n,1,2048});
        auto expectedPos=load("expected_positions.bin",{3,n},true);
        auto expectedDeep=load("expected_deepstack.bin",{3,n,2048});
        llm->setKVCacheInfo(n,0);
        auto mask=llm->gen_attention_mask(n);
        auto values=llm->MNN::Transformer::Llm::forwardRaw(expected,mask,expectedPos,{expectedDeep});
        if(values.empty()) return 9;
        save(root/"reference_logits.bin",values[0]->readMap<float>(),values[0]->getInfo()->size);
        return 0;
    }
    std::ostringstream output;
    llm->response(ids,&output,nullptr,0);
    if(!state->error.empty()) {std::cerr<<state->error; return 5;}
    auto logits=llm->getOutputs()[0];
    save(root/"logits.bin",logits->readMap<float>(),logits->getInfo()->size);
    // MNN's generation loop computes the following decode forward as well;
    // capture prefill logits before entering it, then compare the first token.
    llm->generate(1);
    save(root/"generated.bin",llm->getContext()->output_tokens.data(),llm->getContext()->output_tokens.size());
    // Subsequent independent text request must have no stale visual features.
    llm->reset(); OfflineNpu2::prepareRequest(llm.get(),state);
    llm->response(std::string("你是谁？"),&output,nullptr,3);
    if(!state->error.empty()) {std::cerr<<state->error;return 6;}
    // A failed image request must reject, and a retry must work on the same model.
    llm->reset(); backend->fail=true;
    llm->response(prompt,&output,nullptr,1);
    if(state->error!="injected vision failure") return 7;
    backend->fail=false; OfflineNpu2::prepareRequest(llm.get(),state);
    llm->response(prompt,&output,nullptr,3);
    if(!state->error.empty() || backend->probes!=1) return 8;
    // Agent lifecycle: cache a fixed text prefix, prefill an image, decode,
    // erase to the prefix, and repeat. M-RoPE must follow the surviving KV.
    llm->reset(); OfflineNpu2::prepareRequest(llm.get(),state);
    llm->set_config("{\"reuse_kv\":true,\"use_template\":false}");
    llm->response(std::string("<|im_start|>system\n你是助手。<|im_end|>\n"),nullptr,nullptr,0);
    auto prefix=llm->getCurrentHistory();
    if(prefix==0 || !state->error.empty()) return 10;
    for(int i=0;i<2;++i) {
        if(i) llm->eraseHistory(prefix,0);
        llm->response(prompt,&output,nullptr,2);
        if(!state->error.empty()) {std::cerr<<state->error;return 11;}
    }
    std::cerr<<"HOST SUCCESS probes="<<backend->probes<<" visual_runs="<<backend->runs<<" output="<<output.str()<<'\n';
    return 0;
}
