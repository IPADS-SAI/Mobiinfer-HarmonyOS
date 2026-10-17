#include "OfflineNpu2.hpp"
#include <cassert>
#include <limits>
#include <iostream>
int main(int argc,char** argv) {
    using namespace OfflineNpu2;
    Prompt p; std::string error;
    assert(parseImages("前<img><hw>9,7</hw>/a.jpg</img>后<img>/b.jpg<hw>8,6</hw></img>",p,error));
    assert(p.images==std::vector<std::string>({"/a.jpg","/b.jpg"}));
    std::vector<int> ids{10,151652}; ids.insert(ids.end(),152,151655);
    ids.insert(ids.end(),{151653,11,151652}); ids.insert(ids.end(),152,151655); ids.push_back(151653);
    assert(makePositions(ids,p,7,error));
    assert(p.imageOffsets==std::vector<size_t>({2,157}));
    assert(p.positions[2]==9 && p.positions[ids.size()+153]==27 && p.positions[2*ids.size()+153]==16);
    assert(p.positions[154]==28 && p.positions[157]==31 && p.nextPosition==51);
    std::vector<Features> features(2);
    for(int i=0;i<2;++i) { features[i].image.assign(152*2048,float(i+1));
        features[i].deepstack.resize(3*152*2048);
        for(int layer=0;layer<3;++layer) std::fill(features[i].deepstack.begin()+layer*152*2048,
            features[i].deepstack.begin()+(layer+1)*152*2048,float(10*i+layer+1)); }
    std::vector<float> embeddings(ids.size()*2048,-1),deep;
    assert(injectFeatures(p,features,ids.size(),embeddings.data(),deep,error));
    assert(embeddings[0]==-1 && embeddings[2*2048]==1 && embeddings[157*2048]==2 && embeddings[154*2048]==-1);
    for(int layer=0;layer<3;++layer) {
        assert(deep[(layer*ids.size()+2)*2048]==layer+1);
        assert(deep[(layer*ids.size()+157)*2048]==11+layer);
        assert(deep[(layer*ids.size()+154)*2048]==0);
    }
    features[0].deepstack[0]=std::numeric_limits<float>::infinity();
    assert(!injectFeatures(p,features,ids.size(),embeddings.data(),deep,error));
    ids[3]=42; assert(!makePositions(ids,p,0,error));
    assert(!parseImages("<img>bad",p,error));
    assert(!parseImages("<img><hw>bad</img>",p,error));
    assert(!parseImages("<video>x</video>",p,error));
    assert(parseImages("纯文本",p,error));
    assert(makePositions({1,2,3},p,12,error) && p.nextPosition==15);
    if(argc==3) {
        auto state=std::make_shared<State>();
        // A vision-only/full-NPU config must never be mistaken for a CPU LLM package.
        assert(!create(argv[1],{},{},{},state) && !state->error.empty());
        struct Counters {int probes=0,runs=0,releases=0;} counters;
        struct Backend final:KirinFullNpu::Backend {
            Counters& c; bool fail=false;
            explicit Backend(Counters& c):c(c){}
            void configure(const KirinFullNpu::BackendOptions& o) override {
                assert(o.maxResidentGraphs==1 && o.persistentKvBytes==0 && !o.incrementalKv);
            }
            bool probe(const KirinFullNpu::Graph& g,std::string&) override {
                assert(g.name=="vision_full" && !g.weightDir.empty() && g.inputs.size()==6 && g.outputs.size()==2);
                ++c.probes;return true;
            }
            void release() override {++c.releases;}
            bool run(const KirinFullNpu::Graph&,const std::vector<KirinFullNpu::View>& in,
                     std::vector<std::vector<float>>& out,std::string& error) override {
                ++c.runs;assert(in.size()==6 && in[0].elements==608*1536);
                if(fail) {error="injected";return false;}
                out={std::vector<float>(152*2048,1),std::vector<float>(3*152*2048,2)};return true;
            }
        };
        auto* backend=new Backend(counters);bool cancel=false;
        {
            Vision vision(std::unique_ptr<KirinFullNpu::Backend>(backend),{},[&](){return cancel;});
            assert(vision.load(argv[1],error));
            Features f;
            assert(vision.run(argv[2],f,error));
            assert(vision.run(argv[2],f,error));
            assert(counters.probes==1 && counters.runs==2 && counters.releases==0);
            cancel=true;assert(!vision.run(argv[2],f,error) && error=="cancelled" && counters.runs==2);
            cancel=false;backend->fail=true;assert(!vision.run(argv[2],f,error) && error=="injected");
            backend->fail=false;assert(vision.run(argv[2],f,error));
        }
        assert(counters.releases==1);
        std::cout<<"one vision session retained, zero NNRT KV, cancellation and retry: PASS\n";
    }
    std::cout << "image spans, 3 DeepStack layers, multi-image M-RoPE, malformed input: PASS\n";
}
