#include "PortBinding.hpp"
#include <cassert>
#include <iostream>
using namespace KirinFullNpu;
int main(){
 Port h;h.name="hidden_states";h.dtype=DType::F32;h.shape={1,608,1024};
 Port d=h;d.name="deepstack_11";
 Port t=h;
 std::vector<size_t> m;std::string e;
 std::map<std::string,std::string> a{{"deepstack_11","hidden_states"}};
 assert(bindPorts({h,d},{t},a,m,e)&&m==std::vector<size_t>({0,0}));
 m.clear();t=d;assert(bindPorts({h,d},{t},a,m,e)&&m==std::vector<size_t>({0,0}));
 m.clear();Port x=h;assert(bindPorts({h,d},{x,t},a,m,e)&&m==std::vector<size_t>({0,1}));
 m.clear();assert(!bindPorts({h,d},{x},{},m,e));
 Port k=h;k.name="past_key_0";k.dtype=DType::I8;
 t=k;t.dtype=DType::F32;m.clear();assert(!bindPorts({k},{t},{},m,e));
 t=k;t.name="prefix/past_key_0:0";m.clear();assert(bindPorts({k},{t},{},m,e));
 t.name="past_key_10";m.clear();assert(!bindPorts({k},{t},{},m,e));
 std::cout<<"PASS: NNRT name, shape, dtype, merged-output aliases and ambiguity guards\n";
}
