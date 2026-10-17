// Actual RTLD_LOCAL test DSO. It is deliberately not linked into the test
// executable: InitWeights must be found through its explicit dlopen handle.
#include "FakeNnrt.hpp"

extern "C" int KirinTestFoundationMarker() { return 1; }

#ifndef KIRIN_TEST_MISSING_INIT_WEIGHTS
extern "C" OH_NN_ReturnCode KirinTest_InitWeights(OH_NNExecutor*, const char*);
extern "C" OH_NN_ReturnCode HMS_HiAIExecutor_InitWeights(OH_NNExecutor* executor, const char* directory) {
    return KirinTest_InitWeights(executor, directory);
}
#endif
