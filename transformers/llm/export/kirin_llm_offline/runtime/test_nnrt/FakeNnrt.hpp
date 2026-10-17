#pragma once
// Host-only call substitutes. These are deliberately not vendor SDK headers
// and test wiring/error propagation, never device compatibility or accuracy.
#include <cstddef>
#include <cstdint>

enum OH_NN_ReturnCode { OH_NN_SUCCESS, OH_NN_FAILED, OH_NN_INVALID_PARAMETER,
                        OH_NN_MEMORY_ERROR, OH_NN_OPERATION_FORBIDDEN };
enum OH_NN_DataType { OH_NN_UNKNOWN, OH_NN_FLOAT32, OH_NN_FLOAT16, OH_NN_INT8, OH_NN_INT32 };
enum OH_NN_Format { OH_NN_TEST_FORMAT, OH_NN_TEST_FORMAT_ALTERNATIVE };
enum HiAI_Compatibility { HIAI_COMPATIBILITY_COMPATIBLE, HIAI_COMPATIBILITY_INCOMPATIBLE };
enum HiAI_ExecuteDevice { HIAI_EXECUTE_DEVICE_NPU };
enum HiAI_FallbackMode { HIAI_FALLBACK_DISABLED };
enum HiAI_BandMode { HIAI_BANDMODE_NORMAL };
struct OH_NNCompilation;
struct OH_NNExecutor;
struct NN_Tensor;
struct NN_TensorDesc;
const char* HMS_HiAI_GetVersion();
HiAI_Compatibility HMS_HiAICompatibility_CheckFromFile(const char*);
OH_NN_ReturnCode OH_NNDevice_GetAllDevicesID(const size_t**, uint32_t*);
OH_NN_ReturnCode OH_NNDevice_GetName(size_t, const char**);
OH_NNCompilation* OH_NNCompilation_ConstructWithOfflineModelFile(const char*);
OH_NN_ReturnCode OH_NNCompilation_SetDevice(OH_NNCompilation*, size_t);
OH_NN_ReturnCode HMS_HiAIOptions_SetModelDeviceOrder(OH_NNCompilation*, HiAI_ExecuteDevice*, size_t);
OH_NN_ReturnCode HMS_HiAIOptions_SetFallbackMode(OH_NNCompilation*, HiAI_FallbackMode);
OH_NN_ReturnCode HMS_HiAIOptions_SetBandMode(OH_NNCompilation*, HiAI_BandMode);
OH_NN_ReturnCode OH_NNCompilation_Build(OH_NNCompilation*);
void OH_NNCompilation_Destroy(OH_NNCompilation**);
OH_NNExecutor* OH_NNExecutor_Construct(OH_NNCompilation*);
void OH_NNExecutor_Destroy(OH_NNExecutor**);
OH_NN_ReturnCode OH_NNExecutor_GetInputCount(const OH_NNExecutor*, size_t*);
OH_NN_ReturnCode OH_NNExecutor_GetOutputCount(const OH_NNExecutor*, size_t*);
NN_TensorDesc* OH_NNExecutor_CreateInputTensorDesc(const OH_NNExecutor*, size_t);
NN_TensorDesc* OH_NNExecutor_CreateOutputTensorDesc(const OH_NNExecutor*, size_t);
OH_NN_ReturnCode OH_NNTensorDesc_GetName(const NN_TensorDesc*, const char**);
OH_NN_ReturnCode OH_NNTensorDesc_GetDataType(const NN_TensorDesc*, OH_NN_DataType*);
OH_NN_ReturnCode OH_NNTensorDesc_GetFormat(const NN_TensorDesc*, OH_NN_Format*);
OH_NN_ReturnCode OH_NNTensorDesc_GetShape(const NN_TensorDesc*, int32_t**, size_t*);
OH_NN_ReturnCode OH_NNTensorDesc_GetByteSize(const NN_TensorDesc*, size_t*);
void OH_NNTensorDesc_Destroy(NN_TensorDesc**);
NN_Tensor* OH_NNTensor_Create(size_t, const NN_TensorDesc*);
void OH_NNTensor_Destroy(NN_Tensor**);
void* OH_NNTensor_GetDataBuffer(NN_Tensor*);
OH_NN_ReturnCode OH_NNTensor_GetSize(NN_Tensor*, size_t*);
OH_NN_ReturnCode OH_NNExecutor_RunSync(OH_NNExecutor*, NN_Tensor*[], size_t, NN_Tensor*[], size_t);
