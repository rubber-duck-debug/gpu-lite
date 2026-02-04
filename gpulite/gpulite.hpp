// gpu-lite - Combined Header
// A lightweight C++ library for dynamic CUDA runtime compilation and kernel caching
#ifndef GPULITE_HPP
#define GPULITE_HPP

// =============================================================================
// CUDA Types Wrapper - Minimal CUDA type definitions for build-time independence
// =============================================================================

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA Driver API types (from cuda.h)
typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef struct CUstream_st* CUstream;
typedef unsigned long long CUdeviceptr;

// CUDA Runtime API types (from cuda_runtime.h)
typedef int cudaError_t;
typedef void* cudaStream_t;

// NVRTC types (from nvrtc.h)
typedef int nvrtcResult;
typedef struct _nvrtcProgram* nvrtcProgram;

// dim3 structure for kernel launch parameters
struct dim3 {
    unsigned int x, y, z;

    dim3(unsigned int x = 1, unsigned int y = 1, unsigned int z = 1) : x(x), y(y), z(z) {}
};

// CUDA memory copy kinds
typedef enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
} cudaMemcpyKind;

//global definition for CUDA memory types
typedef enum cudaMemoryType {
      cudaMemoryTypeUnregistered = 0,
      cudaMemoryTypeHost = 1,
      cudaMemoryTypeDevice = 2,
      cudaMemoryTypeManaged = 3
} cudaMemoryType;


// CUDA pointer attributes structure
typedef struct cudaPointerAttributes {
    enum cudaMemoryType type;
    int device;
    void* devicePointer;
    void* hostPointer;
} cudaPointerAttributes;

// CUDA Driver API constants
enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_NOT_INITIALIZED = 3
};

// CUDA Runtime API constants
enum {
    cudaSuccess = 0
};

// CUDA Host Alloc flags
enum {
    cudaHostAllocDefault = 0x00,
    cudaHostAllocPortable = 0x01,
    cudaHostAllocMapped = 0x02,
    cudaHostAllocWriteCombined = 0x04
};

// NVRTC constants
enum {
    NVRTC_SUCCESS = 0
};

// CUDA device attributes
typedef enum CUdevice_attribute {
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 83,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
} CUdevice_attribute;

// CUDA function attributes
typedef enum CUfunction_attribute {
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
    CU_FUNC_ATTRIBUTE_NUM_REGS = 4
} CUfunction_attribute;

// CUDA pointer attributes for cuPointerGetAttribute
typedef enum CUpointer_attribute {
    CU_POINTER_ATTRIBUTE_CONTEXT = 1,
    CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2
} CUpointer_attribute;

// CUDA memory types
enum {
    CU_MEMORYTYPE_HOST = 0x01,
    CU_MEMORYTYPE_DEVICE = 0x02,
    CU_MEMORYTYPE_ARRAY = 0x03,
    CU_MEMORYTYPE_UNIFIED = 0x04
};

typedef enum CUjit_option_enum {
    CU_JIT_GENERATE_DEBUG_INFO = 11,
} CUjit_option;

#ifdef __cplusplus
}
#endif

// =============================================================================
// Dynamic CUDA - Dynamic loading of CUDA runtime libraries
// =============================================================================

#if defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#include <unistd.h>  // for getcwd
#elif defined(_WIN32)
#include <windows.h>
#include <direct.h>  // for _getcwd
#define getcwd _getcwd
#else
#error "Platform not supported"
#endif
#include <stdexcept>
#include <string>
#include <functional>
#include <unordered_map>
#include <any>
#include <sstream>

#define NVRTC_SAFE_CALL(x)                                                                         \
    do {                                                                                           \
        nvrtcResult result = x;                                                                    \
        if (result != NVRTC_SUCCESS) {                                                             \
            std::ostringstream errorMsg;                                                           \
            errorMsg << "\nerror: " #x " failed with error "                                       \
                     << NVRTC_INSTANCE.nvrtcGetErrorString(result) << '\n'                         \
                     << "File: " << __FILE__ << '\n'                                               \
                     << "Line: " << static_cast<int>(__LINE__) << '\n';                            \
            throw std::runtime_error(errorMsg.str());                                              \
        }                                                                                          \
    } while (0)

#define CUDADRIVER_SAFE_CALL(x)                                                                    \
    do {                                                                                           \
        CUresult result = x;                                                                       \
        if (result != CUDA_SUCCESS) {                                                              \
            const char* msg;                                                                       \
            CUDA_DRIVER_INSTANCE.cuGetErrorName(result, &msg);                                     \
            std::ostringstream errorMsg;                                                           \
            errorMsg << "\nerror: " #x " failed with error " << (msg ? msg : "Unknown error")      \
                     << '\n'                                                                       \
                     << "File: " << __FILE__ << '\n'                                               \
                     << "Line: " << static_cast<int>(__LINE__) << '\n';                            \
            throw std::runtime_error(errorMsg.str());                                              \
        }                                                                                          \
    } while (0)

#define CUDART_SAFE_CALL(call)                                                                     \
    do {                                                                                           \
        cudaError_t cudaStatus = (call);                                                           \
        if (cudaStatus != cudaSuccess) {                                                           \
            std::ostringstream errorMsg;                                                           \
            const char* error = CUDART_INSTANCE.cudaGetErrorString(cudaStatus);                    \
            errorMsg << "\nfailed with error " << (error ? error : "Unknown error") << '\n'        \
                     << "File: " << __FILE__ << '\n'                                               \
                     << "Line: " << static_cast<int>(__LINE__) << '\n';                            \
            throw std::runtime_error(errorMsg.str());                                              \
        }                                                                                          \
    } while (0)

// Define a template to dynamically load symbols
template <typename FuncType> FuncType load(void* handle, const char* functionName) {
#if defined(__linux__) || defined(__APPLE__)
    auto func = reinterpret_cast<FuncType>(dlsym(handle, functionName));
#elif defined(_WIN32)
    auto func = reinterpret_cast<FuncType>(GetProcAddress(static_cast<HMODULE>(handle), functionName));
#endif
    if (!func) {
        throw std::runtime_error(std::string("Failed to load function: ") + functionName);
    }
    return func;
}

/*
This class allows us to dynamically load the CUDA runtime and reference the functions contained
within the libcudart.so library (see CUDA Runtime API:
https://docs.nvidia.com/cuda/cuda-runtime-api/index.html).
*/
class CUDART {
  public:
    static CUDART& instance() {
        static CUDART instance;
        return instance;
    }

    bool loaded() { return cudartHandle != nullptr; }

    using cudaGetDeviceCount_t = cudaError_t (*)(int*);
    using cudaGetDevice_t = cudaError_t (*)(int*);
    using cudaSetDevice_t = cudaError_t (*)(int);
    using cudaMalloc_t = cudaError_t (*)(void**, size_t);
    using cudaMemcpy_t = cudaError_t (*)(void*, const void*, size_t, cudaMemcpyKind);
    using cudaMemset_t = cudaError_t (*)(void*, int, size_t);
    using cudaGetErrorName_t = const char* (*)(cudaError_t);
    using cudaGetErrorString_t = const char* (*)(cudaError_t);
    using cudaDeviceSynchronize_t = cudaError_t (*)(void);
    using cudaPointerGetAttributes_t = cudaError_t (*)(cudaPointerAttributes*, const void*);
    using cudaFree_t = cudaError_t (*)(void*);
    using cudaRuntimeGetVersion_t = cudaError_t (*)(int*);
    using cudaStreamCreate_t = cudaError_t (*)(cudaStream_t*);
    using cudaStreamDestroy_t = cudaError_t (*)(cudaStream_t);
    using cudaStreamSynchronize_t = cudaError_t (*)(cudaStream_t);
    using cudaHostAlloc_t = cudaError_t (*)(void**, size_t, unsigned int);
    using cudaFreeHost_t = cudaError_t (*)(void*);
    using cudaHostGetDevicePointer_t = cudaError_t (*)(void**, void*, unsigned int);
    using cudaMemcpyAsync_t = cudaError_t (*)(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);

    cudaGetDeviceCount_t cudaGetDeviceCount;
    cudaGetDevice_t cudaGetDevice;
    cudaSetDevice_t cudaSetDevice;
    cudaMalloc_t cudaMalloc;
    cudaMemset_t cudaMemset;
    cudaMemcpy_t cudaMemcpy;
    cudaGetErrorName_t cudaGetErrorName;
    cudaGetErrorString_t cudaGetErrorString;
    cudaDeviceSynchronize_t cudaDeviceSynchronize;
    cudaPointerGetAttributes_t cudaPointerGetAttributes;
    cudaFree_t cudaFree;
    cudaRuntimeGetVersion_t cudaRuntimeGetVersion;
    cudaStreamCreate_t cudaStreamCreate;
    cudaStreamDestroy_t cudaStreamDestroy;
    cudaStreamSynchronize_t cudaStreamSynchronize;
    cudaHostAlloc_t cudaHostAlloc;
    cudaFreeHost_t cudaFreeHost;
    cudaHostGetDevicePointer_t cudaHostGetDevicePointer;
    cudaMemcpyAsync_t cudaMemcpyAsync;

    CUDART() {
#if defined(__linux__) || defined(__APPLE__)
        cudartHandle = dlopen("libcudart.so", RTLD_NOW);
#elif defined(_WIN32)
        cudartHandle = LoadLibraryA("cudart64_12.dll");
        if (!cudartHandle) {
            cudartHandle = LoadLibraryA("cudart64_11.dll");
        }
        if (!cudartHandle) {
            cudartHandle = LoadLibraryA("cudart64_10.dll");
        }
#else
#error "Platform not supported"
#endif
        if (cudartHandle) {
            // load cudart function pointers using template
            cudaGetDeviceCount = load<cudaGetDeviceCount_t>(cudartHandle, "cudaGetDeviceCount");
            cudaGetDevice = load<cudaGetDevice_t>(cudartHandle, "cudaGetDevice");
            cudaSetDevice = load<cudaSetDevice_t>(cudartHandle, "cudaSetDevice");
            cudaMalloc = load<cudaMalloc_t>(cudartHandle, "cudaMalloc");
            cudaMemset = load<cudaMemset_t>(cudartHandle, "cudaMemset");
            cudaMemcpy = load<cudaMemcpy_t>(cudartHandle, "cudaMemcpy");
            cudaGetErrorName = load<cudaGetErrorName_t>(cudartHandle, "cudaGetErrorName");
            cudaGetErrorString = load<cudaGetErrorString_t>(cudartHandle, "cudaGetErrorString");
            cudaDeviceSynchronize =
                load<cudaDeviceSynchronize_t>(cudartHandle, "cudaDeviceSynchronize");
            cudaPointerGetAttributes =
                load<cudaPointerGetAttributes_t>(cudartHandle, "cudaPointerGetAttributes");
            cudaFree = load<cudaFree_t>(cudartHandle, "cudaFree");
            cudaRuntimeGetVersion =
                load<cudaRuntimeGetVersion_t>(cudartHandle, "cudaRuntimeGetVersion");
            cudaStreamCreate = load<cudaStreamCreate_t>(cudartHandle, "cudaStreamCreate");
            cudaStreamDestroy = load<cudaStreamDestroy_t>(cudartHandle, "cudaStreamDestroy");
            cudaStreamSynchronize = load<cudaStreamSynchronize_t>(cudartHandle, "cudaStreamSynchronize");
            cudaHostAlloc = load<cudaHostAlloc_t>(cudartHandle, "cudaHostAlloc");
            cudaFreeHost = load<cudaFreeHost_t>(cudartHandle, "cudaFreeHost");
            cudaHostGetDevicePointer = load<cudaHostGetDevicePointer_t>(cudartHandle, "cudaHostGetDevicePointer");
            cudaMemcpyAsync = load<cudaMemcpyAsync_t>(cudartHandle, "cudaMemcpyAsync");
        }
    }

    ~CUDART() {
#if defined(__linux__) || defined(__APPLE__)
        if (cudartHandle) {
            dlclose(cudartHandle);
        }
#elif defined(_WIN32)
        if (cudartHandle) {
            FreeLibrary(static_cast<HMODULE>(cudartHandle));
        }
#else
#error "Platform not supported"
#endif
    }

    // Prevent copying
    CUDART(const CUDART&) = delete;
    CUDART& operator=(const CUDART&) = delete;

    void* cudartHandle = nullptr;
};

/*
This class allows us to dynamically load the CUDA Driver and reference the functions contained
within the libcuda.so library (CUDA Driver API:
https://docs.nvidia.com/cuda/cuda-driver-api/index.html).
*/
class CUDADriver {

  public:
    static CUDADriver& instance() {
        static CUDADriver instance;
        return instance;
    }

    bool loaded() { return cudaHandle != nullptr; }

    using cuInit_t = CUresult (*)(unsigned int);
    using cuDeviceGetCount_t = CUresult (*)(int*);
    using cuDevicePrimaryCtxRetain_t = CUresult (*)(CUcontext*, CUdevice);
    using cuDevicePrimaryCtxRelease_t = CUresult (*)(CUdevice);
    using cuCtxCreate_t = CUresult (*)(CUcontext*, unsigned int, CUdevice);
    using cuCtxDestroy_t = CUresult (*)(CUcontext);
    using cuCtxGetCurrent_t = CUresult (*)(CUcontext*);
    using cuCtxSetCurrent_t = CUresult (*)(CUcontext);
    using cuModuleLoadDataEx_t = CUresult (*)(CUmodule*, const void*, unsigned int, CUjit_option*, void**);
    using cuModuleGetFunction_t = CUresult (*)(CUfunction*, CUmodule, const char*);
    using cuFuncSetAttribute_t = CUresult (*)(CUfunction, CUfunction_attribute, int);
    using cuFuncGetAttribute_t = CUresult (*)(int*, CUfunction_attribute, CUfunction);
    using cuCtxGetDevice_t = CUresult (*)(CUdevice*);
    using cuDeviceGetAttribute_t = CUresult (*)(int*, CUdevice_attribute, CUdevice);
    using cuDeviceGetName_t = CUresult (*)(char*, int, CUdevice);
    using cuDeviceTotalMem_t = CUresult (*)(size_t*, CUdevice);
    using cuLaunchKernel_t = CUresult (*)(
        CUfunction,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        size_t,
        CUstream,
        void**,
        void*
    );
    using cuStreamCreate_t = CUresult (*)(CUstream*, unsigned int);
    using cuStreamDestroy_t = CUresult (*)(CUstream);
    using cuCtxSynchronize_t = CUresult (*)(void);
    using cuGetErrorName_t = CUresult (*)(CUresult, const char**);
    using cuCtxPushCurrent_t = CUresult (*)(CUcontext);
    using cuPointerGetAttribute_t = CUresult (*)(void*, CUpointer_attribute, CUdeviceptr);

    cuInit_t cuInit;
    cuDeviceGetCount_t cuDeviceGetCount;
    cuCtxCreate_t cuCtxCreate;
    cuCtxDestroy_t cuCtxDestroy;
    cuDevicePrimaryCtxRetain_t cuDevicePrimaryCtxRetain;
    cuDevicePrimaryCtxRelease_t cuDevicePrimaryCtxRelease;
    cuCtxGetCurrent_t cuCtxGetCurrent;
    cuCtxSetCurrent_t cuCtxSetCurrent;
    cuModuleLoadDataEx_t cuModuleLoadDataEx;
    cuModuleGetFunction_t cuModuleGetFunction;
    cuFuncSetAttribute_t cuFuncSetAttribute;
    cuFuncGetAttribute_t cuFuncGetAttribute;
    cuCtxGetDevice_t cuCtxGetDevice;
    cuDeviceGetAttribute_t cuDeviceGetAttribute;
    cuDeviceGetName_t cuDeviceGetName;
    cuDeviceTotalMem_t cuDeviceTotalMem;
    cuLaunchKernel_t cuLaunchKernel;
    cuStreamCreate_t cuStreamCreate;
    cuStreamDestroy_t cuStreamDestroy;
    cuGetErrorName_t cuGetErrorName;
    cuCtxSynchronize_t cuCtxSynchronize;
    cuCtxPushCurrent_t cuCtxPushCurrent;
    cuPointerGetAttribute_t cuPointerGetAttribute;

    CUDADriver() {
#if defined(__linux__) || defined(__APPLE__)
        cudaHandle = dlopen("libcuda.so", RTLD_NOW);
#elif defined(_WIN32)
        cudaHandle = LoadLibraryA("nvcuda.dll");
#else
#error "Platform not supported"
#endif
        if (cudaHandle) {
            // Load CUDA driver function pointers using template
            cuInit = load<cuInit_t>(cudaHandle, "cuInit");
            cuDeviceGetCount = load<cuDeviceGetCount_t>(cudaHandle, "cuDeviceGetCount");
            cuCtxCreate = load<cuCtxCreate_t>(cudaHandle, "cuCtxCreate");
            cuCtxDestroy = load<cuCtxDestroy_t>(cudaHandle, "cuCtxDestroy");
            cuDevicePrimaryCtxRetain =
                load<cuDevicePrimaryCtxRetain_t>(cudaHandle, "cuDevicePrimaryCtxRetain");
            cuDevicePrimaryCtxRelease =
                load<cuDevicePrimaryCtxRelease_t>(cudaHandle, "cuDevicePrimaryCtxRelease");
            cuCtxGetCurrent = load<cuCtxGetCurrent_t>(cudaHandle, "cuCtxGetCurrent");
            cuCtxSetCurrent = load<cuCtxSetCurrent_t>(cudaHandle, "cuCtxSetCurrent");
            cuModuleLoadDataEx = load<cuModuleLoadDataEx_t>(cudaHandle, "cuModuleLoadDataEx");
            cuModuleGetFunction = load<cuModuleGetFunction_t>(cudaHandle, "cuModuleGetFunction");
            cuFuncSetAttribute = load<cuFuncSetAttribute_t>(cudaHandle, "cuFuncSetAttribute");
            cuFuncGetAttribute = load<cuFuncGetAttribute_t>(cudaHandle, "cuFuncGetAttribute");
            cuCtxGetDevice = load<cuCtxGetDevice_t>(cudaHandle, "cuCtxGetDevice");
            cuDeviceGetAttribute = load<cuDeviceGetAttribute_t>(cudaHandle, "cuDeviceGetAttribute");
            cuDeviceGetName = load<cuDeviceGetName_t>(cudaHandle, "cuDeviceGetName");
            cuDeviceTotalMem = load<cuDeviceTotalMem_t>(cudaHandle, "cuDeviceTotalMem");
            cuLaunchKernel = load<cuLaunchKernel_t>(cudaHandle, "cuLaunchKernel");
            cuStreamCreate = load<cuStreamCreate_t>(cudaHandle, "cuStreamCreate");
            cuStreamDestroy = load<cuStreamDestroy_t>(cudaHandle, "cuStreamDestroy");
            cuCtxSynchronize = load<cuCtxSynchronize_t>(cudaHandle, "cuCtxSynchronize");
            cuGetErrorName = load<cuGetErrorName_t>(cudaHandle, "cuGetErrorName");
            cuCtxPushCurrent = load<cuCtxPushCurrent_t>(cudaHandle, "cuCtxPushCurrent");
            cuPointerGetAttribute =
                load<cuPointerGetAttribute_t>(cudaHandle, "cuPointerGetAttribute");
        }
    }

    ~CUDADriver() {
#if defined(__linux__) || defined(__APPLE__)
        if (cudaHandle) {
            dlclose(cudaHandle);
        }
#elif defined(_WIN32)
        if (cudaHandle) {
            FreeLibrary(static_cast<HMODULE>(cudaHandle));
        }
#else
#error "Platform not supported"
#endif
    }

    // Prevent copying
    CUDADriver(const CUDADriver&) = delete;
    CUDADriver& operator=(const CUDADriver&) = delete;

    void* cudaHandle = nullptr;
};

/*
This class allows us to dynamically load NVRTC and reference the functions contained within the
libnvrtc.so library (see NVRTC API: https://docs.nvidia.com/cuda/nvrtc/index.html).
*/
class NVRTC {

  public:
    static NVRTC& instance() {
        static NVRTC instance;
        return instance;
    }

    bool loaded() { return nvrtcHandle != nullptr; }

    using nvrtcCreateProgram_t =
        nvrtcResult (*)(nvrtcProgram*, const char*, const char*, int, const char*[], const char*[]);
    using nvrtcCompileProgram_t = nvrtcResult (*)(nvrtcProgram, int, const char*[]);
    using nvrtcGetPTX_t = nvrtcResult (*)(nvrtcProgram, char*);
    using nvrtcGetPTXSize_t = nvrtcResult (*)(nvrtcProgram, size_t*);
    using nvrtcGetCUBIN_t = nvrtcResult (*)(nvrtcProgram, char*);
    using nvrtcGetCUBINSize_t = nvrtcResult (*)(nvrtcProgram, size_t*);
    using nvrtcGetProgramLog_t = nvrtcResult (*)(nvrtcProgram, char*);
    using nvrtcGetProgramLogSize_t = nvrtcResult (*)(nvrtcProgram, size_t*);
    using nvrtcAddNameExpression_t = nvrtcResult (*)(nvrtcProgram, const char* const);
    using nvrtcGetLoweredName_t = nvrtcResult (*)(nvrtcProgram, const char*, const char**);
    using nvrtcDestroyProgram_t = nvrtcResult (*)(nvrtcProgram*);
    using nvrtcGetErrorString_t = const char* (*)(nvrtcResult);

    nvrtcCreateProgram_t nvrtcCreateProgram;
    nvrtcCompileProgram_t nvrtcCompileProgram;
    nvrtcGetPTX_t nvrtcGetPTX;
    nvrtcGetPTXSize_t nvrtcGetPTXSize;
    nvrtcGetCUBIN_t nvrtcGetCUBIN;
    nvrtcGetCUBINSize_t nvrtcGetCUBINSize;
    nvrtcGetProgramLog_t nvrtcGetProgramLog;
    nvrtcGetProgramLogSize_t nvrtcGetProgramLogSize;
    nvrtcGetLoweredName_t nvrtcGetLoweredName;
    nvrtcAddNameExpression_t nvrtcAddNameExpression;
    nvrtcDestroyProgram_t nvrtcDestroyProgram;
    nvrtcGetErrorString_t nvrtcGetErrorString;

    NVRTC() {
#if defined(__linux__) || defined(__APPLE__)
        nvrtcHandle = dlopen("libnvrtc.so", RTLD_NOW);
#elif defined(_WIN32)
        nvrtcHandle = LoadLibraryA("nvrtc64_12.dll");
        if (!nvrtcHandle) {
            nvrtcHandle = LoadLibraryA("nvrtc64_11.dll");
        }
        if (!nvrtcHandle) {
            nvrtcHandle = LoadLibraryA("nvrtc64_10.dll");
        }
#else
#error "Platform not supported"
#endif

        if (nvrtcHandle) {
            // Load NVRTC function pointers using template
            nvrtcCreateProgram = load<nvrtcCreateProgram_t>(nvrtcHandle, "nvrtcCreateProgram");
            nvrtcCompileProgram = load<nvrtcCompileProgram_t>(nvrtcHandle, "nvrtcCompileProgram");
            nvrtcGetPTX = load<nvrtcGetPTX_t>(nvrtcHandle, "nvrtcGetPTX");
            nvrtcGetPTXSize = load<nvrtcGetPTXSize_t>(nvrtcHandle, "nvrtcGetPTXSize");
            nvrtcGetCUBIN = load<nvrtcGetCUBIN_t>(nvrtcHandle, "nvrtcGetCUBIN");
            nvrtcGetCUBINSize = load<nvrtcGetCUBINSize_t>(nvrtcHandle, "nvrtcGetCUBINSize");
            nvrtcGetProgramLog = load<nvrtcGetProgramLog_t>(nvrtcHandle, "nvrtcGetProgramLog");
            nvrtcGetProgramLogSize =
                load<nvrtcGetProgramLogSize_t>(nvrtcHandle, "nvrtcGetProgramLogSize");
            nvrtcGetLoweredName = load<nvrtcGetLoweredName_t>(nvrtcHandle, "nvrtcGetLoweredName");
            nvrtcAddNameExpression =
                load<nvrtcAddNameExpression_t>(nvrtcHandle, "nvrtcAddNameExpression");
            nvrtcDestroyProgram = load<nvrtcDestroyProgram_t>(nvrtcHandle, "nvrtcDestroyProgram");
            nvrtcGetErrorString = load<nvrtcGetErrorString_t>(nvrtcHandle, "nvrtcGetErrorString");
        }
    }

    ~NVRTC() {
#if defined(__linux__) || defined(__APPLE__)
        if (nvrtcHandle) {
            dlclose(nvrtcHandle);
        }
#elif defined(_WIN32)
        if (nvrtcHandle) {
            FreeLibrary(static_cast<HMODULE>(nvrtcHandle));
        }
#else
#error "Platform not supported"
#endif
    }

    // Prevent copying
    NVRTC(const NVRTC&) = delete;
    NVRTC& operator=(const NVRTC&) = delete;

    void* nvrtcHandle = nullptr;
};

#define CUDART_INSTANCE CUDART::instance()
#define CUDA_DRIVER_INSTANCE CUDADriver::instance()
#define NVRTC_INSTANCE NVRTC::instance()

// Convenience macros for cleaner API - hides the global instance from users
// Usage: GPULITE_CUDART_CALL(cudaMalloc(&ptr, size))
//        GPULITE_DRIVER_CALL(cuCtxGetCurrent(&ctx))
#define GPULITE_CUDART_CALL(func) CUDART_SAFE_CALL(CUDART_INSTANCE.func)
#define GPULITE_DRIVER_CALL(func) CUDADRIVER_SAFE_CALL(CUDA_DRIVER_INSTANCE.func)

// =============================================================================
// CUDA Kernel Cache Manager - Runtime compilation and caching system
// =============================================================================

#include <fstream>
#include <vector>
#include <unordered_map>
#include <memory>
#include <typeinfo>
#include <cxxabi.h>

// Helper function to demangle the type name if necessary
std::string demangleTypeName(const std::string& name) {
#if defined(__GNUC__) || defined(__clang__)
    int status = 0;
    std::unique_ptr<char, void (*)(void*)> demangled_name(
        abi::__cxa_demangle(name.c_str(), 0, 0, &status), std::free
    );
    return (status == 0) ? demangled_name.get() : name;
#else
    throw std::runtime_error("demangling not supported using this toolchain.");
#endif
}

// Base case: No template arguments, return function name without any type information
std::string getKernelName(const std::string& fn_name) { return fn_name; }

// Function to get type name of a single type
template <typename T> std::string typeName() { return demangleTypeName(typeid(T).name()); }

// Variadic template function to build type list
template <typename T, typename... Ts> void buildTemplateTypes(std::string& base) {
    base += typeName<T>(); // Add the first type
    // If there are more types, add a comma and recursively call for the remaining types
    if constexpr (sizeof...(Ts) > 0) {
        base += ", ";
        buildTemplateTypes<Ts...>(base); // Recursively call for the rest of the types
    }
}

// Helper function to start building the types
template <typename T, typename... Ts> std::string buildTemplateTypes() {
    std::string result;
    buildTemplateTypes<T, Ts...>(result); // Use recursive variadic template
    return result;
}

/*
Function to get the kernel name with the list of templated types if any:
*/
template <typename T, typename... Ts> std::string getKernelName(const std::string& fn_name) {
    std::string type_list = buildTemplateTypes<T, Ts...>(); // Build type list
    return fn_name + "<" + type_list + ">"; // Return function name with type list in angle brackets
}

// Function to load CUDA source code from a file
std::string load_cuda_source(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

/*
Container class for the cached kernels. Provides functionality for launching compiled kernels as
well as automatically resizing dynamic shared memory allocations, when needed. Kernels are compiled
on first launch.
*/
class CachedKernel {

  public:
    CachedKernel(
        std::string kernel_name,
        std::string kernel_code,
        std::string source_name,
        std::vector<std::string> options
    ) {
        this->kernel_name = kernel_name;
        this->kernel_code = kernel_code;
        this->source_name = source_name;
        this->options = options;
    }

    CachedKernel() = default;

    // Copy constructor
    CachedKernel(const CachedKernel&) = default;

    // Copy assignment operator
    CachedKernel& operator=(const CachedKernel&) = default;

    inline void setFuncAttribute(CUfunction_attribute attribute, int value) const {
        CUDADRIVER_SAFE_CALL(CUDA_DRIVER_INSTANCE.cuFuncSetAttribute(function, attribute, value));
    }

    int getFuncAttribute(CUfunction_attribute attribute) const {
        int value;
        CUDADRIVER_SAFE_CALL(CUDA_DRIVER_INSTANCE.cuFuncGetAttribute(&value, attribute, function));
        return value;
    }

    /*
    launches the kernel, and optionally synchronizes until control can be passed back to host.
    */
    void launch(
        dim3 grid,
        dim3 block,
        size_t shared_mem_size,
        void* cuda_stream,
        std::vector<void*> args,
        bool synchronize = true
    ) {

        if (!compiled) {
            this->compileKernel(args);
        }

        CUcontext currentContext = nullptr;
        // Get current context
        CUresult result = CUDA_DRIVER_INSTANCE.cuCtxGetCurrent(&currentContext);

        if (result != CUDA_SUCCESS || !currentContext) {
            throw std::runtime_error("CachedKernel::launch error getting current context.");
        }

        if (currentContext != context) {
            CUDADRIVER_SAFE_CALL(CUDA_DRIVER_INSTANCE.cuCtxSetCurrent(context));
        }

        this->checkAndAdjustSharedMem(shared_mem_size);

        CUstream cstream = reinterpret_cast<CUstream>(cuda_stream);

        CUDADRIVER_SAFE_CALL(CUDA_DRIVER_INSTANCE.cuLaunchKernel(
            function,
            grid.x,
            grid.y,
            grid.z,
            block.x,
            block.y,
            block.z,
            shared_mem_size,
            cstream,
            args.data(),
            0
        ));

        if (synchronize) {
            CUDADRIVER_SAFE_CALL(CUDA_DRIVER_INSTANCE.cuCtxSynchronize());
        }

        if (currentContext != context) {
            CUDADRIVER_SAFE_CALL(CUDA_DRIVER_INSTANCE.cuCtxSetCurrent(currentContext));
        }
    }

  private:
    /*
    The default shared memory space on most recent NVIDIA cards is defaulted
    49152 bytes. This method
    attempts to adjust the shared memory to fit the requested configuration if
    the kernel launch parameters exceeds the default 49152 bytes.
    */
    void checkAndAdjustSharedMem(int query_shared_mem_size) {
        if (current_smem_size == 0) {
            CUdevice cuDevice;
            CUDADRIVER_SAFE_CALL(CUDA_DRIVER_INSTANCE.cuCtxGetDevice(&cuDevice));

            CUDADRIVER_SAFE_CALL(CUDA_DRIVER_INSTANCE.cuDeviceGetAttribute(
                &max_smem_size_optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, cuDevice
            ));

            int reserved_smem_per_block = 0;

            CUDADRIVER_SAFE_CALL(CUDA_DRIVER_INSTANCE.cuDeviceGetAttribute(
                &reserved_smem_per_block, CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK, cuDevice
            ));

            int curr_max_smem_per_block = 0;

            CUDADRIVER_SAFE_CALL(CUDA_DRIVER_INSTANCE.cuDeviceGetAttribute(
                &curr_max_smem_per_block, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, cuDevice
            ));

            current_smem_size = (curr_max_smem_per_block - reserved_smem_per_block);
        }

        if (query_shared_mem_size > current_smem_size) {

            if (query_shared_mem_size > max_smem_size_optin) {
                throw std::runtime_error(
                    "CachedKernel::launch requested more smem than available on card."
                );
            } else {
                CUDADRIVER_SAFE_CALL(CUDA_DRIVER_INSTANCE.cuFuncSetAttribute(
                    function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, query_shared_mem_size
                ));
                current_smem_size = query_shared_mem_size;
            }
        }
    }

    /*
        Compiles the kernel "kernel_name" located in source file "kernel_code", which additional
        parameters "options" passed to NVRTC_INSTANCE. Will auto-detect the compute capability of
       the available card. args for the launch need to be queried as we need to grab the CUcontext
       in which these ptrs exist.
        */
    void compileKernel(std::vector<void*>& kernel_args) {

        this->initCudaDriver();

        CUcontext currentContext = nullptr;

        for (size_t ptr_id = 0; ptr_id < kernel_args.size(); ptr_id++) {
            unsigned int memtype = 0;
            CUdeviceptr device_ptr = *reinterpret_cast<CUdeviceptr*>(kernel_args[ptr_id]);

            CUresult res = CUDA_DRIVER_INSTANCE.cuPointerGetAttribute(
                &memtype, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, device_ptr
            );

            if (res == CUDA_SUCCESS && memtype == CU_MEMORYTYPE_DEVICE) {
                CUDADRIVER_SAFE_CALL(CUDA_DRIVER_INSTANCE.cuPointerGetAttribute(
                    &currentContext, CU_POINTER_ATTRIBUTE_CONTEXT, device_ptr
                ));

                if (currentContext) {
                    break;
                }
            }
        }

        CUcontext query = nullptr;
        CUDADRIVER_SAFE_CALL(CUDA_DRIVER_INSTANCE.cuCtxGetCurrent(&query));

        if (query != currentContext) {
            CUDADRIVER_SAFE_CALL(CUDA_DRIVER_INSTANCE.cuCtxSetCurrent(currentContext));
        }

        CUdevice cuDevice;
        CUDADRIVER_SAFE_CALL(CUDA_DRIVER_INSTANCE.cuCtxGetDevice(&cuDevice));

        // Check if debug option is enabled - need to know early for source file handling
        const bool enableDebug = std::any_of(
            this->options.cbegin(), this->options.cend(),
            [](const std::string& opt) {
                return opt == "-G" || opt == "--device-debug";
            }
        );

        // When debugging, write source to a real file so cuda-gdb can find it
        std::string effective_source_name = this->source_name;
        if (enableDebug) {
            // Create a debug source file in the current working directory
            // Use absolute path so cuda-gdb can reliably find it
            char cwd[4096];
            if (getcwd(cwd, sizeof(cwd)) != nullptr) {
                effective_source_name = std::string(cwd) + "/" + this->source_name;
            }
            // Write the kernel source code to the file
            std::ofstream debug_source_file(effective_source_name);
            if (debug_source_file.is_open()) {
                debug_source_file << this->kernel_code;
                debug_source_file.close();
            } else {
                throw std::runtime_error(
                    "Failed to write debug source file: " + effective_source_name
                );
            }
        }

        nvrtcProgram prog;

        NVRTC_SAFE_CALL(NVRTC_INSTANCE.nvrtcCreateProgram(
            &prog, this->kernel_code.c_str(), effective_source_name.c_str(), 0, nullptr, nullptr
        ));

        NVRTC_SAFE_CALL(NVRTC_INSTANCE.nvrtcAddNameExpression(prog, this->kernel_name.c_str()));

        std::vector<const char*> c_options;
        c_options.reserve(this->options.size());
        for (const auto& option : this->options) {
            c_options.push_back(option.c_str());
        }

        int major = 0;
        int minor = 0;
        CUDADRIVER_SAFE_CALL(CUDA_DRIVER_INSTANCE.cuDeviceGetAttribute(
            &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice
        ));
        CUDADRIVER_SAFE_CALL(CUDA_DRIVER_INSTANCE.cuDeviceGetAttribute(
            &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice
        ));
        int arch = major * 10 + minor;
        std::string smbuf = "--gpu-architecture=sm_" + std::to_string(arch);

        c_options.push_back(smbuf.c_str());

        nvrtcResult compileResult =
            NVRTC_INSTANCE.nvrtcCompileProgram(prog, c_options.size(), c_options.data());
        if (compileResult != NVRTC_SUCCESS) {
            size_t logSize;
            NVRTC_SAFE_CALL(NVRTC_INSTANCE.nvrtcGetProgramLogSize(prog, &logSize));
            std::string log(logSize, '\0');
            NVRTC_SAFE_CALL(NVRTC_INSTANCE.nvrtcGetProgramLog(prog, &log[0]));
            throw std::runtime_error(
                "KernelFactory::compileAndCacheKernel: Failed to compile CUDA program:\n" + log
            );
        }

        // fetch CUBIN
        size_t cubinSize = 0;
        NVRTC_SAFE_CALL(NVRTC_INSTANCE.nvrtcGetCUBINSize(prog, &cubinSize));
        std::vector<char> cubin(cubinSize);
        NVRTC_SAFE_CALL(NVRTC_INSTANCE.nvrtcGetCUBIN(prog, cubin.data()));

        // load the module from cubin
        CUmodule module = nullptr;
        CUresult cuResult;

        if (enableDebug) {
            // Load with JIT debug info
            CUjit_option opts[1];
            opts[0] = CU_JIT_GENERATE_DEBUG_INFO;
            void** vals = new void*[1];
            vals[0] = (void*)(size_t)1;
            cuResult = CUDA_DRIVER_INSTANCE.cuModuleLoadDataEx(
                &module, cubin.data(), 1, opts, vals
            );
            delete[] vals;
        } else {
            // Load without JIT options
            cuResult = CUDA_DRIVER_INSTANCE.cuModuleLoadDataEx(
                &module, cubin.data(), 0, 0, 0
            );
        }

        if (cuResult != CUDA_SUCCESS) {
            throw std::runtime_error(
                "KernelFactory::compileAndCacheKernel: Failed to load PTX code into CUDA "
                "module "
                "(error code: " +
                std::to_string(cuResult) + ")"
            );
        }

        const char* lowered_name;
        NVRTC_SAFE_CALL(
            NVRTC_INSTANCE.nvrtcGetLoweredName(prog, this->kernel_name.c_str(), &lowered_name)
        );
        CUfunction kernel;
        CUDADRIVER_SAFE_CALL(CUDA_DRIVER_INSTANCE.cuModuleGetFunction(&kernel, module, lowered_name));

        this->module = module;
        this->function = kernel;
        this->context = currentContext;
        this->compiled = true;

        NVRTC_SAFE_CALL(NVRTC_INSTANCE.nvrtcDestroyProgram(&prog));
    }

    void initCudaDriver() {

        int deviceCount = 0;
        // Check if CUDA has already been initialized
        CUresult res = CUDA_DRIVER_INSTANCE.cuDeviceGetCount(&deviceCount);
        if (res == CUDA_ERROR_NOT_INITIALIZED) {
            // CUDA hasn't been initialized, so we initialize it now
            res = CUDA_DRIVER_INSTANCE.cuInit(0);
            if (res != CUDA_SUCCESS) {
                throw std::runtime_error(
                    "KernelFactory::initCudaDriver: Failed to initialize CUDA CUDA_DRIVER_INSTANCE."
                );
                return;
            }
        }
    }

    int current_smem_size = 0;
    int max_smem_size_optin = 0;
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    CUcontext context = nullptr;
    bool compiled = false;

    std::string kernel_name;
    std::string kernel_code;
    std::string source_name;
    std::vector<std::string> options;
};

/*
Factory class to create and store compiled cuda kernels for caching as a simple name-based hashmap.
Allows both compiling from a source file, or for compiling from a variable containing CUDA code.
*/
class KernelFactory {

  public:
    static KernelFactory& instance() {
        static KernelFactory instance;
        return instance;
    }

    void cacheKernel(
        const std::string& kernel_name,
        const std::string& source_path,
        const std::string& source_name,
        const std::vector<std::string>& options
    ) {
        kernel_cache[kernel_name] =
            std::make_unique<CachedKernel>(kernel_name, source_path, source_name, options);
    }

    bool hasKernel(const std::string& kernel_name) const {
        return kernel_cache.find(kernel_name) != kernel_cache.end();
    }

    CachedKernel* getKernel(const std::string& kernel_name) const {
        auto it = kernel_cache.find(kernel_name);
        if (it != kernel_cache.end()) {
            return it->second.get();
        }
        throw std::runtime_error("Kernel not found in cache.");
    }

    /*
    Tries to retrieve the kernel "kernel_name". If not found, instantiate it and save to cache.
    */
    CachedKernel* createFromSource(
        const std::string& kernel_name,
        const std::string& source_path,
        const std::string& source_name,
        const std::vector<std::string>& options
    ) {
        if (!this->hasKernel(kernel_name)) {
            std::string kernel_code = load_cuda_source(source_path);
            this->cacheKernel(kernel_name, kernel_code, source_name, options);
        }
        return this->getKernel(kernel_name);
    }

    /*
    Tries to retrieve the kernel "kernel_name". If not found, instantiate it and save to cache.
    */
    CachedKernel* create(
        const std::string& kernel_name,
        const std::string& source_variable,
        const std::string& source_name,
        const std::vector<std::string>& options
    ) {
        if (!this->hasKernel(kernel_name)) {
            this->cacheKernel(kernel_name, source_variable, source_name, options);
        }

        return this->getKernel(kernel_name);
    }

  private:
    KernelFactory() {}
    std::unordered_map<std::string, std::unique_ptr<CachedKernel>> kernel_cache;

    KernelFactory(const KernelFactory&) = delete;
    KernelFactory& operator=(const KernelFactory&) = delete;
};

#endif // GPULITE_HPP
