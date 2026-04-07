// gpu-lite - Combined Header
// A lightweight C++ library for dynamic CUDA runtime compilation and kernel caching
#ifndef GPULITE_HPP
#define GPULITE_HPP

#include <cstddef>

#include <list>
#include <mutex>
#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <unordered_map>


#if defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#include <unistd.h>  // for getcwd
#elif defined(_WIN32)
#include <windows.h>
#include <libloaderapi.h>

#include <direct.h>  // for _getcwd
#define getcwd _getcwd

#include <filesystem>
#else
#error "Platform not supported"
#endif

#if defined(_MSC_VER)
  // MSVC historically reports __cplusplus wrong unless /Zc:__cplusplus is enabled,
  // so prefer _MSVC_LANG there.
  #if !defined(_MSVC_LANG) || _MSVC_LANG < 201703L
    #error "This project requires C++17 or newer (/std:c++17)."
  #endif
#else
  #if __cplusplus < 201703L
    #error "This project requires C++17 or newer (-std=c++17)."
  #endif
#endif

// =============================================================================
// CUDA Types Wrapper - Minimal CUDA type definitions for build-time independence
// =============================================================================

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
#define CUDA_SUCCESS 0
#define CUDA_ERROR_NOT_INITIALIZED 3

// CUDA Runtime API constants
#define cudaSuccess 0

// CUDA Host Alloc flags
enum {
    cudaHostAllocDefault = 0x00,
    cudaHostAllocPortable = 0x01,
    cudaHostAllocMapped = 0x02,
    cudaHostAllocWriteCombined = 0x04
};

// NVRTC constants
#define NVRTC_SUCCESS 0

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

namespace gpulite {

namespace details {

// Define a template to dynamically load symbols
template <typename FuncType> FuncType loadSymbol(void* handle, const char* functionName) {
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

#ifdef _WIN32

inline std::wstring GetEnvVar(const wchar_t* name) {
    DWORD n = GetEnvironmentVariableW(name, nullptr, 0);
    if (n == 0) return L"";
    std::wstring val(n, L'\0');
    GetEnvironmentVariableW(name, val.data(), n);
    if (!val.empty() && val.back() == L'\0') val.pop_back();
    return val;
}

// Parse versions from filenames like cudart64_90.dll, cudart64_12.dll, cudart64_120.dll
inline int ParseCudartVersionScore(std::wstring prefix, const std::wstring& filename) {
    // Return a score; higher = preferred. Unknown parse => 0.

    prefix = prefix + L"_";
    // We try to parse digits after `prefix` and before ".dll".
    if (filename.rfind(prefix, 0) != 0) {
        return 0;
    }

    size_t start = prefix.size();
    size_t end = filename.find(L".dll");
    if (end == std::wstring::npos || end <= start) {
        return 0;
    }

    std::wstring num = filename.substr(start, end - start);
    if (num.empty()) {
        return 0;
    }
    for (wchar_t c : num) {
        if (c < L'0' || c > L'9') {
            return 0;
        }
    }

    // e.g. "90" -> 90, "12" -> 12, "120" -> 120
    return std::stoi(num);
}


inline std::vector<std::filesystem::path> CandidateCudaDirs() {
    std::vector<std::filesystem::path> dirs;

    // 1) CUDA_PATH\bin and CUDA_PATH\bin\x64
    std::wstring cudaPath = GetEnvVar(L"CUDA_PATH");
    if (!cudaPath.empty()) {
        dirs.push_back(std::filesystem::path(cudaPath) / L"bin");
        dirs.push_back(std::filesystem::path(cudaPath) / L"bin" / L"x64");
    }

    // 2) Search in PATH
    std::wstring pathEnv = GetEnvVar(L"PATH");
    if (!pathEnv.empty()) {
        size_t start = 0;
        size_t end = pathEnv.find(L';');
        while (end != std::wstring::npos) {
            std::wstring token = pathEnv.substr(start, end - start);
            if (!token.empty()) {
                dirs.push_back(std::filesystem::path(token));
            }
            start = end + 1;
            end = pathEnv.find(L';', start);
        }
        std::wstring token = pathEnv.substr(start);
        if (!token.empty()) {
            dirs.push_back(std::filesystem::path(token));
        }
    }

    // 3) Default toolkit install root (scan versions)
    //    C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin
    std::filesystem::path root = L"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA";

    std::error_code ec;
    std::filesystem::directory_options options = std::filesystem::directory_options::skip_permission_denied;
    for (auto& entry: std::filesystem::directory_iterator(root, options, ec)) {
        if (ec) {
            break;
        }

        // folders like v12.4, v13.0, etc.
        if (!entry.is_directory(ec) || ec) {
            continue;
        }

        dirs.push_back(entry.path() / L"bin");
        dirs.push_back(entry.path() / L"bin" / L"x64");
    }

    // De-dup + keep only existing dirs
    std::sort(dirs.begin(), dirs.end());
    dirs.erase(std::unique(dirs.begin(), dirs.end()), dirs.end());
    dirs.erase(
        std::remove_if(dirs.begin(), dirs.end(), [](const std::filesystem::path& p) {
            assert(!p.empty());

            std::error_code ec;
            if (!std::filesystem::exists(p, ec) || ec) {
                return true;
            }

            if (!std::filesystem::is_directory(p, ec) || ec) {
                return true;
            }

            return false;

        }),
        dirs.end()
    );
    return dirs;
}

inline std::optional<std::filesystem::path> FindBestCudaDll(const std::wstring& prefix) {
    auto dirs = CandidateCudaDirs();

    struct Match {
        std::filesystem::path path;
        int score;
    };
    std::vector<Match> matches;

    for (const auto& d: dirs) {
        std::error_code ec;
        std::filesystem::directory_options options = std::filesystem::directory_options::skip_permission_denied;
        for (auto& e: std::filesystem::directory_iterator(d, options, ec)) {
            if (ec) {
                break;
            }

            if (!e.is_regular_file(ec) || ec) {
                continue;
            }

            auto name = e.path().filename().wstring();
            // Must look like <prefix>*.dll
            if (name.size() < prefix.size() + 4) {
                continue;
            }

            if (name.rfind(prefix, 0) != 0) {
                continue;
            }

            if (e.path().extension().wstring() != L".dll") {
                continue;
            }

            int score = ParseCudartVersionScore(prefix, name);
            // Prefer versioned DLLs; still accept plain "<prefix>.dll" with score 1
            if (name == prefix + L".dll") {
                score = std::max(score, 1);
            }

            matches.push_back({ e.path(), score });
        }
    }

    if (matches.empty()) {
        return std::nullopt;
    }

    // Prefer highest score; if tie, prefer shortest path (arbitrary stable tie-break)
    std::sort(matches.begin(), matches.end(), [](const Match& a, const Match& b) {
        if (a.score != b.score) return a.score > b.score;
        return a.path.wstring().size() < b.path.wstring().size();
    });

    return matches.front().path;
}

#endif

} // namespace details

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

    static bool loaded() { return instance().cudartHandle != nullptr; }

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
        static const char* CANDIDATES[] = {
            "libcudart.so",
            "libcudart.so.11",
            "libcudart.so.12",
            "libcudart.so.13",
            "libcudart.so.14",
            "libcudart.so.15",
        };
        for (auto* candidate: CANDIDATES) {
            cudartHandle = dlopen(candidate, RTLD_NOW);
            if (cudartHandle) {
                break;
            }
        }
#elif defined(_WIN32)
        auto dllPathOpt = details::FindBestCudaDll(L"cudart64");
        if (dllPathOpt) {
            auto dllPath = *dllPathOpt;
            auto dir = dllPath.parent_path();
            // add the directory containing the DLL to the search path
            SetDllDirectoryW(dir.c_str());

            cudartHandle = LoadLibraryExW(
                dllPath.c_str(),
                nullptr,
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                LOAD_LIBRARY_SEARCH_DEFAULT_DIRS |
                LOAD_LIBRARY_SEARCH_USER_DIRS
            );
        }
#else
#error "Platform not supported"
#endif
        if (cudartHandle) {
            // load cudart function pointers using template
            cudaGetDeviceCount = details::loadSymbol<cudaGetDeviceCount_t>(cudartHandle, "cudaGetDeviceCount");
            cudaGetDevice = details::loadSymbol<cudaGetDevice_t>(cudartHandle, "cudaGetDevice");
            cudaSetDevice = details::loadSymbol<cudaSetDevice_t>(cudartHandle, "cudaSetDevice");
            cudaMalloc = details::loadSymbol<cudaMalloc_t>(cudartHandle, "cudaMalloc");
            cudaMemset = details::loadSymbol<cudaMemset_t>(cudartHandle, "cudaMemset");
            cudaMemcpy = details::loadSymbol<cudaMemcpy_t>(cudartHandle, "cudaMemcpy");
            cudaGetErrorName = details::loadSymbol<cudaGetErrorName_t>(cudartHandle, "cudaGetErrorName");
            cudaGetErrorString = details::loadSymbol<cudaGetErrorString_t>(cudartHandle, "cudaGetErrorString");
            cudaDeviceSynchronize = details::loadSymbol<cudaDeviceSynchronize_t>(cudartHandle, "cudaDeviceSynchronize");
            cudaPointerGetAttributes = details::loadSymbol<cudaPointerGetAttributes_t>(cudartHandle, "cudaPointerGetAttributes");
            cudaFree = details::loadSymbol<cudaFree_t>(cudartHandle, "cudaFree");
            cudaRuntimeGetVersion = details::loadSymbol<cudaRuntimeGetVersion_t>(cudartHandle, "cudaRuntimeGetVersion");
            cudaStreamCreate = details::loadSymbol<cudaStreamCreate_t>(cudartHandle, "cudaStreamCreate");
            cudaStreamDestroy = details::loadSymbol<cudaStreamDestroy_t>(cudartHandle, "cudaStreamDestroy");
            cudaStreamSynchronize = details::loadSymbol<cudaStreamSynchronize_t>(cudartHandle, "cudaStreamSynchronize");
            cudaHostAlloc = details::loadSymbol<cudaHostAlloc_t>(cudartHandle, "cudaHostAlloc");
            cudaFreeHost = details::loadSymbol<cudaFreeHost_t>(cudartHandle, "cudaFreeHost");
            cudaHostGetDevicePointer = details::loadSymbol<cudaHostGetDevicePointer_t>(cudartHandle, "cudaHostGetDevicePointer");
            cudaMemcpyAsync = details::loadSymbol<cudaMemcpyAsync_t>(cudartHandle, "cudaMemcpyAsync");
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

    static bool loaded() { return instance().cudaHandle != nullptr; }

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
            cuInit = details::loadSymbol<cuInit_t>(cudaHandle, "cuInit");
            cuDeviceGetCount = details::loadSymbol<cuDeviceGetCount_t>(cudaHandle, "cuDeviceGetCount");
            cuCtxCreate = details::loadSymbol<cuCtxCreate_t>(cudaHandle, "cuCtxCreate");
            cuCtxDestroy = details::loadSymbol<cuCtxDestroy_t>(cudaHandle, "cuCtxDestroy");
            cuDevicePrimaryCtxRetain = details::loadSymbol<cuDevicePrimaryCtxRetain_t>(cudaHandle, "cuDevicePrimaryCtxRetain");
            cuDevicePrimaryCtxRelease = details::loadSymbol<cuDevicePrimaryCtxRelease_t>(cudaHandle, "cuDevicePrimaryCtxRelease");
            cuCtxGetCurrent = details::loadSymbol<cuCtxGetCurrent_t>(cudaHandle, "cuCtxGetCurrent");
            cuCtxSetCurrent = details::loadSymbol<cuCtxSetCurrent_t>(cudaHandle, "cuCtxSetCurrent");
            cuModuleLoadDataEx = details::loadSymbol<cuModuleLoadDataEx_t>(cudaHandle, "cuModuleLoadDataEx");
            cuModuleGetFunction = details::loadSymbol<cuModuleGetFunction_t>(cudaHandle, "cuModuleGetFunction");
            cuFuncSetAttribute = details::loadSymbol<cuFuncSetAttribute_t>(cudaHandle, "cuFuncSetAttribute");
            cuFuncGetAttribute = details::loadSymbol<cuFuncGetAttribute_t>(cudaHandle, "cuFuncGetAttribute");
            cuCtxGetDevice = details::loadSymbol<cuCtxGetDevice_t>(cudaHandle, "cuCtxGetDevice");
            cuDeviceGetAttribute = details::loadSymbol<cuDeviceGetAttribute_t>(cudaHandle, "cuDeviceGetAttribute");
            cuDeviceGetName = details::loadSymbol<cuDeviceGetName_t>(cudaHandle, "cuDeviceGetName");
            cuDeviceTotalMem = details::loadSymbol<cuDeviceTotalMem_t>(cudaHandle, "cuDeviceTotalMem");
            cuLaunchKernel = details::loadSymbol<cuLaunchKernel_t>(cudaHandle, "cuLaunchKernel");
            cuStreamCreate = details::loadSymbol<cuStreamCreate_t>(cudaHandle, "cuStreamCreate");
            cuStreamDestroy = details::loadSymbol<cuStreamDestroy_t>(cudaHandle, "cuStreamDestroy");
            cuCtxSynchronize = details::loadSymbol<cuCtxSynchronize_t>(cudaHandle, "cuCtxSynchronize");
            cuGetErrorName = details::loadSymbol<cuGetErrorName_t>(cudaHandle, "cuGetErrorName");
            cuCtxPushCurrent = details::loadSymbol<cuCtxPushCurrent_t>(cudaHandle, "cuCtxPushCurrent");
            cuPointerGetAttribute = details::loadSymbol<cuPointerGetAttribute_t>(cudaHandle, "cuPointerGetAttribute");
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

    static bool loaded() { return instance().nvrtcHandle != nullptr; }

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
        static const char* CANDIDATES[] = {
            "libnvrtc.so",
            "libnvrtc.so.11",
            "libnvrtc.so.12",
            "libnvrtc.so.13",
            "libnvrtc.so.14",
            "libnvrtc.so.15",
        };
        for (auto* candidate: CANDIDATES) {
            nvrtcHandle = dlopen(candidate, RTLD_NOW);
            if (nvrtcHandle != nullptr) {
                break;
            }
        }

#elif defined(_WIN32)
        auto dllPathOpt = details::FindBestCudaDll(L"nvrtc64");
        if (dllPathOpt) {
            auto dllPath = *dllPathOpt;
            // add the directory containing the DLL to the search path
            auto dir = dllPath.parent_path();
            SetDllDirectoryW(dir.c_str());

            nvrtcHandle = LoadLibraryExW(
                dllPath.c_str(),
                nullptr,
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                LOAD_LIBRARY_SEARCH_DEFAULT_DIRS |
                LOAD_LIBRARY_SEARCH_USER_DIRS
            );
        }
#else
#error "Platform not supported"
#endif

        if (nvrtcHandle) {
            // Load NVRTC function pointers using template
            nvrtcCreateProgram = details::loadSymbol<nvrtcCreateProgram_t>(nvrtcHandle, "nvrtcCreateProgram");
            nvrtcCompileProgram = details::loadSymbol<nvrtcCompileProgram_t>(nvrtcHandle, "nvrtcCompileProgram");
            nvrtcGetPTX = details::loadSymbol<nvrtcGetPTX_t>(nvrtcHandle, "nvrtcGetPTX");
            nvrtcGetPTXSize = details::loadSymbol<nvrtcGetPTXSize_t>(nvrtcHandle, "nvrtcGetPTXSize");
            nvrtcGetCUBIN = details::loadSymbol<nvrtcGetCUBIN_t>(nvrtcHandle, "nvrtcGetCUBIN");
            nvrtcGetCUBINSize = details::loadSymbol<nvrtcGetCUBINSize_t>(nvrtcHandle, "nvrtcGetCUBINSize");
            nvrtcGetProgramLog = details::loadSymbol<nvrtcGetProgramLog_t>(nvrtcHandle, "nvrtcGetProgramLog");
            nvrtcGetProgramLogSize = details::loadSymbol<nvrtcGetProgramLogSize_t>(nvrtcHandle, "nvrtcGetProgramLogSize");
            nvrtcGetLoweredName = details::loadSymbol<nvrtcGetLoweredName_t>(nvrtcHandle, "nvrtcGetLoweredName");
            nvrtcAddNameExpression = details::loadSymbol<nvrtcAddNameExpression_t>(nvrtcHandle, "nvrtcAddNameExpression");
            nvrtcDestroyProgram = details::loadSymbol<nvrtcDestroyProgram_t>(nvrtcHandle, "nvrtcDestroyProgram");
            nvrtcGetErrorString = details::loadSymbol<nvrtcGetErrorString_t>(nvrtcHandle, "nvrtcGetErrorString");
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

namespace details {

template <typename Res, typename ErrFuncType>
auto checkCall(Res status, Res sucess, ErrFuncType errFunc, const char* file, int line, const char* call) {
    if (status != sucess) {
        auto errorString = errFunc(status);
        auto functionCall = std::string(call);
        throw std::runtime_error(
            functionCall + ": failed with error " + errorString +
            " (error code " + std::to_string(status) + ") at " +
            file + ':' + std::to_string(line)
        );
    }
}

inline const char* cudaDriverErrorString(cudaError_t error) {
    if (CUDADriver::instance().loaded()) {
        const char* errorStr = nullptr;
        CUDADriver::instance().cuGetErrorName(error, &errorStr);
        return errorStr ? errorStr : "Unknown CUDA driver error";
    }
    return "CUDA driver library not loaded";
}

inline const char* cudartErrorString(CUresult error) {
    if (CUDART::instance().loaded()) {
        const char* errorStr = CUDART::instance().cudaGetErrorString(error);
        return errorStr ? errorStr : "Unknown CUDA runtime error";
    }
    return "CUDA runtime library not loaded";
}

inline const char* nvrtcErrorString(nvrtcResult error) {
    if (NVRTC::instance().loaded()) {
        const char* errorStr = NVRTC::instance().nvrtcGetErrorString(error);
        return errorStr ? errorStr : "Unknown NVRTC error";
    }
    return "NVRTC library not loaded";
}

} // namespace details


#define GPULITE_CUDA_DRIVER_CALL(func) \
    gpulite::details::checkCall(gpulite::CUDADriver::instance().func, CUDA_SUCCESS, gpulite::details::cudaDriverErrorString, __FILE__, __LINE__, #func)

#define GPULITE_CUDART_CALL(func) \
    gpulite::details::checkCall(gpulite::CUDART::instance().func, cudaSuccess, gpulite::details::cudartErrorString, __FILE__, __LINE__, #func)

#define GPULITE_NVRTC_CALL(func) \
    gpulite::details::checkCall(gpulite::NVRTC::instance().func, NVRTC_SUCCESS, gpulite::details::nvrtcErrorString, __FILE__, __LINE__, #func)


// =============================================================================
// CUDA Kernel Cache Manager - Runtime compilation and caching system
// =============================================================================

/// Container class for the cached kernels. Provides functionality for launching
/// compiled kernels as well as automatically resizing dynamic shared memory
/// allocations, when needed. Kernels are compiled on first launch.
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
        GPULITE_CUDA_DRIVER_CALL(cuFuncSetAttribute(function, attribute, value));
    }

    int getFuncAttribute(CUfunction_attribute attribute) const {
        int value;
        GPULITE_CUDA_DRIVER_CALL(cuFuncGetAttribute(&value, attribute, function));
        return value;
    }

    /// Launch the kernel, and optionally synchronizes until control can be
    /// passed back to host.
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
        CUresult result = CUDADriver::instance().cuCtxGetCurrent(&currentContext);

        if (result != CUDA_SUCCESS || !currentContext) {
            throw std::runtime_error("CachedKernel::launch error getting current context.");
        }

        if (currentContext != context) {
            GPULITE_CUDA_DRIVER_CALL(cuCtxSetCurrent(context));
        }

        this->checkAndAdjustSharedMem(shared_mem_size);

        CUstream cstream = reinterpret_cast<CUstream>(cuda_stream);

        GPULITE_CUDA_DRIVER_CALL(cuLaunchKernel(
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
            GPULITE_CUDA_DRIVER_CALL(cuCtxSynchronize());
        }

        if (currentContext != context) {
            GPULITE_CUDA_DRIVER_CALL(cuCtxSetCurrent(currentContext));
        }
    }

  private:
    /// The default shared memory space on most recent NVIDIA cards is 49152
    /// bytes. This method attempts to adjust the shared memory to fit the
    /// requested configuration if the kernel launch parameters exceeds the
    /// default 49152 bytes.
    void checkAndAdjustSharedMem(int query_shared_mem_size) {
        if (current_smem_size == 0) {
            CUdevice cuDevice;
            GPULITE_CUDA_DRIVER_CALL(cuCtxGetDevice(&cuDevice));

            GPULITE_CUDA_DRIVER_CALL(cuDeviceGetAttribute(
                &max_smem_size_optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, cuDevice
            ));

            int reserved_smem_per_block = 0;

            GPULITE_CUDA_DRIVER_CALL(cuDeviceGetAttribute(
                &reserved_smem_per_block, CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK, cuDevice
            ));

            int curr_max_smem_per_block = 0;

            GPULITE_CUDA_DRIVER_CALL(cuDeviceGetAttribute(
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
                GPULITE_CUDA_DRIVER_CALL(cuFuncSetAttribute(
                    function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, query_shared_mem_size
                ));
                current_smem_size = query_shared_mem_size;
            }
        }
    }

    /// Compiles the kernel "kernel_name" located in source file "kernel_code",
    /// which additional parameters "options" passed to the NVRTC instance. Will
    /// auto-detect the compute capability of the available card. args for the
    /// launch need to be queried as we need to grab the CUcontext in which
    /// these ptrs exist.
    void compileKernel(std::vector<void*>& kernel_args) {
        this->initCudaDriver();

        CUcontext currentContext = nullptr;

        for (size_t ptr_id = 0; ptr_id < kernel_args.size(); ptr_id++) {
            unsigned int memtype = 0;
            CUdeviceptr device_ptr = *reinterpret_cast<CUdeviceptr*>(kernel_args[ptr_id]);

            CUresult res = CUDADriver::instance().cuPointerGetAttribute(
                &memtype, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, device_ptr
            );

            if (res == CUDA_SUCCESS && memtype == CU_MEMORYTYPE_DEVICE) {
                GPULITE_CUDA_DRIVER_CALL(cuPointerGetAttribute(
                    &currentContext, CU_POINTER_ATTRIBUTE_CONTEXT, device_ptr
                ));

                if (currentContext) {
                    break;
                }
            }
        }

        CUcontext query = nullptr;
        GPULITE_CUDA_DRIVER_CALL(cuCtxGetCurrent(&query));

        if (query != currentContext) {
            GPULITE_CUDA_DRIVER_CALL(cuCtxSetCurrent(currentContext));
        }

        CUdevice cuDevice;
        GPULITE_CUDA_DRIVER_CALL(cuCtxGetDevice(&cuDevice));

        // Check if debug option is enabled
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

        GPULITE_NVRTC_CALL(nvrtcCreateProgram(
            &prog, this->kernel_code.c_str(), effective_source_name.c_str(), 0, nullptr, nullptr
        ));

        GPULITE_NVRTC_CALL(nvrtcAddNameExpression(prog, this->kernel_name.c_str()));

        std::vector<const char*> c_options;
        c_options.reserve(this->options.size());
        for (const auto& option : this->options) {
            c_options.push_back(option.c_str());
        }

        int major = 0;
        int minor = 0;
        GPULITE_CUDA_DRIVER_CALL(cuDeviceGetAttribute(
            &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice
        ));
        GPULITE_CUDA_DRIVER_CALL(cuDeviceGetAttribute(
            &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice
        ));
        int arch = major * 10 + minor;
        std::string smbuf = "--gpu-architecture=sm_" + std::to_string(arch);

        c_options.push_back(smbuf.c_str());

        nvrtcResult compileResult = NVRTC::instance().nvrtcCompileProgram(prog, c_options.size(), c_options.data());
        if (compileResult != NVRTC_SUCCESS) {
            size_t logSize;
            GPULITE_NVRTC_CALL(nvrtcGetProgramLogSize(prog, &logSize));
            std::string log(logSize, '\0');
            GPULITE_NVRTC_CALL(nvrtcGetProgramLog(prog, &log[0]));
            throw std::runtime_error(
                "KernelFactory::compileAndCacheKernel: Failed to compile CUDA program:\n" + log
            );
        }

        // fetch CUBIN
        size_t cubinSize = 0;
        GPULITE_NVRTC_CALL(nvrtcGetCUBINSize(prog, &cubinSize));
        std::vector<char> cubin(cubinSize);
        GPULITE_NVRTC_CALL(nvrtcGetCUBIN(prog, cubin.data()));

        // load the module from cubin
        CUmodule module = nullptr;
        CUresult cuResult;

        if (enableDebug) {
            // Load with JIT debug info
            CUjit_option opts[1];
            opts[0] = CU_JIT_GENERATE_DEBUG_INFO;
            void** vals = new void*[1];
            vals[0] = (void*)(size_t)1;
            cuResult = CUDADriver::instance().cuModuleLoadDataEx(
                &module, cubin.data(), 1, opts, vals
            );
            delete[] vals;
        } else {
            // Load without JIT options
            cuResult = CUDADriver::instance().cuModuleLoadDataEx(
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
        GPULITE_NVRTC_CALL(nvrtcGetLoweredName(prog, this->kernel_name.c_str(), &lowered_name));
        CUfunction kernel;
        GPULITE_CUDA_DRIVER_CALL(cuModuleGetFunction(&kernel, module, lowered_name));

        this->module = module;
        this->function = kernel;
        this->context = currentContext;
        this->compiled = true;

        GPULITE_NVRTC_CALL(nvrtcDestroyProgram(&prog));
    }

    void initCudaDriver() {
        int deviceCount = 0;
        // Check if CUDA has already been initialized
        CUresult res = CUDADriver::instance().cuDeviceGetCount(&deviceCount);
        if (res == CUDA_ERROR_NOT_INITIALIZED) {
            // CUDA hasn't been initialized, so we initialize it now
            res = CUDADriver::instance().cuInit(0);
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


/// Factory class to create and store compiled cuda kernels for caching as a
/// simple name-based hashmap. Allows both compiling from a source file, or for
/// compiling from a variable containing CUDA code.
class KernelFactory {
  public:
    KernelFactory(const KernelFactory&) = delete;
    KernelFactory& operator=(const KernelFactory&) = delete;

    KernelFactory(KernelFactory&&) = default;
    KernelFactory& operator=(KernelFactory&&) = default;

    /// Get the singleton instance of the KernelFactory for a given CUDA device.
    /// This ensures that each CUDA device has its own kernel cache.
    static KernelFactory& instance(CUdevice device) {
        static std::list<KernelFactory> INSTANCES;
        for (size_t i = INSTANCES.size(); i < device + 1; i++) {
            INSTANCES.emplace_back(KernelFactory());
        }

        // get the element at index "device" in the list and return it
        auto it = INSTANCES.begin();
        std::advance(it, device);
        return *it;
    }

    void cacheKernel(
        const std::string& kernel_name,
        const std::string& source_path,
        const std::string& source_name,
        const std::vector<std::string>& options
    ) {
        std::lock_guard<std::mutex> kernel_cache_lock(kernel_cache_mutex_);
        kernel_cache_[kernel_name] =
            std::make_unique<CachedKernel>(kernel_name, source_path, source_name, options);
    }

    bool hasKernel(const std::string& kernel_name) {
        std::lock_guard<std::mutex> kernel_cache_lock(kernel_cache_mutex_);
        return kernel_cache_.find(kernel_name) != kernel_cache_.end();
    }

    CachedKernel* getKernel(const std::string& kernel_name) {
        std::lock_guard<std::mutex> kernel_cache_lock(kernel_cache_mutex_);
        auto it = kernel_cache_.find(kernel_name);
        if (it != kernel_cache_.end()) {
            return it->second.get();
        } else {
            throw std::runtime_error("Kernel not found in cache.");
        }
    }

    /// Tries to retrieve the kernel "kernel_name". If not found, instantiate it
    /// and save to cache.
    CachedKernel* createFromSource(
        const std::string& kernel_name,
        const std::string& source_path,
        const std::string& source_name,
        const std::vector<std::string>& options
    ) {
        if (!this->hasKernel(kernel_name)) {
            std::ifstream file(source_path);
            if (!file.is_open()) {
                throw std::runtime_error("Failed to open file: " + source_path);
            }
            std::ostringstream ss;
            ss << file.rdbuf();

            std::string kernel_code = ss.str();
            this->cacheKernel(kernel_name, kernel_code, source_name, options);
        }
        return this->getKernel(kernel_name);
    }

    /// Tries to retrieve the kernel "kernel_name". If not found, instantiate it
    /// and save to cache.
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
    std::unordered_map<std::string, std::unique_ptr<CachedKernel>> kernel_cache_;

    static std::mutex kernel_cache_mutex_;
};

inline std::mutex KernelFactory::kernel_cache_mutex_;

} // namespace gpulite

#endif // GPULITE_HPP
