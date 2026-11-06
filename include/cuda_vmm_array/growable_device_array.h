#pragma once

#include <cuda.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

// Helper for cuda driver api error checking
#define CU_CHECK(call) \
    do { \
        const CUresult cuCheckResult = (call); \
        if (cuCheckResult != CUDA_SUCCESS) { \
            const char* cuErrorName = nullptr; \
            cuGetErrorName(cuCheckResult, &cuErrorName); \
            throw std::runtime_error(std::string(#call) + " failed at " + __FILE__ + ":" + \
                                     std::to_string(__LINE__) + " with error: " + \
                                     (cuErrorName ? cuErrorName : "unknown error")); \
        } \
    } while (0)


// Device array using CUDA vmm for backing virtual memory.
// The addr never changes as a result. Newly mapped memory is
// uninitialized. Growing is synchronous and must
// be called from the host.
// Reserve = address space only, Map = physical memory backs
// addr space
template<typename T> 
class GrowableDeviceArray {
public:
    GrowableDeviceArray() = default;
    GrowableDeviceArray(int device, size_t maxElements) {
        reserve(device, maxElements);
    }
    ~GrowableDeviceArray() {
        release();
    }

    GrowableDeviceArray(const GrowableDeviceArray&) = delete;
    GrowableDeviceArray& operator=(const GrowableDeviceArray&) = delete;
    GrowableDeviceArray(GrowableDeviceArray&&) = delete;
    GrowableDeviceArray& operator=(GrowableDeviceArray&&) = delete;

    // Reserve addr space for up to maxElements (no physical memory)
    void reserve(int device, size_t maxElements) {
        release();

        m_prop = {};
        m_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        m_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        m_prop.location.id = device;
        CU_CHECK(cuMemGetAllocationGranularity(&m_granularity, &m_prop,
                                               CU_MEM_ALLOC_GRANULARITY_MINIMUM));

        // Round up to a multiple of minimum granularity of cuda allocation
        m_reservedBytes = roundUp(std::max<size_t>(maxElements * sizeof(T), 1), m_granularity);
        CU_CHECK(cuMemAddressReserve(&m_base, m_reservedBytes, 0, 0, 0));

        m_access = {};
        m_access.location = m_prop.location;
        m_access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }

    // Map physical memory until capacity() >= elements
    void ensureCapacity(size_t elements) {
        const size_t neededBytes = elements * sizeof(T);
        if (neededBytes <= m_mappedBytes) {
            // Already have enough mapped bytes
            return;
        }
        if (neededBytes > m_reservedBytes) {
            throw std::runtime_error(
                "GrowableDeviceArray: capacity requested beyond the reservation");
        }

        // 1.5x mapped bytes
        size_t targetBytes = std::max(neededBytes, m_mappedBytes + m_mappedBytes / 2);
        // Don't allocate more then the reserved bytes
        targetBytes = std::min(roundUp(targetBytes, m_granularity), m_reservedBytes);
        const size_t chunkBytes = targetBytes - m_mappedBytes;
        // Address of the chunk being allocated
        const CUdeviceptr chunkAddress = m_base + m_mappedBytes;

        CUmemGenericAllocationHandle handle;
        CU_CHECK(cuMemCreate(&handle, chunkBytes, &m_prop, 0));

        // If mapping fails partly, undo what happened
        if (cuMemMap(chunkAddress, chunkBytes, 0, handle, 0) != CUDA_SUCCESS) {
            cuMemRelease(handle);
            throw std::runtime_error("GrowableDeviceArray: cuMemMap failed");
        }
        if (cuMemSetAccess(chunkAddress, chunkBytes, &m_access, 1) != CUDA_SUCCESS) {
            cuMemUnmap(chunkAddress, chunkBytes);
            cuMemRelease(handle);
            throw std::runtime_error("GrowableDeviceArray: cuMemSetAccess failed");
        }

        m_chunks.push_back({handle, chunkAddress, chunkBytes});
        m_mappedBytes = targetBytes;
    }

    T* data() const {
        return reinterpret_cast<T*>(m_base);
    }

    size_t capacity() const {
        return m_mappedBytes / sizeof(T);
    }

    size_t maxCapacity() const {
        return m_reservedBytes / sizeof(T);
    }

private:
    struct Chunk {
        CUmemGenericAllocationHandle handle;
        CUdeviceptr address;
        size_t bytes;
    };

    static size_t roundUp(size_t value, size_t multiple) {
        return (value + multiple - 1) / multiple * multiple;
    }

    void release()
    {
        for (auto it = m_chunks.rbegin(); it != m_chunks.rend(); ++it) {
            cuMemUnmap(it->address, it->bytes);
            cuMemRelease(it->handle);
        }
        m_chunks.clear();
        if (m_base != 0) {
            cuMemAddressFree(m_base, m_reservedBytes);
        }
        m_base = 0;
        m_reservedBytes = 0;
        m_mappedBytes = 0;
    }

    CUdeviceptr m_base = 0;
    size_t m_reservedBytes = 0;
    size_t m_mappedBytes = 0;
    size_t m_granularity = 0;
    CUmemAllocationProp m_prop{};
    CUmemAccessDesc m_access{};
    std::vector<Chunk> m_chunks;
};