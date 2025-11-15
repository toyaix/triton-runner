/*
 * Copyright 2011-2024 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#if !defined(_CUPTI_ACTIVITY_DEPRECATED_H_)
#define _CUPTI_ACTIVITY_DEPRECATED_H_

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility push(default)
#endif

/**
 * \brief The kinds of activity records.
 *
 * Each activity record kind represents information about a GPU or an
 * activity occurring on a CPU or GPU. Each kind is associated with a
 * activity record structure that holds the information associated
 * with the kind.
 * \see CUpti_ActivityOverhead
 * \see CUpti_ActivityOverhead2
 * \see CUpti_ActivityDevice
 * \see CUpti_ActivityDevice2
 * \see CUpti_ActivityDevice3
 * \see CUpti_ActivityDevice4
 * \see CUpti_ActivityKernel
 * \see CUpti_ActivityKernel2
 * \see CUpti_ActivityKernel3
 * \see CUpti_ActivityKernel4
 * \see CUpti_ActivityKernel5
 * \see CUpti_ActivityKernel6
 * \see CUpti_ActivityKernel7
 * \see CUpti_ActivityKernel8
 * \see CUpti_ActivityMemcpy
 * \see CUpti_ActivityMemcpy3
 * \see CUpti_ActivityMemcpy4
 * \see CUpti_ActivityMemcpyPtoP
 * \see CUpti_ActivityMemcpyPtoP2
 * \see CUpti_ActivityMemcpyPtoP3
 * \see CUpti_ActivityMemset
 * \see CUpti_ActivityMemset2
 * \see CUpti_ActivityMemset3
 * \see CUpti_ActivityMemory2
 * \see CUpti_ActivityMemory3
 * \see CUpti_ActivityMemoryPool
 * \see CUpti_ActivityMarker
 * \see CUpti_ActivityGlobalAccess
 * \see CUpti_ActivityGlobalAccess2
 * \see CUpti_ActivityBranch
 * \see CUpti_ActivityPCSampling
 * \see CUpti_ActivityPCSampling2
 * \see CUpti_ActivityUnifiedMemoryCounter
 * \see CUpti_ActivityUnifiedMemoryCounter2
 * \see CUpti_ActivityNvLink
 * \see CUpti_ActivityNvLink2
 * \see CUpti_ActivityNvLink3
 */

/**
 * \brief The activity record for CUPTI and driver overheads.
 * (Deprecated in CUDA 12.2)
 *
 * This activity record provides CUPTI and driver overhead information
 * (CUPTI_ACTIVITY_OVERHEAD). These records are now reported using
 * CUpti_ActivityOverhead3
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_OVERHEAD.
   */
  CUpti_ActivityKind kind;

  /**
   * The kind of overhead, CUPTI, DRIVER, COMPILER etc.
   */
  CUpti_ActivityOverheadKind overheadKind;

  /**
   * The kind of activity object that the overhead is associated with.
   */
  CUpti_ActivityObjectKind objectKind;

  /**
   * The identifier for the activity object. 'objectKind' indicates
   * which ID is valid for this record.
   */
  CUpti_ActivityObjectKindId objectId;

  /**
   * The start timestamp for the overhead, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the overhead.
   */
  uint64_t start;

  /**
   * The end timestamp for the overhead, in ns. A value of 0 for both
   * the start and end timestamps indicates that timestamp information
   * could not be collected for the overhead.
   */
  uint64_t end;
} CUpti_ActivityOverhead;

/**
 * \brief The activity record for CUPTI and driver overheads.
 *
 * This activity record provides CUPTI and driver overhead information
 * (CUPTI_ACTIVITY_OVERHEAD).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_OVERHEAD.
   */
  CUpti_ActivityKind kind;

  /**
   * The kind of overhead, CUPTI, DRIVER, COMPILER etc.
   */
  CUpti_ActivityOverheadKind overheadKind;

  /**
   * The kind of activity object that the overhead is associated with.
   */
  CUpti_ActivityObjectKind objectKind;

  /**
   * The identifier for the activity object. 'objectKind' indicates
   * which ID is valid for this record.
   */
  CUpti_ActivityObjectKindId objectId;

  /**
   * The start timestamp for the overhead, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the overhead.
   */
  uint64_t start;

  /**
   * The end timestamp for the overhead, in ns. A value of 0 for both
   * the start and end timestamps indicates that timestamp information
   * could not be collected for the overhead.
   */
  uint64_t end;

  /**
   * The correlation ID of the overhead operation to which
   * records belong to. This ID is identical to the
   * correlation ID in the driver or runtime API activity record that
   * launched the overhead operation.
   * In some cases, it can be zero, such as for CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH records.
   */
  uint32_t correlationId;

  /**
   * Reserved for internal use.
   */
  uint32_t reserved0;
} CUpti_ActivityOverhead2;

/**
 * \brief The activity record for a device. (deprecated)
 *
 * This activity record represents information about a GPU device
 * (CUPTI_ACTIVITY_KIND_DEVICE).
 * Device activity is now reported using the
 * CUpti_ActivityDevice5 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_DEVICE.
   */
  CUpti_ActivityKind kind;

  /**
   * The flags associated with the device. \see CUpti_ActivityFlag
   */
  CUpti_ActivityFlag flags;

  /**
   * The global memory bandwidth available on the device, in
   * kBytes/sec.
   */
  uint64_t globalMemoryBandwidth;

  /**
   * The amount of global memory on the device, in bytes.
   */
  uint64_t globalMemorySize;

  /**
   * The amount of constant memory on the device, in bytes.
   */
  uint32_t constantMemorySize;

  /**
   * The size of the L2 cache on the device, in bytes.
   */
  uint32_t l2CacheSize;

  /**
   * The number of threads per warp on the device.
   */
  uint32_t numThreadsPerWarp;

  /**
   * The core clock rate of the device, in kHz.
   */
  uint32_t coreClockRate;

  /**
   * Number of memory copy engines on the device.
   */
  uint32_t numMemcpyEngines;

  /**
   * Number of multiprocessors on the device.
   */
  uint32_t numMultiprocessors;

  /**
   * The maximum "instructions per cycle" possible on each device
   * multiprocessor.
   */
  uint32_t maxIPC;

  /**
   * Maximum number of warps that can be present on a multiprocessor
   * at any given time.
   */
  uint32_t maxWarpsPerMultiprocessor;

  /**
   * Maximum number of blocks that can be present on a multiprocessor
   * at any given time.
   */
  uint32_t maxBlocksPerMultiprocessor;

  /**
   * Maximum number of registers that can be allocated to a block.
   */
  uint32_t maxRegistersPerBlock;

  /**
   * Maximum amount of shared memory that can be assigned to a block,
   * in bytes.
   */
  uint32_t maxSharedMemoryPerBlock;

  /**
   * Maximum number of threads allowed in a block.
   */
  uint32_t maxThreadsPerBlock;

  /**
   * Maximum allowed X dimension for a block.
   */
  uint32_t maxBlockDimX;

  /**
   * Maximum allowed Y dimension for a block.
   */
  uint32_t maxBlockDimY;

  /**
   * Maximum allowed Z dimension for a block.
   */
  uint32_t maxBlockDimZ;

  /**
   * Maximum allowed X dimension for a grid.
   */
  uint32_t maxGridDimX;

  /**
   * Maximum allowed Y dimension for a grid.
   */
  uint32_t maxGridDimY;

  /**
   * Maximum allowed Z dimension for a grid.
   */
  uint32_t maxGridDimZ;

  /**
   * Compute capability for the device, major number.
   */
  uint32_t computeCapabilityMajor;

  /**
   * Compute capability for the device, minor number.
   */
  uint32_t computeCapabilityMinor;

  /**
   * The device ID.
   */
  uint32_t id;

#ifdef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * The device name. This name is shared across all activity records
   * representing instances of the device, and so should not be
   * modified.
   */
  const char *name;
} CUpti_ActivityDevice;

/**
 * \brief The activity record for a device. (deprecated)
 *
 * This activity record represents information about a GPU device
 * (CUPTI_ACTIVITY_KIND_DEVICE).
 * Device activity is now reported using the
 * CUpti_ActivityDevice5 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_DEVICE.
   */
  CUpti_ActivityKind kind;

  /**
   * The flags associated with the device. \see CUpti_ActivityFlag
   */
  CUpti_ActivityFlag flags;

  /**
   * The global memory bandwidth available on the device, in
   * kBytes/sec.
   */
  uint64_t globalMemoryBandwidth;

  /**
   * The amount of global memory on the device, in bytes.
   */
  uint64_t globalMemorySize;

  /**
   * The amount of constant memory on the device, in bytes.
   */
  uint32_t constantMemorySize;

  /**
   * The size of the L2 cache on the device, in bytes.
   */
  uint32_t l2CacheSize;

  /**
   * The number of threads per warp on the device.
   */
  uint32_t numThreadsPerWarp;

  /**
   * The core clock rate of the device, in kHz.
   */
  uint32_t coreClockRate;

  /**
   * Number of memory copy engines on the device.
   */
  uint32_t numMemcpyEngines;

  /**
   * Number of multiprocessors on the device.
   */
  uint32_t numMultiprocessors;

  /**
   * The maximum "instructions per cycle" possible on each device
   * multiprocessor.
   */
  uint32_t maxIPC;

  /**
   * Maximum number of warps that can be present on a multiprocessor
   * at any given time.
   */
  uint32_t maxWarpsPerMultiprocessor;

  /**
   * Maximum number of blocks that can be present on a multiprocessor
   * at any given time.
   */
  uint32_t maxBlocksPerMultiprocessor;

  /**
   * Maximum amount of shared memory available per multiprocessor, in bytes.
   */
  uint32_t maxSharedMemoryPerMultiprocessor;

  /**
   * Maximum number of 32-bit registers available per multiprocessor.
   */
  uint32_t maxRegistersPerMultiprocessor;

  /**
   * Maximum number of registers that can be allocated to a block.
   */
  uint32_t maxRegistersPerBlock;

  /**
   * Maximum amount of shared memory that can be assigned to a block,
   * in bytes.
   */
  uint32_t maxSharedMemoryPerBlock;

  /**
   * Maximum number of threads allowed in a block.
   */
  uint32_t maxThreadsPerBlock;

  /**
   * Maximum allowed X dimension for a block.
   */
  uint32_t maxBlockDimX;

  /**
   * Maximum allowed Y dimension for a block.
   */
  uint32_t maxBlockDimY;

  /**
   * Maximum allowed Z dimension for a block.
   */
  uint32_t maxBlockDimZ;

  /**
   * Maximum allowed X dimension for a grid.
   */
  uint32_t maxGridDimX;

  /**
   * Maximum allowed Y dimension for a grid.
   */
  uint32_t maxGridDimY;

  /**
   * Maximum allowed Z dimension for a grid.
   */
  uint32_t maxGridDimZ;

  /**
   * Compute capability for the device, major number.
   */
  uint32_t computeCapabilityMajor;

  /**
   * Compute capability for the device, minor number.
   */
  uint32_t computeCapabilityMinor;

  /**
   * The device ID.
   */
  uint32_t id;

  /**
   * ECC enabled flag for device
   */
  uint32_t eccEnabled;

  /**
   * The device UUID. This value is the globally unique immutable
   * alphanumeric identifier of the device.
   */
  CUuuid uuid;

#ifndef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * The device name. This name is shared across all activity records
   * representing instances of the device, and so should not be
   * modified.
   */
  const char *name;
} CUpti_ActivityDevice2;

/**
 * \brief The activity record for a device. (CUDA 7.0 onwards)
 *
 * This activity record represents information about a GPU device
 * (CUPTI_ACTIVITY_KIND_DEVICE).
 * Device activity is now reported using the
 * CUpti_ActivityDevice5 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_DEVICE.
   */
  CUpti_ActivityKind kind;

  /**
   * The flags associated with the device. \see CUpti_ActivityFlag
   */
  CUpti_ActivityFlag flags;

  /**
   * The global memory bandwidth available on the device, in
   * kBytes/sec.
   */
  uint64_t globalMemoryBandwidth;

  /**
   * The amount of global memory on the device, in bytes.
   */
  uint64_t globalMemorySize;

  /**
   * The amount of constant memory on the device, in bytes.
   */
  uint32_t constantMemorySize;

  /**
   * The size of the L2 cache on the device, in bytes.
   */
  uint32_t l2CacheSize;

  /**
   * The number of threads per warp on the device.
   */
  uint32_t numThreadsPerWarp;

  /**
   * The core clock rate of the device, in kHz.
   */
  uint32_t coreClockRate;

  /**
   * Number of memory copy engines on the device.
   */
  uint32_t numMemcpyEngines;

  /**
   * Number of multiprocessors on the device.
   */
  uint32_t numMultiprocessors;

  /**
   * The maximum "instructions per cycle" possible on each device
   * multiprocessor.
   */
  uint32_t maxIPC;

  /**
   * Maximum number of warps that can be present on a multiprocessor
   * at any given time.
   */
  uint32_t maxWarpsPerMultiprocessor;

  /**
   * Maximum number of blocks that can be present on a multiprocessor
   * at any given time.
   */
  uint32_t maxBlocksPerMultiprocessor;

  /**
   * Maximum amount of shared memory available per multiprocessor, in bytes.
   */
  uint32_t maxSharedMemoryPerMultiprocessor;

  /**
   * Maximum number of 32-bit registers available per multiprocessor.
   */
  uint32_t maxRegistersPerMultiprocessor;

  /**
   * Maximum number of registers that can be allocated to a block.
   */
  uint32_t maxRegistersPerBlock;

  /**
   * Maximum amount of shared memory that can be assigned to a block,
   * in bytes.
   */
  uint32_t maxSharedMemoryPerBlock;

  /**
   * Maximum number of threads allowed in a block.
   */
  uint32_t maxThreadsPerBlock;

  /**
   * Maximum allowed X dimension for a block.
   */
  uint32_t maxBlockDimX;

  /**
   * Maximum allowed Y dimension for a block.
   */
  uint32_t maxBlockDimY;

  /**
   * Maximum allowed Z dimension for a block.
   */
  uint32_t maxBlockDimZ;

  /**
   * Maximum allowed X dimension for a grid.
   */
  uint32_t maxGridDimX;

  /**
   * Maximum allowed Y dimension for a grid.
   */
  uint32_t maxGridDimY;

  /**
   * Maximum allowed Z dimension for a grid.
   */
  uint32_t maxGridDimZ;

  /**
   * Compute capability for the device, major number.
   */
  uint32_t computeCapabilityMajor;

  /**
   * Compute capability for the device, minor number.
   */
  uint32_t computeCapabilityMinor;

  /**
   * The device ID.
   */
  uint32_t id;

  /**
   * ECC enabled flag for device
   */
  uint32_t eccEnabled;

  /**
   * The device UUID. This value is the globally unique immutable
   * alphanumeric identifier of the device.
   */
  CUuuid uuid;

#ifndef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * The device name. This name is shared across all activity records
   * representing instances of the device, and so should not be
   * modified.
   */
  const char *name;

  /**
   * Flag to indicate whether the device is visible to CUDA. Users can
   * set the device visibility using CUDA_VISIBLE_DEVICES environment
   */
  uint8_t isCudaVisible;

  uint8_t reserved[7];
} CUpti_ActivityDevice3;

/**
 * \brief The activity record for a device. (CUDA 11.6 onwards)
 *
 * This activity record represents information about a GPU device
 * (CUPTI_ACTIVITY_KIND_DEVICE).
 * Device activity is now reported using the
 * CUpti_ActivityDevice5 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_DEVICE.
   */
  CUpti_ActivityKind kind;

  /**
   * The flags associated with the device. \see CUpti_ActivityFlag
   */
  CUpti_ActivityFlag flags;

  /**
   * The global memory bandwidth available on the device, in
   * kBytes/sec.
   */
  uint64_t globalMemoryBandwidth;

  /**
   * The amount of global memory on the device, in bytes.
   */
  uint64_t globalMemorySize;

  /**
   * The amount of constant memory on the device, in bytes.
   */
  uint32_t constantMemorySize;

  /**
   * The size of the L2 cache on the device, in bytes.
   */
  uint32_t l2CacheSize;

  /**
   * The number of threads per warp on the device.
   */
  uint32_t numThreadsPerWarp;

  /**
   * The core clock rate of the device, in kHz.
   */
  uint32_t coreClockRate;

  /**
   * Number of memory copy engines on the device.
   */
  uint32_t numMemcpyEngines;

  /**
   * Number of multiprocessors on the device.
   */
  uint32_t numMultiprocessors;

  /**
   * The maximum "instructions per cycle" possible on each device
   * multiprocessor.
   */
  uint32_t maxIPC;

  /**
   * Maximum number of warps that can be present on a multiprocessor
   * at any given time.
   */
  uint32_t maxWarpsPerMultiprocessor;

  /**
   * Maximum number of blocks that can be present on a multiprocessor
   * at any given time.
   */
  uint32_t maxBlocksPerMultiprocessor;

  /**
   * Maximum amount of shared memory available per multiprocessor, in bytes.
   */
  uint32_t maxSharedMemoryPerMultiprocessor;

  /**
   * Maximum number of 32-bit registers available per multiprocessor.
   */
  uint32_t maxRegistersPerMultiprocessor;

  /**
   * Maximum number of registers that can be allocated to a block.
   */
  uint32_t maxRegistersPerBlock;

  /**
   * Maximum amount of shared memory that can be assigned to a block,
   * in bytes.
   */
  uint32_t maxSharedMemoryPerBlock;

  /**
   * Maximum number of threads allowed in a block.
   */
  uint32_t maxThreadsPerBlock;

  /**
   * Maximum allowed X dimension for a block.
   */
  uint32_t maxBlockDimX;

  /**
   * Maximum allowed Y dimension for a block.
   */
  uint32_t maxBlockDimY;

  /**
   * Maximum allowed Z dimension for a block.
   */
  uint32_t maxBlockDimZ;

  /**
   * Maximum allowed X dimension for a grid.
   */
  uint32_t maxGridDimX;

  /**
   * Maximum allowed Y dimension for a grid.
   */
  uint32_t maxGridDimY;

  /**
   * Maximum allowed Z dimension for a grid.
   */
  uint32_t maxGridDimZ;

  /**
   * Compute capability for the device, major number.
   */
  uint32_t computeCapabilityMajor;

  /**
   * Compute capability for the device, minor number.
   */
  uint32_t computeCapabilityMinor;

  /**
   * The device ID.
   */
  uint32_t id;

  /**
   * ECC enabled flag for device
   */
  uint32_t eccEnabled;

  /**
   * The device UUID. This value is the globally unique immutable
   * alphanumeric identifier of the device.
   */
  CUuuid uuid;

#ifndef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * The device name. This name is shared across all activity records
   * representing instances of the device, and so should not be
   * modified.
   */
  const char *name;

  /**
   * Flag to indicate whether the device is visible to CUDA. Users can
   * set the device visibility using CUDA_VISIBLE_DEVICES environment
   */
  uint8_t isCudaVisible;

  /**
   * MIG enabled flag for device
   */
  uint8_t isMigEnabled;

  uint8_t reserved[6];

  /**
   * GPU Instance id for MIG enabled devices.
   * If mig mode is disabled value is set to UINT32_MAX
   */
  uint32_t gpuInstanceId;

  /**
   * Compute Instance id for MIG enabled devices.
   * If mig mode is disabled value is set to UINT32_MAX
   */
  uint32_t computeInstanceId;

  /**
   * The MIG UUID. This value is the globally unique immutable
   * alphanumeric identifier of the device.
   */
  CUuuid migUuid;

} CUpti_ActivityDevice4;

/**
 * \brief The activity record for kernel. (deprecated)
 *
 * This activity record represents a kernel execution
 * (CUPTI_ACTIVITY_KIND_KERNEL and
 * CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) but is no longer generated
 * by CUPTI. Kernel activities are now reported using the
 * CUpti_ActivityKernel9 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_KERNEL
   * or CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL.
   */
  CUpti_ActivityKind kind;

  /**
   * The cache configuration requested by the kernel. The value is one
   * of the CUfunc_cache enumeration values from cuda.h.
   */
  uint8_t cacheConfigRequested;

  /**
   * The cache configuration used for the kernel. The value is one of
   * the CUfunc_cache enumeration values from cuda.h.
   */
  uint8_t cacheConfigExecuted;

  /**
   * The number of registers required for each thread executing the
   * kernel.
   */
  uint16_t registersPerThread;

  /**
   * The start timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t start;

  /**
   * The end timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t end;

  /**
   * The ID of the device where the kernel is executing.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the kernel is executing.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the kernel is executing.
   */
  uint32_t streamId;

  /**
   * The X-dimension grid size for the kernel.
   */
  int32_t gridX;

  /**
   * The Y-dimension grid size for the kernel.
   */
  int32_t gridY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t gridZ;

  /**
   * The X-dimension block size for the kernel.
   */
  int32_t blockX;

  /**
   * The Y-dimension block size for the kernel.
   */
  int32_t blockY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t blockZ;

  /**
   * The static shared memory allocated for the kernel, in bytes.
   */
  int32_t staticSharedMemory;

  /**
   * The dynamic shared memory reserved for the kernel, in bytes.
   */
  int32_t dynamicSharedMemory;

  /**
   * The amount of local memory reserved for each thread, in bytes.
   */
  uint32_t localMemoryPerThread;

  /**
   * The total amount of local memory reserved for the kernel, in
   * bytes.
   */
  uint32_t localMemoryTotal;

  /**
   * The correlation ID of the kernel. Each kernel execution is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver API activity record that launched
   * the kernel.
   */
  uint32_t correlationId;

  /**
   * The runtime correlation ID of the kernel. Each kernel execution
   * is assigned a unique runtime correlation ID that is identical to
   * the correlation ID in the runtime API activity record that
   * launched the kernel.
   */
  uint32_t runtimeCorrelationId;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;

  /**
   * The name of the kernel. This name is shared across all activity
   * records representing the same kernel, and so should not be
   * modified.
   */
  const char *name;

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;
} CUpti_ActivityKernel;

/**
 * \brief The activity record for kernel. (deprecated)
 *
 * This activity record represents a kernel execution
 * (CUPTI_ACTIVITY_KIND_KERNEL and
 * CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) but is no longer generated
 * by CUPTI. Kernel activities are now reported using the
 * CUpti_ActivityKernel9 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_KERNEL or
   * CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL.
   */
  CUpti_ActivityKind kind;

  union {
    uint8_t both;
    struct {
      /**
       * The cache configuration requested by the kernel. The value is one
       * of the CUfunc_cache enumeration values from cuda.h.
       */
      uint8_t requested:4;

      /**
       * The cache configuration used for the kernel. The value is one of
       * the CUfunc_cache enumeration values from cuda.h.
       */
      uint8_t executed:4;
    } config;
  } cacheConfig;

  /**
   * The shared memory configuration used for the kernel. The value is one of
   * the CUsharedconfig enumeration values from cuda.h.
   */
  uint8_t sharedMemoryConfig;

  /**
   * The number of registers required for each thread executing the
   * kernel.
   */
  uint16_t registersPerThread;

  /**
   * The start timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t start;

  /**
   * The end timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t end;

  /**
   * The completed timestamp for the kernel execution, in ns.  It
   * represents the completion of all it's child kernels and the
   * kernel itself. A value of CUPTI_TIMESTAMP_UNKNOWN indicates that
   * the completion time is unknown.
   */
  uint64_t completed;

  /**
   * The ID of the device where the kernel is executing.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the kernel is executing.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the kernel is executing.
   */
  uint32_t streamId;

  /**
   * The X-dimension grid size for the kernel.
   */
  int32_t gridX;

  /**
   * The Y-dimension grid size for the kernel.
   */
  int32_t gridY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t gridZ;

  /**
   * The X-dimension block size for the kernel.
   */
  int32_t blockX;

  /**
   * The Y-dimension block size for the kernel.
   */
  int32_t blockY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t blockZ;

  /**
   * The static shared memory allocated for the kernel, in bytes.
   */
  int32_t staticSharedMemory;

  /**
   * The dynamic shared memory reserved for the kernel, in bytes.
   */
  int32_t dynamicSharedMemory;

  /**
   * The amount of local memory reserved for each thread, in bytes.
   */
  uint32_t localMemoryPerThread;

  /**
   * The total amount of local memory reserved for the kernel, in
   * bytes.
   */
  uint32_t localMemoryTotal;

  /**
   * The correlation ID of the kernel. Each kernel execution is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver or runtime API activity record that
   * launched the kernel.
   */
  uint32_t correlationId;

  /**
   * The grid ID of the kernel. Each kernel is assigned a unique
   * grid ID at runtime.
   */
  int64_t gridId;

  /**
   * The name of the kernel. This name is shared across all activity
   * records representing the same kernel, and so should not be
   * modified.
   */
  const char *name;

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;
} CUpti_ActivityKernel2;

/**
 * \brief The activity record for a kernel (CUDA 6.5(with sm_52 support) onwards).
 * (deprecated in CUDA 9.0)
 *
 * This activity record represents a kernel execution
 * (CUPTI_ACTIVITY_KIND_KERNEL and
 * CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL).
 * Kernel activities are now reported using the CUpti_ActivityKernel9 activity
 * record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_KERNEL or
   * CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL.
   */
  CUpti_ActivityKind kind;

  union {
    uint8_t both;
    struct {
      /**
       * The cache configuration requested by the kernel. The value is one
       * of the CUfunc_cache enumeration values from cuda.h.
       */
      uint8_t requested:4;

      /**
       * The cache configuration used for the kernel. The value is one of
       * the CUfunc_cache enumeration values from cuda.h.
       */
      uint8_t executed:4;
    } config;
  } cacheConfig;

  /**
   * The shared memory configuration used for the kernel. The value is one of
   * the CUsharedconfig enumeration values from cuda.h.
   */
  uint8_t sharedMemoryConfig;

  /**
   * The number of registers required for each thread executing the
   * kernel.
   */
  uint16_t registersPerThread;

  /**
   * The partitioned global caching requested for the kernel. Partitioned
   * global caching is required to enable caching on certain chips, such as
   * devices with compute capability 5.2.
   */
  CUpti_ActivityPartitionedGlobalCacheConfig partitionedGlobalCacheRequested;

  /**
   * The partitioned global caching executed for the kernel. Partitioned
   * global caching is required to enable caching on certain chips, such as
   * devices with compute capability 5.2. Partitioned global caching can be
   * automatically disabled if the occupancy requirement of the launch cannot
   * support caching.
   */
  CUpti_ActivityPartitionedGlobalCacheConfig partitionedGlobalCacheExecuted;

  /**
   * The start timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t start;

  /**
   * The end timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t end;

  /**
   * The completed timestamp for the kernel execution, in ns.  It
   * represents the completion of all it's child kernels and the
   * kernel itself. A value of CUPTI_TIMESTAMP_UNKNOWN indicates that
   * the completion time is unknown.
   */
  uint64_t completed;

  /**
   * The ID of the device where the kernel is executing.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the kernel is executing.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the kernel is executing.
   */
  uint32_t streamId;

  /**
   * The X-dimension grid size for the kernel.
   */
  int32_t gridX;

  /**
   * The Y-dimension grid size for the kernel.
   */
  int32_t gridY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t gridZ;

  /**
   * The X-dimension block size for the kernel.
   */
  int32_t blockX;

  /**
   * The Y-dimension block size for the kernel.
   */
  int32_t blockY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t blockZ;

  /**
   * The static shared memory allocated for the kernel, in bytes.
   */
  int32_t staticSharedMemory;

  /**
   * The dynamic shared memory reserved for the kernel, in bytes.
   */
  int32_t dynamicSharedMemory;

  /**
   * The amount of local memory reserved for each thread, in bytes.
   */
  uint32_t localMemoryPerThread;

  /**
   * The total amount of local memory reserved for the kernel, in
   * bytes.
   */
  uint32_t localMemoryTotal;

  /**
   * The correlation ID of the kernel. Each kernel execution is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver or runtime API activity record that
   * launched the kernel.
   */
  uint32_t correlationId;

  /**
   * The grid ID of the kernel. Each kernel is assigned a unique
   * grid ID at runtime.
   */
  int64_t gridId;

  /**
   * The name of the kernel. This name is shared across all activity
   * records representing the same kernel, and so should not be
   * modified.
   */
  const char *name;

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;
} CUpti_ActivityKernel3;

/**
 * \brief The activity record for a kernel (CUDA 9.0(with sm_70 support) onwards).
 * (deprecated in CUDA 11.0)
 *
 * This activity record represents a kernel execution
 * (CUPTI_ACTIVITY_KIND_KERNEL and
 * CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL).
 * Kernel activities are now reported using the CUpti_ActivityKernel9 activity
 * record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_KERNEL or
   * CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL.
   */
  CUpti_ActivityKind kind;

  /**
   * For devices with compute capability 7.0+ cacheConfig values are not updated
   * in case field isSharedMemoryCarveoutRequested is set
   */
  union {
    uint8_t both;
    struct {
      /**
       * The cache configuration requested by the kernel. The value is one
       * of the CUfunc_cache enumeration values from cuda.h.
       */
      uint8_t requested:4;

      /**
       * The cache configuration used for the kernel. The value is one of
       * the CUfunc_cache enumeration values from cuda.h.
       */
      uint8_t executed:4;
    } config;
  } cacheConfig;

  /**
   * The shared memory configuration used for the kernel. The value is one of
   * the CUsharedconfig enumeration values from cuda.h.
   */
  uint8_t sharedMemoryConfig;

  /**
   * The number of registers required for each thread executing the
   * kernel.
   */
  uint16_t registersPerThread;

  /**
   * The partitioned global caching requested for the kernel. Partitioned
   * global caching is required to enable caching on certain chips, such as
   * devices with compute capability 5.2.
   */
  CUpti_ActivityPartitionedGlobalCacheConfig partitionedGlobalCacheRequested;

  /**
   * The partitioned global caching executed for the kernel. Partitioned
   * global caching is required to enable caching on certain chips, such as
   * devices with compute capability 5.2. Partitioned global caching can be
   * automatically disabled if the occupancy requirement of the launch cannot
   * support caching.
   */
  CUpti_ActivityPartitionedGlobalCacheConfig partitionedGlobalCacheExecuted;

  /**
   * The start timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t start;

  /**
   * The end timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t end;

  /**
   * The completed timestamp for the kernel execution, in ns.  It
   * represents the completion of all it's child kernels and the
   * kernel itself. A value of CUPTI_TIMESTAMP_UNKNOWN indicates that
   * the completion time is unknown.
   */
  uint64_t completed;

  /**
   * The ID of the device where the kernel is executing.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the kernel is executing.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the kernel is executing.
   */
  uint32_t streamId;

  /**
   * The X-dimension grid size for the kernel.
   */
  int32_t gridX;

  /**
   * The Y-dimension grid size for the kernel.
   */
  int32_t gridY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t gridZ;

  /**
   * The X-dimension block size for the kernel.
   */
  int32_t blockX;

  /**
   * The Y-dimension block size for the kernel.
   */
  int32_t blockY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t blockZ;

  /**
   * The static shared memory allocated for the kernel, in bytes.
   */
  int32_t staticSharedMemory;

  /**
   * The dynamic shared memory reserved for the kernel, in bytes.
   */
  int32_t dynamicSharedMemory;

  /**
   * The amount of local memory reserved for each thread, in bytes.
   */
  uint32_t localMemoryPerThread;

  /**
   * The total amount of local memory reserved for the kernel, in
   * bytes.
   */
  uint32_t localMemoryTotal;

  /**
   * The correlation ID of the kernel. Each kernel execution is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver or runtime API activity record that
   * launched the kernel.
   */
  uint32_t correlationId;

  /**
   * The grid ID of the kernel. Each kernel is assigned a unique
   * grid ID at runtime.
   */
  int64_t gridId;

  /**
   * The name of the kernel. This name is shared across all activity
   * records representing the same kernel, and so should not be
   * modified.
   */
  const char *name;

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;

  /**
   * The timestamp when the kernel is queued up in the command buffer, in ns.
   * A value of CUPTI_TIMESTAMP_UNKNOWN indicates that the queued time
   * could not be collected for the kernel. This timestamp is not collected
   * by default. Use API \ref cuptiActivityEnableLatencyTimestamps() to
   * enable collection.
   *
   * Command buffer is a buffer written by CUDA driver to send commands
   * like kernel launch, memory copy etc to the GPU. All launches of CUDA
   * kernels are asynchronous with respect to the host, the host requests
   * the launch by writing commands into the command buffer, then returns
   * without checking the GPU's progress.
   */
  uint64_t queued;

  /**
   * The timestamp when the command buffer containing the kernel launch
   * is submitted to the GPU, in ns. A value of CUPTI_TIMESTAMP_UNKNOWN
   * indicates that the submitted time could not be collected for the kernel.
   * This timestamp is not collected by default. Use API \ref
   * cuptiActivityEnableLatencyTimestamps() to enable collection.
   */
  uint64_t submitted;

  /**
   * The indicates if the kernel was executed via a regular launch or via a
   * single/multi device cooperative launch. \see CUpti_ActivityLaunchType
   */
  uint8_t launchType;

  /**
   * This indicates if CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT was
   * updated for the kernel launch
   */
  uint8_t isSharedMemoryCarveoutRequested;

  /**
   * Shared memory carveout value requested for the function in percentage of
   * the total resource. The value will be updated only if field
   * isSharedMemoryCarveoutRequested is set.
   */
  uint8_t sharedMemoryCarveoutRequested;

  /**
   * Undefined. Reserved for internal use.
   */
  uint8_t padding;

 /**
  * Shared memory size set by the driver.
  */
  uint32_t sharedMemoryExecuted;
} CUpti_ActivityKernel4;

/**
 * \brief The activity record for a kernel (CUDA 11.0(with sm_80 support) onwards).
 * (deprecated in CUDA 11.2)
 * This activity record represents a kernel execution
 * (CUPTI_ACTIVITY_KIND_KERNEL and
 * CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) but is no longer generated
 * by CUPTI. Kernel activities are now reported using the
 * CUpti_ActivityKernel9 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_KERNEL or
   * CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL.
   */
  CUpti_ActivityKind kind;

  /**
   * For devices with compute capability 7.0+ cacheConfig values are not updated
   * in case field isSharedMemoryCarveoutRequested is set
   */
  union {
    uint8_t both;
    struct {
      /**
       * The cache configuration requested by the kernel. The value is one
       * of the CUfunc_cache enumeration values from cuda.h.
       */
      uint8_t requested:4;

      /**
       * The cache configuration used for the kernel. The value is one of
       * the CUfunc_cache enumeration values from cuda.h.
       */
      uint8_t executed:4;
    } config;
  } cacheConfig;

  /**
   * The shared memory configuration used for the kernel. The value is one of
   * the CUsharedconfig enumeration values from cuda.h.
   */
  uint8_t sharedMemoryConfig;

  /**
   * The number of registers required for each thread executing the
   * kernel.
   */
  uint16_t registersPerThread;

  /**
   * The partitioned global caching requested for the kernel. Partitioned
   * global caching is required to enable caching on certain chips, such as
   * devices with compute capability 5.2.
   */
  CUpti_ActivityPartitionedGlobalCacheConfig partitionedGlobalCacheRequested;

  /**
   * The partitioned global caching executed for the kernel. Partitioned
   * global caching is required to enable caching on certain chips, such as
   * devices with compute capability 5.2. Partitioned global caching can be
   * automatically disabled if the occupancy requirement of the launch cannot
   * support caching.
   */
  CUpti_ActivityPartitionedGlobalCacheConfig partitionedGlobalCacheExecuted;

  /**
   * The start timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t start;

  /**
   * The end timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t end;

  /**
   * The completed timestamp for the kernel execution, in ns.  It
   * represents the completion of all it's child kernels and the
   * kernel itself. A value of CUPTI_TIMESTAMP_UNKNOWN indicates that
   * the completion time is unknown.
   */
  uint64_t completed;

  /**
   * The ID of the device where the kernel is executing.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the kernel is executing.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the kernel is executing.
   */
  uint32_t streamId;

  /**
   * The X-dimension grid size for the kernel.
   */
  int32_t gridX;

  /**
   * The Y-dimension grid size for the kernel.
   */
  int32_t gridY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t gridZ;

  /**
   * The X-dimension block size for the kernel.
   */
  int32_t blockX;

  /**
   * The Y-dimension block size for the kernel.
   */
  int32_t blockY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t blockZ;

  /**
   * The static shared memory allocated for the kernel, in bytes.
   */
  int32_t staticSharedMemory;

  /**
   * The dynamic shared memory reserved for the kernel, in bytes.
   */
  int32_t dynamicSharedMemory;

  /**
   * The amount of local memory reserved for each thread, in bytes.
   */
  uint32_t localMemoryPerThread;

  /**
   * The total amount of local memory reserved for the kernel, in
   * bytes.
   */
  uint32_t localMemoryTotal;

  /**
   * The correlation ID of the kernel. Each kernel execution is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver or runtime API activity record that
   * launched the kernel.
   */
  uint32_t correlationId;

  /**
   * The grid ID of the kernel. Each kernel is assigned a unique
   * grid ID at runtime.
   */
  int64_t gridId;

  /**
   * The name of the kernel. This name is shared across all activity
   * records representing the same kernel, and so should not be
   * modified.
   */
  const char *name;

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;

  /**
   * The timestamp when the kernel is queued up in the command buffer, in ns.
   * A value of CUPTI_TIMESTAMP_UNKNOWN indicates that the queued time
   * could not be collected for the kernel. This timestamp is not collected
   * by default. Use API \ref cuptiActivityEnableLatencyTimestamps() to
   * enable collection.
   *
   * Command buffer is a buffer written by CUDA driver to send commands
   * like kernel launch, memory copy etc to the GPU. All launches of CUDA
   * kernels are asynchronous with respect to the host, the host requests
   * the launch by writing commands into the command buffer, then returns
   * without checking the GPU's progress.
   */
  uint64_t queued;

  /**
   * The timestamp when the command buffer containing the kernel launch
   * is submitted to the GPU, in ns. A value of CUPTI_TIMESTAMP_UNKNOWN
   * indicates that the submitted time could not be collected for the kernel.
   * This timestamp is not collected by default. Use API \ref
   * cuptiActivityEnableLatencyTimestamps() to enable collection.
   */
  uint64_t submitted;

  /**
   * The indicates if the kernel was executed via a regular launch or via a
   * single/multi device cooperative launch. \see CUpti_ActivityLaunchType
   */
  uint8_t launchType;

  /**
   * This indicates if CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT was
   * updated for the kernel launch
   */
  uint8_t isSharedMemoryCarveoutRequested;

  /**
   * Shared memory carveout value requested for the function in percentage of
   * the total resource. The value will be updated only if field
   * isSharedMemoryCarveoutRequested is set.
   */
  uint8_t sharedMemoryCarveoutRequested;

  /**
   * Undefined. Reserved for internal use.
   */
  uint8_t padding;

 /**
  * Shared memory size set by the driver.
  */
  uint32_t sharedMemoryExecuted;

  /**
   * The unique ID of the graph node that launched this kernel through graph launch APIs.
   * This field will be 0 if the kernel is not launched through graph launch APIs.
   */
  uint64_t graphNodeId;

  /**
   * The shared memory limit config for the kernel. This field shows whether user has opted for a
   * higher per block limit of dynamic shared memory.
   */
  CUpti_FuncShmemLimitConfig shmemLimitConfig;

  /**
   * The unique ID of the graph that launched this kernel through graph launch APIs.
   * This field will be 0 if the kernel is not launched through graph launch APIs.
   */
  uint32_t graphId;
} CUpti_ActivityKernel5;

/**
 * \brief The activity record for kernel. (deprecated in CUDA 11.6)
 *
 * This activity record represents a kernel execution
 * (CUPTI_ACTIVITY_KIND_KERNEL and
 * CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) but is no longer generated
 * by CUPTI. Kernel activities are now reported using the
 * CUpti_ActivityKernel9 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_KERNEL or
   * CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL.
   */
  CUpti_ActivityKind kind;

  /**
   * For devices with compute capability 7.0+ cacheConfig values are not updated
   * in case field isSharedMemoryCarveoutRequested is set
   */
  union {
    uint8_t both;
    struct {
      /**
       * The cache configuration requested by the kernel. The value is one
       * of the CUfunc_cache enumeration values from cuda.h.
       */
      uint8_t requested:4;

      /**
       * The cache configuration used for the kernel. The value is one of
       * the CUfunc_cache enumeration values from cuda.h.
       */
      uint8_t executed:4;
    } config;
  } cacheConfig;

  /**
   * The shared memory configuration used for the kernel. The value is one of
   * the CUsharedconfig enumeration values from cuda.h.
   */
  uint8_t sharedMemoryConfig;

  /**
   * The number of registers required for each thread executing the
   * kernel.
   */
  uint16_t registersPerThread;

  /**
   * The partitioned global caching requested for the kernel. Partitioned
   * global caching is required to enable caching on certain chips, such as
   * devices with compute capability 5.2.
   */
  CUpti_ActivityPartitionedGlobalCacheConfig partitionedGlobalCacheRequested;

  /**
   * The partitioned global caching executed for the kernel. Partitioned
   * global caching is required to enable caching on certain chips, such as
   * devices with compute capability 5.2. Partitioned global caching can be
   * automatically disabled if the occupancy requirement of the launch cannot
   * support caching.
   */
  CUpti_ActivityPartitionedGlobalCacheConfig partitionedGlobalCacheExecuted;

  /**
   * The start timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t start;

  /**
   * The end timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t end;

  /**
   * The completed timestamp for the kernel execution, in ns.  It
   * represents the completion of all it's child kernels and the
   * kernel itself. A value of CUPTI_TIMESTAMP_UNKNOWN indicates that
   * the completion time is unknown.
   */
  uint64_t completed;

  /**
   * The ID of the device where the kernel is executing.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the kernel is executing.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the kernel is executing.
   */
  uint32_t streamId;

  /**
   * The X-dimension grid size for the kernel.
   */
  int32_t gridX;

  /**
   * The Y-dimension grid size for the kernel.
   */
  int32_t gridY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t gridZ;

  /**
   * The X-dimension block size for the kernel.
   */
  int32_t blockX;

  /**
   * The Y-dimension block size for the kernel.
   */
  int32_t blockY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t blockZ;

  /**
   * The static shared memory allocated for the kernel, in bytes.
   */
  int32_t staticSharedMemory;

  /**
   * The dynamic shared memory reserved for the kernel, in bytes.
   */
  int32_t dynamicSharedMemory;

  /**
   * The amount of local memory reserved for each thread, in bytes.
   */
  uint32_t localMemoryPerThread;

  /**
   * The total amount of local memory reserved for the kernel, in
   * bytes.
   */
  uint32_t localMemoryTotal;

  /**
   * The correlation ID of the kernel. Each kernel execution is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver or runtime API activity record that
   * launched the kernel.
   */
  uint32_t correlationId;

  /**
   * The grid ID of the kernel. Each kernel is assigned a unique
   * grid ID at runtime.
   */
  int64_t gridId;

  /**
   * The name of the kernel. This name is shared across all activity
   * records representing the same kernel, and so should not be
   * modified.
   */
  const char *name;

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;

  /**
   * The timestamp when the kernel is queued up in the command buffer, in ns.
   * A value of CUPTI_TIMESTAMP_UNKNOWN indicates that the queued time
   * could not be collected for the kernel. This timestamp is not collected
   * by default. Use API \ref cuptiActivityEnableLatencyTimestamps() to
   * enable collection.
   *
   * Command buffer is a buffer written by CUDA driver to send commands
   * like kernel launch, memory copy etc to the GPU. All launches of CUDA
   * kernels are asynchronous with respect to the host, the host requests
   * the launch by writing commands into the command buffer, then returns
   * without checking the GPU's progress.
   */
  uint64_t queued;

  /**
   * The timestamp when the command buffer containing the kernel launch
   * is submitted to the GPU, in ns. A value of CUPTI_TIMESTAMP_UNKNOWN
   * indicates that the submitted time could not be collected for the kernel.
   * This timestamp is not collected by default. Use API \ref
   * cuptiActivityEnableLatencyTimestamps() to enable collection.
   */
  uint64_t submitted;

  /**
   * The indicates if the kernel was executed via a regular launch or via a
   * single/multi device cooperative launch. \see CUpti_ActivityLaunchType
   */
  uint8_t launchType;

  /**
   * This indicates if CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT was
   * updated for the kernel launch
   */
  uint8_t isSharedMemoryCarveoutRequested;

  /**
   * Shared memory carveout value requested for the function in percentage of
   * the total resource. The value will be updated only if field
   * isSharedMemoryCarveoutRequested is set.
   */
  uint8_t sharedMemoryCarveoutRequested;

  /**
   * Undefined. Reserved for internal use.
   */
  uint8_t padding;

 /**
  * Shared memory size set by the driver.
  */
  uint32_t sharedMemoryExecuted;

  /**
   * The unique ID of the graph node that launched this kernel through graph launch APIs.
   * This field will be 0 if the kernel is not launched through graph launch APIs.
   */
  uint64_t graphNodeId;

  /**
   * The shared memory limit config for the kernel. This field shows whether user has opted for a
   * higher per block limit of dynamic shared memory.
   */
  CUpti_FuncShmemLimitConfig shmemLimitConfig;

  /**
   * The unique ID of the graph that launched this kernel through graph launch APIs.
   * This field will be 0 if the kernel is not launched through graph launch APIs.
   */
  uint32_t graphId;

  /**
   * The pointer to the access policy window. The structure CUaccessPolicyWindow is
   * defined in cuda.h.
   */
  CUaccessPolicyWindow *pAccessPolicyWindow;
} CUpti_ActivityKernel6;

/**
 * \brief The activity record for kernel. (deprecated in CUDA 11.8)
 *
 * This activity record represents a kernel execution
 * (CUPTI_ACTIVITY_KIND_KERNEL and
 * CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) but is no longer generated
 * by CUPTI. Kernel activities are now reported using the
 * CUpti_ActivityKernel9 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_KERNEL or
   * CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL.
   */
  CUpti_ActivityKind kind;

  /**
   * For devices with compute capability 7.0+ cacheConfig values are not updated
   * in case field isSharedMemoryCarveoutRequested is set
   */
  union {
    uint8_t both;
    struct {
      /**
       * The cache configuration requested by the kernel. The value is one
       * of the CUfunc_cache enumeration values from cuda.h.
       */
      uint8_t requested:4;

      /**
       * The cache configuration used for the kernel. The value is one of
       * the CUfunc_cache enumeration values from cuda.h.
       */
      uint8_t executed:4;
    } config;
  } cacheConfig;

  /**
   * The shared memory configuration used for the kernel. The value is one of
   * the CUsharedconfig enumeration values from cuda.h.
   */
  uint8_t sharedMemoryConfig;

  /**
   * The number of registers required for each thread executing the
   * kernel.
   */
  uint16_t registersPerThread;

  /**
   * The partitioned global caching requested for the kernel. Partitioned
   * global caching is required to enable caching on certain chips, such as
   * devices with compute capability 5.2.
   */
  CUpti_ActivityPartitionedGlobalCacheConfig partitionedGlobalCacheRequested;

  /**
   * The partitioned global caching executed for the kernel. Partitioned
   * global caching is required to enable caching on certain chips, such as
   * devices with compute capability 5.2. Partitioned global caching can be
   * automatically disabled if the occupancy requirement of the launch cannot
   * support caching.
   */
  CUpti_ActivityPartitionedGlobalCacheConfig partitionedGlobalCacheExecuted;

  /**
   * The start timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t start;

  /**
   * The end timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t end;

  /**
   * The completed timestamp for the kernel execution, in ns.  It
   * represents the completion of all it's child kernels and the
   * kernel itself. A value of CUPTI_TIMESTAMP_UNKNOWN indicates that
   * the completion time is unknown.
   */
  uint64_t completed;

  /**
   * The ID of the device where the kernel is executing.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the kernel is executing.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the kernel is executing.
   */
  uint32_t streamId;

  /**
   * The X-dimension grid size for the kernel.
   */
  int32_t gridX;

  /**
   * The Y-dimension grid size for the kernel.
   */
  int32_t gridY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t gridZ;

  /**
   * The X-dimension block size for the kernel.
   */
  int32_t blockX;

  /**
   * The Y-dimension block size for the kernel.
   */
  int32_t blockY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t blockZ;

  /**
   * The static shared memory allocated for the kernel, in bytes.
   */
  int32_t staticSharedMemory;

  /**
   * The dynamic shared memory reserved for the kernel, in bytes.
   */
  int32_t dynamicSharedMemory;

  /**
   * The amount of local memory reserved for each thread, in bytes.
   */
  uint32_t localMemoryPerThread;

  /**
   * The total amount of local memory reserved for the kernel, in
   * bytes.
   */
  uint32_t localMemoryTotal;

  /**
   * The correlation ID of the kernel. Each kernel execution is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver or runtime API activity record that
   * launched the kernel.
   */
  uint32_t correlationId;

  /**
   * The grid ID of the kernel. Each kernel is assigned a unique
   * grid ID at runtime.
   */
  int64_t gridId;

  /**
   * The name of the kernel. This name is shared across all activity
   * records representing the same kernel, and so should not be
   * modified.
   */
  const char *name;

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;

  /**
   * The timestamp when the kernel is queued up in the command buffer, in ns.
   * A value of CUPTI_TIMESTAMP_UNKNOWN indicates that the queued time
   * could not be collected for the kernel. This timestamp is not collected
   * by default. Use API \ref cuptiActivityEnableLatencyTimestamps() to
   * enable collection.
   *
   * Command buffer is a buffer written by CUDA driver to send commands
   * like kernel launch, memory copy etc to the GPU. All launches of CUDA
   * kernels are asynchronous with respect to the host, the host requests
   * the launch by writing commands into the command buffer, then returns
   * without checking the GPU's progress.
   */
  uint64_t queued;

  /**
   * The timestamp when the command buffer containing the kernel launch
   * is submitted to the GPU, in ns. A value of CUPTI_TIMESTAMP_UNKNOWN
   * indicates that the submitted time could not be collected for the kernel.
   * This timestamp is not collected by default. Use API \ref
   * cuptiActivityEnableLatencyTimestamps() to enable collection.
   */
  uint64_t submitted;

  /**
   * The indicates if the kernel was executed via a regular launch or via a
   * single/multi device cooperative launch. \see CUpti_ActivityLaunchType
   */
  uint8_t launchType;

  /**
   * This indicates if CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT was
   * updated for the kernel launch
   */
  uint8_t isSharedMemoryCarveoutRequested;

  /**
   * Shared memory carveout value requested for the function in percentage of
   * the total resource. The value will be updated only if field
   * isSharedMemoryCarveoutRequested is set.
   */
  uint8_t sharedMemoryCarveoutRequested;

  /**
   * Undefined. Reserved for internal use.
   */
  uint8_t padding;

 /**
  * Shared memory size set by the driver.
  */
  uint32_t sharedMemoryExecuted;

  /**
   * The unique ID of the graph node that launched this kernel through graph launch APIs.
   * This field will be 0 if the kernel is not launched through graph launch APIs.
   */
  uint64_t graphNodeId;

  /**
   * The shared memory limit config for the kernel. This field shows whether user has opted for a
   * higher per block limit of dynamic shared memory.
   */
  CUpti_FuncShmemLimitConfig shmemLimitConfig;

  /**
   * The unique ID of the graph that launched this kernel through graph launch APIs.
   * This field will be 0 if the kernel is not launched through graph launch APIs.
   */
  uint32_t graphId;

  /**
   * The pointer to the access policy window. The structure CUaccessPolicyWindow is
   * defined in cuda.h.
   */
  CUaccessPolicyWindow *pAccessPolicyWindow;

  /**
   * The ID of the HW channel on which the kernel is launched.
   */
  uint32_t channelID;

  /**
   * The type of the channel
   */
  CUpti_ChannelType channelType;
} CUpti_ActivityKernel7;

/**
 * \brief The activity record for kernel.
 *
 * This activity record represents a kernel execution
 * (CUPTI_ACTIVITY_KIND_KERNEL and
 * CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_KERNEL or
   * CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL.
   */
  CUpti_ActivityKind kind;

  /**
   * For devices with compute capability 7.0+ cacheConfig values are not updated
   * in case field isSharedMemoryCarveoutRequested is set
   */
  union {
    uint8_t both;
    struct {
      /**
       * The cache configuration requested by the kernel. The value is one
       * of the CUfunc_cache enumeration values from cuda.h.
       */
      uint8_t requested:4;

      /**
       * The cache configuration used for the kernel. The value is one of
       * the CUfunc_cache enumeration values from cuda.h.
       */
      uint8_t executed:4;
    } config;
  } cacheConfig;

  /**
   * The shared memory configuration used for the kernel. The value is one of
   * the CUsharedconfig enumeration values from cuda.h.
   */
  uint8_t sharedMemoryConfig;

  /**
   * The number of registers required for each thread executing the
   * kernel.
   */
  uint16_t registersPerThread;

  /**
   * The partitioned global caching requested for the kernel. Partitioned
   * global caching is required to enable caching on certain chips, such as
   * devices with compute capability 5.2.
   */
  CUpti_ActivityPartitionedGlobalCacheConfig partitionedGlobalCacheRequested;

  /**
   * The partitioned global caching executed for the kernel. Partitioned
   * global caching is required to enable caching on certain chips, such as
   * devices with compute capability 5.2. Partitioned global caching can be
   * automatically disabled if the occupancy requirement of the launch cannot
   * support caching.
   */
  CUpti_ActivityPartitionedGlobalCacheConfig partitionedGlobalCacheExecuted;

  /**
   * The start timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t start;

  /**
   * The end timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t end;

  /**
   * The completed timestamp for the kernel execution, in ns.  It
   * represents the completion of all it's child kernels and the
   * kernel itself. A value of CUPTI_TIMESTAMP_UNKNOWN indicates that
   * the completion time is unknown.
   */
  uint64_t completed;

  /**
   * The ID of the device where the kernel is executing.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the kernel is executing.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the kernel is executing.
   */
  uint32_t streamId;

  /**
   * The X-dimension grid size for the kernel.
   */
  int32_t gridX;

  /**
   * The Y-dimension grid size for the kernel.
   */
  int32_t gridY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t gridZ;

  /**
   * The X-dimension block size for the kernel.
   */
  int32_t blockX;

  /**
   * The Y-dimension block size for the kernel.
   */
  int32_t blockY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t blockZ;

  /**
   * The static shared memory allocated for the kernel, in bytes.
   */
  int32_t staticSharedMemory;

  /**
   * The dynamic shared memory reserved for the kernel, in bytes.
   */
  int32_t dynamicSharedMemory;

  /**
   * The amount of local memory reserved for each thread, in bytes.
   */
  uint32_t localMemoryPerThread;

  /**
   * The total amount of local memory reserved for the kernel, in
   * bytes (deprecated in CUDA 11.8).
   * Refer field localMemoryTotal_v2
   */
  uint32_t localMemoryTotal;

  /**
   * The correlation ID of the kernel. Each kernel execution is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver or runtime API activity record that
   * launched the kernel.
   */
  uint32_t correlationId;

  /**
   * The grid ID of the kernel. Each kernel is assigned a unique
   * grid ID at runtime.
   */
  int64_t gridId;

  /**
   * The name of the kernel. This name is shared across all activity
   * records representing the same kernel, and so should not be
   * modified.
   */
  const char *name;

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;

  /**
   * The timestamp when the kernel is queued up in the command buffer, in ns.
   * A value of CUPTI_TIMESTAMP_UNKNOWN indicates that the queued time
   * could not be collected for the kernel. This timestamp is not collected
   * by default. Use API \ref cuptiActivityEnableLatencyTimestamps() to
   * enable collection.
   *
   * Command buffer is a buffer written by CUDA driver to send commands
   * like kernel launch, memory copy etc to the GPU. All launches of CUDA
   * kernels are asynchronous with respect to the host, the host requests
   * the launch by writing commands into the command buffer, then returns
   * without checking the GPU's progress.
   */
  uint64_t queued;

  /**
   * The timestamp when the command buffer containing the kernel launch
   * is submitted to the GPU, in ns. A value of CUPTI_TIMESTAMP_UNKNOWN
   * indicates that the submitted time could not be collected for the kernel.
   * This timestamp is not collected by default. Use API \ref
   * cuptiActivityEnableLatencyTimestamps() to enable collection.
   */
  uint64_t submitted;

  /**
   * The indicates if the kernel was executed via a regular launch or via a
   * single/multi device cooperative launch. \see CUpti_ActivityLaunchType
   */
  uint8_t launchType;

  /**
   * This indicates if CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT was
   * updated for the kernel launch
   */
  uint8_t isSharedMemoryCarveoutRequested;

  /**
   * Shared memory carveout value requested for the function in percentage of
   * the total resource. The value will be updated only if field
   * isSharedMemoryCarveoutRequested is set.
   */
  uint8_t sharedMemoryCarveoutRequested;

  /**
   * Undefined. Reserved for internal use.
   */
  uint8_t padding;

 /**
  * Shared memory size set by the driver.
  */
  uint32_t sharedMemoryExecuted;

  /**
   * The unique ID of the graph node that launched this kernel through graph launch APIs.
   * This field will be 0 if the kernel is not launched through graph launch APIs.
   */
  uint64_t graphNodeId;

  /**
   * The shared memory limit config for the kernel. This field shows whether user has opted for a
   * higher per block limit of dynamic shared memory.
   */
  CUpti_FuncShmemLimitConfig shmemLimitConfig;

  /**
   * The unique ID of the graph that launched this kernel through graph launch APIs.
   * This field will be 0 if the kernel is not launched through graph launch APIs.
   */
  uint32_t graphId;

  /**
   * The pointer to the access policy window. The structure CUaccessPolicyWindow is
   * defined in cuda.h.
   */
  CUaccessPolicyWindow *pAccessPolicyWindow;

  /**
   * The ID of the HW channel on which the kernel is launched.
   */
  uint32_t channelID;

  /**
   * The type of the channel
   */
  CUpti_ChannelType channelType;

  /**
   * The X-dimension cluster size for the kernel.
   * Field is valid for devices with compute capability 9.0 and higher
   */
  uint32_t clusterX;

  /**
   * The Y-dimension cluster size for the kernel.
   * Field is valid for devices with compute capability 9.0 and higher
   */
  uint32_t clusterY;

  /**
   * The Z-dimension cluster size for the kernel.
   * Field is valid for devices with compute capability 9.0 and higher
   */
  uint32_t clusterZ;

  /**
   * The cluster scheduling policy for the kernel. Refer CUclusterSchedulingPolicy
   * Field is valid for devices with compute capability 9.0 and higher
   */
  uint32_t clusterSchedulingPolicy;

  /**
   * The total amount of local memory reserved for the kernel, in
   * bytes.
   */
  uint64_t localMemoryTotal_v2;
} CUpti_ActivityKernel8;

/**
 * \brief The activity record for memory copies. (deprecated)
 *
 * This activity record represents a memory copy
 * (CUPTI_ACTIVITY_KIND_MEMCPY).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_MEMCPY.
   */
  CUpti_ActivityKind kind;

  /**
   * The kind of the memory copy, stored as a byte to reduce record
   * size. \see CUpti_ActivityMemcpyKind
   */
  uint8_t copyKind;

  /**
   * The source memory kind read by the memory copy, stored as a byte
   * to reduce record size. \see CUpti_ActivityMemoryKind
   */
  uint8_t srcKind;

  /**
   * The destination memory kind read by the memory copy, stored as a
   * byte to reduce record size. \see CUpti_ActivityMemoryKind
   */
  uint8_t dstKind;

  /**
   * The flags associated with the memory copy. \see CUpti_ActivityFlag
   */
  uint8_t flags;

  /**
   * The number of bytes transferred by the memory copy.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t start;

  /**
   * The end timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t end;

  /**
   * The ID of the device where the memory copy is occurring.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the memory copy is occurring.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the memory copy is occurring.
   */
  uint32_t streamId;

  /**
   * The correlation ID of the memory copy. Each memory copy is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver API activity record that launched
   * the memory copy.
   */
  uint32_t correlationId;

  /**
   * The runtime correlation ID of the memory copy. Each memory copy
   * is assigned a unique runtime correlation ID that is identical to
   * the correlation ID in the runtime API activity record that
   * launched the memory copy.
   */
  uint32_t runtimeCorrelationId;

#ifdef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;
} CUpti_ActivityMemcpy;

/**
 * \brief The activity record for memory copies. (deprecated in CUDA 11.1)
 *
 * This activity record represents a memory copy
 * (CUPTI_ACTIVITY_KIND_MEMCPY).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_MEMCPY.
   */
  CUpti_ActivityKind kind;

  /**
   * The kind of the memory copy, stored as a byte to reduce record
   * size. \see CUpti_ActivityMemcpyKind
   */
  uint8_t copyKind;

  /**
   * The source memory kind read by the memory copy, stored as a byte
   * to reduce record size. \see CUpti_ActivityMemoryKind
   */
  uint8_t srcKind;

  /**
   * The destination memory kind read by the memory copy, stored as a
   * byte to reduce record size. \see CUpti_ActivityMemoryKind
   */
  uint8_t dstKind;

  /**
   * The flags associated with the memory copy. \see CUpti_ActivityFlag
   */
  uint8_t flags;

  /**
   * The number of bytes transferred by the memory copy.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t start;

  /**
   * The end timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t end;

  /**
   * The ID of the device where the memory copy is occurring.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the memory copy is occurring.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the memory copy is occurring.
   */
  uint32_t streamId;

  /**
   * The correlation ID of the memory copy. Each memory copy is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver API activity record that launched
   * the memory copy.
   */
  uint32_t correlationId;

  /**
   * The runtime correlation ID of the memory copy. Each memory copy
   * is assigned a unique runtime correlation ID that is identical to
   * the correlation ID in the runtime API activity record that
   * launched the memory copy.
   */
  uint32_t runtimeCorrelationId;

#ifdef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;

  /**
   * The unique ID of the graph node that executed this memcpy through graph launch.
   * This field will be 0 if the memcpy is not done through graph launch.
   */
  uint64_t graphNodeId;
} CUpti_ActivityMemcpy3;

/**
 * \brief The activity record for memory copies. (deprecated in CUDA 11.6)
 *
 * This activity record represents a memory copy
 * (CUPTI_ACTIVITY_KIND_MEMCPY).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_MEMCPY.
   */
  CUpti_ActivityKind kind;

  /**
   * The kind of the memory copy, stored as a byte to reduce record
   * size. \see CUpti_ActivityMemcpyKind
   */
  uint8_t copyKind;

  /**
   * The source memory kind read by the memory copy, stored as a byte
   * to reduce record size. \see CUpti_ActivityMemoryKind
   */
  uint8_t srcKind;

  /**
   * The destination memory kind read by the memory copy, stored as a
   * byte to reduce record size. \see CUpti_ActivityMemoryKind
   */
  uint8_t dstKind;

  /**
   * The flags associated with the memory copy. \see CUpti_ActivityFlag
   */
  uint8_t flags;

  /**
   * The number of bytes transferred by the memory copy.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t start;

  /**
   * The end timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t end;

  /**
   * The ID of the device where the memory copy is occurring.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the memory copy is occurring.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the memory copy is occurring.
   */
  uint32_t streamId;

  /**
   * The correlation ID of the memory copy. Each memory copy is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver API activity record that launched
   * the memory copy.
   */
  uint32_t correlationId;

  /**
   * The runtime correlation ID of the memory copy. Each memory copy
   * is assigned a unique runtime correlation ID that is identical to
   * the correlation ID in the runtime API activity record that
   * launched the memory copy.
   */
  uint32_t runtimeCorrelationId;

#ifdef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;

  /**
   * The unique ID of the graph node that executed this memcpy through graph launch.
   * This field will be 0 if the memcpy is not done through graph launch.
   */
  uint64_t graphNodeId;

  /**
   * The unique ID of the graph that executed this memcpy through graph launch.
   * This field will be 0 if the memcpy is not done through graph launch.
   */
  uint32_t graphId;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t padding;
} CUpti_ActivityMemcpy4;

/**
 * \brief The activity record for peer-to-peer memory copies.
 *
 * This activity record represents a peer-to-peer memory copy
 * (CUPTI_ACTIVITY_KIND_MEMCPY2) but is no longer generated
 * by CUPTI. Peer-to-peer memory copy activities are now reported using the
 * CUpti_ActivityMemcpyPtoP2 activity record..
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_MEMCPY2.
   */
  CUpti_ActivityKind kind;

  /**
   * The kind of the memory copy, stored as a byte to reduce record
   * size.  \see CUpti_ActivityMemcpyKind
   */
  uint8_t copyKind;

  /**
   * The source memory kind read by the memory copy, stored as a byte
   * to reduce record size.  \see CUpti_ActivityMemoryKind
   */
  uint8_t srcKind;

  /**
   * The destination memory kind read by the memory copy, stored as a
   * byte to reduce record size.  \see CUpti_ActivityMemoryKind
   */
  uint8_t dstKind;

  /**
   * The flags associated with the memory copy. \see
   * CUpti_ActivityFlag
   */
  uint8_t flags;

  /**
   * The number of bytes transferred by the memory copy.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t start;

  /**
   * The end timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t end;

  /**
  * The ID of the device where the memory copy is occurring.
  */
  uint32_t deviceId;

  /**
   * The ID of the context where the memory copy is occurring.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the memory copy is occurring.
   */
  uint32_t streamId;

  /**
   * The ID of the device where memory is being copied from.
   */
  uint32_t srcDeviceId;

  /**
   * The ID of the context owning the memory being copied from.
   */
  uint32_t srcContextId;

  /**
   * The ID of the device where memory is being copied to.
   */
  uint32_t dstDeviceId;

  /**
   * The ID of the context owning the memory being copied to.
   */
  uint32_t dstContextId;

  /**
   * The correlation ID of the memory copy. Each memory copy is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver and runtime API activity record that
   * launched the memory copy.
   */
  uint32_t correlationId;

#ifndef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;
} CUpti_ActivityMemcpyPtoP;

typedef CUpti_ActivityMemcpyPtoP CUpti_ActivityMemcpy2;

/**
 * \brief The activity record for peer-to-peer memory copies.
 * (deprecated in CUDA 11.1)
 *
 * This activity record represents a peer-to-peer memory copy
 * (CUPTI_ACTIVITY_KIND_MEMCPY2).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_MEMCPY2.
   */
  CUpti_ActivityKind kind;

  /**
   * The kind of the memory copy, stored as a byte to reduce record
   * size.  \see CUpti_ActivityMemcpyKind
   */
  uint8_t copyKind;

  /**
   * The source memory kind read by the memory copy, stored as a byte
   * to reduce record size.  \see CUpti_ActivityMemoryKind
   */
  uint8_t srcKind;

  /**
   * The destination memory kind read by the memory copy, stored as a
   * byte to reduce record size.  \see CUpti_ActivityMemoryKind
   */
  uint8_t dstKind;

  /**
   * The flags associated with the memory copy. \see
   * CUpti_ActivityFlag
   */
  uint8_t flags;

  /**
   * The number of bytes transferred by the memory copy.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t start;

  /**
   * The end timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t end;

  /**
  * The ID of the device where the memory copy is occurring.
  */
  uint32_t deviceId;

  /**
   * The ID of the context where the memory copy is occurring.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the memory copy is occurring.
   */
  uint32_t streamId;

  /**
   * The ID of the device where memory is being copied from.
   */
  uint32_t srcDeviceId;

  /**
   * The ID of the context owning the memory being copied from.
   */
  uint32_t srcContextId;

  /**
   * The ID of the device where memory is being copied to.
   */
  uint32_t dstDeviceId;

  /**
   * The ID of the context owning the memory being copied to.
   */
  uint32_t dstContextId;

  /**
   * The correlation ID of the memory copy. Each memory copy is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver and runtime API activity record that
   * launched the memory copy.
   */
  uint32_t correlationId;

#ifndef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;

  /**
   * The unique ID of the graph node that executed the memcpy through graph launch.
   * This field will be 0 if memcpy is not done using graph launch.
   */
  uint64_t graphNodeId;
} CUpti_ActivityMemcpyPtoP2;

/**
 * \brief The activity record for peer-to-peer memory copies.
 * (deprecated in CUDA 11.6)
 *
 * This activity record represents a peer-to-peer memory copy
 * (CUPTI_ACTIVITY_KIND_MEMCPY2).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_MEMCPY2.
   */
  CUpti_ActivityKind kind;

  /**
   * The kind of the memory copy, stored as a byte to reduce record
   * size.  \see CUpti_ActivityMemcpyKind
   */
  uint8_t copyKind;

  /**
   * The source memory kind read by the memory copy, stored as a byte
   * to reduce record size.  \see CUpti_ActivityMemoryKind
   */
  uint8_t srcKind;

  /**
   * The destination memory kind read by the memory copy, stored as a
   * byte to reduce record size.  \see CUpti_ActivityMemoryKind
   */
  uint8_t dstKind;

  /**
   * The flags associated with the memory copy. \see
   * CUpti_ActivityFlag
   */
  uint8_t flags;

  /**
   * The number of bytes transferred by the memory copy.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t start;

  /**
   * The end timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t end;

  /**
  * The ID of the device where the memory copy is occurring.
  */
  uint32_t deviceId;

  /**
   * The ID of the context where the memory copy is occurring.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the memory copy is occurring.
   */
  uint32_t streamId;

  /**
   * The ID of the device where memory is being copied from.
   */
  uint32_t srcDeviceId;

  /**
   * The ID of the context owning the memory being copied from.
   */
  uint32_t srcContextId;

  /**
   * The ID of the device where memory is being copied to.
   */
  uint32_t dstDeviceId;

  /**
   * The ID of the context owning the memory being copied to.
   */
  uint32_t dstContextId;

  /**
   * The correlation ID of the memory copy. Each memory copy is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver and runtime API activity record that
   * launched the memory copy.
   */
  uint32_t correlationId;

#ifndef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;

  /**
   * The unique ID of the graph node that executed the memcpy through graph launch.
   * This field will be 0 if memcpy is not done using graph launch.
   */
  uint64_t graphNodeId;

  /**
   * The unique ID of the graph that executed this memcpy through graph launch.
   * This field will be 0 if the memcpy is not done through graph launch.
   */
  uint32_t graphId;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t padding;
} CUpti_ActivityMemcpyPtoP3;

/**
 * \brief The activity record for memset. (deprecated)
 *
 * This activity record represents a memory set operation
 * (CUPTI_ACTIVITY_KIND_MEMSET).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_MEMSET.
   */
  CUpti_ActivityKind kind;

  /**
   * The value being assigned to memory by the memory set.
   */
  uint32_t value;

  /**
   * The number of bytes being set by the memory set.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory set, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory set.
   */
  uint64_t start;

  /**
   * The end timestamp for the memory set, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory set.
   */
  uint64_t end;

  /**
   * The ID of the device where the memory set is occurring.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the memory set is occurring.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the memory set is occurring.
   */
  uint32_t streamId;

  /**
   * The correlation ID of the memory set. Each memory set is assigned
   * a unique correlation ID that is identical to the correlation ID
   * in the driver API activity record that launched the memory set.
   */
  uint32_t correlationId;

  /**
   * The flags associated with the memset. \see CUpti_ActivityFlag
   */
  uint16_t flags;

  /**
   * The memory kind of the memory set \see CUpti_ActivityMemoryKind
   */
  uint16_t memoryKind;

#ifdef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;
} CUpti_ActivityMemset;

/**
 * \brief The activity record for memset. (deprecated in CUDA 11.1)
 *
 * This activity record represents a memory set operation
 * (CUPTI_ACTIVITY_KIND_MEMSET).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_MEMSET.
   */
  CUpti_ActivityKind kind;

  /**
   * The value being assigned to memory by the memory set.
   */
  uint32_t value;

  /**
   * The number of bytes being set by the memory set.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory set, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory set.
   */
  uint64_t start;

  /**
   * The end timestamp for the memory set, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory set.
   */
  uint64_t end;

  /**
   * The ID of the device where the memory set is occurring.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the memory set is occurring.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the memory set is occurring.
   */
  uint32_t streamId;

  /**
   * The correlation ID of the memory set. Each memory set is assigned
   * a unique correlation ID that is identical to the correlation ID
   * in the driver API activity record that launched the memory set.
   */
  uint32_t correlationId;

  /**
   * The flags associated with the memset. \see CUpti_ActivityFlag
   */
  uint16_t flags;

  /**
   * The memory kind of the memory set \see CUpti_ActivityMemoryKind
   */
  uint16_t memoryKind;

#ifdef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;

  /**
   * The unique ID of the graph node that executed this memset through graph launch.
   * This field will be 0 if the memset is not executed through graph launch.
   */
  uint64_t graphNodeId;
} CUpti_ActivityMemset2;

/**
 * \brief The activity record for memset. (deprecated in CUDA 11.6)
 *
 * This activity record represents a memory set operation
 * (CUPTI_ACTIVITY_KIND_MEMSET).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_MEMSET.
   */
  CUpti_ActivityKind kind;

  /**
   * The value being assigned to memory by the memory set.
   */
  uint32_t value;

  /**
   * The number of bytes being set by the memory set.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory set, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory set.
   */
  uint64_t start;

  /**
   * The end timestamp for the memory set, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory set.
   */
  uint64_t end;

  /**
   * The ID of the device where the memory set is occurring.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the memory set is occurring.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the memory set is occurring.
   */
  uint32_t streamId;

  /**
   * The correlation ID of the memory set. Each memory set is assigned
   * a unique correlation ID that is identical to the correlation ID
   * in the driver API activity record that launched the memory set.
   */
  uint32_t correlationId;

  /**
   * The flags associated with the memset. \see CUpti_ActivityFlag
   */
  uint16_t flags;

  /**
   * The memory kind of the memory set \see CUpti_ActivityMemoryKind
   */
  uint16_t memoryKind;

#ifdef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;

  /**
   * The unique ID of the graph node that executed this memset through graph launch.
   * This field will be 0 if the memset is not executed through graph launch.
   */
  uint64_t graphNodeId;

  /**
   * The unique ID of the graph that executed this memset through graph launch.
   * This field will be 0 if the memset is not executed through graph launch.
   */
  uint32_t graphId;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t padding;
} CUpti_ActivityMemset3;

/**
 * \brief The activity record for memory.
 *
 * This activity record represents a memory allocation and free operation
 * (CUPTI_ACTIVITY_KIND_MEMORY2).
 * This activity record provides separate records for memory allocation and
 * memory release operations.
 * This allows to correlate the corresponding driver and runtime API
 * activity record with the memory operation.
 *
 * Note: This activity record is an upgrade over \ref CUpti_ActivityMemory
 * enabled using the kind \ref CUPTI_ACTIVITY_KIND_MEMORY.
 * \ref CUpti_ActivityMemory provides a single record for the memory
 * allocation and memory release operations.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_MEMORY2
   */
  CUpti_ActivityKind kind;

  /**
   * The memory operation requested by the user, \ref CUpti_ActivityMemoryOperationType.
   */
  CUpti_ActivityMemoryOperationType memoryOperationType;

  /**
   * The memory kind requested by the user, \ref CUpti_ActivityMemoryKind.
   */
  CUpti_ActivityMemoryKind memoryKind;

  /**
   * The correlation ID of the memory operation. Each memory operation is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver and runtime API activity record that
   * launched the memory operation.
   */
  uint32_t correlationId;

  /**
   * The virtual address of the allocation.
   */
  uint64_t address;

  /**
   * The number of bytes of memory allocated.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory operation, in ns.
   */
  uint64_t timestamp;

  /**
   * The program counter of the memory operation.
   */
  uint64_t PC;

  /**
   * The ID of the process to which this record belongs to.
   */
  uint32_t processId;

  /**
   * The ID of the device where the memory operation is taking place.
   */
  uint32_t deviceId;

  /**
   * The ID of the context. If context is NULL, \p contextId is set to CUPTI_INVALID_CONTEXT_ID.
   */
  uint32_t contextId;

  /**
   * The ID of the stream. If memory operation is not async, \p streamId is set to CUPTI_INVALID_STREAM_ID.
   */
  uint32_t streamId;

  /**
   * Variable name. This name is shared across all activity
   * records representing the same symbol, and so should not be
   * modified.
   */
  const char* name;

  /**
   * \p isAsync is set if memory operation happens through async memory APIs.
   */
  uint32_t isAsync;

#ifdef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad1;
#endif

  /**
   * The memory pool configuration used for the memory operations.
   */
  struct {
    /**
     * The type of the memory pool, \ref CUpti_ActivityMemoryPoolType
     */
    CUpti_ActivityMemoryPoolType memoryPoolType;

#ifdef CUPTILP64
    /**
     * Undefined. Reserved for internal use.
     */
    uint32_t pad2;
#endif

    /**
     * The base address of the memory pool.
     */
    uint64_t address;

    /**
     * The release threshold of the memory pool in bytes. \p releaseThreshold is
     * valid for CUPTI_ACTIVITY_MEMORY_POOL_TYPE_LOCAL, \ref CUpti_ActivityMemoryPoolType.
     */
    uint64_t releaseThreshold;

   /**
   * The size of the memory pool in bytes and the processID of the memory pool.
   * \p size is valid if \p memoryPoolType is
   * CUPTI_ACTIVITY_MEMORY_POOL_TYPE_LOCAL, \ref CUpti_ActivityMemoryPoolType.
   * \p processId is valid if \p memoryPoolType is
   * CUPTI_ACTIVITY_MEMORY_POOL_TYPE_IMPORTED, \ref CUpti_ActivityMemoryPoolType.
   */
   union {
      uint64_t size;
      uint64_t processId;
    } pool;
  } memoryPoolConfig;

} CUpti_ActivityMemory2;

/**
 * \brief The activity record for memory.
 *
 * This activity record represents a memory allocation and free operation
 * (CUPTI_ACTIVITY_KIND_MEMORY2).
 * This activity record provides separate records for memory allocation and
 * memory release operations.
 * This allows to correlate the corresponding driver and runtime API
 * activity record with the memory operation.
 *
 * Note: This activity record is an upgrade over \ref CUpti_ActivityMemory2
 * enabled using the kind \ref CUPTI_ACTIVITY_KIND_MEMORY.
 * \ref CUpti_ActivityMemory provides a single record for the memory
 * allocation and memory release operations.
 */

typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_MEMORY2
   */
  CUpti_ActivityKind kind;

  /**
   * The memory operation requested by the user, \ref CUpti_ActivityMemoryOperationType.
   */
  CUpti_ActivityMemoryOperationType memoryOperationType;

  /**
   * The memory kind requested by the user, \ref CUpti_ActivityMemoryKind.
   */
  CUpti_ActivityMemoryKind memoryKind;

  /**
   * The correlation ID of the memory operation. Each memory operation is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver and runtime API activity record that
   * launched the memory operation.
   */
  uint32_t correlationId;

  /**
   * The virtual address of the allocation.
   */
  uint64_t address;

  /**
   * The number of bytes of memory allocated.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory operation, in ns.
   */
  uint64_t timestamp;

  /**
   * The program counter of the memory operation.
   */
  uint64_t PC;

  /**
   * The ID of the process to which this record belongs to.
   */
  uint32_t processId;

  /**
   * The ID of the device where the memory operation is taking place.
   */
  uint32_t deviceId;

  /**
   * The ID of the context. If context is NULL, \p contextId is set to CUPTI_INVALID_CONTEXT_ID.
   */
  uint32_t contextId;

  /**
   * The ID of the stream. If memory operation is not async, \p streamId is set to CUPTI_INVALID_STREAM_ID.
   */
  uint32_t streamId;

  /**
   * Variable name. This name is shared across all activity
   * records representing the same symbol, and so should not be
   * modified.
   */
  const char* name;

  /**
   * \p isAsync is set if memory operation happens through async memory APIs.
   */
  uint32_t isAsync;

#ifdef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad1;
#endif

  /**
   * The memory pool configuration used for the memory operations.
   */
  struct PACKED_ALIGNMENT {
    /**
     * The type of the memory pool, \ref CUpti_ActivityMemoryPoolType
     */
    CUpti_ActivityMemoryPoolType memoryPoolType;

#ifdef CUPTILP64
    /**
     * Undefined. Reserved for internal use.
     */
    uint32_t pad2;
#endif

    /**
     * The base address of the memory pool.
     */
    uint64_t address;

    /**
     * The release threshold of the memory pool in bytes. \p releaseThreshold is
     * valid for CUPTI_ACTIVITY_MEMORY_POOL_TYPE_LOCAL, \ref CUpti_ActivityMemoryPoolType.
     */
    uint64_t releaseThreshold;

    /**
     * The size of memory pool in bytes and the processId of the memory pools
     * \p size is valid if \p memoryPoolType is
     * CUPTI_ACTIVITY_MEMORY_POOL_TYPE_LOCAL, \ref CUpti_ActivityMemoryPoolType.
     * \p processId is valid if \p memoryPoolType is
     * CUPTI_ACTIVITY_MEMORY_POOL_TYPE_IMPORTED, \ref CUpti_ActivityMemoryPoolType
     */
    union {
      uint64_t size;
      uint64_t processId;
    } pool;

    /**
     * The utilized size of the memory pool. \p utilizedSize is
     * valid for CUPTI_ACTIVITY_MEMORY_POOL_TYPE_LOCAL, \ref CUpti_ActivityMemoryPoolType.
     */
    uint64_t utilizedSize;
  } memoryPoolConfig;

} CUpti_ActivityMemory3;

/**
 * \brief The activity record for memory pool.
 *
 * This activity record represents a memory pool creation, destruction and
 * trimming (CUPTI_ACTIVITY_KIND_MEMORY_POOL).
 * This activity record provides separate records for memory pool creation,
 * destruction and trimming operations.
 * This allows to correlate the corresponding driver and runtime API
 * activity record with the memory pool operation.
 *
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_MEMORY_POOL
   */
  CUpti_ActivityKind kind;

  /**
   * The memory operation requested by the user, \ref CUpti_ActivityMemoryPoolOperationType.
   */
  CUpti_ActivityMemoryPoolOperationType memoryPoolOperationType;

  /**
   * The type of the memory pool, \ref CUpti_ActivityMemoryPoolType
   */
  CUpti_ActivityMemoryPoolType memoryPoolType;

  /**
   * The correlation ID of the memory pool operation. Each memory pool
   * operation is assigned a unique correlation ID that is identical to the
   * correlation ID in the driver and runtime API activity record that
   * launched the memory operation.
   */
  uint32_t correlationId;

  /**
   * The ID of the process to which this record belongs to.
   */
  uint32_t processId;

  /**
   * The ID of the device where the memory pool is created.
   */
  uint32_t deviceId;

  /**
   * The minimum bytes to keep of the memory pool. \p minBytesToKeep is
   * valid for CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_TRIMMED,
   * \ref CUpti_ActivityMemoryPoolOperationType
   */
  size_t minBytesToKeep;

#ifndef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * The virtual address of the allocation.
   */
  uint64_t address;

  /**
   * The size of the memory pool operation in bytes. \p size is
   * valid for CUPTI_ACTIVITY_MEMORY_POOL_TYPE_LOCAL, \ref CUpti_ActivityMemoryPoolType.
   */
  uint64_t size;

  /**
   * The release threshold of the memory pool. \p releaseThreshold is
   * valid for CUPTI_ACTIVITY_MEMORY_POOL_TYPE_LOCAL, \ref CUpti_ActivityMemoryPoolType.
   */
  uint64_t releaseThreshold;

  /**
   * The start timestamp for the memory operation, in ns.
   */
  uint64_t timestamp;
} CUpti_ActivityMemoryPool;

/**
 * \brief The activity record providing a marker which is an
 * instantaneous point in time. (deprecated in CUDA 8.0)
 *
 * The marker is specified with a descriptive name and unique id
 * (CUPTI_ACTIVITY_KIND_MARKER).
 * Marker activity is now reported using the
 * CUpti_ActivityMarker2 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_MARKER.
   */
  CUpti_ActivityKind kind;

  /**
   * The flags associated with the marker. \see CUpti_ActivityFlag
   */
  CUpti_ActivityFlag flags;

  /**
   * The timestamp for the marker, in ns. A value of 0 indicates that
   * timestamp information could not be collected for the marker.
   */
  uint64_t timestamp;

  /**
   * The marker ID.
   */
  uint32_t id;

  /**
   * The kind of activity object associated with this marker.
   */
  CUpti_ActivityObjectKind objectKind;

  /**
   * The identifier for the activity object associated with this
   * marker. 'objectKind' indicates which ID is valid for this record.
   */
  CUpti_ActivityObjectKindId objectId;

#ifdef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * The marker name for an instantaneous or start marker. This will
   * be NULL for an end marker.
   */
  const char *name;

} CUpti_ActivityMarker;

/**
 * \brief The activity record for source-level global
 * access. (deprecated)
 *
 * This activity records the locations of the global
 * accesses in the source (CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS).
 * Global access activities are now reported using the
 * CUpti_ActivityGlobalAccess3 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS.
   */
  CUpti_ActivityKind kind;

  /**
   * The properties of this global access.
   */
  CUpti_ActivityFlag flags;

  /**
   * The ID for source locator.
   */
  uint32_t sourceLocatorId;

  /**
   * The correlation ID of the kernel to which this result is associated.
   */
  uint32_t correlationId;

  /**
   * The pc offset for the access.
   */
  uint32_t pcOffset;

  /**
   * The number of times this instruction was executed per warp. It will be incremented
   * when at least one of thread among warp is active with predicate and condition code
   * evaluating to true.
   */
  uint32_t executed;

  /**
   * This increments each time when this instruction is executed by number
   * of threads that executed this instruction with predicate and condition code evaluating to true.
   */
  uint64_t threadsExecuted;

  /**
   * The total number of 32 bytes transactions to L2 cache generated by this access
   */
  uint64_t l2_transactions;
} CUpti_ActivityGlobalAccess;

/**
 * \brief The activity record for source-level global
 * access. (deprecated in CUDA 9.0)
 *
 * This activity records the locations of the global
 * accesses in the source (CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS).
 * Global access activities are now reported using the
 * CUpti_ActivityGlobalAccess3 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS.
   */
  CUpti_ActivityKind kind;

  /**
   * The properties of this global access.
   */
  CUpti_ActivityFlag flags;

  /**
   * The ID for source locator.
   */
  uint32_t sourceLocatorId;

  /**
   * The correlation ID of the kernel to which this result is associated.
   */
  uint32_t correlationId;

  /**
  * Correlation ID with global/device function name
  */
  uint32_t functionId;

  /**
   * The pc offset for the access.
   */
  uint32_t pcOffset;

  /**
   * This increments each time when this instruction is executed by number
   * of threads that executed this instruction with predicate and condition code evaluating to true.
   */
  uint64_t threadsExecuted;

  /**
   * The total number of 32 bytes transactions to L2 cache generated by this access
   */
  uint64_t l2_transactions;

  /**
   * The minimum number of L2 transactions possible based on the access pattern.
   */
  uint64_t theoreticalL2Transactions;

  /**
   * The number of times this instruction was executed per warp. It will be incremented
   * when at least one of thread among warp is active with predicate and condition code
   * evaluating to true.
   */
  uint32_t executed;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
} CUpti_ActivityGlobalAccess2;

/**
 * \brief The activity record for source level result
 * branch. (deprecated)
 *
 * This activity record the locations of the branches in the
 * source (CUPTI_ACTIVITY_KIND_BRANCH).
 * Branch activities are now reported using the
 * CUpti_ActivityBranch2 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_BRANCH.
   */
  CUpti_ActivityKind kind;

  /**
   * The ID for source locator.
   */
  uint32_t sourceLocatorId;

  /**
   * The correlation ID of the kernel to which this result is associated.
   */
  uint32_t correlationId;

  /**
   * The pc offset for the branch.
   */
  uint32_t pcOffset;

  /**
   * The number of times this instruction was executed per warp. It will be incremented
   * regardless of predicate or condition code.
   */
  uint32_t executed;

  /**
   * Number of times this branch diverged
   */
  uint32_t diverged;

  /**
   * This increments each time when this instruction is executed by number
   * of threads that executed this instruction
   */
  uint64_t threadsExecuted;
} CUpti_ActivityBranch;

/**
 * \brief The activity record for PC sampling. (deprecated in CUDA 8.0)
 *
 * This activity records information obtained by sampling PC
 * (CUPTI_ACTIVITY_KIND_PC_SAMPLING).
 * PC sampling activities are now reported using the
 * CUpti_ActivityPCSampling2 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_PC_SAMPLING.
   */
  CUpti_ActivityKind kind;

  /**
   * The properties of this instruction.
   */
  CUpti_ActivityFlag flags;

  /**
   * The ID for source locator.
   */
  uint32_t sourceLocatorId;

  /**
   * The correlation ID of the kernel to which this result is associated.
   */
  uint32_t correlationId;

  /**
  * Correlation ID with global/device function name
  */
  uint32_t functionId;

  /**
   * The pc offset for the instruction.
   */
  uint32_t pcOffset;

  /**
   * Number of times the PC was sampled with the stallReason in the record.
   * The same PC can be sampled with different stall reasons.
   */
  uint32_t samples;

  /**
   * Current stall reason. Includes one of the reasons from
   * \ref CUpti_ActivityPCSamplingStallReason
   */
  CUpti_ActivityPCSamplingStallReason stallReason;
} CUpti_ActivityPCSampling;

/**
 * \brief The activity record for PC sampling. (deprecated in CUDA 9.0)
 *
 * This activity records information obtained by sampling PC
 * (CUPTI_ACTIVITY_KIND_PC_SAMPLING).
 * PC sampling activities are now reported using the
 * CUpti_ActivityPCSampling3 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_PC_SAMPLING.
   */
  CUpti_ActivityKind kind;

  /**
   * The properties of this instruction.
   */
  CUpti_ActivityFlag flags;

  /**
   * The ID for source locator.
   */
  uint32_t sourceLocatorId;

  /**
   * The correlation ID of the kernel to which this result is associated.
   */
  uint32_t correlationId;

  /**
  * Correlation ID with global/device function name
  */
  uint32_t functionId;

  /**
   * The pc offset for the instruction.
   */
  uint32_t pcOffset;

  /**
   * Number of times the PC was sampled with the stallReason in the record.
   * These samples indicate that no instruction was issued in that cycle from
   * the warp scheduler from where the warp was sampled.
   * Field is valid for devices with compute capability 6.0 and higher
   */
  uint32_t latencySamples;

  /**
   * Number of times the PC was sampled with the stallReason in the record.
   * The same PC can be sampled with different stall reasons. The count includes
   * latencySamples.
   */
  uint32_t samples;

  /**
   * Current stall reason. Includes one of the reasons from
   * \ref CUpti_ActivityPCSamplingStallReason
   */
  CUpti_ActivityPCSamplingStallReason stallReason;

  uint32_t pad;
} CUpti_ActivityPCSampling2;

/**
 * \brief The activity record for Unified Memory counters (deprecated in CUDA 7.0)
 *
 * This activity record represents a Unified Memory counter
 * (CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER
   */
  CUpti_ActivityKind kind;

  /**
   * The Unified Memory counter kind. See \ref CUpti_ActivityUnifiedMemoryCounterKind
   */
  CUpti_ActivityUnifiedMemoryCounterKind counterKind;

  /**
   * Scope of the Unified Memory counter. See \ref CUpti_ActivityUnifiedMemoryCounterScope
   */
  CUpti_ActivityUnifiedMemoryCounterScope scope;

  /**
   * The ID of the device involved in the memory transfer operation.
   * It is not relevant if the scope of the counter is global (all devices).
   */
  uint32_t deviceId;

  /**
   * Value of the counter
   *
   */
  uint64_t value;

  /**
   * The timestamp when this sample was retrieved, in ns. A value of 0
   * indicates that timestamp information could not be collected
   */
  uint64_t timestamp;

  /**
   * The ID of the process to which this record belongs to. In case of
   * global scope, processId is undefined.
   */
  uint32_t processId;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
} CUpti_ActivityUnifiedMemoryCounter;

/**
 * \brief The activity record for Unified Memory counters (deprecated in 12.8)
 *
 * This activity record represents a Unified Memory counter
 * (CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER
   */
  CUpti_ActivityKind kind;

  /**
   * The Unified Memory counter kind
   */
  CUpti_ActivityUnifiedMemoryCounterKind counterKind;

  /**
   * Value of the counter
   * For counterKind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD,
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH,
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THREASHING and
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_REMOTE_MAP, it is the size of the
   * memory region in bytes.
   * For counterKind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT, it
   * is the number of page fault groups for the same page.
   * For counterKind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT,
   * it is the program counter for the instruction that caused fault.
   */
  uint64_t value;

  /**
   * The start timestamp of the counter, in ns.
   * For counterKind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD and
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH, timestamp is
   * captured when activity starts on GPU.
   * For counterKind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT and
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT, timestamp is
   * captured when CUDA driver started processing the fault.
   * For counterKind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING, timestamp
   * is captured when CUDA driver detected thrashing of memory region.
   * For counterKind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING,
   * timestamp is captured when throttling operation was started by CUDA driver.
   * For counterKind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_REMOTE_MAP,
   * timestamp is captured when CUDA driver has pushed all required operations
   * to the processor specified by dstId.
   */
  uint64_t start;

  /**
   * The end timestamp of the counter, in ns.
   * Ignore this field if counterKind is
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT or
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING or
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_REMOTE_MAP.
   * For counterKind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD and
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH, timestamp is
   * captured when activity finishes on GPU.
   * For counterKind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT, timestamp is
   * captured when CUDA driver queues the replay of faulting memory accesses on the GPU
   * For counterKind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING, timestamp
   * is captured when throttling operation was finished by CUDA driver
   */
  uint64_t end;

  /**
   * This is the virtual base address of the page/s being transferred. For cpu and
   * gpu faults, the virtual address for the page that faulted.
   */
  uint64_t address;

  /**
   * The ID of the source CPU/device involved in the memory transfer, page fault, thrashing,
   * throttling or remote map operation. For counterKind
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING, it is a bitwise ORing of the
   * device IDs fighting for the memory region. Ignore this field if counterKind is
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT
   */
  uint32_t srcId;

  /**
   * The ID of the destination CPU/device involved in the memory transfer or remote map
   * operation. Ignore this field if counterKind is
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT or
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT or
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING or
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING
   */
  uint32_t dstId;

  /**
   * The ID of the stream causing the transfer.
   * This value of this field is invalid.
   */
  uint32_t streamId;

  /**
   * The ID of the process to which this record belongs to.
   */
  uint32_t processId;

  /**
   * The flags associated with this record. See enums \ref CUpti_ActivityUnifiedMemoryAccessType
   * if counterKind is CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT
   * and \ref CUpti_ActivityUnifiedMemoryMigrationCause if counterKind is
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD or
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD
   * and \ref CUpti_ActivityUnifiedMemoryRemoteMapCause if counterKind is
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_REMOTE_MAP and \ref CUpti_ActivityFlag
   * if counterKind is CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING or
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING
   */
  uint32_t flags;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
} CUpti_ActivityUnifiedMemoryCounter2;

/**
* \brief NVLink information. (deprecated in CUDA 9.0)
*
* This structure gives capabilities of each logical NVLink connection between two devices,
* gpu<->gpu or gpu<->CPU which can be used to understand the topology.
* NVLink information are now reported using the
* CUpti_ActivityNvLink2 activity record.
*/
typedef struct PACKED_ALIGNMENT {
  /**
  * The activity record kind, must be CUPTI_ACTIVITY_KIND_NVLINK.
  */
  CUpti_ActivityKind kind;

  /**
  * NVLink version.
  */
  uint32_t nvlinkVersion;

  /**
  * Type of device 0 \ref CUpti_DevType
  */
  CUpti_DevType typeDev0;

  /**
  * Type of device 1 \ref CUpti_DevType
  */
  CUpti_DevType typeDev1;

  /**
  * If typeDev0 is CUPTI_DEV_TYPE_GPU, UUID for device 0. \ref CUpti_ActivityDevice5.
  * If typeDev0 is CUPTI_DEV_TYPE_NPU, struct npu for NPU.
  */
  union {
    CUuuid uuidDev;
    struct {
      /**
      * Index of the NPU. First index will always be zero.
      */
      uint32_t index;

      /**
      * Domain ID of NPU. On Linux, this can be queried using lspci.
      */
      uint32_t domainId;
    } npu;
  } idDev0;

  /**
  * If typeDev1 is CUPTI_DEV_TYPE_GPU, UUID for device 1. \ref CUpti_ActivityDevice5.
  * If typeDev1 is CUPTI_DEV_TYPE_NPU, struct npu for NPU.
  */
  union {
    CUuuid uuidDev;
    struct {
      /**
      * Index of the NPU. First index will always be zero.
      */
      uint32_t index;

      /**
      * Domain ID of NPU. On Linux, this can be queried using lspci.
      */
      uint32_t domainId;
    } npu;
  } idDev1;

  /**
  * Flag gives capabilities of the link \see CUpti_LinkFlag
  */
  uint32_t flag;

  /**
  * Number of physical NVLinks present between two devices.
  */
  uint32_t physicalNvLinkCount;

  /**
  * Port numbers for maximum 4 NVLinks connected to device 0.
  * If typeDev0 is CUPTI_DEV_TYPE_NPU, ignore this field.
  * In case of invalid/unknown port number, this field will be set
  * to value CUPTI_NVLINK_INVALID_PORT.
  * This will be used to correlate the metric values to individual
  * physical link and attribute traffic to the logical NVLink in
  * the topology.
  */
  int8_t portDev0[4];

  /**
  * Port numbers for maximum 4 NVLinks connected to device 1.
  * If typeDev1 is CUPTI_DEV_TYPE_NPU, ignore this field.
  * In case of invalid/unknown port number, this field will be set
  * to value CUPTI_NVLINK_INVALID_PORT.
  * This will be used to correlate the metric values to individual
  * physical link and attribute traffic to the logical NVLink in
  * the topology.
  */
  int8_t portDev1[4];

  /**
  * Bandwidth of NVLink in kbytes/sec
  */
  uint64_t bandwidth;
} CUpti_ActivityNvLink;

/**
* \brief NVLink information. (deprecated in CUDA 10.0)
*
* This structure gives capabilities of each logical NVLink connection between two devices,
* gpu<->gpu or gpu<->CPU which can be used to understand the topology.
* NvLink information are now reported using the
* CUpti_ActivityNvLink4 activity record.
*/
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_NVLINK.
   */
  CUpti_ActivityKind kind;

  /**
   * NvLink version.
   */
  uint32_t nvlinkVersion;

  /**
   * Type of device 0 \ref CUpti_DevType
   */
  CUpti_DevType typeDev0;

  /**
   * Type of device 1 \ref CUpti_DevType
   */
  CUpti_DevType typeDev1;

  /**
  * If typeDev0 is CUPTI_DEV_TYPE_GPU, UUID for device 0. \ref CUpti_ActivityDevice5.
  * If typeDev0 is CUPTI_DEV_TYPE_NPU, struct npu for NPU.
  */
  union {
    CUuuid uuidDev;
    struct {
      /**
       * Index of the NPU. First index will always be zero.
       */
      uint32_t index;

      /**
       * Domain ID of NPU. On Linux, this can be queried using lspci.
       */
      uint32_t domainId;
    } npu;
  } idDev0;

  /**
  * If typeDev1 is CUPTI_DEV_TYPE_GPU, UUID for device 1. \ref CUpti_ActivityDevice5.
  * If typeDev1 is CUPTI_DEV_TYPE_NPU, struct npu for NPU.
  */
  union {
    CUuuid uuidDev;
    struct {
      /**
       * Index of the NPU. First index will always be zero.
       */
      uint32_t index;

      /**
       * Domain ID of NPU. On Linux, this can be queried using lspci.
       */
      uint32_t domainId;
    } npu;
  } idDev1;

  /**
   * Flag gives capabilities of the link \see CUpti_LinkFlag
   */
  uint32_t flag;

  /**
   * Number of physical NVLinks present between two devices.
   */
  uint32_t physicalNvLinkCount;

  /**
   * Port numbers for maximum 16 NVLinks connected to device 0.
   * If typeDev0 is CUPTI_DEV_TYPE_NPU, ignore this field.
   * In case of invalid/unknown port number, this field will be set
   * to value CUPTI_NVLINK_INVALID_PORT.
   * This will be used to correlate the metric values to individual
   * physical link and attribute traffic to the logical NVLink in
   * the topology.
   */
  int8_t portDev0[CUPTI_MAX_NVLINK_PORTS];

  /**
   * Port numbers for maximum 16 NVLinks connected to device 1.
   * If typeDev1 is CUPTI_DEV_TYPE_NPU, ignore this field.
   * In case of invalid/unknown port number, this field will be set
   * to value CUPTI_NVLINK_INVALID_PORT.
   * This will be used to correlate the metric values to individual
   * physical link and attribute traffic to the logical NVLink in
   * the topology.
   */
  int8_t portDev1[CUPTI_MAX_NVLINK_PORTS];

  /**
   * Bandwidth of NVLink in kbytes/sec
   */
  uint64_t  bandwidth;
} CUpti_ActivityNvLink2;

/**
* \brief NVLink information.
*
* This structure gives capabilities of each logical NVLink connection between two devices,
* gpu<->gpu or gpu<->CPU which can be used to understand the topology.
* NvLink information are now reported using the
* CUpti_ActivityNvLink4 activity record.
*/
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_NVLINK.
   */
  CUpti_ActivityKind kind;
  /**
   * NvLink version.
   */
  uint32_t nvlinkVersion;

  /**
   * Type of device 0 \ref CUpti_DevType
   */
  CUpti_DevType typeDev0;

  /**
   * Type of device 1 \ref CUpti_DevType
   */
  CUpti_DevType typeDev1;

  /**
  * If typeDev0 is CUPTI_DEV_TYPE_GPU, UUID for device 0. \ref CUpti_ActivityDevice5.
  * If typeDev0 is CUPTI_DEV_TYPE_NPU, struct npu for NPU.
  */
  union {
    CUuuid uuidDev;
    struct {
      /**
       * Index of the NPU. First index will always be zero.
       */
      uint32_t index;

      /**
       * Domain ID of NPU. On Linux, this can be queried using lspci.
       */
      uint32_t domainId;
    } npu;
  } idDev0;

  /**
  * If typeDev1 is CUPTI_DEV_TYPE_GPU, UUID for device 1. \ref CUpti_ActivityDevice5.
  * If typeDev1 is CUPTI_DEV_TYPE_NPU, struct npu for NPU.
  */
  union {
    CUuuid uuidDev;
    struct {
      /**
       * Index of the NPU. First index will always be zero.
       */
      uint32_t index;

      /**
       * Domain ID of NPU. On Linux, this can be queried using lspci.
       */
      uint32_t domainId;
    } npu;
  } idDev1;

  /**
   * Flag gives capabilities of the link \see CUpti_LinkFlag
   */
  uint32_t flag;

  /**
   * Number of physical NVLinks present between two devices.
   */
  uint32_t physicalNvLinkCount;

  /**
   * Port numbers for maximum 16 NVLinks connected to device 0.
   * If typeDev0 is CUPTI_DEV_TYPE_NPU, ignore this field.
   * In case of invalid/unknown port number, this field will be set
   * to value CUPTI_NVLINK_INVALID_PORT.
   * This will be used to correlate the metric values to individual
   * physical link and attribute traffic to the logical NVLink in
   * the topology.
   */
  int8_t portDev0[CUPTI_MAX_NVLINK_PORTS];

  /**
   * Port numbers for maximum 16 NVLinks connected to device 1.
   * If typeDev1 is CUPTI_DEV_TYPE_NPU, ignore this field.
   * In case of invalid/unknown port number, this field will be set
   * to value CUPTI_NVLINK_INVALID_PORT.
   * This will be used to correlate the metric values to individual
   * physical link and attribute traffic to the logical NVLink in
   * the topology.
   */
  int8_t portDev1[CUPTI_MAX_NVLINK_PORTS];

  /**
   * Bandwidth of NVLink in kbytes/sec
   */
  uint64_t bandwidth;

  /**
   * NVSwitch is connected as an intermediate node.
   */
  uint8_t nvswitchConnected;

  /**
   * Undefined. reserved for internal use
   */
  uint8_t pad[7];
} CUpti_ActivityNvLink3;

/**
 * \brief The activity record for trace of graph execution.
 *
 * This activity record represents execution for a graph without giving visibility
 * about the execution of its nodes. This is intended to reduce overheads in tracing
 * each node. The activity kind is CUPTI_ACTIVITY_KIND_GRAPH_TRACE
 * Graph trace activity is now reported using CUpti_ActivityGraphTrace2 record.
 */
typedef struct {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_GRAPH_TRACE
   */
  CUpti_ActivityKind kind;

  /**
   * The correlation ID of the graph launch. Each graph launch is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver API activity record that launched
   * the graph.
   */
  uint32_t correlationId;

  /**
   * The start timestamp for the graph execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the graph.
   */
  uint64_t start;

  /**
   * The end timestamp for the graph execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the graph.
   */
  uint64_t end;

  /**
   * The ID of the device where the graph execution is occurring.
   */
  uint32_t deviceId;

  /**
   * The unique ID of the graph that is launched.
   */
  uint32_t graphId;

  /**
   * The ID of the context where the graph is being launched.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the graph is being launched.
   */
  uint32_t streamId;

  /**
   * This field is reserved for internal use
   */
  void *reserved;
} CUpti_ActivityGraphTrace;

/**
 * \brief The activity record for a context.
 *
 * This activity record represents information about a context
 * (CUPTI_ACTIVITY_KIND_CONTEXT).
 * Context activity is now reported using CUpti_ActivityContext3 record
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_CONTEXT.
   */
  CUpti_ActivityKind kind;

  /**
   * The context ID.
   */
  uint32_t contextId;

  /**
   * The device ID.
   */
  uint32_t deviceId;

  /**
   * The compute API kind. \see CUpti_ActivityComputeApiKind
   */
  uint16_t computeApiKind;

  /**
   * The ID for the NULL stream in this context
   */
  uint16_t nullStreamId;
} CUpti_ActivityContext;

/**
 * \brief The activity record for a context.
 *
 * This activity record represents information about a context
 * (CUPTI_ACTIVITY_KIND_CONTEXT).
 * Context activity is now reported using CUpti_ActivityContext3 record
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_CONTEXT.
   */
  CUpti_ActivityKind kind;

  /**
   * The context ID.
   */
  uint32_t contextId;

  /**
   * The device ID.
   */
  uint32_t deviceId;

  /**
   * The compute API kind. \see CUpti_ActivityComputeApiKind
   */
  uint16_t computeApiKind;

  /**
   * The ID for the NULL stream in this context
   */
  uint16_t nullStreamId;

  /**
   * The ID of the parent context. It would be 0 if
   * context does not have parent
   */
  uint32_t parentContextId;

  /**
   * This field indicates whether the context is a green context
   */
  uint8_t isGreenContext;

  uint8_t padding;

  /**
   * Number of multiprocessors assigned to the green context
   * Invalid if the field 'isGreenContext' is 0
   */
  uint16_t numMultiprocessors;
} CUpti_ActivityContext2;

/**
 * \brief The activity record for JIT operations.
 * This activity represents the JIT operations (compile, load, store) of a CUmodule
 * from the Compute Cache.
 * Gives the exact hashed path of where the cached module is loaded from,
 * or where the module will be stored after Just-In-Time (JIT) compilation.
 *
 * JIT activity is now reported using CUpti_ActivityJit2 record
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind must be CUPTI_ACTIVITY_KIND_JIT.
   */
  CUpti_ActivityKind kind;

  /**
    * The JIT entry type.
    */
  CUpti_ActivityJitEntryType jitEntryType;

  /**
   * The JIT operation type.
   */
  CUpti_ActivityJitOperationType jitOperationType;

  /**
   * The device ID.
   */
  uint32_t deviceId;

  /**
   * The start timestamp for the JIT operation, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the JIT operation.
   */
  uint64_t start;

  /**
   * The end timestamp for the JIT operation, in ns. A value of 0 for both
   * the start and end timestamps indicates that timestamp information
   * could not be collected for the JIT operation.
   */
  uint64_t end;

  /**
   * The correlation ID of the JIT operation to which
   * records belong to. Each JIT operation is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver or runtime API activity record that
   * launched the JIT operation.
   */
  uint32_t correlationId;

  /**
   * Internal use.
   */
  uint32_t padding;

  /**
   * The correlation ID to correlate JIT compilation, load and store operations.
   * Each JIT compilation unit is assigned a unique correlation ID
   * at the time of the JIT compilation. This correlation id can be used
   * to find the matching JIT cache load/store records.
   */
  uint64_t jitOperationCorrelationId;

  /**
   * The size of compute cache.
   */
  uint64_t cacheSize;

  /**
   * The path where the fat binary is cached.
   */
  const char* cachePath;
} CUpti_ActivityJit;

/**
 * \brief The activity record for CUDA event.
 *
 * This activity is used to track recorded events.
 * (CUPTI_ACTIVITY_KIND_CUDA_EVENT).
 *
 * Structure deprecated in CUDA 12.8: Refer to CUpti_ActivityCudaEvent2
 * for the latest structure.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_CUDA_EVENT.
   */
  CUpti_ActivityKind kind;

  /**
   * The correlation ID of the API to which this result is associated.
   */
  uint32_t correlationId;

  /**
   * The ID of the context where the event was recorded.
   */
  uint32_t contextId;

  /**
   * The compute stream where the event was recorded.
   */
  uint32_t streamId;

  /**
   * A unique event ID to identify the event record.
   */
  uint32_t eventId;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
} CUpti_ActivityCudaEvent;

/**
 * \brief The activity record for synchronization management.
 *
 * This activity is used to track various CUDA synchronization APIs.
 * (CUPTI_ACTIVITY_KIND_SYNCHRONIZATION).
 *
 * Structure deprecated in CUDA 12.8: Refer to CUpti_ActivitySynchronization2
 * for the latest structure.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_SYNCHRONIZATION.
   */
  CUpti_ActivityKind kind;

  /**
   * The type of record.
   */
  CUpti_ActivitySynchronizationType type;

  /**
   * The start timestamp for the function, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the function.
   */
  uint64_t start;

  /**
   * The end timestamp for the function, in ns. A value of 0 for both
   * the start and end timestamps indicates that timestamp information
   * could not be collected for the function.
   */
  uint64_t end;

  /**
   * The correlation ID of the API to which this result is associated.
   */
  uint32_t correlationId;

  /**
   * The ID of the context for which the synchronization API is called.
   * In case of context synchronization API it is the context id for which the API is called.
   * In case of stream/event synchronization it is the ID of the context where the stream/event was created.
   */
  uint32_t contextId;

  /**
   * The compute stream for which the synchronization API is called.
   * A CUPTI_SYNCHRONIZATION_INVALID_VALUE value indicate the field is not applicable for this record.
   * Not valid for cuCtxSynchronize, cuEventSynchronize.
   */
  uint32_t streamId;

  /**
   * The event ID for which the synchronization API is called.
   * A CUPTI_SYNCHRONIZATION_INVALID_VALUE value indicate the field is not applicable for this record.
   * Not valid for cuCtxSynchronize, cuStreamSynchronize.
   */
  uint32_t cudaEventId;
} CUpti_ActivitySynchronization;

/**
 * \brief The activity record for memory copies.
 *
 * This activity record represents a memory copy
 * (CUPTI_ACTIVITY_KIND_MEMCPY).
 *
 * Structure deprecated in CUDA 12.8: Refer to CUpti_ActivityMemcpy6
 * for the latest structure.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_MEMCPY.
   */
  CUpti_ActivityKind kind;

  /**
   * The kind of the memory copy, stored as a byte to reduce record
   * size. \see CUpti_ActivityMemcpyKind
   */
  uint8_t copyKind;

  /**
   * The source memory kind read by the memory copy, stored as a byte
   * to reduce record size. \see CUpti_ActivityMemoryKind
   */
  uint8_t srcKind;

  /**
   * The destination memory kind read by the memory copy, stored as a
   * byte to reduce record size. \see CUpti_ActivityMemoryKind
   */
  uint8_t dstKind;

  /**
   * The flags associated with the memory copy. \see CUpti_ActivityFlag
   */
  uint8_t flags;

  /**
   * The number of bytes transferred by the memory copy.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t start;

  /**
   * The end timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t end;

  /**
   * The ID of the device where the memory copy is occurring.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the memory copy is occurring.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the memory copy is occurring.
   */
  uint32_t streamId;

  /**
   * The correlation ID of the memory copy. Each memory copy is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver API activity record that launched
   * the memory copy.
   */
  uint32_t correlationId;

  /**
   * The runtime correlation ID of the memory copy. Each memory copy
   * is assigned a unique runtime correlation ID that is identical to
   * the correlation ID in the runtime API activity record that
   * launched the memory copy.
   */
  uint32_t runtimeCorrelationId;

#ifdef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;

  /**
   * The unique ID of the graph node that executed this memcpy through graph launch.
   * This field will be 0 if the memcpy is not done through graph launch.
   */
  uint64_t graphNodeId;

  /**
   * The unique ID of the graph that executed this memcpy through graph launch.
   * This field will be 0 if the memcpy is not done through graph launch.
   */
  uint32_t graphId;

  /**
   * The ID of the HW channel on which the memory copy is occurring.
   */
  uint32_t channelID;

  /**
   * The type of the channel
   */
  CUpti_ChannelType channelType;

  /**
   *  Reserved for internal use.
   */
  uint32_t pad2;
} CUpti_ActivityMemcpy5;

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility pop
#endif

#if defined(__cplusplus)
}
#endif

#endif /*_CUPTI_ACTIVITY_DEPRECATED_H_*/
