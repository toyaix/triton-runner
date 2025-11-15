/*
 * Copyright 2024 NVIDIA Corporation. All rights reserved.
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

#if !defined(_CUPTI_PMSAMPLING_H_)
#define _CUPTI_PMSAMPLING_H_

#include <cuda.h>
#include <cupti_result.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility push(default)
#endif

#ifndef CUPTI_PROFILER_STRUCT_SIZE
#define CUPTI_PROFILER_STRUCT_SIZE(type_, lastfield_)                     (offsetof(type_, lastfield_) + sizeof(((type_*)0)->lastfield_))
#endif

/* CUPTI PM sampling APIs */
/**
 * \defgroup CUPTI_PM_SAMPLING_API CUPTI PM Sampling API
 * Functions to enable, disable, start, stop, and decode PM sampling.
 * @{
 */
typedef struct CUpti_PmSampling_Object CUpti_PmSampling_Object;

typedef enum CUpti_PmSampling_TriggerMode
{
    /// The trigger is based off of the SYSCLK frequency, note SYS frequency by default is variable.
    /// the sample interval (set in the struct CUpti_PmSampling_SetConfig_Params) is in terms of clocks.
    CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_SYSCLK_INTERVAL = 0,
    /// The trigger is based off of a fixed frequency source.
    /// The sample interval (set in the struct CUpti_PmSampling_SetConfig_Params) is in terms of nanoseconds.
    /// Note: This trigger mode is not supported on Turing GPU architecture and GA100 GPU.
    /// It is supported on Ampere GA10x and later GPU architectures.
    CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_TIME_INTERVAL = 1,
    CUPTI_PM_SAMPLING_TRIGGER_MODE_COUNT
} CUpti_PmSampling_TriggerMode;

typedef enum CUpti_PmSampling_DecodeStopReason
{
    CUPTI_PM_SAMPLING_DECODE_STOP_REASON_OTHER = 0,
    /// Counter data image is full.
    CUPTI_PM_SAMPLING_DECODE_STOP_REASON_COUNTER_DATA_FULL,
    /// All the records in the hardware buffer is decoded.
    CUPTI_PM_SAMPLING_DECODE_STOP_REASON_END_OF_RECORDS,
    CUPTI_PM_SAMPLING_DECODE_STOP_REASON_COUNT
} CUpti_PmSampling_DecodeStopReason;

typedef enum CUpti_PmSampling_HardwareBuffer_AppendMode
{
    /// Keep the oldest records in the hardware buffer.
    /// CUPTI will report error for overflow in case hardware buffer is getting filled up.
    CUPTI_PM_SAMPLING_HARDWARE_BUFFER_APPEND_MODE_KEEP_OLDEST = 0,
    /// Keep the latest records in the hardware buffer.
    /// Note: This mode is not supported on Turing GPU architecture.
    /// It is supported on Ampere and later GPU architectures.
    CUPTI_PM_SAMPLING_HARDWARE_BUFFER_APPEND_MODE_KEEP_LATEST = 1
} CUpti_PmSampling_HardwareBuffer_AppendMode;

/**
 * \brief Params for cuptiPmSamplingSetConfig
 */
typedef struct CUpti_PmSampling_SetConfig_Params
{
    /// [in] Size of the data structure.
    size_t structSize;
    /// [in] Set to NULL.
    void* pPriv;
    /// [in] PM sampling object.
    CUpti_PmSampling_Object* pPmSamplingObject;
    /// [in] Size of the config image.
    size_t configSize;
    /// [in] Config image.
    const uint8_t* pConfig;
    /// [in] The hardware buffer size in which raw PM sampling data
    /// will be stored. These samples will be decoded to counter data
    /// image with \ref cuptiPmSamplingDecodeData call.
    size_t hardwareBufferSize;
    /// [in] For the trigger mode `CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_SYSCLK_INTERVAL`, sampling interval
    /// is the number of sys clock cycles. For the trigger mode `CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_TIME_INTERVAL`,
    /// sampling interval is in nanoseconds.
    uint64_t samplingInterval;
    /// [in] Trigger mode.
    /// Note: CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_TIME_INTERVAL is not supported in Turing and GA100.
    /// Supported from GA10x onwards.
    CUpti_PmSampling_TriggerMode triggerMode;
    /// [in] Append mode for the records in hardware buffer.
    /// For KEEP_OLDEST mode, all the records will be kept in the buffer and in case hardware buffer is getting filled up.
    /// overflow will be set to 1 in \ref CUpti_PmSampling_DecodeData_Params. For KEEP_LATEST mode, the new records will
    /// overwrite the oldest records in the buffer in case of filled buffer.
    CUpti_PmSampling_HardwareBuffer_AppendMode hwBufferAppendMode;
} CUpti_PmSampling_SetConfig_Params;

#define CUpti_PmSampling_SetConfig_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_PmSampling_SetConfig_Params, hwBufferAppendMode)

/**
 * \brief Set the configuration for PM sampling like sampling interval, maximum number of samples
 * filled in HW buffer, trigger mode and the config image which has scheduling info for metric collection.
 *
 * \param pParams A pointer to \ref CUpti_PmSampling_SetConfig_Params
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_NOT_SUPPORTED for config image which require multiple passes for data collection
 */
CUptiResult CUPTIAPI cuptiPmSamplingSetConfig(CUpti_PmSampling_SetConfig_Params* pParams);

/**
 * \brief Params for cuptiPmSamplingEnable
 */
typedef struct CUpti_PmSampling_Enable_Params
{
    /// [in] Size of the data structure.
    size_t structSize;
    /// [in] Set to NULL.
    void* pPriv;
    /// [in] Device index.
    size_t deviceIndex;
    /// [out] PM sampling object.
    CUpti_PmSampling_Object* pPmSamplingObject;
} CUpti_PmSampling_Enable_Params;

#define CUpti_PmSampling_Enable_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_PmSampling_Enable_Params, pPmSamplingObject)

/**
 * \brief Create a PM sampling object and enable PM sampling on the CUDA device.
 *
 * \param pParams A pointer to \ref CUpti_PmSampling_Enable_Params
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_OUT_OF_MEMORY if memory allocation fails while creating the PM sampling object
 * \retval CUPTI_ERROR_INVALID_OPERATION if PM sampling is already enabled on the device
 * \retval CUPTI_ERROR_INSUFFICIENT_PRIVILEGES if the user does not have sufficient privileges to perform the operation
 * \retval CUPTI_ERROR_UNKNOWN for any internal error
 */
CUptiResult CUPTIAPI cuptiPmSamplingEnable(CUpti_PmSampling_Enable_Params* pParams);

/**
 * \brief Params for cuptiPmSamplingDisable
 */
typedef struct CUpti_PmSampling_Disable_Params
{
    /// [in] Size of the data structure.
    size_t structSize;
    /// [in] Set to NULL.
    void* pPriv;
    /// [in] PM sampling object.
    CUpti_PmSampling_Object* pPmSamplingObject;
} CUpti_PmSampling_Disable_Params;

#define CUpti_PmSampling_Disable_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_PmSampling_Disable_Params, pPmSamplingObject)

/**
 * \brief Disable PM sampling on the CUDA device and destroy the PM sampling object.
 *
 * \param pParams A pointer to \ref CUpti_PmSampling_Disable_Params
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_UNKNOWN for any internal error
 */
CUptiResult CUPTIAPI cuptiPmSamplingDisable(CUpti_PmSampling_Disable_Params* pParams);

/**
 * \brief Params for cuptiPmSamplingStart
 */
typedef struct CUpti_PmSampling_Start_Params
{
    /// [in] Size of the data structure.
    size_t structSize;
    /// [in] Set to NULL.
    void* pPriv;
    /// [in] PM sampling object.
    CUpti_PmSampling_Object* pPmSamplingObject;
} CUpti_PmSampling_Start_Params;

#define CUpti_PmSampling_Start_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_PmSampling_Start_Params, pPmSamplingObject)

/**
 * \brief Start the PM sampling. The GPU will start collecting the metrics data
 * periodically based on trigger type and sampling interval passed in CUpti_PmSampling_SetConfig_Params.
 * The collected data will be stored in the hardware buffer.
 *
 * \param pParams A pointer to \ref CUpti_PmSampling_Start_Params
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_INVALID_OPERATION if PM sampling Start is called without enabling PM sampling,
 * and PM sampling is already started
 * \retval CUPTI_ERROR_UNKNOWN for any internal error
 */
CUptiResult CUPTIAPI cuptiPmSamplingStart(CUpti_PmSampling_Start_Params* pParams);

/**
 * \brief Params for cuptiPmSamplingStop
 */
typedef struct CUpti_PmSampling_Stop_Params
{
    /// [in] Size of the data structure.
    size_t structSize;
    /// [in] Set to NULL.
    void* pPriv;
    /// [in] PM sampling object.
    CUpti_PmSampling_Object* pPmSamplingObject;
} CUpti_PmSampling_Stop_Params;

#define CUpti_PmSampling_Stop_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_PmSampling_Stop_Params, pPmSamplingObject)

/**
 * \brief Stop the PM sampling. The GPU will stop collecting the metrics data.
 *
 * \param pParams A pointer to \ref CUpti_PmSampling_Stop_Params
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_INVALID_OPERATION if PM sampling Stop is called without enabling PM sampling,
 * and PM sampling is already stopped
 * \retval CUPTI_ERROR_UNKNOWN for any internal error
 */
CUptiResult CUPTIAPI cuptiPmSamplingStop(CUpti_PmSampling_Stop_Params* pParams);

/**
 * \brief Params for cuptiPmSamplingDecodeData
 */
typedef struct CUpti_PmSampling_DecodeData_Params
{
    /// [in] Size of the data structure.
    size_t structSize;
    /// [in] Set to NULL.
    void* pPriv;
    /// [in] PM sampling object.
    CUpti_PmSampling_Object* pPmSamplingObject;
    /// [in] Counter data image.
    uint8_t* pCounterDataImage;
    /// [in] Size of the counter data image.
    size_t counterDataImageSize;
    /// [out] decode stop reason
    CUpti_PmSampling_DecodeStopReason decodeStopReason;
    /// [out] overflow status for hardware buffer.
    /// To avoid overflow, either increase the maxSamples values in
    /// \ref CUpti_PmSampling_SetConfig_Params or reduce the sampling interval.
    uint8_t overflow;
} CUpti_PmSampling_DecodeData_Params;

#define CUpti_PmSampling_DecodeData_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_PmSampling_DecodeData_Params, overflow)

/**
 * \brief Decode the metrics data stored in the hardware buffer to the counter data image.
 *
 *
 * \param pParams A pointer to \ref CUpti_PmSampling_DecodeData_Params
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_INVALID_OPERATION if PM sampling DecodeData is called without enabling PM sampling
 * \retval CUPTI_ERROR_OUT_OF_MEMORY if there is record overflow in the hardware buffer
 * \retval CUPTI_ERROR_UNKNOWN for any internal error
 */
CUptiResult CUPTIAPI cuptiPmSamplingDecodeData(CUpti_PmSampling_DecodeData_Params* pParams);

/**
 * \brief Params for cuptiPmSamplingGetCounterData
 */
typedef struct CUpti_PmSampling_GetCounterAvailability_Params
{
    /// [in] Size of the data structure.
    size_t structSize;
    /// [in] Set to NULL.
    void* pPriv;
    /// [in] Device index.
    size_t deviceIndex;
    /// [inout] Size of the counter availability image. When pCounterAvailabilityImage is NULL,
    /// this field is used to return the size of the counter availability image.
    size_t counterAvailabilityImageSize;
    /// [out] Counter availability image.
    uint8_t* pCounterAvailabilityImage;
} CUpti_PmSampling_GetCounterAvailability_Params;
#define CUpti_PmSampling_GetCounterAvailability_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_PmSampling_GetCounterAvailability_Params, pCounterAvailabilityImage)

/**
 * \brief Query counter availibility information in a buffer which can be used to filter unavailable raw metrics on host.
 * Note: This API may fail, if any profiling or sampling session is active on the specified device.
 *
 * \param pParams A pointer to \ref CUpti_PmSampling_GetCounterAvailability_Params
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_INSUFFICIENT_PRIVILEGES if the user does not have sufficient privileges to perform the operation
 * \retval CUPTI_ERROR_UNKNOWN for any internal error
 */
CUptiResult CUPTIAPI cuptiPmSamplingGetCounterAvailability(CUpti_PmSampling_GetCounterAvailability_Params* pParams);

/**
 * \brief Params for cuptiPmSamplingGetCounterDataSize
 */
typedef struct CUpti_PmSampling_GetCounterDataSize_Params
{
    /// [in] Size of the data structure.
    size_t structSize;
    /// [in] Set to NULL.
    void* pPriv;
    /// [in] PM sampling object.
    CUpti_PmSampling_Object* pPmSamplingObject;
    /// [in] Names of the metrics to be collected.
    const char** pMetricNames;
    /// [in] Number of metrics to be collected.
    size_t numMetrics;
    /// [in] Maximum number of samples to be stored in the counter data image.
    uint32_t maxSamples;
    /// [out] Size of the counter data image.
    size_t counterDataSize;
} CUpti_PmSampling_GetCounterDataSize_Params;
#define CUpti_PmSampling_GetCounterDataSize_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_PmSampling_GetCounterDataSize_Params, counterDataSize)

/**
 * \brief Query the size of the counter data image which will be used to store the metrics data.
 * User need to allocate the memory for the counter data image based on the size returned by this API.
 *
 * \param pParams A pointer to \ref CUpti_PmSampling_GetCounterDataSize_Params
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_INVALID_OPERATION if PM sampling GetCounterDataSize is called without enabling PM sampling
 * \retval CUPTI_ERROR_UNKNOWN for any internal error
 */
CUptiResult CUPTIAPI cuptiPmSamplingGetCounterDataSize(CUpti_PmSampling_GetCounterDataSize_Params* pParams);

/**
 * \brief Params for cuptiPmSamplingCounterDataImageInitialize
 */
typedef struct CUpti_PmSampling_CounterDataImage_Initialize_Params
{
    /// [in] Size of the data structure.
    size_t structSize;
    /// [in] Set to NULL.
    void* pPriv;
    /// [in] PM sampling object.
    CUpti_PmSampling_Object* pPmSamplingObject;
    /// [in] Size of the counter data image.
    size_t counterDataSize;
    /// [in] Counter data image.
    uint8_t* pCounterData;
} CUpti_PmSampling_CounterDataImage_Initialize_Params;
#define CUpti_PmSampling_CounterDataImage_Initialize_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_PmSampling_CounterDataImage_Initialize_Params, pCounterData)

/**
 * \brief Initialize the counter data to CUPTI record format for storing the metric data.
 *
 * \param pParams A pointer to \ref CUpti_PmSampling_CounterDataImage_Initialize_Params
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_INVALID_OPERATION if PM sampling CounterDataInitialize is called without enabling PM sampling
 * \retval CUPTI_ERROR_UNKNOWN for any internal error
 */
CUptiResult CUPTIAPI cuptiPmSamplingCounterDataImageInitialize(CUpti_PmSampling_CounterDataImage_Initialize_Params* pParams);

/**
 * \brief Params for cuptiPmSamplingGetCounterDataInfo
 */
typedef struct CUpti_PmSampling_GetCounterDataInfo_Params
{
    /// [in] Size of the data structure.
    size_t structSize;
    /// [in] Set to NULL.
    void* pPriv;
    /// [in] Counter data image.
    const uint8_t* pCounterDataImage;
    /// [in] Size of the counter data image.
    size_t counterDataImageSize;
    /// [out] Number of samples in the counter data image.
    size_t numTotalSamples;
    /// [out] Number of populated samples.
    size_t numPopulatedSamples;
    /// [out] Number of samples that have been completed.
    size_t numCompletedSamples;
} CUpti_PmSampling_GetCounterDataInfo_Params;
#define CUpti_PmSampling_GetCounterDataInfo_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_PmSampling_GetCounterDataInfo_Params, numCompletedSamples)

/**
 * \brief Get the counter data info like number of samples, number of populated
 * samples and number of completed samples in a counter data image.
 *
 * \param pParams A pointer to \ref CUpti_PmSampling_GetCounterDataInfo_Params
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_UNKNOWN for any internal error
 */
CUptiResult CUPTIAPI cuptiPmSamplingGetCounterDataInfo(CUpti_PmSampling_GetCounterDataInfo_Params* pParams);

/**
 * \brief Params for cuptiPmSamplingCounterDataGetSampleInfo
 */
typedef struct CUpti_PmSampling_CounterData_GetSampleInfo_Params
{
    /// [in] Size of the data structure.
    size_t structSize;
    /// [in] Set to NULL.
    void* pPriv;
    /// [in] PM sampling object.
    CUpti_PmSampling_Object* pPmSamplingObject;
    /// [in] Counter data image.
    const uint8_t* pCounterDataImage;
    /// [in] Size of the counter data image.
    size_t counterDataImageSize;
    /// [in] Index of the sample.
    size_t sampleIndex;
    /// [out] Start time of the sample.
    uint64_t startTimestamp;
    /// [out] End time of the sample.
    uint64_t endTimestamp;
} CUpti_PmSampling_CounterData_GetSampleInfo_Params;
#define CUpti_PmSampling_CounterData_GetSampleInfo_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_PmSampling_CounterData_GetSampleInfo_Params, endTimestamp)

/**
 * \brief Get the sample info (start and end time stamp) for the given sample index.
 * Each sample is distinguished by the start and end time stamp.
 *
 * \param pParams A pointer to \ref CUpti_PmSampling_CounterData_GetSampleInfo_Params
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_UNKNOWN for any internal error
 */
CUptiResult CUPTIAPI cuptiPmSamplingCounterDataGetSampleInfo(CUpti_PmSampling_CounterData_GetSampleInfo_Params* pParams);

/** @} */ /* END CUPTI_PMSAMPLING_API */
#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility pop
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif // _CUPTI_PMSAMPLING_H_
