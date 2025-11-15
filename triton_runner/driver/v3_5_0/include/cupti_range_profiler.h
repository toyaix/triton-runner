/*
 * Copyright 2024 NVIDIA Corporation.  All rights reserved.
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

#if !defined(_CUPTI_RANGE_PROFILER_H_)
#define _CUPTI_RANGE_PROFILER_H_

#include <cuda.h>
#include <cupti_result.h>
#include <cupti_profiler_target.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility push(default)
#endif

/**
 * \defgroup CUPTI_RANGE_PROFILER_API CUPTI Range Profiling API
 * Functions, types, and enums that implement the CUPTI Range Profiling API.
 * @{
 */
#ifndef CUPTI_PROFILER_STRUCT_SIZE
#define CUPTI_PROFILER_STRUCT_SIZE(type_, lastfield_)                     (offsetof(type_, lastfield_) + sizeof(((type_*)0)->lastfield_))
#endif


typedef struct CUpti_RangeProfiler_Object CUpti_RangeProfiler_Object;

/**
 * \brief Params for cuptiRangeProfilerSetConfig
 */
typedef struct CUpti_RangeProfiler_SetConfig_Params
{
    /// [in] Size of the data structure.
    size_t structSize;
    /// [in] Set to NULL.
    void* pPriv;
    /// [in] Range Profiler Object.
    CUpti_RangeProfiler_Object* pRangeProfilerObject;
    /// [in] Size of the config image.
    size_t configSize;
    /// [in] Config image.
    const uint8_t* pConfig;
    /// [in] Size of the counter data image.
    size_t counterDataImageSize;
    /// [in] Counter data image.
    uint8_t* pCounterDataImage;
    /// [in] Profiling Range mode.
    CUpti_ProfilerRange range;
    /// [in] Replay mode.
    CUpti_ProfilerReplayMode replayMode;
    /// [in] Maximum number of ranges that can be profiled in a pass.
    size_t maxRangesPerPass;
    /// [in] number of nesting level to be profiled. For Auto range mode, this should be set to 1.
    uint16_t numNestingLevels;
    /// [in] minimum nesting level to be profiled.
    uint16_t minNestingLevel;
    /// [in] Pass index for the replay session.
    size_t passIndex;
    /// [in] Target nesting level for the replay session.
    uint16_t targetNestingLevel;
} CUpti_RangeProfiler_SetConfig_Params;

#define CUpti_RangeProfiler_SetConfig_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_RangeProfiler_SetConfig_Params, targetNestingLevel)

/**
 * \brief Set the configuration for range profiler like maximum number of ranges per pass, number of nesting levels,
 * range and replay mode and the config image which has scheduling info for metric collection.
 * 
 * \param pParams A pointer to \ref CUpti_RangeProfiler_SetConfig_Params
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 */
CUptiResult CUPTIAPI cuptiRangeProfilerSetConfig(CUpti_RangeProfiler_SetConfig_Params* pParams);

/**
 * \brief Params for cuptiRangeProfilerEnable
 */
typedef struct CUpti_RangeProfiler_Enable_Params
{
    /// [in] Size of the data structure.
    size_t structSize;
    /// [in] Set to NULL.
    void* pPriv;
    /// [in] Context to be used for profiling.
    CUcontext ctx;
    /// [out] Range Profiler Object.
    CUpti_RangeProfiler_Object* pRangeProfilerObject;
} CUpti_RangeProfiler_Enable_Params;
#define CUpti_RangeProfiler_Enable_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_RangeProfiler_Enable_Params, pRangeProfilerObject)

/**
 * \brief Create a range profiler object and enable range profiling on the CUDA context.
 * 
 * \param pParams A pointer to \ref CUpti_RangeProfiler_Enable_Params
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_OUT_OF_MEMORY if memory allocation fails while creating the PM sampling object
 * \retval CUPTI_ERROR_INSUFFICIENT_PRIVILEGES if the user does not have sufficient privileges to perform the operation
 * \retval CUPTI_ERROR_UNKNOWN for any internal error
 */
CUptiResult CUPTIAPI cuptiRangeProfilerEnable(CUpti_RangeProfiler_Enable_Params* pParams);

/**
 * \brief Params for cuptiRangeProfilerDisable
 */
typedef struct CUpti_RangeProfiler_Disable_Params
{
    /// [in] Size of the data structure.
    size_t structSize;
    /// [in] Set to NULL.
    void* pPriv;
    /// [in] Range Profiler Object.
    CUpti_RangeProfiler_Object* pRangeProfilerObject;
} CUpti_RangeProfiler_Disable_Params;
#define CUpti_RangeProfiler_Disable_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_RangeProfiler_Disable_Params, pRangeProfilerObject)

/**
 * \brief Disable the range profiler on the CUDA context and destroy the range profiler object.
 * 
 * \param pParams A pointer to \ref CUpti_RangeProfiler_Disable_Params
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 */
CUptiResult CUPTIAPI cuptiRangeProfilerDisable(CUpti_RangeProfiler_Disable_Params* pParams);

/**
 * \brief Params for cuptiRangeProfilerStart
 */
typedef struct CUpti_RangeProfiler_Start_Params
{
    /// [in] Size of the data structure.
    size_t structSize;
    /// [in] Set to NULL.
    void* pPriv;
    /// [in] Range Profiler Object.
    CUpti_RangeProfiler_Object* pRangeProfilerObject;
} CUpti_RangeProfiler_Start_Params;
#define CUpti_RangeProfiler_Start_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_RangeProfiler_Start_Params, pRangeProfilerObject)

/**
 * \brief Start the range profiler.
 * 
 * \param pParams A pointer to \ref CUpti_RangeProfiler_Start_Params
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_INVALID_OPERATION if range profiler Start is called without enabling range profiler
 * \retval CUPTI_ERROR_UNKNOWN for any internal error
 */
CUptiResult CUPTIAPI cuptiRangeProfilerStart(CUpti_RangeProfiler_Start_Params* pParams);

/**
 * \brief Params for cuptiRangeProfilerStop
 */
typedef struct CUpti_RangeProfiler_Stop_Params
{
    /// [in] Size of the data structure.
    size_t structSize;
    /// [in] Set to NULL.
    void* pPriv;
    /// [in] Range Profiler Object.
    CUpti_RangeProfiler_Object* pRangeProfilerObject;
    /// [out] pass index for the replay session.
    size_t passIndex;
    /// [out] target nesting level for the replay session.
    size_t targetNestingLevel;
    /// [out] 1 if all passes are submitted to GPU for collection, 0 otherwise.
    uint8_t isAllPassSubmitted;
} CUpti_RangeProfiler_Stop_Params;
#define CUpti_RangeProfiler_Stop_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_RangeProfiler_Stop_Params, isAllPassSubmitted)

/**
 * \brief Stop the range profiler.
 * 
 * \param pParams A pointer to \ref CUpti_RangeProfiler_Stop_Params
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_INVALID_OPERATION if range profiler Stop is called without enabling range profiler
 * \retval CUPTI_ERROR_UNKNOWN for any internal error
 */
CUptiResult CUPTIAPI cuptiRangeProfilerStop(CUpti_RangeProfiler_Stop_Params* pParams);

/**
 * \brief Params for cuptiRangeProfilerPushRange
 */
typedef struct CUpti_RangeProfiler_PushRange_Params
{
    /// [in] Size of the data structure.
    size_t structSize;
    /// [in] Set to NULL.
    void* pPriv;
    /// [in] Range Profiler Object.
    CUpti_RangeProfiler_Object* pRangeProfilerObject;
    /// [in] Name of the range to be profiled (only valid for User range mode).
    const char* pRangeName;
} CUpti_RangeProfiler_PushRange_Params;
#define CUpti_RangeProfiler_PushRange_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_RangeProfiler_PushRange_Params, pRangeName)

/**
 * \brief Add a new range to the Range Profiler with a given range name.
 * For nested ranges, this API should be called again for the innermost range. For profiling the nested
 * range, users need to set the values for minNestingLevel and numNestingLevels in the SetConfig API.
 * 
 * \param pParams A pointer to \ref CUpti_RangeProfiler_PushRange_Params
 * 
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_INVALID_OPERATION if range profiler PushRange is called without enabling range profiler
 * \retval CUPTI_ERROR_UNKNOWN for any internal error
*/
CUptiResult CUPTIAPI cuptiRangeProfilerPushRange(CUpti_RangeProfiler_PushRange_Params* pParams);

/**
 * \brief Params for cuptiRangeProfilerPopRange
 */
typedef struct CUpti_RangeProfiler_PopRange_Params
{
    /// [in] Size of the data structure.
    size_t structSize;
    /// [in] Set to NULL.
    void* pPriv;
    /// [in] Range Profiler Object.
    CUpti_RangeProfiler_Object* pRangeProfilerObject;
} CUpti_RangeProfiler_PopRange_Params;
#define CUpti_RangeProfiler_PopRange_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_RangeProfiler_PopRange_Params, pRangeProfilerObject)

/**
 * \brief pop the current range to the Range Profiler.
 * The number of pop range API call should be same as number of push ranges in the same order.
 * 
 * \param pParams A pointer to \ref CUpti_RangeProfiler_PopRange_Params
 * 
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_INVALID_OPERATION if range profiler PopRange is called without enabling range profiler
 * \retval CUPTI_ERROR_UNKNOWN for any internal error
*/
CUptiResult CUPTIAPI cuptiRangeProfilerPopRange(CUpti_RangeProfiler_PopRange_Params* pParams);

/**
 * \brief Params for cuptiRangeProfilerDecodeData
 */
typedef struct CUpti_RangeProfiler_DecodeData_Params
{
    /// [in] Size of the data structure.
    size_t structSize;
    /// [in] Set to NULL.
    void* pPriv;
    /// [in] Range Profiler Object.
    CUpti_RangeProfiler_Object* pRangeProfilerObject;
    /// [out] Number of ranges dropped in the processed passes.
    size_t numOfRangeDropped;
} CUpti_RangeProfiler_DecodeData_Params;
#define CUpti_RangeProfiler_DecodeData_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_RangeProfiler_DecodeData_Params, numOfRangeDropped)

/**
 * \brief Decode the profiling data stored in the hardware to the counter data image passed in the
 * SetConfig API. This API should be called after cuptiRangeProfilerStop. The counter data image
 * will be updated with the profiling data for the ranges profiled.
 * 
 * For the cases where the number of ranges counter data image can store is less than the number of ranges
 * profiled (= maxRangesPerPass in SetConfig API), the counter data image will report dropped ranges.
 * 
 * \param pParams A pointer to \ref CUpti_RangeProfiler_DecodeData_Params
 * 
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_INVALID_OPERATION if range profiler DecodeData is called without enabling range profiler
 * \retval CUPTI_ERROR_UNKNOWN for any internal error
*/
CUptiResult CUPTIAPI cuptiRangeProfilerDecodeData(CUpti_RangeProfiler_DecodeData_Params* pParams);

/**
 * \brief Params for cuptiRangeProfilerGetCounterDataSize
 */
typedef struct CUpti_RangeProfiler_GetCounterDataSize_Params
{
    /// [in] Size of the data structure.
    size_t structSize;
    /// [in] Set to NULL.
    void* pPriv;
    /// [in] Periodic sampler object.
    CUpti_RangeProfiler_Object* pRangeProfilerObject;
    /// [in] Names of the metrics to be collected.
    const char** pMetricNames;
    /// [in] Number of metrics to be collected.
    size_t numMetrics;
    /// [in] Maximum number of ranges to be stored in the counter data image.
    size_t maxNumOfRanges;
    /// [in] Maximum number of RangeTree nodes; must be >= maxNumOfRanges
    uint32_t maxNumRangeTreeNodes;
    /// [out] Size of the counter data image.
    size_t counterDataSize;
} CUpti_RangeProfiler_GetCounterDataSize_Params;
#define CUpti_RangeProfiler_GetCounterDataSize_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_RangeProfiler_GetCounterDataSize_Params, counterDataSize)

/**
 * \brief Get the size of the counter data image required to store the profiling data for the ranges profiled.
 * 
 * \param pParams A pointer to \ref CUpti_RangeProfiler_GetCounterDataSize_Params
 * 
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_INVALID_OPERATION if range profiler GetCounterDataSize is called without enabling range profiler
 * \retval CUPTI_ERROR_UNKNOWN for any internal error
*/
CUptiResult CUPTIAPI cuptiRangeProfilerGetCounterDataSize(CUpti_RangeProfiler_GetCounterDataSize_Params* pParams);

/**
 * \brief Params for cuptiRangeProfilerCounterDataImageInitialize
 */
typedef struct CUpti_RangeProfiler_CounterDataImage_Initialize_Params
{
    /// [in] Size of the data structure.
    size_t structSize;
    /// [in] Set to NULL.
    void* pPriv;
    /// [in] Periodic sampler object.
    CUpti_RangeProfiler_Object* pRangeProfilerObject;
    /// [in] Size of the counter data image.
    size_t counterDataSize;
    /// [in] Counter data image.
    uint8_t* pCounterData;
} CUpti_RangeProfiler_CounterDataImage_Initialize_Params;
#define CUpti_RangeProfiler_CounterDataImage_Initialize_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_RangeProfiler_CounterDataImage_Initialize_Params, pCounterData)

/**
 * \brief Initialize the counter data image with the profiling data for the ranges profiled.
 * 
 * \param pParams A pointer to \ref CUpti_RangeProfiler_CounterDataImage_Initialize_Params
 * 
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_INVALID_OPERATION if range profiler CounterDataImageInitialize is called without enabling range profiler
 * \retval CUPTI_ERROR_UNKNOWN for any internal error
*/
CUptiResult CUPTIAPI cuptiRangeProfilerCounterDataImageInitialize(CUpti_RangeProfiler_CounterDataImage_Initialize_Params* pParams);

/**
 * \brief Params for cuptiRangeProfilerGetCounterDataInfo
 */
typedef struct CUpti_RangeProfiler_GetCounterDataInfo_Params
{
    /// [in] Size of the data structure.
    size_t structSize;
    /// [in] Set to NULL.
    void* pPriv;
    /// [in] Counter data image.
    const uint8_t* pCounterDataImage;
    /// [in] Size of the counter data image.
    size_t counterDataImageSize;
    /// [out] Number of ranges in the counter data image.
    size_t numTotalRanges;
} CUpti_RangeProfiler_GetCounterDataInfo_Params;
#define CUpti_RangeProfiler_GetCounterDataInfo_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_RangeProfiler_GetCounterDataInfo_Params, numTotalRanges)

/**
 * \brief Get the number of ranges stored in the counter data image.
 * 
 * \param pParams A pointer to \ref CUpti_RangeProfiler_GetCounterDataInfo_Params
 * 
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_UNKNOWN for any internal error
*/
CUptiResult CUPTIAPI cuptiRangeProfilerGetCounterDataInfo(CUpti_RangeProfiler_GetCounterDataInfo_Params* pParams);

/**
 * \brief Params for cuptiRangeProfilerCounterDataGetRangeInfo
 */
typedef struct CUpti_RangeProfiler_CounterData_GetRangeInfo_Params
{
    /// [in] Size of the data structure.
    size_t structSize;
    /// [in] Set to NULL.
    void* pPriv;
    /// [in] Counter data image.
    const uint8_t* pCounterDataImage;
    /// [in] Size of the counter data image.
    size_t counterDataImageSize;
    /// [in] Index of the sample.
    size_t rangeIndex;
    /// [in] range delimiter.
    const char* rangeDelimiter;
    /// [out] RangeName;
    const char* rangeName;
} CUpti_RangeProfiler_CounterData_GetRangeInfo_Params;
#define CUpti_RangeProfiler_CounterData_GetRangeInfo_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_RangeProfiler_CounterData_GetRangeInfo_Params, rangeName)

/**
 * \brief Get the range name for the given range index.
 * 
 * \param pParams A pointer to \ref CUpti_RangeProfiler_CounterData_GetRangeInfo_Params
 * 
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_UNKNOWN for any internal error
*/
CUptiResult CUPTIAPI cuptiRangeProfilerCounterDataGetRangeInfo(CUpti_RangeProfiler_CounterData_GetRangeInfo_Params* pParams);


#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility pop
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /*_CUPTI_RANGE_PROFILER_H_*/
