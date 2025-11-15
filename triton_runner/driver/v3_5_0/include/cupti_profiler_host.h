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

#if !defined(_CUPTI_PROFILER_HOST_H_)
#define _CUPTI_PROFILER_HOST_H_

/*
CUPTI profiler host API's
This file contains the CUPTI profiling host API's.
*/
#include <cupti_result.h>
#include <stdint.h>
#include <stddef.h>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility push(default)
#endif

/**
 * \defgroup CUPTI_PROFILER_HOST_API CUPTI Profiler Host API
 * Functions, types, and enums that implement the CUPTI Profiler Host API.
 * @{
 */
#ifndef CUPTI_PROFILER_STRUCT_SIZE
#define CUPTI_PROFILER_STRUCT_SIZE(type_, lastfield_)                     (offsetof(type_, lastfield_) + sizeof(((type_*)0)->lastfield_))
#endif

typedef enum CUpti_MetricType
{
    CUPTI_METRIC_TYPE_COUNTER = 0,
    CUPTI_METRIC_TYPE_RATIO,
    CUPTI_METRIC_TYPE_THROUGHPUT,
    CUPTI_METRIC_TYPE__COUNT
} CUpti_MetricType;

typedef enum CUpti_ProfilerType
{
    CUPTI_PROFILER_TYPE_RANGE_PROFILER,
    CUPTI_PROFILER_TYPE_PM_SAMPLING,
    CUPTI_PROFILER_TYPE_PROFILER_INVALID
} CUpti_ProfilerType;

typedef struct CUpti_Profiler_Host_Object CUpti_Profiler_Host_Object;

/**
 * \brief Params for cuptiProfilerHostInitialize
 */
typedef struct CUpti_Profiler_Host_Initialize_Params
{
    /// [in] Size of the data structure.
    /// CUPTI client should set the size of the structure. It will be used in CUPTI to check what fields are
    /// available in the structure. Used to preserve backward compatibility.
    size_t structSize;
    /// [in] Assign to NULL
    void* pPriv;
    /// [in] the profiler kind one from CUpti_ProfilerType
    CUpti_ProfilerType profilerType;
    /// [in] accepted for chips supported at the time-of-release.
    const char* pChipName;
    /// [in] buffer with counter availability image - required for future chip support
    const uint8_t* pCounterAvailabilityImage;
    /// [out] binary blob allocated by CUPTI and operations associated with this object.
    CUpti_Profiler_Host_Object* pHostObject;
} CUpti_Profiler_Host_Initialize_Params;

#define CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_Host_Initialize_Params, pHostObject)

/**
 * \brief Create and initialize the profiler host object (CUpti_Profiler_Host_Object).
 * 
 * \param pParams A pointer to \ref CUpti_Profiler_Host_Initialize_Params
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_UNKNOWN for any internal error
 */
CUptiResult CUPTIAPI cuptiProfilerHostInitialize(CUpti_Profiler_Host_Initialize_Params* pParams);

/**
 * \brief Params for cuptiProfilerHostDeinitialize
 */
typedef struct CUpti_Profiler_Host_Deinitialize_Params
{
    /// [in] Size of the data structure.
    /// CUPTI client should set the size of the structure. It will be used in CUPTI to check what fields are
    /// available in the structure. Used to preserve backward compatibility.
    size_t structSize;
    /// [in] Assign to NULL
    void* pPriv;
    /// [in] reference to the profiler host object allocated by CUPTI in cuptiProfilerHostInitialize
    struct CUpti_Profiler_Host_Object* pHostObject;
} CUpti_Profiler_Host_Deinitialize_Params;

#define CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_Host_Deinitialize_Params, pHostObject)

/**
 * \brief Deinitialize and destroy the profiler host object (CUpti_Profiler_Host_Object).
 * 
 * \param pParams A pointer to \ref CUpti_Profiler_Host_Deinitialize_Params
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_UNKNOWN for any internal error
 */
CUptiResult CUPTIAPI cuptiProfilerHostDeinitialize(CUpti_Profiler_Host_Deinitialize_Params* pParams);

/**
 * \brief Params for cuptiProfilerHostGetSupportedChips
 */
typedef struct CUpti_Profiler_Host_GetSupportedChips_Params
{
    /// [in] Size of the data structure.
    size_t structSize;
    /// [in] Assign to NULL
    void* pPriv;
    /// [out] number of supported chips
    size_t numChips;
    /// [out] list of supported chips
    const char* const* ppChipNames;
} CUpti_Profiler_Host_GetSupportedChips_Params;

#define CUpti_Profiler_Host_GetSupportedChips_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_Host_GetSupportedChips_Params, ppChipNames)

/**
 * \brief Get the list of supported chips.
 * 
 * \param pParams A pointer to \ref CUpti_Profiler_Host_GetSupportedChips_Params
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_UNKNOWN for any internal error
 */
CUptiResult CUPTIAPI cuptiProfilerHostGetSupportedChips(CUpti_Profiler_Host_GetSupportedChips_Params* pParams);

/**
 * \brief Params for cuptiProfilerHostGetSupportedMetrics
 */
typedef struct CUpti_Profiler_Host_GetBaseMetrics_Params
{
    /// [in] Size of the data structure.
    size_t structSize;
    /// [in] Assign to NULL
    void* pPriv;
    /// [in] reference to the profiler host object allocated by CUPTI in cuptiProfilerHostInitialize
    struct CUpti_Profiler_Host_Object* pHostObject;
    /// [in] metric type (counter, ratio, throughput)
    CUpti_MetricType metricType;
    /// [out] list of base metrics supported of queried metric type for the chip
    const char** ppMetricNames;
    /// [out] number of metrics
    size_t numMetrics;
} CUpti_Profiler_Host_GetBaseMetrics_Params;

#define CUpti_Profiler_Host_GetBaseMetrics_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_Host_GetBaseMetrics_Params, numMetrics)

/**
 * \brief Get the list of supported base metrics for the chip.
 * 
 * \param pParams A pointer to \ref CUpti_Profiler_Host_GetBaseMetrics_Params
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_UNKNOWN for any internal error
 */
CUptiResult CUPTIAPI cuptiProfilerHostGetBaseMetrics(CUpti_Profiler_Host_GetBaseMetrics_Params* pParams);

/**
 * \brief Params for cuptiProfilerHostGetSubMetrics
 */
typedef struct CUpti_Profiler_Host_GetSubMetrics_Params
{
    /// [in] Size of the data structure.
    size_t structSize;
    /// [in] Assign to NULL
    void* pPriv;
    /// [in] reference to the profiler host object allocated by CUPTI in cuptiProfilerHostInitialize
    CUpti_Profiler_Host_Object* pHostObject;
    /// [in] the metric type for queried metric
    CUpti_MetricType metricType;
    /// [in] metric name for which sub-metric will be listed
    const char* pMetricName;
    /// [out] number of submetrics supported
    size_t numOfSubmetrics;
    /// [out] list of submetrics supported for the metric.
    const char** ppSubMetrics;
} CUpti_Profiler_Host_GetSubMetrics_Params;

#define CUpti_Profiler_Host_GetSubMetrics_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_Host_GetSubMetrics_Params, ppSubMetrics)

/**
 * \brief Get the list of supported sub-metrics for the metric.
 * 
 * \param pParams A pointer to \ref CUpti_Profiler_Host_GetSubMetrics_Params
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_INVALID_METRIC_NAME if the metric name is not valid or not supported for the chip
 * \retval CUPTI_ERROR_UNKNOWN for any internal error
 */
CUptiResult CUPTIAPI cuptiProfilerHostGetSubMetrics(CUpti_Profiler_Host_GetSubMetrics_Params* pParams);

/**
 * \brief Params for cuptiProfilerHostGetMetricProperties
 */
typedef struct CUpti_Profiler_Host_GetMetricProperties_Params
{
    /// [in] Size of the data structure.
    size_t structSize;
    /// [in] Assign to NULL
    void* pPriv;
    /// [in] reference to the profiler host object allocated by CUPTI in cuptiProfilerHostInitialize
    CUpti_Profiler_Host_Object* pHostObject;
    /// [in] metric name for which its properties will be listed
    const char* pMetricName;
    /// [out] a short description about the metric
    const char* pDescription;
    /// [out] associated hw unit for the metric
    const char* pHwUnit;
    /// [out] the dimension of the metric values
    const char* pDimUnit;
    /// [out] the metric type (counter, ratio or throughput)
    CUpti_MetricType metricType;
} CUpti_Profiler_Host_GetMetricProperties_Params;

#define CUpti_Profiler_Host_GetMetricProperties_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_Host_GetMetricProperties_Params, metricType)

/**
 * \brief Get the properties of the metric.
 * 
 * \param pParams A pointer to \ref CUpti_Profiler_Host_GetMetricProperties_Params
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_INVALID_METRIC_NAME if the metric name is not valid or not supported for the chip
 * \retval CUPTI_ERROR_UNKNOWN for any internal error
 */
CUptiResult CUPTIAPI cuptiProfilerHostGetMetricProperties(CUpti_Profiler_Host_GetMetricProperties_Params* pParams);

/**
 * \brief Params for cuptiProfilerHostGetRangeName
 */
typedef struct CUpti_Profiler_Host_GetRangeName_Params
{
    /// [in] Size of the data structure.
    size_t structSize;
    /// [in] Assign to NULL
    void* pPriv;
    /// [in] the counter data image where profiling data has been decoded
    const uint8_t* pCounterDataImage;
    /// [in] size of counter data image
    size_t counterDataImageSize;
    /// [in] range index for which the range name will be queried
    size_t rangeIndex;
    /// [in] used in case of nested ranges, default="/". Range1<delimiter>Range2 
    const char* delimiter;
    /// [out] the range name.
    /// Note: that the CUPTI allocate the memory internal and
    /// its user responsibility to free up the allocated memory
    const char* pRangeName;
} CUpti_Profiler_Host_GetRangeName_Params;

#define CUpti_Profiler_Host_GetRangeName_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_Host_GetRangeName_Params, pRangeName)

/**
 * \brief Get the range name for the range index stored in the counter data.
 * In Range profiler, for Auto range mode the range name will be numeric value
 * assigned to the kernel based on execution order. For user range mode, the 
 * name of range will be based on the range name provided by the user using
 * Push range API.
 * 
 * \param pParams A pointer to \ref CUpti_Profiler_Host_GetRangeName_Params
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_UNKNOWN for any internal error
 */
CUptiResult CUPTIAPI cuptiProfilerHostGetRangeName(CUpti_Profiler_Host_GetRangeName_Params* pParams);

/**
 * \brief Params for cuptiProfilerHostEvaluateToGpuValues
 */
typedef struct CUpti_Profiler_Host_EvaluateToGpuValues_Params
{
    /// [in] Size of the data structure.
    size_t structSize;
    /// [in] Assign to NULL
    void* pPriv;
    /// [in] reference to the profiler host object allocated by CUPTI in cuptiProfilerHostInitialize
    CUpti_Profiler_Host_Object* pHostObject;
    /// [in] the counter data image where profiling data has been decoded
    const uint8_t* pCounterDataImage;
    /// [in] size of counter data image
    size_t counterDataImageSize;
    /// [in] range index for which the range name will be queried
    size_t rangeIndex;
    /// [in] the metrics for which GPU values will be evaluated for the range
    const char** ppMetricNames;
    /// [in] number of metrics
    size_t numMetrics;
    /// [out] output value for given metric and range index
    double* pMetricValues;
} CUpti_Profiler_Host_EvaluateToGpuValues_Params;

#define CUpti_Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_Host_EvaluateToGpuValues_Params, pMetricValues)

/**
 * \brief Evaluate the metric values for the range index stored in the counter data.
 * 
 * \param pParams A pointer to \ref CUpti_Profiler_Host_EvaluateToGpuValues_Params
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_INVALID_METRIC_NAME if the metric name is not valid or not supported for the chip
 * \retval CUPTI_ERROR_UNKNOWN for any internal error
 */
CUptiResult CUPTIAPI cuptiProfilerHostEvaluateToGpuValues(CUpti_Profiler_Host_EvaluateToGpuValues_Params* pParams);

/**
 * \brief Params for cuptiProfilerHostConfigAddMetrics
 */
typedef struct CUpti_Profiler_Host_ConfigAddMetrics_Params
{
    /// [in] Size of the data structure.
    size_t structSize;
    /// [in] Assign to NULL
    void* pPriv;
    /// [in] reference to the profiler host object allocated by CUPTI in cuptiProfilerHostInitialize
    struct CUpti_Profiler_Host_Object* pHostObject;
    /// [in] metric names for which config image will be generated
    const char** ppMetricNames;
    /// [in] number of metrics
    size_t numMetrics;
} CUpti_Profiler_Host_ConfigAddMetrics_Params;

#define CUpti_Profiler_Host_ConfigAddMetrics_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_Host_ConfigAddMetrics_Params, numMetrics)

/**
 * \brief Add the metrics to the profiler host object for generating the config image.
 * The config image will have the required information to schedule the metrics for
 * collecting the profiling data. 
 * Note: PM sampling only supports single pass config image.
 * 
 * \param pParams A pointer to \ref CUpti_Profiler_Host_ConfigAddMetrics_Params
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_INVALID_METRIC_NAME if the metric name is not valid or not supported for the chip
 * \retval CUPTI_ERROR_UNKNOWN for any internal error
 */
CUptiResult CUPTIAPI cuptiProfilerHostConfigAddMetrics(CUpti_Profiler_Host_ConfigAddMetrics_Params* pParams);

/**
 * \brief Params for cuptiProfilerHostGetConfigImageSize
 */
typedef struct CUpti_Profiler_Host_GetConfigImageSize_Params
{
    /// [in] Size of the data structure.
    size_t structSize;
    /// [in] Assign to NULL
    void* pPriv;
    /// [in] reference to the profiler host object allocated by CUPTI in cuptiProfilerHostInitialize
    CUpti_Profiler_Host_Object* pHostObject;
    /// [out] the size of config image, users need to allocate the buffer for storing
    size_t configImageSize;
} CUpti_Profiler_Host_GetConfigImageSize_Params;

#define CUpti_Profiler_Host_GetConfigImageSize_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_Host_GetConfigImageSize_Params, configImageSize)

/**
 * \brief Get the size of the config image for the metrics added to the profiler host object.
 * Users need to allocate the buffer for storing the config image.
 * 
 * \param pParams A pointer to \ref CUpti_Profiler_Host_GetConfigImageSize_Params
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_UNKNOWN for any internal error
 */
CUptiResult CUPTIAPI cuptiProfilerHostGetConfigImageSize(CUpti_Profiler_Host_GetConfigImageSize_Params* pParams);

/**
 * \brief Params for cuptiProfilerHostGetConfigImage
 */
typedef struct CUpti_Profiler_Host_GetConfigImage_Params
{
    /// [in] Size of the data structure.
    size_t structSize;
    /// [in] Assign to NULL
    void* pPriv;
    /// [in] reference to the profiler host object allocated by CUPTI in cuptiProfilerHostInitialize
    CUpti_Profiler_Host_Object* pHostObject;
    /// [in] Number of bytes allocated for pBuffer
    size_t configImageSize;
    /// [out] Buffer receiving the config image
    uint8_t* pConfigImage;
} CUpti_Profiler_Host_GetConfigImage_Params;

#define CUpti_Profiler_Host_GetConfigImage_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_Host_GetConfigImage_Params, pConfigImage)

/**
 * \brief Get the config image for the metrics added to the profiler host object.
 * User will pass the allocated buffer to store the config image.
 * 
 * \param pParams A pointer to \ref CUpti_Profiler_Host_GetConfigImage_Params
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_UNKNOWN for any internal error
 */
CUptiResult CUPTIAPI cuptiProfilerHostGetConfigImage(CUpti_Profiler_Host_GetConfigImage_Params* pParams);

/**
 * \brief Params for cuptiProfilerHostGetNumOfPasses
 */
typedef struct CUpti_Profiler_Host_GetNumOfPasses_Params
{
    /// [in] Size of the data structure.
    size_t structSize;
    /// [in] Assign to NULL
    void* pPriv;
    /// [in] Number of bytes allocated for pConfigImage
    size_t configImageSize;
    /// [in] the config image buffer
    uint8_t* pConfigImage;
    /// [out] number of passes required for profiling scheduled metrics in the config image
    size_t numOfPasses;
} CUpti_Profiler_Host_GetNumOfPasses_Params;

#define CUpti_Profiler_Host_GetNumOfPasses_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_Host_GetNumOfPasses_Params, numOfPasses)

/**
 * \brief Get the number of passes required for profiling the scheduled metrics in the config image.
 * 
 * \param pParams A pointer to \ref CUpti_Profiler_Host_GetNumOfPasses_Params
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_UNKNOWN for any internal error
 */
CUptiResult CUPTIAPI cuptiProfilerHostGetNumOfPasses(CUpti_Profiler_Host_GetNumOfPasses_Params* pParams);

/**
 * \brief Params for cuptiProfilerHostGetMaxNumHardwareMetricsPerPass
 */
typedef struct CUpti_Profiler_Host_GetMaxNumHardwareMetricsPerPass_Params
{
    /// [in] Size of the data structure.
    size_t structSize;
    /// [in] Assign to NULL
    void* pPriv;
    /// [in] the profiler kind one from CUpti_ProfilerType
    CUpti_ProfilerType profilerType;
    /// [in] accepted for chips supported at the time-of-release.
    const char* pChipName;
    /// [in] buffer with counter availability image - required for future chip support
    uint8_t* pCounterAvailabilityImage;
    /// [out] maximum number of metrics that can be scheduled in a pass
    size_t maxMetricsPerPass;
} CUpti_Profiler_Host_GetMaxNumHardwareMetricsPerPass_Params;

#define CUpti_Profiler_Host_GetMaxNumHardwareMetricsPerPass_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_Host_GetMaxNumHardwareMetricsPerPass_Params, maxMetricsPerPass)

/**
 * \brief Get the maximum number of hardware metrics (metric names which doesn't include _sass_ keyword)
 * that can be scheduled in a single pass for a chip. While this represents a theoretical upper limit,
 * practical constraints may prevent reaching this threshold for a specific set of metrics. Furthermore,
 * the maximum achievable value is contingent upon the characteristics and architecture of the chip in question.
 * 
 * Use cuptiProfilerHostGetNumOfPasses API for getting the actual number of passes required for the
 * for collecting the profiling data for the scheduled metrics in a config image.
 * 
 * \param pParams A pointer to \ref CUpti_Profiler_Host_GetMaxNumHardwareMetricsPerPass_Params
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_UNKNOWN for any internal error
 */
CUptiResult CUPTIAPI cuptiProfilerHostGetMaxNumHardwareMetricsPerPass(CUpti_Profiler_Host_GetMaxNumHardwareMetricsPerPass_Params* pParams);

/** @} */ /* END CUPTI_METRIC_API */
#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility pop
#endif


#ifdef __cplusplus
} /* extern "C" */
#endif

#endif