/*
 * Copyright 2023 NVIDIA Corporation. All rights reserved.
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

#if !defined(_CUPTI_SASS_METRICS_H_)
#define _CUPTI_SASS_METRICS_H_

#include <cuda.h>
#include <cupti_result.h>
#include <cupti_profiler_target.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility push(default)
#endif

/**
 * \defgroup CUPTI_SASS_METRICS_API CUPTI SASS Metrics API
 * Functions, types, and enums that implement the CUPTI SASS Metrics API.
 * @{
 */

typedef enum
{
    /// SASS metric data will be collected at GPU level. 
    /// In CUpti_SassMetricsGetDataProperties_Params struct the numOfInstances will be equal to 1
    CUPTI_SASS_METRICS_OUTPUT_GRANULARITY_GPU = 0,

    /// SASS metric data will be collected at SM level
    /// In CUpti_SassMetricsGetDataProperties_Params struct the numOfInstances will be equal to number of SMs in the GPU
    CUPTI_SASS_METRICS_OUTPUT_GRANULARITY_SM = 1,

    /// SASS metric data will be collected at SM sub-partition level
    /// In CUpti_SassMetricsGetDataProperties_Params struct the numOfInstances will be equal to number of SM sub-partitions in the GPU
    CUPTI_SASS_METRICS_OUTPUT_GRANULARITY_SMSP = 2,

    CUPTI_SASS_METRICS_OUTPUT_GRANULARITY_INVALID
} CUpti_SassMetrics_OutputGranularity;

typedef struct CUpti_SassMetrics_MetricDetails
{
    /// unique ID for the SASS metric
    uint64_t metricId;
    /// metric name
    const char* pMetricName;
    /// metric description
    const char* pMetricDescription;
} CUpti_SassMetrics_MetricDetails;

/**
 * \brief Params for cuptiSassMetricsGetNumOfMetrics
 */
typedef struct CUpti_SassMetrics_GetNumOfMetrics_Params
{
    /// [in] should be equal to CUpti_SassMetrics_GetNumOfMetrics_Params_STRUCT_SIZE
    size_t structSize;
    /// [in] assign to NULL
    void* pPriv;
    /// [in] chip name for which metrics will be queried
    const char* pChipName;
    /// [out] number of metrics supported for the queried chip
    size_t numOfMetrics;
} CUpti_SassMetrics_GetNumOfMetrics_Params;

#define CUpti_SassMetrics_GetNumOfMetrics_Params_STRUCT_SIZE                 CUPTI_PROFILER_STRUCT_SIZE(CUpti_SassMetrics_GetNumOfMetrics_Params, numOfMetrics)

/**
 * \brief Get the number of supported SASS metrics for the chip.
 * 
 * \param pParams A pointer to \ref CUpti_SassMetrics_GetNumOfMetrics_Params
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_NOT_SUPPORTED indicates that the system/device doesn't support SASS metric collection
 */
CUptiResult CUPTIAPI cuptiSassMetricsGetNumOfMetrics(CUpti_SassMetrics_GetNumOfMetrics_Params* pParams);

/**
 * \brief Params for cuptiSassMetricsGetMetrics
 */
typedef struct CUpti_SassMetrics_GetMetrics_Params
{
    /// [in] should be equal to CUpti_SassMetrics_GetMetrics_Params_STRUCT_SIZE
    size_t structSize;
    /// [in] assign to NULL
    void* pPriv;
    /// [in] chip name for which metrics will be queried
    const char* pChipName;
    /// [in] number of metrics supported for the queried chip (can be queried using cuptiSassMetricsGetNumOfMetrics())
    size_t numOfMetrics;
    /// [out] list of metrics supported for queried chip
    CUpti_SassMetrics_MetricDetails* pMetricsList;
} CUpti_SassMetrics_GetMetrics_Params;
#define CUpti_SassMetrics_GetMetrics_Params_STRUCT_SIZE                 CUPTI_PROFILER_STRUCT_SIZE(CUpti_SassMetrics_GetMetrics_Params, pMetricsList)

/**
 * \brief Get the list of all supported SASS metrics for the chip.
 * 
 * \param pParams A pointer to \ref CUpti_SassMetrics_GetMetrics_Params
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_NOT_SUPPORTED indicates that the system/device doesn't support SASS metric collection
 */
CUptiResult CUPTIAPI cuptiSassMetricsGetMetrics(CUpti_SassMetrics_GetMetrics_Params* pParams);

/**
 * \brief Params for cuptiSassMetricsGetProperties
 */
typedef struct CUpti_SassMetrics_GetProperties_Params
{
    /// [in] should be equal to CUpti_SassMetrics_GetProperties_Params_STRUCT_SIZE
    size_t structSize;
    /// [in] assign to NULL
    void* pPriv;
    /// [in] chip name for which metric will be queried
    const char* pChipName;
    /// [in] metric name
    const char* pMetricName;
    /// [out] returns the metric ID and the metric description
    CUpti_SassMetrics_MetricDetails metric;
} CUpti_SassMetrics_GetProperties_Params;
#define CUpti_SassMetrics_GetProperties_Params_STRUCT_SIZE        CUPTI_PROFILER_STRUCT_SIZE(CUpti_SassMetrics_GetProperties_Params, metric)

/**
 * \brief Get metric properties for the queried metric.
 * For a given metric the results will be put in CUpti_SassMetrics_MetricDetails which
 * stores metric ID, description of the metric.
 * 
 * \param pParams A pointer to \ref CUpti_SassMetrics_GetProperties_Params
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_NOT_SUPPORTED indicates that the system/device doesn't support SASS metric data collection
 */
CUptiResult CUPTIAPI cuptiSassMetricsGetProperties(CUpti_SassMetrics_GetProperties_Params *pParams);

typedef struct CUpti_SassMetrics_Config
{
    /// [in] unique id for the SASS metric, can be queried using cuptiSassMetricsGetProperties()
    uint64_t metricId;
    /// [in] CUpti_SassMetrics_OutputGranularity
    uint8_t outputGranularity;
} CUpti_SassMetrics_Config;

/**
 * \brief Params for cuptiSassMetricsSetConfig
 */
typedef struct CUpti_SassMetricsSetConfig_Params
{
    /// [in] equal to CUpti_SassMetricsSetConfig_Params_STRUCT_SIZE
    size_t structSize;
    /// [in] assign to NULL
    void* pPriv;
    /// [in] num of metric configs, will be equal to number of metrics queried
    size_t numOfMetricConfig;
    /// [in] list of metric config generated for given sass metrics
    CUpti_SassMetrics_Config* pConfigs;
    /// [in] device index for which config will be set, user can call this once for
    /// the device on which the the SASS metric data will be collected
    uint32_t deviceIndex;
} CUpti_SassMetricsSetConfig_Params;
#define CUpti_SassMetricsSetConfig_Params_STRUCT_SIZE                    CUPTI_PROFILER_STRUCT_SIZE(CUpti_SassMetricsSetConfig_Params, deviceIndex)

/**
 * \brief Set config for the SASS metric data collection for a device.
 * User need to call this API before calling any of the SASS metric data collection APIs.
 * Each set config API call need to be followed by cuptiSassPatchingUnSetConfig API
 * before calling the cuptiSassMetricsSetConfig() API again for the same device.
 * 
 * \param pParams A pointer to \ref CUpti_SassMetricsSetConfig_Params
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_INVALID_CONTEXT if any cuda context has not been created prior to this API call
 * \retval CUPTI_ERROR_INVALID_OPERATION if this is called multiple times for the device without calling unset config API
 * \retval CUPTI_ERROR_NOT_SUPPORTED indicates that the system/device doesn't support SASS metric data collection
 */
CUptiResult CUPTIAPI cuptiSassMetricsSetConfig(CUpti_SassMetricsSetConfig_Params *pParams);

/**
 * \brief Params for cuptiSassMetricsUnsetConfig
 */
typedef struct CUpti_SassMetricsUnsetConfig_Params
{
    /// [in] equal to CUpti_SassMetricsUnsetConfig_Params_STRUCT_SIZE
    size_t structSize;
    /// [in] assign to NULL
    void* pPriv;
    /// [in] device index for which SASS metric data collection config will get reset, user need to call this API for
    /// all the devices on which the the SASS metric data collection have been configured.
    uint32_t deviceIndex;
} CUpti_SassMetricsUnsetConfig_Params;
#define CUpti_SassMetricsUnsetConfig_Params_STRUCT_SIZE                  CUPTI_PROFILER_STRUCT_SIZE(CUpti_SassMetricsUnsetConfig_Params, deviceIndex)

/**
 * \brief Unset config API will reset the SASS metric data collection configuration for the device.
 * Once this API called CUPTI will deallocate all the memory allocated and remove all
 * the configuration for SASS metric data collection. User can only call this API for a device where
 * cuptiSassMetricsSetConfig() API has been called earlier for the device.
 * 
 * \param pParams A pointer to \ref CUpti_SassMetricsSetConfig_Params
 * 
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_INVALID_CONTEXT if any cuda context has not been created prior to this API call
 * \retval CUPTI_ERROR_INVALID_OPERATION if this is called multiple times for the device without calling set config API
 * \retval CUPTI_ERROR_NOT_SUPPORTED indicates that the system/device doesn't support SASS metric data collection
 */
CUptiResult CUPTIAPI cuptiSassMetricsUnsetConfig(CUpti_SassMetricsUnsetConfig_Params *pParams);

/**
 * \brief Params for cuptiSassMetricsEnable
 */
typedef struct CUpti_SassMetricsEnable_Params
{
    /// [in] equal to CUpti_SassMetricsEnable_Params_STRUCT_SIZE
    size_t structSize;
    /// [in] assign to NULL
    void* pPriv;
    /// [in] CUDA context on which SASS metric data collection will be enabled.
    /// If set NULL, default context will be consider for SASS metric data collection.
    CUcontext ctx;
    /// [in] if false, all the functions will patched regardless of their execution with cuptiSassMetricsEnable() API call.
    /// when this parameter is set to true, metric data collection for the function will be done at the very first execution in the enable/disble
    /// range.
    uint8_t enableLazyPatching;
} CUpti_SassMetricsEnable_Params;
#define CUpti_SassMetricsEnable_Params_STRUCT_SIZE                       CUPTI_PROFILER_STRUCT_SIZE(CUpti_SassMetricsEnable_Params, enableLazyPatching)

/**
 * \brief Sass metric data collection enable API will mark the start of a range, between which kernel
 *  will be profiled for SASS metrics.
 *
 * \param pParams A pointer to \ref CUpti_SassMetricsEnable_Params
 * 
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_NOT_SUPPORTED indicates that the system/device doesn't support SASS metric data collection
 * \retval CUPTI_ERROR_INVALID_CONTEXT if any cuda context has not been created prior to this API call
 * \retval CUPTI_ERROR_INVALID_OPERATION if this API is called multiple times for a cuda context without calling 
 * cuptiSassMetricsDisable() API or called before cuptiSassMetricsSetConfig() API call.
 */
CUptiResult CUPTIAPI cuptiSassMetricsEnable(CUpti_SassMetricsEnable_Params* pParams);

/**
 * \brief Params for cuptiSassMetricsDisable
 */
typedef struct CUpti_SassMetricsDisable_Params
{
    /// [in] equal to CUpti_SassMetricsDisable_Params_STRUCT_SIZE
    size_t structSize;
    /// [in] assign to NULL
    void* pPriv;
    /// [in] CUDA context on which SASS metric data collection will be disabled.
    /// If set NULL, default context will be consider for SASS metric data collection.
    CUcontext ctx;
    /// [out] Num of dropped SASS records will be equal to numOfPatchedInstructions * numOfInstances.
    /// Number of dropped records will be zero when data is flushed prior to calling the disable API.
    size_t numOfDroppedRecords;
} CUpti_SassMetricsDisable_Params;
#define CUpti_SassMetricsDisable_Params_STRUCT_SIZE                      CUPTI_PROFILER_STRUCT_SIZE(CUpti_SassMetricsDisable_Params, numOfDroppedRecords)

/**
 * \brief SASS metric data collection disable API will mark the end of a range, any kernel launched after this
 * API call will not be profiled for the SASS metrics.
 *
 * \param pParams A pointer to \ref CUpti_SassMetricsDisable_Params
 * 
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_NOT_SUPPORTED indicates that the system/device doesn't support SASS metric data collection
 * \retval CUPTI_ERROR_INVALID_CONTEXT if any cuda context has not been created prior to this API call
 * \retval CUPTI_ERROR_INVALID_OPERATION if this API is called multiple times for a cuda context without calling 
 * cuptiSassMetricsEnable() API or called before cuptiSassMetricsSetConfig() API call.
 */
CUptiResult CUPTIAPI cuptiSassMetricsDisable(CUpti_SassMetricsDisable_Params* pParams);

/**
 * \brief Params for cuptiSassMetricsGetDataProperties
 */
typedef struct CUpti_SassMetricsGetDataProperties_Params
{
    /// [in] equal to CUpti_SassMetricsGetDataProperties_Params_STRUCT_SIZE
    size_t structSize;
    /// [in] assign to NULL
    void* pPriv;
    /// [in] CUDA context on which SASS metric data collection was enabled.
    /// If set NULL, default context will be consider for SASS metric data collection.
    CUcontext ctx;
    /// [out] total number of SASS records has been collected
    size_t numOfPatchedInstructionRecords;
    /// [out] number of instances for each metric value per instruction.
    /// This will depend on CUpti_SassPatching_OutputGranularity level set for the metric config.
    size_t numOfInstances;
} CUpti_SassMetricsGetDataProperties_Params;

#define CUpti_SassMetricsGetDataProperties_Params_STRUCT_SIZE           CUPTI_PROFILER_STRUCT_SIZE(CUpti_SassMetricsGetDataProperties_Params, numOfInstances)
/**
 * \brief SASS metric data properties API will give the data regarding number of instances of a metric
 * value and number of SASS instruction data has been collected. The number of instances of a metric
 * will vary as per user set the output granularity level with CUpti_SassMetrics_OutputGranularity value.
 * User need to allocate memory for retriving the SASS data using cuptiSassMetricsFlushData() API.
 * 
 * \param pParams A pointer to \ref CUpti_SassMetricsGetDataProperties_Params
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_NOT_SUPPORTED indicates that the system/device doesn't support SASS metric data collection
 * \retval CUPTI_ERROR_INVALID_OPERATION if this API is called outside the enable/disable range.
 */
CUptiResult CUPTIAPI cuptiSassMetricsGetDataProperties(CUpti_SassMetricsGetDataProperties_Params* pParams);

typedef struct CUpti_SassMetrics_InstanceValue
{
    // unique id of the metric
    uint64_t metricId;
    // metric value 
    uint64_t value;
} CUpti_SassMetrics_InstanceValue;
#define CUpti_SassMetrics_InstanceValue_STRUCT_SIZE                      CUPTI_PROFILER_STRUCT_SIZE(CUpti_SassMetrics_InstanceValue, value)

typedef struct CUpti_SassMetrics_Data
{
    /// [in] equal to CUpti_SassMetricsFlushData_Params_STRUCT_SIZE
    size_t structSize;
    /// [in] assign to NULL
    void* pPriv;
    /// [out] Unique cubin id
    uint32_t cubinCrc;
    /// [out] function's unique symbol index in the module.
    uint32_t functionIndex;
    /// [out] The function name
    const char* functionName;
    /// [out] pc offset for the function in a module
    uint32_t pcOffset;
    /// [out] array of size equal to number of instances per metric, which contains the metric ID and metric value.
    CUpti_SassMetrics_InstanceValue* pInstanceValues;
} CUpti_SassMetrics_Data;

/**
 * \brief Params for cuptiSassMetricsFlushData
 */
typedef struct CUpti_SassMetricsFlushData_Params
{
    /// [in] equal to CUpti_SassMetricsFlushData_Params_STRUCT_SIZE
    size_t structSize;
    /// [in] assign to NULL
    void* pPriv;
    /// [in] CUDA context on which SASS metric data collection was enabled.
    /// If set NULL, default context will be consider for SASS metric data collection.
    CUcontext ctx;
    /// [in] number of patched instruction record will be retrived, user can call cuptiSassMetricsGetDataProperties()
    /// for getting total number of records available.
    size_t numOfPatchedInstructionRecords;
    /// [in] number of patched instruction record instances for a metric, user can call cuptiSassMetricsGetDataProperties()
    /// for getting total number of instances for each record per metric available.
    size_t numOfInstances;
    /// [out] 
    CUpti_SassMetrics_Data* pMetricsData;
} CUpti_SassMetricsFlushData_Params;
#define CUpti_SassMetricsFlushData_Params_STRUCT_SIZE                      CUPTI_PROFILER_STRUCT_SIZE(CUpti_SassMetricsFlushData_Params, numOfInstances)

/**
 * \brief Flush SASS metrics data from CUPTI internal buffer to the user buffer.
 * User needs to allocate the buffer for retrieving the data. The number of records collected
 * can be queried using the API cuptiSassMetricsGetDataProperties().
 *
 * \param pParams A pointer to \ref CUpti_SassMetricsFlushData_Params
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if any \p pParams is not valid
 * \retval CUPTI_ERROR_NOT_SUPPORTED indicates that the system/device doesn't support SASS metric data collection.
 * \retval CUPTI_ERROR_INVALID_OPERATION if this API is called outside the enable/disable range.
 */
CUptiResult CUPTIAPI cuptiSassMetricsFlushData(CUpti_SassMetricsFlushData_Params* pParams);

/** @} */ /* END CUPTI_SASS_METRICS_API */

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility pop
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif // _CUPTI_SASS_METRICS_H_