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

// to easily switch off fp128 device functions if needed
#ifndef __NV_DISABLE_DEVICE_FP128_FUNCTIONS__

#if !defined(__DEVICE_FP128_FUNCTIONS_H__)
#define __DEVICE_FP128_FUNCTIONS_H__

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if defined(__cplusplus) && defined(__CUDACC__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "builtin_types.h"
#include "device_types.h"

#if !defined(__CUDA_ARCH__) && !defined(_NVHPC_CUDA)
#define __DEF_IF_HOST { }
#define __INLINE_IF_HOST__ __inline__
#else  /* !__CUDA_ARCH__ */
#define __DEF_IF_HOST ;
#define __INLINE_IF_HOST__
#endif /* __CUDA_ARCH__ */

#define __DEVICE_FP128_FUNCTIONS_DECL__ __device__ __cudart_builtin__ __INLINE_IF_HOST__

/*******************************************************************************
*                                                                              *
* Support for __float128 on:                                                   *
*    - NVRTC on Linux                                                          *
*    - GCC version 4.1 or later on x86_64/amd64                                *
*    - Clang version 3.9 or later on x86_64/amd64                              *
*    - NVHPC version 21.1 or later on x86_64/amd64                             *
*                                                                              *
*******************************************************************************/
#if defined(__CUDACC_RTC__)
#if !_WIN64
#define __FLOAT128_CPP_SPELLING_ENABLED__
#endif
#else /* !__CUDACC_RTC__ */

#if (defined __NVCOMPILER_MAJOR__)
    #if (defined(__x86_64__) || defined(__amd64__)) && \
        ((__NVCOMPILER_MAJOR__ > 21) || \
            (__NVCOMPILER_MAJOR__ == 21 && __NVCOMPILER_MINOR__ >= 1))
        #define __FLOAT128_CPP_SPELLING_ENABLED__
    #endif
#elif defined(__clang__)
    #if (defined(__x86_64__) || defined(__amd64__)) && \
        ((__clang_major__ > 3) || \
            (__clang_major__ == 3 && __clang_minor__ >= 9))
        #define __FLOAT128_CPP_SPELLING_ENABLED__
    #endif
#elif defined(__GNUC__)
    // check gcc version if no other host compiler is used
    #if (defined(__x86_64__) || defined(__amd64__)) && \
        ((__GNUC__ > 4) || \
            (__GNUC__ == 4 && __GNUC_MINOR__ >= 1))
        #define __FLOAT128_CPP_SPELLING_ENABLED__
    #endif
#endif /* (defined __NVCOMPILER_MAJOR__) */

#endif /* !__CUDACC_RTC__ */

/*******************************************************************************
*                                                                              *
* Support for _Float128 on:                                                    *
*    - GCC version 13.1 or later on x86_64/amd64/aarch64                       *
*                                                                              *
*******************************************************************************/
#if defined(__GNUC__) && !defined(__clang__) && !defined(__NVCOMPILER_MAJOR__)
    // check gcc version if no other host compiler is used
    #if (defined(__x86_64__) || defined(__amd64__) || defined(__aarch64__)) && \
        ((__GNUC__ > 13) || \
            (__GNUC__ == 13 && __GNUC_MINOR__ >= 1))
        #define __FLOAT128_C_SPELLING_ENABLED__
    #endif
#endif /* defined(__GNUC__) && !defined(__clang__) && !defined(__NVCOMPILER_MAJOR__) */

/**
 * \defgroup CUDA_MATH_QUAD FP128 Quad Precision Mathematical Functions
 * This section describes quad precision mathematical functions.
 * To use these functions, include the header file \p device_fp128_functions.h in your program.
 * 
 * Functions declared here have \p __nv_fp128_ prefix to distinguish them
 * from other global namespace symbols.
 *
 * Note that FP128 CUDA Math functions are only available to device programs
 * on platforms where host compiler supports the basic quad precision datatype
 * \p __float128 or \p _Float128.
 * 
 * Every FP128 CUDA Math function name is overloaded to support either of these
 * host-compiler-specific types, whenever the types are available. See for example:
 * \code
 * #ifdef __FLOAT128_CPP_SPELLING_ENABLED__
 *     __float128 __nv_fp128_sqrt(__float128 x);
 * #endif
 * #ifdef __FLOAT128_C_SPELLING_ENABLED__
 *     _Float128 __nv_fp128_sqrt(_Float128 x);
 * #endif
 * \endcode
 *
 * \note_fp128_target_arch
 */

#ifdef __FLOAT128_CPP_SPELLING_ENABLED__
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Calculate \cuda_math_formula \sqrt{x} \end_cuda_math_formula, the square root of the input argument.
 *
 * \return 
 * \cuda_math_formula \sqrt{x} \end_cuda_math_formula.
 * - __nv_fp128_sqrt(
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * ) returns 
 * \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - __nv_fp128_sqrt(
 * \cuda_math_formula +\infty \end_cuda_math_formula
 * ) returns 
 * \cuda_math_formula +\infty \end_cuda_math_formula.
 * - __nv_fp128_sqrt(\p x) returns NaN if \p x is less than 0.
 * - __nv_fp128_sqrt(NaN) returns NaN.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_sqrt(__float128 x) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Calculate \cuda_math_formula \sin{x} \end_cuda_math_formula, the sine of input argument (measured in radians).
 * 
 * \return 
 * \cuda_math_formula \sin{x} \end_cuda_math_formula.
 * - __nv_fp128_sin(
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * ) returns 
 * \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - __nv_fp128_sin(
 * \cuda_math_formula \pm \infty \end_cuda_math_formula
 * ) returns NaN.
 * - __nv_fp128_sin(NaN) returns NaN.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_sin(__float128 x) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Calculate \cuda_math_formula \cos{x} \end_cuda_math_formula, the cosine of input argument (measured in radians).
 * 
 * \return 
 * \cuda_math_formula \cos{x} \end_cuda_math_formula.
 * - __nv_fp128_cos(
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * ) returns 
 * \cuda_math_formula 1 \end_cuda_math_formula.
 * - __nv_fp128_cos(
 * \cuda_math_formula \pm \infty \end_cuda_math_formula
 * ) returns NaN.
 * - __nv_fp128_cos(NaN) returns NaN.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_cos(__float128 x) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Calculate \cuda_math_formula \tan{x} \end_cuda_math_formula, the tangent of input argument (measured in radians).
 * 
 * \return 
 * \cuda_math_formula \tan{x} \end_cuda_math_formula.
 * - __nv_fp128_tan(
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * ) returns 
 * \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - __nv_fp128_tan(
 * \cuda_math_formula \pm \infty \end_cuda_math_formula
 * ) returns NaN.
 * - __nv_fp128_tan(NaN) returns NaN.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_tan(__float128 x) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Calculate \cuda_math_formula \sin^{-1}{x} \end_cuda_math_formula, the arc sine of input argument.
 * 
 * \return 
 * The principal value of the arc sine of the input argument \p x.
 * Result will be in radians, in the interval [-
 * \cuda_math_formula \pi/2 \end_cuda_math_formula
 * , +
 * \cuda_math_formula \pi/2 \end_cuda_math_formula
 * ] for \p x inside [-1, +1].
 * - __nv_fp128_asin(
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * ) returns
 * \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - __nv_fp128_asin(\p x) returns NaN for \p x outside [-1, +1].
 * - __nv_fp128_asin(NaN) returns NaN.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_asin(__float128 x) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Calculate \cuda_math_formula \cos^{-1}{x} \end_cuda_math_formula, the arc cosine of input argument.
 *
 * \return 
 * The principal value of the arc cosine of the input argument \p x.
 * Result will be in radians, in the interval [0, 
 * \cuda_math_formula \pi \end_cuda_math_formula
 * ] for \p x inside [-1, +1].
 * - __nv_fp128_acos(1) returns +0.
 * - __nv_fp128_acos(\p x) returns NaN for \p x outside [-1, +1].
 * - __nv_fp128_acos(NaN) returns NaN.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_acos(__float128 x) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Calculate \cuda_math_formula \tan^{-1}{x} \end_cuda_math_formula, the arc tangent of input argument.
 *
 * \return 
 * The principal value of the arc tangent of the input argument \p x.
 * Result will be in radians, in the interval [-
 * \cuda_math_formula \pi/2 \end_cuda_math_formula
 * , +
 * \cuda_math_formula \pi/2 \end_cuda_math_formula
 * ].
 * - __nv_fp128_atan(
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * ) returns
 * \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - __nv_fp128_atan(
 * \cuda_math_formula \pm \infty \end_cuda_math_formula
 * ) returns
 * \cuda_math_formula \pm \pi \end_cuda_math_formula
 * /2.
 * - __nv_fp128_atan(NaN) returns NaN.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_atan(__float128 x) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Calculate \cuda_math_formula e^x \end_cuda_math_formula, the base 
 * \cuda_math_formula e \end_cuda_math_formula
 *  exponential of the input argument.
 *
 * \return
 * - __nv_fp128_exp(
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * ) returns 1.
 * - __nv_fp128_exp(
 * \cuda_math_formula -\infty \end_cuda_math_formula
 * ) returns +0.
 * - __nv_fp128_exp(
 * \cuda_math_formula +\infty \end_cuda_math_formula
 * ) returns
 * \cuda_math_formula +\infty \end_cuda_math_formula.
 * - __nv_fp128_exp(NaN) returns NaN.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_exp(__float128 x) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Calculate \cuda_math_formula 2^x \end_cuda_math_formula, the base 2 exponential of the input argument.
 *
 * \return
 * - __nv_fp128_exp2(
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * ) returns 1.
 * - ex__nv_fp128_exp2p2f(
 * \cuda_math_formula -\infty \end_cuda_math_formula
 * ) returns +0.
 * - __nv_fp128_exp2(
 * \cuda_math_formula +\infty \end_cuda_math_formula
 * ) returns
 * \cuda_math_formula +\infty \end_cuda_math_formula.
 * - __nv_fp128_exp2(NaN) returns NaN.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_exp2(__float128 x) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Calculate \cuda_math_formula 10^x \end_cuda_math_formula, the base 10 exponential of the input argument.
 *
 * \return
 * - __nv_fp128_exp10(
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * ) returns 1.
 * - __nv_fp128_exp10(
 * \cuda_math_formula -\infty \end_cuda_math_formula
 * ) returns +0.
 * - __nv_fp128_exp10(
 * \cuda_math_formula +\infty \end_cuda_math_formula
 * ) returns
 * \cuda_math_formula +\infty \end_cuda_math_formula.
 * - __nv_fp128_exp10(NaN) returns NaN.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_exp10(__float128 x) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Calculate 
 * \cuda_math_formula e^x - 1 \end_cuda_math_formula,
 * the base e exponential of the input argument, minus 1.
 *
 * \return
 * - __nv_fp128_expm1(
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * ) returns
 * \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - __nv_fp128_expm1(
 * \cuda_math_formula -\infty \end_cuda_math_formula
 * ) returns -1.
 * - __nv_fp128_expm1(
 * \cuda_math_formula +\infty \end_cuda_math_formula
 * ) returns
 * \cuda_math_formula +\infty \end_cuda_math_formula.
 * - __nv_fp128_expm1(NaN) returns NaN.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_expm1(__float128 x) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Calculate \cuda_math_formula \log_{e}{x} \end_cuda_math_formula, the base 
 * \cuda_math_formula e \end_cuda_math_formula
 *  logarithm of the input argument.
 *
 * \return
 * - __nv_fp128_log(
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * ) returns 
 * \cuda_math_formula -\infty \end_cuda_math_formula.
 * - __nv_fp128_log(1) returns +0.
 * - __nv_fp128_log(\p x) returns NaN for \p x < 0.
 * - __nv_fp128_log(
 * \cuda_math_formula +\infty \end_cuda_math_formula
 * ) returns 
 * \cuda_math_formula +\infty \end_cuda_math_formula.
 * - __nv_fp128_log(NaN) returns NaN.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_log(__float128 x) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Calculate \cuda_math_formula \log_{2}{x} \end_cuda_math_formula, the base 2 logarithm of the input argument.
 *
 * \return 
 * - __nv_fp128_log2(
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * ) returns 
 * \cuda_math_formula -\infty \end_cuda_math_formula.
 * - __nv_fp128_log2(1) returns +0.
 * - __nv_fp128_log2(\p x) returns NaN for \p x < 0.
 * - __nv_fp128_log2(
 * \cuda_math_formula +\infty \end_cuda_math_formula
 * ) returns 
 * \cuda_math_formula +\infty \end_cuda_math_formula.
 * - __nv_fp128_log2(NaN) returns NaN.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_log2(__float128 x) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Calculate \cuda_math_formula \log_{10}{x} \end_cuda_math_formula, the base 10 logarithm of the input argument.
 *
 * \return 
 * - __nv_fp128_log10(
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * ) returns 
 * \cuda_math_formula -\infty \end_cuda_math_formula.
 * - __nv_fp128_log10(1) returns +0.
 * - __nv_fp128_log10(\p x) returns NaN for \p x < 0.
 * - __nv_fp128_log10(
 * \cuda_math_formula +\infty \end_cuda_math_formula
 * ) returns 
 * \cuda_math_formula +\infty \end_cuda_math_formula.
 * - __nv_fp128_log10(NaN) returns NaN.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_log10(__float128 x) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Calculate the value of 
 * \cuda_math_formula \log_{e}(1+x) \end_cuda_math_formula.
 *
 * \return
 * - __nv_fp128_log1p(
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * ) returns 
 * \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - __nv_fp128_log1p(-1) returns
 * \cuda_math_formula -\infty \end_cuda_math_formula.
 * - __nv_fp128_log1p(\p x) returns NaN for \p x < -1.
 * - __nv_fp128_log1p(
 * \cuda_math_formula +\infty \end_cuda_math_formula
 * ) returns 
 * \cuda_math_formula +\infty \end_cuda_math_formula.
 * - __nv_fp128_log1p(NaN) returns NaN.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_log1p(__float128 x) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Calculate the value of \cuda_math_formula x^{y} \end_cuda_math_formula, first argument to the power of second argument.
 *
 * \return 
 * - __nv_fp128_pow(
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * , \p y) returns 
 * \cuda_math_formula \pm \infty \end_cuda_math_formula
 *  for \p y an odd integer less than 0.
 * - __nv_fp128_pow(
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * , \p y) returns 
 * \cuda_math_formula +\infty \end_cuda_math_formula
 *  for \p y less than 0 and not an odd integer.
 * - __nv_fp128_pow(
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * , \p y) returns 
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 *  for \p y an odd integer greater than 0.
 * - __nv_fp128_pow(
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * , \p y) returns +0 for \p y > 0 and not an odd integer.
 * - __nv_fp128_pow(-1, 
 * \cuda_math_formula \pm \infty \end_cuda_math_formula
 * ) returns 1.
 * - __nv_fp128_pow(+1, \p y) returns 1 for any \p y, even a NaN.
 * - __nv_fp128_pow(\p x, 
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * ) returns 1 for any \p x, even a NaN.
 * - __nv_fp128_pow(\p x, \p y) returns a NaN for finite \p x < 0 and finite non-integer \p y.
 * - __nv_fp128_pow(\p x, 
 * \cuda_math_formula -\infty \end_cuda_math_formula
 * ) returns 
 * \cuda_math_formula +\infty \end_cuda_math_formula
 *  for 
 * \cuda_math_formula | x | < 1 \end_cuda_math_formula.
 * - __nv_fp128_pow(\p x, 
 * \cuda_math_formula -\infty \end_cuda_math_formula
 * ) returns +0 for 
 * \cuda_math_formula | x | > 1 \end_cuda_math_formula.
 * - __nv_fp128_pow(\p x, 
 * \cuda_math_formula +\infty \end_cuda_math_formula
 * ) returns +0 for 
 * \cuda_math_formula | x | < 1 \end_cuda_math_formula.
 * - __nv_fp128_pow(\p x, 
 * \cuda_math_formula +\infty \end_cuda_math_formula
 * ) returns 
 * \cuda_math_formula +\infty \end_cuda_math_formula
 *  for 
 * \cuda_math_formula | x | > 1 \end_cuda_math_formula.
 * - __nv_fp128_pow(
 * \cuda_math_formula -\infty \end_cuda_math_formula
 * , \p y) returns -0 for \p y an odd integer less than 0.
 * - __nv_fp128_pow(
 * \cuda_math_formula -\infty \end_cuda_math_formula
 * , \p y) returns +0 for \p y < 0 and not an odd integer.
 * - __nv_fp128_pow(
 * \cuda_math_formula -\infty \end_cuda_math_formula
 * , \p y) returns 
 * \cuda_math_formula -\infty \end_cuda_math_formula
 *  for \p y an odd integer greater than 0.
 * - __nv_fp128_pow(
 * \cuda_math_formula -\infty \end_cuda_math_formula
 * , \p y) returns 
 * \cuda_math_formula +\infty \end_cuda_math_formula
 *  for \p y > 0 and not an odd integer.
 * - __nv_fp128_pow(
 * \cuda_math_formula +\infty \end_cuda_math_formula
 * , \p y) returns +0 for \p y < 0.
 * - __nv_fp128_pow(
 * \cuda_math_formula +\infty \end_cuda_math_formula
 * , \p y) returns 
 * \cuda_math_formula +\infty \end_cuda_math_formula
 *  for \p y > 0.
 * - __nv_fp128_pow(\p x, \p y) returns NaN if either \p x or \p y or both are NaN and \p x \cuda_math_formula \neq \end_cuda_math_formula +1 and \p y \cuda_math_formula \neq\pm 0 \end_cuda_math_formula.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_pow(__float128 x, __float128 y) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Calculate \cuda_math_formula \sinh{x} \end_cuda_math_formula, the hyperbolic sine of the input argument.
 *
 * Calculate \cuda_math_formula \sinh{x} \end_cuda_math_formula, the hyperbolic sine of the input argument \p x.
 *
 * \return
 * - __nv_fp128_sinhinh(
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * ) returns 
 * \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - __nv_fp128_sinh(
 * \cuda_math_formula \pm \infty \end_cuda_math_formula
 * ) returns
 * \cuda_math_formula \pm \infty \end_cuda_math_formula.
 * - __nv_fp128_sinh(NaN) returns NaN.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_sinh(__float128 x) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Calculate \cuda_math_formula \cosh{x} \end_cuda_math_formula, the hyperbolic cosine of the input argument.
 *
 * \return
 * - __nv_fp128_cosh(
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * ) returns 1.
 * - __nv_fp128_cosh(
 * \cuda_math_formula \pm \infty \end_cuda_math_formula
 * ) returns 
 * \cuda_math_formula +\infty \end_cuda_math_formula.
 * - __nv_fp128_cosh(NaN) returns NaN.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_cosh(__float128 x) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Calculate \cuda_math_formula \tanh{x} \end_cuda_math_formula, the hyperbolic tangent of the input argument.
 *
 * \return
 * - __nv_fp128_tanh(
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * ) returns 
 * \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - __nv_fp128_tanh( 
 * \cuda_math_formula \pm \infty \end_cuda_math_formula
 * ) returns
 * \cuda_math_formula \pm 1 \end_cuda_math_formula.
 * - __nv_fp128_tanh(NaN) returns NaN.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_tanh(__float128 x) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Calculate \cuda_math_formula \sinh^{-1}{x} \end_cuda_math_formula, the inverse hyperbolic sine of the input argument.
 *
 * \return
 * - __nv_fp128_asinh(
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * ) returns 
 * \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - __nv_fp128_asinh(
 * \cuda_math_formula \pm \infty \end_cuda_math_formula
 * ) returns 
 * \cuda_math_formula \pm \infty \end_cuda_math_formula. 
 * - __nv_fp128_asinh(NaN) returns NaN.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_asinh(__float128 x) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Calculate \cuda_math_formula \cosh^{-1}{x} \end_cuda_math_formula, the nonnegative inverse hyperbolic cosine of the input argument.
 *
 * \return 
 * Result will be in the interval [0, 
 * \cuda_math_formula +\infty \end_cuda_math_formula
 * ].
 * - __nv_fp128_acosh(1) returns 0.
 * - __nv_fp128_acosh(\p x) returns NaN for \p x in the interval [
 * \cuda_math_formula -\infty \end_cuda_math_formula
 * , 1).
 * - __nv_fp128_acosh( 
 * \cuda_math_formula +\infty \end_cuda_math_formula
 * ) returns
 * \cuda_math_formula +\infty \end_cuda_math_formula.
 * - __nv_fp128_acosh(NaN) returns NaN.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_acosh(__float128 x) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Calculate \cuda_math_formula \tanh^{-1}{x} \end_cuda_math_formula, the inverse hyperbolic tangent of the input argument.
 *
 * \return 
 * - __nv_fp128_atanh(
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * ) returns 
 * \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - __nv_fp128_atanh(
 * \cuda_math_formula \pm 1 \end_cuda_math_formula
 * ) returns 
 * \cuda_math_formula \pm \infty \end_cuda_math_formula.
 * - __nv_fp128_atanh(\p x) returns NaN for \p x outside interval [-1, 1].
 * - __nv_fp128_atanh(NaN) returns NaN.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_atanh(__float128 x) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Truncate input argument to the integral part.
 *
 * \return 
 * Rounded \p x to the nearest integer value in floating-point format, that does not exceed \p x in 
 * magnitude.
 * - __nv_fp128_trunc(
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * ) returns 
 * \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - __nv_fp128_trunc(
 * \cuda_math_formula \pm \infty \end_cuda_math_formula
 * ) returns 
 * \cuda_math_formula \pm \infty \end_cuda_math_formula.
 * - __nv_fp128_trunc(NaN) returns NaN.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_trunc(__float128 x) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Calculate \cuda_math_formula \lfloor x \rfloor \end_cuda_math_formula, the largest integer less than or equal to \p x.
 * 
 * \return
 * \cuda_math_formula \lfloor x \rfloor \end_cuda_math_formula
 *  expressed as a floating-point number.
 * - __nv_fp128_floor(
 * \cuda_math_formula \pm \infty \end_cuda_math_formula
 * ) returns 
 * \cuda_math_formula \pm \infty \end_cuda_math_formula.
 * - __nv_fp128_floor(
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * ) returns 
 * \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - __nv_fp128_floor(NaN) returns NaN.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_floor(__float128 x) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Calculate \cuda_math_formula \lceil x \rceil \end_cuda_math_formula, the smallest integer greater than or equal to \p x.
 * 
 * \return
 * \cuda_math_formula \lceil x \rceil \end_cuda_math_formula
 *  expressed as a floating-point number.
 * - __nv_fp128_ceil(
 * \cuda_math_formula \pm \infty \end_cuda_math_formula
 * ) returns 
 * \cuda_math_formula \pm \infty \end_cuda_math_formula.
 * - __nv_fp128_ceil(
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * ) returns 
 * \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - __nv_fp128_ceil(NaN) returns NaN.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_ceil(__float128 x) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Round to nearest integer value in floating-point format,
 * with halfway cases rounded away from zero.
 *
 * \return 
 * - __nv_fp128_round(
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * ) returns 
 * \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - __nv_fp128_round(
 * \cuda_math_formula \pm \infty \end_cuda_math_formula
 * ) returns 
 * \cuda_math_formula \pm \infty \end_cuda_math_formula.
 * - __nv_fp128_round(NaN) returns NaN.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_round(__float128 x) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Round to nearest integer value in floating-point format,
 * with halfway cases rounded to the nearest even integer value.
 *
 * \return 
 * - __nv_fp128_rint(
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * ) returns 
 * \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - __nv_fp128_rint(
 * \cuda_math_formula \pm \infty \end_cuda_math_formula
 * ) returns 
 * \cuda_math_formula \pm \infty \end_cuda_math_formula.
 * - __nv_fp128_rint(NaN) returns NaN.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_rint(__float128 x) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Calculate \cuda_math_formula |x| \end_cuda_math_formula, the absolute value of the input argument.
 *
 * \return
 * - __nv_fp128_fabs(
 * \cuda_math_formula \pm \infty \end_cuda_math_formula
 * ) returns 
 * \cuda_math_formula +\infty \end_cuda_math_formula.
 * - __nv_fp128_fabs(
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * ) returns +0.
 * - __nv_fp128_fabs(NaN) returns an unspecified NaN.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_fabs(__float128 x) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Create value with the magnitude of the first agument \p x, and the sign of the second argument \p y.
 *
 * \return
 * - copysign(\p NaN, \p y) returns a \p NaN with the sign of \p y.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_copysign(__float128 x, __float128 y) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Determine the maximum numeric value of the arguments.
 *
 * \return
 * The maximum numeric value of the arguments \p x and \p y. Treats NaN 
 * arguments as missing data.
 * - If both arguments are NaN, returns NaN.
 * - If one argument is NaN, returns the numeric argument.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_fmax(__float128 x, __float128 y) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Determine the minimum numeric value of the arguments.
 *
 * \return
 * The minimum numeric value of the arguments \p x and \p y. Treats NaN 
 * arguments as missing data.
 * - If both arguments are NaN, returns NaN.
 * - If one argument is NaN, returns the numeric argument.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_fmin(__float128 x, __float128 y) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Compute the positive difference between \p x and \p y.
 *
 * \return
 * - __nv_fp128_fdim(\p x, \p y) returns \p x - \p y if \cuda_math_formula x > y \end_cuda_math_formula.
 * - __nv_fp128_fdim(\p x, \p y) returns +0 if \cuda_math_formula x \leq y \end_cuda_math_formula.
 * - If either argument is NaN, NaN is returned.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_fdim(__float128 x, __float128 y) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Calculate the floating-point remainder of \p x / \p y.
 *
 * \return
 * The floating-point remainder of the division operation \p x / \p y calculated
 * by this function is exactly the value <tt>x - n*y</tt>, where \p n is \p x / \p y with its fractional part truncated.
 * - The computed value will have the same sign as \p x, and its magnitude will be less than the magnitude of \p y.
 * - __nv_fp128_fmod(
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * , \p y) returns 
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 *  if \p y is not zero.
 * - __nv_fp128_fmod(\p x, 
 * \cuda_math_formula \pm \infty \end_cuda_math_formula
 * ) returns \p x if \p x is finite.
 * - __nv_fp128_fmod(\p x, \p y) returns NaN if \p x is 
 * \cuda_math_formula \pm\infty \end_cuda_math_formula
 *  or \p y is zero.
 * - If either argument is NaN, NaN is returned.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_fmod(__float128 x, __float128 y) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Compute the floating-point remainder function.
 *
 * \return 
 * The floating-point remainder \p r of dividing 
 * \p x by \p y for nonzero \p y is defined as 
 * \cuda_math_formula r = x - n y \end_cuda_math_formula.
 * The value \p n is the integer value nearest 
 * \cuda_math_formula \frac{x}{y} \end_cuda_math_formula. 
 * In the halfway cases when 
 * \cuda_math_formula | n -\frac{x}{y} | = \frac{1}{2} \end_cuda_math_formula
 * , the
 * even \p n value is chosen.
 * - __nv_fp128_remainder(\p x,
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * ) returns NaN.
 * - __nv_fp128_remainder(
 * \cuda_math_formula \pm \infty \end_cuda_math_formula
 * , \p y) returns NaN.
 * - __nv_fp128_remainder(\p x, 
 * \cuda_math_formula \pm \infty \end_cuda_math_formula
 * ) returns \p x for finite \p x.
 * - If either argument is NaN, NaN is returned.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_remainder(__float128 x, __float128 y) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Extract mantissa and exponent of the floating-point input argument.
 * 
 * Decompose the floating-point value \p x into a component \p m for the 
 * normalized fraction element and an integral term \p n for the exponent.
 * The absolute value of \p m will be greater than or equal to 0.5 and 
 * less than 1.0 or it will be equal to 0; 
 * \cuda_math_formula x = m\cdot 2^n \end_cuda_math_formula.
 * The integer exponent \p n will be stored in the location to which \p nptr points.
 *
 * \return
 * The fractional component \p m.
 * - __nv_fp128_frexp(
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * , \p nptr) returns 
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 *  and stores zero in the location pointed to by \p nptr.
 * - __nv_fp128_frexp(
 * \cuda_math_formula \pm \infty \end_cuda_math_formula
 * , \p nptr) returns 
 * \cuda_math_formula \pm \infty \end_cuda_math_formula
 *  and stores an unspecified value in the 
 * location to which \p nptr points.
 * - __nv_fp128_frexp(NaN, \p y) returns a NaN and stores an unspecified value in the location to which \p nptr points.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_frexp(__float128 x, int* nptr) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Break down the input argument into fractional and integral parts.
 *
 * Break down the argument \p x into fractional and integral parts. The 
 * integral part is stored in floating-point format in the location to which \p iptr points.
 * Fractional and integral parts are given the same sign as the argument \p x.
 *
 * \return 
 * - __nv_fp128_modf(
 * \cuda_math_formula \pm x \end_cuda_math_formula
 * , \p iptr) returns a result with the same sign as \p x.
 * - __nv_fp128_modf(
 * \cuda_math_formula \pm \infty \end_cuda_math_formula
 * , \p iptr) returns 
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 *  and stores 
 * \cuda_math_formula \pm \infty \end_cuda_math_formula
 *   in the object pointed to by \p iptr.
 * - __nv_fp128_modf(NaN, \p iptr) stores a NaN in the object pointed to by \p iptr and returns a NaN.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_modf(__float128 x, __float128* iptr) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Calculate \cuda_math_formula \sqrt{x^2+y^2} \end_cuda_math_formula, the square root of the sum of squares of two arguments.
 *
 * \return
 * The length of the hypotenuse of a right triangle whose two sides have lengths 
 * \cuda_math_formula |x| \end_cuda_math_formula and \cuda_math_formula |y| \end_cuda_math_formula without undue overflow or underflow.
 * - __nv_fp128_hypot(\p x,\p y), __nv_fp128_hypot(\p y,\p x), and __nv_fp128_hypot(\p x, \p -y) are equivalent.
 * - __nv_fp128_hypot(\p x,
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * ) is equivalent to __nv_fp128_fabs(\p x).
 * - __nv_fp128_hypot(
 * \cuda_math_formula \pm \infty \end_cuda_math_formula
 * ,\p y) returns
 * \cuda_math_formula +\infty \end_cuda_math_formula,
 * even if \p y is a NaN.
 * - __nv_fp128_hypot(NaN, \p y) returns NaN, when \p y is not \cuda_math_formula \pm\infty \end_cuda_math_formula.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_hypot(__float128 x, __float128 y) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Compute 
 * \cuda_math_formula x \times y + z \end_cuda_math_formula
 * as a single operation using round-to-nearest-even rounding mode.
 *
 * \return
 * The value of 
 * \cuda_math_formula x \times y + z \end_cuda_math_formula
 * as a single ternary operation, rounded once using round-to-nearest,
 * ties-to-even rounding mode.
 * - __nv_fp128_fma(
 * \cuda_math_formula \pm \infty \end_cuda_math_formula
 * , 
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * , \p z) returns NaN.
 * - __nv_fp128_fma(
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * , 
 * \cuda_math_formula \pm \infty \end_cuda_math_formula
 * , \p z) returns NaN.
 * - __nv_fp128_fma(\p x, \p y, 
 * \cuda_math_formula -\infty \end_cuda_math_formula
 * ) returns NaN if 
 * \cuda_math_formula x \times y \end_cuda_math_formula
 *  is an exact 
 * \cuda_math_formula +\infty \end_cuda_math_formula.
 * - __nv_fp128_fma(\p x, \p y, 
 * \cuda_math_formula +\infty \end_cuda_math_formula
 * ) returns NaN if 
 * \cuda_math_formula x \times y \end_cuda_math_formula
 *  is an exact 
 * \cuda_math_formula -\infty \end_cuda_math_formula.
 * - __nv_fp128_fma(\p x, \p y, \cuda_math_formula \pm 0 \end_cuda_math_formula) returns \cuda_math_formula \pm 0 \end_cuda_math_formula if \cuda_math_formula x \times y \end_cuda_math_formula is exact \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - __nv_fp128_fma(\p x, \p y, \cuda_math_formula \mp 0 \end_cuda_math_formula) returns \cuda_math_formula +0 \end_cuda_math_formula if \cuda_math_formula x \times y \end_cuda_math_formula is exact \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - __nv_fp128_fma(\p x, \p y, \p z) returns \cuda_math_formula +0 \end_cuda_math_formula if \cuda_math_formula x \times y + z \end_cuda_math_formula is exactly zero and \cuda_math_formula z \neq 0 \end_cuda_math_formula.
 * - If either argument is NaN, NaN is returned.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_fma(__float128 x, __float128 y, __float128 c) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Calculate the value of 
 * \cuda_math_formula x\cdot 2^{exp} \end_cuda_math_formula.
 *
 * \return
 * - __nv_fp128_ldexp(
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * , \p exp) returns 
 * \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - __nv_fp128_ldexp(\p x, 0) returns \p x.
 * - __nv_fp128_ldexp(
 * \cuda_math_formula \pm \infty \end_cuda_math_formula
 * , \p exp) returns 
 * \cuda_math_formula \pm \infty \end_cuda_math_formula.
 * - __nv_fp128_ldexp(NaN, \p exp) returns NaN.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_ldexp(__float128 x, int exp) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Compute the unbiased integer exponent of the input argument.
 *
 * \return
 * - If successful, returns the unbiased exponent of the argument.
 * - __nv_fp128_ilogb(
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * ) returns <tt>INT_MIN</tt>.
 * - __nv_fp128_ilogb(NaN) returns <tt>INT_MIN</tt>.
 * - __nv_fp128_ilogb(
 * \cuda_math_formula \pm \infty \end_cuda_math_formula
 * ) returns <tt>INT_MAX</tt>.
 * - Note: above behavior does not take into account <tt>FP_ILOGB0</tt> nor <tt>FP_ILOGBNAN</tt>.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ int __nv_fp128_ilogb(__float128 x) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Compute \cuda_math_formula x \cdot y \end_cuda_math_formula, the product of the two floating-point inputs using round-to-nearest-even rounding mode.
 *
 * \return Returns \p x * \p y.
 * - sign of the product \p x * \p y is XOR of the signs of \p x and \p y when neither inputs nor result are NaN.
 * - __nv_fp128_mul(\p x, \p y) is equivalent to __nv_fp128_mul(\p y, \p x).
 * - __nv_fp128_mul(\p x, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns \cuda_math_formula \infty \end_cuda_math_formula of appropriate sign for \p x \cuda_math_formula \neq 0 \end_cuda_math_formula.
 * - __nv_fp128_mul(\cuda_math_formula \pm 0 \end_cuda_math_formula, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns NaN.
 * - __nv_fp128_mul(\cuda_math_formula \pm 0 \end_cuda_math_formula, \p y) returns \cuda_math_formula 0 \end_cuda_math_formula of appropriate sign for finite \p y.
 * - If either argument is NaN, NaN is returned.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_mul(__float128 x, __float128 y) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Compute \cuda_math_formula x + y \end_cuda_math_formula, the sum of the two floating-point inputs using round-to-nearest-even rounding mode.
 *
 * \return Returns \p x + \p y.
 * - __nv_fp128_add(\p x, \p y) is equivalent to __nv_fp128_add(\p y, \p x).
 * - __nv_fp128_add(\p x, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns \cuda_math_formula \pm\infty \end_cuda_math_formula for finite \p x.
 * - __nv_fp128_add(\cuda_math_formula \pm\infty \end_cuda_math_formula, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns \cuda_math_formula \pm\infty \end_cuda_math_formula.
 * - __nv_fp128_add(\cuda_math_formula \pm\infty \end_cuda_math_formula, \cuda_math_formula \mp\infty \end_cuda_math_formula) returns NaN.
 * - __nv_fp128_add(\cuda_math_formula \pm 0 \end_cuda_math_formula, \cuda_math_formula \pm 0 \end_cuda_math_formula) returns \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - __nv_fp128_add(\p x, \p -x) returns \cuda_math_formula +0 \end_cuda_math_formula for finite \p x, including \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - If either argument is NaN, NaN is returned.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_add(__float128 x, __float128 y) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Compute \cuda_math_formula x - y \end_cuda_math_formula, the difference of the two floating-point inputs using round-to-nearest-even rounding mode.
 *
 * \return Returns \p x - \p y.
 * - __nv_fp128_sub(\cuda_math_formula \pm\infty \end_cuda_math_formula, \p y) returns \cuda_math_formula \pm\infty \end_cuda_math_formula for finite \p y.
 * - __nv_fp128_sub(\p x, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns \cuda_math_formula \mp\infty \end_cuda_math_formula for finite \p x.
 * - __nv_fp128_sub(\cuda_math_formula \pm\infty \end_cuda_math_formula, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns NaN.
 * - __nv_fp128_sub(\cuda_math_formula \pm\infty \end_cuda_math_formula, \cuda_math_formula \mp\infty \end_cuda_math_formula) returns \cuda_math_formula \pm\infty \end_cuda_math_formula.
 * - __nv_fp128_sub(\cuda_math_formula \pm 0 \end_cuda_math_formula, \cuda_math_formula \mp 0 \end_cuda_math_formula) returns \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - __nv_fp128_sub(\p x, \p x) returns \cuda_math_formula +0 \end_cuda_math_formula for finite \p x, including \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - If either argument is NaN, NaN is returned.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_sub(__float128 x, __float128 y) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Compute \cuda_math_formula \frac{x}{y} \end_cuda_math_formula, the quotient of the two floating-point inputs using round-to-nearest-even rounding mode.
 *
 * \return
 * - sign of the quotient \p x / \p y is XOR of the signs of \p x and \p y when neither inputs nor result are NaN.
 * - __nv_fp128_div(\cuda_math_formula \pm 0 \end_cuda_math_formula, \cuda_math_formula \pm 0 \end_cuda_math_formula) returns NaN.
 * - __nv_fp128_div(\cuda_math_formula \pm\infty \end_cuda_math_formula, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns NaN.
 * - __nv_fp128_div(\p x, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns \cuda_math_formula 0 \end_cuda_math_formula of appropriate sign for finite \p x.
 * - __nv_fp128_div(\cuda_math_formula \pm\infty \end_cuda_math_formula, \p y) returns \cuda_math_formula \infty \end_cuda_math_formula of appropriate sign for finite \p y.
 * - __nv_fp128_div(\p x, \cuda_math_formula \pm 0 \end_cuda_math_formula) returns \cuda_math_formula \infty \end_cuda_math_formula of appropriate sign for \p x \cuda_math_formula \neq 0 \end_cuda_math_formula.
 * - __nv_fp128_div(\cuda_math_formula \pm 0 \end_cuda_math_formula, \p y) returns \cuda_math_formula 0 \end_cuda_math_formula of appropriate sign for \p y \cuda_math_formula \neq 0 \end_cuda_math_formula.
 * - If either argument is NaN, NaN is returned.
 *
 * \note_accuracy_quad
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ __float128 __nv_fp128_div(__float128 x, __float128 y) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Determine whether the input argument is a NaN.
 *
 * \return
 * A nonzero value if and only if \p x is a NaN value.
 *
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ int __nv_fp128_isnan(__float128 x) __DEF_IF_HOST
/**
 * \ingroup CUDA_MATH_QUAD
 * \brief Determine whether the pair of inputs is unordered.
 *
 * \return
 * - nonzero value if at least one of input values is a NaN.
 * - zero otherwise
 *
 * \note_fp128_target_arch
 */
__DEVICE_FP128_FUNCTIONS_DECL__ int __nv_fp128_isunordered(__float128 x, __float128 y) __DEF_IF_HOST
#endif /* __FLOAT128_CPP_SPELLING_ENABLED__ */


#ifdef __FLOAT128_C_SPELLING_ENABLED__
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_sqrt(_Float128 x) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_sin(_Float128 x) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_cos(_Float128 x) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_tan(_Float128 x) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_asin(_Float128 x) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_acos(_Float128 x) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_atan(_Float128 x) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_exp(_Float128 x) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_exp2(_Float128 x) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_exp10(_Float128 x) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_expm1(_Float128 x) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_log(_Float128 x) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_log2(_Float128 x) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_log10(_Float128 x) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_log1p(_Float128 x) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_pow(_Float128 x, _Float128 y) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_sinh(_Float128 x) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_cosh(_Float128 x) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_tanh(_Float128 x) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_asinh(_Float128 x) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_acosh(_Float128 x) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_atanh(_Float128 x) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_trunc(_Float128 x) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_floor(_Float128 x) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_ceil(_Float128 x) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_round(_Float128 x) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_rint(_Float128 x) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_fabs(_Float128 x) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_copysign(_Float128 x, _Float128 y) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_fmax(_Float128 x, _Float128 y) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_fmin(_Float128 x, _Float128 y) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_fdim(_Float128 x, _Float128 y) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_fmod(_Float128 x, _Float128 y) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_remainder(_Float128 x, _Float128 y) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_frexp(_Float128 x, int* nptr) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_modf(_Float128 x, _Float128* iptr) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_hypot(_Float128 x, _Float128 y) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_fma(_Float128 x, _Float128 y, _Float128 c) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_ldexp(_Float128 x, int exp) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ int __nv_fp128_ilogb(_Float128 x) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_mul(_Float128 x, _Float128 y) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_add(_Float128 x, _Float128 y) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_sub(_Float128 x, _Float128 y) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ _Float128 __nv_fp128_div(_Float128 x, _Float128 y) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ int __nv_fp128_isnan(_Float128 x) __DEF_IF_HOST
__DEVICE_FP128_FUNCTIONS_DECL__ int __nv_fp128_isunordered(_Float128 x, _Float128 y) __DEF_IF_HOST
#endif /* __FLOAT_C_SPELLING_ENABLED */


#undef __DEVICE_FP128_FUNCTIONS_DECL__

#endif /* __cplusplus && __CUDACC__ */

#endif /* !__DEVICE_FP128_FUNCTIONS_H__ */

#endif /* !__NV_DISABLE_DEVICE_FP128_FUNCTIONS__ */
