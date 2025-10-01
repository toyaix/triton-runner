## 1. Python Debug

In addition to using `@triton_runner.jit` instead of `@triton.jit`, you also need use `triton_runner.language.dump()` in your Triton kernel. And we allocate a temporary tensor called debug_tensor, and simply pass it to the kernel through the debug_tensor parameter. Here are some example commands for debugging.

```shell
python debug_tool/python/01-vec_add/debug_output.py
python debug_tool/python/01-vec_add/debug_x.py
python debug_tool/python/01-vec_add/debug_y.py

python debug_tool/python/02-matrix_transpose/debug_2d_load.py
python debug_tool/python/02-matrix_transpose/debug_2d_trans.py

python debug_tool/python/03-matrix_multiplication/debug_acc.py
python debug_tool/python/03-matrix_multiplication/debug_out.py

python debug_tool/python/04-softmax/debug_exp_shifted.py
python debug_tool/python/04-softmax/debug_max_in_loop.py
python debug_tool/python/04-softmax/debug_max_out_loop.py
python debug_tool/python/04-softmax/debug_normalize_by_sum.py
python debug_tool/python/04-softmax/debug_sub.py
python debug_tool/python/04-softmax/debug_sum_in_loop.py
python debug_tool/python/04-softmax/debug_sum_out_loop.py

python debug_tool/python/05-softmax_lse/debug_log_acc.py
python debug_tool/python/05-softmax_lse/debug_max_acc.py

python debug_tool/python/06-attention/debug_out.py

python debug_tool/python/07-debug_not_f32/debug_bf16.py
```

## 2. TTIR Debug

Debugging is supported for TTIR ops like `tt.load`, `arith.addf`, and `tt.trans` in Triton v3.4.0.

```shell
python debug_tool/ttir/01-vector_add/debug_load.py
python debug_tool/ttir/01-vector_add/debug_addf.py

python debug_tool/ttir/02-matrix_transpose/debug_2d_load.py
python debug_tool/ttir/02-matrix_transpose/debug_2d_trans.py

python debug_tool/ttir/03-matrix_multiplication/debug_acc.py

python debug_tool/ttir/04-softmax/debug_maxnumf.py
python debug_tool/ttir/04-softmax/debug_addf-sum.py
python debug_tool/ttir/04-softmax/debug_subf.py
python debug_tool/ttir/04-softmax/debug_exp-exp_shifted.py
python debug_tool/ttir/04-softmax/debug_divf-normalize_by_sum.py

python debug_tool/ttir/05-softmax_lse/debug_log_acc.py
python debug_tool/ttir/05-softmax_lse/debug_max_acc.py
python debug_tool/ttir/05-softmax_lse/debug_more.py

python debug_tool/ttir/06-attention/debug_out.py

python debug_tool/ttir/07-debug_not_f32/debug_bf16.py
```

## 3. TTGIR Debug

Debugging is supported for TTGIR level like `tt.load`, `arith.addf`, and `tt.trans` in Triton v3.4.0. Here are some example commands for debugging.

```shell
python debug_tool/ttgir/01-vec_add/debug_addf.py
python debug_tool/ttgir/01-vec_add/debug_load.py

python debug_tool/ttgir/02-matrix_transpose/debug_2d_load.py
python debug_tool/ttgir/02-matrix_transpose/debug_2d_trans.py

python debug_tool/ttgir/03-matrix_multiplication/debug_acc.py

python debug_tool/ttgir/04-softmax/debug_addf-sum.py
python debug_tool/ttgir/04-softmax/debug_divf-normalize_by_sum.py
python debug_tool/ttgir/04-softmax/debug_exp-exp_shifted.py
python debug_tool/ttgir/04-softmax/debug_maxnumf.py
python debug_tool/ttgir/04-softmax/debug_subf.py

python debug_tool/ttgir/05-softmax_lse/debug_log_acc.py
python debug_tool/ttgir/05-softmax_lse/debug_max_acc.py
python debug_tool/ttgir/05-softmax_lse/debug_more.py

python debug_tool/ttgir/06-attention/debug_out.py
```
