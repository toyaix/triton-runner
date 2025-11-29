Python/TTIR/TTGIR now support dump on Triton v3.5.x, v3.4.0, v3.3.x.

## 1. Python Dump

In addition to using `@triton_runner.jit` instead of `@triton.jit`, you also need use `triton_runner.language.dump()` in your Triton kernel. And we allocate a temporary tensor called dump_tensor, and simply pass it to the kernel through the dump_tensor parameter. Here are some example commands for dump.

```shell
python examples/dump/python/01-vec_add/dump_output.py
python examples/dump/python/01-vec_add/dump_with_grid.py
python examples/dump/python/01-vec_add/dump_with_offset.py
python examples/dump/python/01-vec_add/dump_x.py
python examples/dump/python/01-vec_add/dump_y.py

python examples/dump/python/02-matrix_transpose/dump_2d_load.py
python examples/dump/python/02-matrix_transpose/dump_2d_trans.py

python examples/dump/python/03-matrix_multiplication/dump_acc.py
python examples/dump/python/03-matrix_multiplication/dump_out.py
python examples/dump/python/03-matrix_multiplication/dump_with_grid.py
python examples/dump/python/03-matrix_multiplication/dump_with_offset.py

python examples/dump/python/04-softmax/dump_exp_shifted.py
python examples/dump/python/04-softmax/dump_max_in_loop.py
python examples/dump/python/04-softmax/dump_max_out_loop.py
python examples/dump/python/04-softmax/dump_normalize_by_sum.py
python examples/dump/python/04-softmax/dump_sub.py
python examples/dump/python/04-softmax/dump_sum_in_loop.py
python examples/dump/python/04-softmax/dump_sum_out_loop.py

python examples/dump/python/05-softmax_lse/dump_log_acc.py
python examples/dump/python/05-softmax_lse/dump_max_acc.py
python examples/dump/python/05-softmax_lse/dump_more_with_offset.py

python examples/dump/python/06-attention/dump_out.py

python examples/dump/python/07-dump_not_f32/dump_bf16.py
```

## 2. TTIR Dump

dump is supported for TTIR ops like `tt.load`, `arith.addf`, and `tt.trans` in Triton v3.4.0.

```shell
python examples/dump/ttir/01-vector_add/dump_load.py
python examples/dump/ttir/01-vector_add/dump_addf.py

python examples/dump/ttir/02-matrix_transpose/dump_2d_load.py
python examples/dump/ttir/02-matrix_transpose/dump_2d_trans.py

python examples/dump/ttir/03-matrix_multiplication/dump_acc.py

python examples/dump/ttir/04-softmax/dump_maxnumf.py
python examples/dump/ttir/04-softmax/dump_addf-sum.py
python examples/dump/ttir/04-softmax/dump_subf.py
python examples/dump/ttir/04-softmax/dump_exp-exp_shifted.py
python examples/dump/ttir/04-softmax/dump_divf-normalize_by_sum.py

python examples/dump/ttir/05-softmax_lse/dump_log_acc.py
python examples/dump/ttir/05-softmax_lse/dump_max_acc.py
python examples/dump/ttir/05-softmax_lse/dump_more.py

python examples/dump/ttir/06-attention/dump_out.py

python examples/dump/ttir/07-dump_not_f32/dump_bf16.py
```

## 3. TTGIR Dump

dump is supported for TTGIR level like `tt.load`, `arith.addf`, and `tt.trans` in Triton v3.4.0. Here are some example commands for dump.

```shell
python examples/dump/ttgir/01-vec_add/dump_addf.py
python examples/dump/ttgir/01-vec_add/dump_load.py

python examples/dump/ttgir/02-matrix_transpose/dump_2d_load.py
python examples/dump/ttgir/02-matrix_transpose/dump_2d_trans.py

python examples/dump/ttgir/03-matrix_multiplication/dump_acc.py

python examples/dump/ttgir/04-softmax/dump_addf-sum.py
python examples/dump/ttgir/04-softmax/dump_divf-normalize_by_sum.py
python examples/dump/ttgir/04-softmax/dump_exp-exp_shifted.py
python examples/dump/ttgir/04-softmax/dump_maxnumf.py
python examples/dump/ttgir/04-softmax/dump_subf.py

python examples/dump/ttgir/05-softmax_lse/dump_log_acc.py
python examples/dump/ttgir/05-softmax_lse/dump_max_acc.py
python examples/dump/ttgir/05-softmax_lse/dump_more.py

python examples/dump/ttgir/06-attention/dump_out.py
```
