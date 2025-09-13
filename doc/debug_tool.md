## TTIR Debug

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