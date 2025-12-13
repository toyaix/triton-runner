import triton

triton_version = triton.__version__
version_str = ".".join(triton_version.split('.')[:2])

is_support_version = version_str in ["3.6", "3.5", "3.4", "3.3", "3.2", "3.1", "3.0"]

is_triton_v3_6 = version_str == "3.6"
is_triton_v3_5 = version_str == "3.5"
is_triton_v3_4 = version_str == "3.4"
is_triton_v3_3 = version_str == "3.3"
is_triton_v3_2 = version_str == "3.2"
is_triton_v3_1 = version_str == "3.1"
is_triton_v3_0 = version_str == "3.0"

is_triton_geq_v3_3 = version_str in ["3.3", "3.4", "3.5", "3.6"]
is_triton_geq_v3_4 = version_str in ["3.4", "3.5", "3.6"]
is_triton_geq_v3_5 = version_str in ["3.5", "3.6"]

is_triton_leq_v3_2 = version_str in ["3.2", "3.1", "3.0"]
is_triton_leq_v3_1 = version_str in ["3.1", "3.0"]
is_disable_multithreading = version_str in ["3.5", "3.4", "3.3", "3.2"]

if is_triton_v3_5:
    uni_triton_version = "3.5.x"
elif is_triton_v3_3:
    uni_triton_version = "3.3.x"
else:
    uni_triton_version = triton_version

try:
    import triton.language.extra.tlx as tlx
    is_tlx = True
except ImportError as e:
    is_tlx = False
