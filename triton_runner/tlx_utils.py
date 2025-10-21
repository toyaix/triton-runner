try:
    import triton.language.extra.tlx as tlx
    is_tlx = True
except ImportError as e:
    is_tlx = False
