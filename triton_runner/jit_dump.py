import os
import re
from .dump_utils import get_injected_ir
from .compile import get_source_ir


class DumpMixin:

    def need_dump(self, kwargs):
        return "dump_tensor" in kwargs

    def is_python_dump(self, kwargs, source_dir_type):
        return self.need_dump(kwargs) and source_dir_type not in ["ttir_dir", "ttgir_dir"]

    def get_dump_key(self, key, kwargs):
        if self.need_dump(kwargs):
            key += "|dump_tensor"
        if "dump_value" in kwargs:
            key += f"|dump_value={kwargs['dump_value']}"
        if "dump_grid" in kwargs:
            key += f"|dump_grid={kwargs['dump_grid']}"
        if (runner_source_dir_str := self.get_runner_source_dir_str(kwargs)):
            key += f"|runner_src={runner_source_dir_str}"
        return key

    def insert_dump_tensor_param(self, full_text):
        pattern = re.compile(r'(tt\.func\s+public\s+@\w+\s*)\((.*?)\)(\s*attributes\s*{[^}]*}\s*{)', re.DOTALL)

        def replacer(match):
            prefix, args_str, suffix = match.groups()
            if "%runner_dump_tensor: !tt.ptr<f32>" in args_str:
                return match.group(0)
            new_args_str = args_str + ', %runner_dump_tensor: !tt.ptr<f32>'
            return f"{prefix}({new_args_str}){suffix}"

        return pattern.sub(replacer, full_text, count=1)

    def inject_ssa_ir_dump_store(self, full_text, ssa_value, dump_grid):
        pattern = re.compile(
            rf'^(?P<indent>\s*){ssa_value}\s*=\s*'
            r'(?P<op>\S+)\s+'
            r'.*'
            r'tensor<'
            r'(?P<size>(?:\d+x)*\d+)'
            r'x(?P<elem_ty>(?:[^,<>]|<[^>]*>)+)'
            r'(?:,\s*(?P<encoding>#[^>]+))?'
            r'>'
            r'[^<]*?'
            r'loc\((?P<loc>#[^)]+)\)',
            re.MULTILINE
        )
        def make_replacer(dump_grid):
            def replacer(match):
                original_line = match.group(0)
                indent = match.group("indent")
                op = match.group("op")
                size = match.group("size")
                elem_ty = match.group("elem_ty")
                ptr_match = re.match(r'!tt\.ptr<(.+)>', elem_ty)
                if ptr_match:
                    elem_ty = ptr_match.group(1)
                loc = match.group("loc")
                encoding = match.group("encoding")
                return get_injected_ir(ssa_value, op, original_line, indent, size, elem_ty, encoding, loc, dump_grid=dump_grid)
            return replacer
        return pattern.sub(make_replacer(dump_grid), full_text, count=1)

    def inject_dump_op_dump_store(self, full_text):
        ssa_value = r'%\d+'
        pattern = re.compile(
            rf'^(?P<indent>[ \t]*)'
            rf'(?P<ssa_value>{ssa_value})\s*=\s*'
            r'(?P<op>\S+)\s+'
            r'(?P<args>[^\n{}]*)'
            r'\{(?P<attrs>[^\n}]*\btt\.dump\s*=\s*[^\n}]+)\}\s*'
            r':\s*tensor<(?P<size>(?:\d+x)*\d+)x(?P<elem_ty>(?:[^,<>]|<[^>]*>)+)(?:,\s*#[^>]+)?>'
            r'[^\n]*?loc\((?P<loc>#[^)]+)\)'
            r'\n.*=\s*(?P<offset_val>.*)',
            re.MULTILINE
        )
        def make_replacer(replace_id):
            def replacer(match):
                original_line = match.group(0).split('\n')[0]
                indent = match.group("indent")
                op = match.group("op")
                size = match.group("size")
                elem_ty = match.group("elem_ty")
                loc = match.group("loc")
                ssa_value = match.group("ssa_value")
                offset_val = match.group("offset_val")
                clean_line = re.sub(r"\s*\{[^{}]*\}", "", original_line)
                return get_injected_ir(ssa_value, op, clean_line, indent, size, elem_ty, None, loc,
                                       python_dump=True, offset_val=offset_val, replace_id=replace_id)
            return replacer
        full_text, count = pattern.subn(make_replacer(0), full_text, count=1)
        replace_id = 0
        while count > 0:
            replace_id = replace_id + 1
            full_text, count = pattern.subn(make_replacer(replace_id), full_text, count=1)
        return full_text

    def get_src_and_save_dump_file(self, kwargs, source_dir_type, signature, constexprs, attrs, target, options, bound_args):
        src = None
        if self.need_dump(kwargs):
            dump_tensor = kwargs["dump_tensor"]
            from .color_print import check_dump_tensor_dtype
            check_dump_tensor_dtype(dump_tensor)
            if self.is_python_dump(kwargs, source_dir_type):
                src = self.ASTSource(self, signature, constexprs, attrs)
                module = get_source_ir(src, target=target, options=options.__dict__)
                runner_cache_dir = self.get_runner_cache_dir()
                dump_content = self.insert_dump_tensor_param(str(module))
                dump_content = self.inject_dump_op_dump_store(dump_content)
                src = os.path.join(runner_cache_dir, f"{self.__name__}-dump.ttir")
                with open(src, "w") as file:
                    file.write(dump_content)
            signature["dump_tensor"], bound_args["dump_tensor"] = "*fp32", dump_tensor
        return src
