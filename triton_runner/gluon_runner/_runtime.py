from ..jit import RunnerJITFunctionV3_5_0, RunnerJITFunctionV3_4_0, RunnerJITFunction, JITFunction
from triton.experimental.gluon._runtime import GluonASTSource, T
from typing import Optional, Callable, Iterable, Union
import triton


class RunnerGluonJITFunction(RunnerJITFunction[T]):
    pass

class RunnerGluonJITFunctionV3_5_0(RunnerJITFunctionV3_5_0[T]):
    def create_binder(self):
        result = super().create_binder()
        self.ASTSource = GluonASTSource
        return result

    def is_gluon(self):
        return True

class RunnerGluonJITFunctionV3_4_0(RunnerJITFunctionV3_4_0[T]):

    def create_binder(self):
        result = super().create_binder()
        self.ASTSource = GluonASTSource
        return result

    def is_gluon(self):
        return True


def jit(
    fn: Optional[T] = None,
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int | str]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int | str]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> Union[RunnerGluonJITFunction[T], Callable[[T], JITFunction[T]]]:
    """
    Decorator for JIT-compiling a function using the Triton compiler.

    :note: When a jit'd function is called, arguments are
        implicitly converted to pointers if they have a :code:`.data_ptr()` method
        and a `.dtype` attribute.

    :note: This function will be compiled and run on the GPU. It will only have access to:

           * python primitives,
           * builtins within the triton package,
           * arguments to this function,
           * other jit'd functions

    :param fn: the function to be jit-compiled
    :type fn: Callable
    """

    def decorator(fn: T) -> JITFunction[T]:
        assert callable(fn)
        if triton.__version__ == "3.5.0":
            return RunnerGluonJITFunctionV3_5_0(
                fn,
                version=version,
                do_not_specialize=do_not_specialize,
                do_not_specialize_on_alignment=do_not_specialize_on_alignment,
                debug=debug,
                noinline=noinline,
                repr=repr,
                launch_metadata=launch_metadata,
            )
        elif triton.__version__ == "3.4.0":
            return RunnerGluonJITFunctionV3_4_0(
                fn,
                version=version,
                do_not_specialize=do_not_specialize,
                do_not_specialize_on_alignment=do_not_specialize_on_alignment,
                debug=debug,
                noinline=noinline,
                repr=repr,
                launch_metadata=launch_metadata,
            )
        else:
            raise RuntimeError(f"Can't support Triton v{triton.__version__}.")

    if fn is not None:
        return decorator(fn)

    else:
        return decorator
