# necessary to surface onnx.ModelProto through ExportOutput:
from __future__ import annotations

import abc
import io
import logging
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Mapping,
    Optional,
    Protocol,
    runtime_checkable,
    Sequence,
    Tuple,
    Type,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

import torch
import torch._ops

from torch.onnx._internal import _beartype

# We can only import onnx from this module in a type-checking context to ensure that
# 'import torch.onnx' continues to work without having 'onnx' installed. We fully
# 'import onnx' inside of dynamo_export (by way of _assert_dependencies).
if TYPE_CHECKING:
    import onnx


_DEFAULT_OPSET_VERSION: Final[int] = 18
"""The default ONNX opset version the exporter will use if one is not specified explicitly
through ``ExportOptions``. This should NEVER be accessed outside of this module! Users
should reference ``ExportOptions.opset_version``."""


class ExportOptions:
    """Options to influence the TorchDynamo ONNX exporter."""

    opset_version: Optional[int] = None
    """The ONNX opset version the exporter should target. Defaults to the latest
    supported ONNX opset version. The default version will increment over time as
    ONNX continues to evolve."""

    dynamic_shapes: Optional[bool] = None
    """Shape information hint for input/output tensors.

    - ``None``: the exporter determines the most compatible setting.
    - ``True``: all input shapes are considered dynamic.
    - ``False``: all input shapes are considered static."""

    op_level_debug: Optional[bool] = None
    """Whether to export the model with op-level debug information by evaluating
    ops through ONNX Runtime."""

    logger: Optional[logging.Logger] = None
    """The logger for the ONNX exporter to use. Defaults to creating a child
    logger named "torch.onnx" under the current logger (as returned by
    :py:meth:`logging.getLogger`)."""

    @_beartype.beartype
    def __init__(
        self,
        *,
        opset_version: Optional[int] = None,
        dynamic_shapes: Optional[bool] = None,
        op_level_debug: Optional[bool] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.opset_version = opset_version
        self.dynamic_shapes = dynamic_shapes
        self.op_level_debug = op_level_debug
        self.logger = logger


class ResolvedExportOptions(ExportOptions):
    """Consolidates `ExportOptions` with default values.
    All unspecified options from `ExportOptions` are assigned a default value.
    This is an internal class and its API may be changed at any time without notice.
    """

    # Public attributes MUST be redefined below without ``Optional[]`` from ``ExportOptions``
    opset_version: int
    dynamic_shapes: bool
    op_level_debug: bool
    logger: logging.Logger

    # Private only attributes
    decomposition_table: Dict[torch._ops.OpOverload, Callable]
    """A dictionary that maps operators to their decomposition functions."""

    fx_tracer: FXGraphExtractor
    """The FXGraphExtractor instance used to extract the FX graph from the model."""

    @_beartype.beartype
    def __init__(
        self, options: Optional[Union[ExportOptions, "ResolvedExportOptions"]]
    ):
        if options is None:
            options = ExportOptions()
        if isinstance(options, ResolvedExportOptions):
            self.opset_version = options.opset_version
            self.dynamic_shapes = options.dynamic_shapes
            self.op_level_debug = options.op_level_debug
            self.logger = options.logger
            self.fx_tracer = options.fx_tracer
            self.decomposition_table = options.decomposition_table
        else:
            T = TypeVar("T")

            @_beartype.beartype
            def resolve(value: Optional[T], fallback: Union[T, Callable[[], T]]) -> T:
                if value is not None:
                    return value
                if callable(fallback):
                    return fallback()
                return fallback

            self.opset_version = resolve(options.opset_version, _DEFAULT_OPSET_VERSION)
            self.dynamic_shapes = resolve(options.dynamic_shapes, False)
            import torch.onnx._internal.fx.dynamo_graph_extractor as dynamo_graph_extractor  # TODO: Prevent circular dep
            from torch.onnx._internal.fx import (  # TODO: PyTorch does not take dep on onnxscript outside torch.onnx context
                function_dispatcher,
            )

            self.fx_tracer = dynamo_graph_extractor.DynamoExport()
            self.decomposition_table = (
                function_dispatcher.DEFAULT_ONNX_EXPORTER_DECOMPOSITION_TABLE
            )
            self.op_level_debug = resolve(options.op_level_debug, False)
            self.logger = resolve(
                options.logger, lambda: logging.getLogger().getChild("torch.onnx")
            )

            for key in dir(options):
                if not key.startswith("_"):  # skip private attributes
                    assert hasattr(self, key), f"Unresolved option '{key}'"


@runtime_checkable
class ExportOutputSerializer(Protocol):
    """Protocol for serializing an ONNX graph into a specific format (e.g. Protobuf).
    Note that this is an advanced usage scenario."""

    def serialize(
        self, export_output: ExportOutput, destination: io.BufferedIOBase
    ) -> None:
        """Protocol method that must be implemented for serialization.

        Args:
            export_output: Represents the in-memory exported ONNX model
            destination: A binary IO stream or pre-allocated buffer into which
                the serialized model should be written.

        Example:

            A simple serializer that writes the exported ``onnx.ModelProto`` in Protobuf
            format to ``destination``:

            ::

                class ProtobufExportOutputSerializer:
                    def serialize(
                        self, export_output: ExportOutput, destination: io.BufferedIOBase
                    ) -> None:
                        destination.write(export_output.model_proto.SerializeToString())

                torch.onnx.dynamo_export(...).save(
                    destination="exported_model.onnx",
                    serializer=ProtobufExportOutputSerializer(),
                )
        """
        ...


class ProtobufExportOutputSerializer:
    """Serializes ONNX graph as Protobuf."""

    @_beartype.beartype
    def serialize(
        self, export_output: ExportOutput, destination: io.BufferedIOBase
    ) -> None:
        import onnx

        if not isinstance(export_output.model_proto, onnx.ModelProto):
            raise ValueError("export_output.ModelProto is not an onnx.ModelProto")
        destination.write(export_output.model_proto.SerializeToString())


# TODO(bowbao): Add diagnostics for IO adapters.
@runtime_checkable
class InputAdaptStep(Protocol):
    """A protocol that defines a step in the input adapting process.

    The input adapting process is a sequence of steps that are applied to the
    PyTorch model inputs to transform them into the inputs format expected by the
    exported ONNX model. Each step takes the PyTorch model inputs as arguments and
    returns the transformed inputs.

    This serves as a base formalized construct for the transformation done to model
    input signature by any individual component in the exporter.
    """

    def apply(
        self, model_args: Sequence[Any], model_kwargs: Mapping[str, Any]
    ) -> Tuple[Sequence[Any], Mapping[str, Any]]:
        ...


class InputAdapter:
    """A class that adapts the PyTorch model inputs to exported ONNX model inputs format."""

    _input_adapt_steps: List[InputAdaptStep]

    def __init__(self, input_adapt_steps: Optional[List[InputAdaptStep]] = None):
        self._input_adapt_steps = input_adapt_steps or []

    @_beartype.beartype
    def append_step(self, step: InputAdaptStep) -> None:
        """Appends a step to the input adapt steps.

        Args:
            step: The step to append.
        """
        self._input_adapt_steps.append(step)

    @_beartype.beartype
    def apply(
        self, *model_args, **model_kwargs
    ) -> Sequence[Union[int, float, bool, torch.Tensor]]:
        """Converts the PyTorch model inputs to exported ONNX model inputs format.

        Args:
            model_args: The PyTorch model inputs.
            model_kwargs: The PyTorch model keyword inputs.

        Returns:
            A sequence of tensors converted from PyTorch model inputs.
        """
        args: Sequence[Any] = model_args
        kwargs: Mapping[str, Any] = model_kwargs
        for step in self._input_adapt_steps:
            args, kwargs = step.apply(args, kwargs)
        assert not kwargs
        return args


@runtime_checkable
class OutputAdaptStep(Protocol):
    """A protocol that defines a step in the output adapting process.

    The output adapting process is a sequence of steps that are applied to the
    PyTorch model outputs to transform them into the outputs format produced by the
    exported ONNX model. Each step takes the PyTorch model outputs as arguments and
    returns the transformed outputs.

    This serves as a base formalized construct for the transformation done to model
    output signature by any individual component in the exporter.
    """

    def apply(self, model_outputs: Any) -> Any:
        ...


class OutputAdapter:
    """A class that adapts the PyTorch model outputs to exported ONNX model outputs format."""

    _output_adapt_steps: List[OutputAdaptStep]

    def __init__(self, output_adapt_steps: Optional[List[OutputAdaptStep]] = None):
        self._output_adapt_steps = output_adapt_steps or []

    @_beartype.beartype
    def append_step(self, step: OutputAdaptStep) -> None:
        """Appends a step to the output format steps.

        Args:
            step: The step to append.
        """
        self._output_adapt_steps.append(step)

    @_beartype.beartype
    def apply(
        self, model_outputs: Any
    ) -> Sequence[Union[torch.Tensor, int, float, bool]]:
        """Converts the PyTorch model outputs to exported ONNX model outputs format.

        Args:
            model_outputs: The PyTorch model outputs.

        Returns:
            PyTorch model outputs in exported ONNX model outputs format.
        """
        for step in self._output_adapt_steps:
            model_outputs = step.apply(model_outputs)
        return model_outputs


class ExportOutput:
    """An in-memory representation of a PyTorch model that has been exported to ONNX."""

    _model_proto: Final[onnx.ModelProto]
    _input_adapter: Final[InputAdapter]
    _output_adapter: Final[OutputAdapter]

    @_beartype.beartype
    def __init__(
        self,
        model_proto: onnx.ModelProto,
        input_adapter: InputAdapter,
        output_adapter: OutputAdapter,
    ):
        self._model_proto = model_proto
        self._input_adapter = input_adapter
        self._output_adapter = output_adapter

    @property
    def model_proto(self) -> onnx.ModelProto:
        """The exported ONNX model as an ``onnx.ModelProto``."""

        return self._model_proto

    @_beartype.beartype
    def adapt_torch_inputs_to_onnx(
        self, *model_args, **model_kwargs
    ) -> Sequence[Union[torch.Tensor, int, float, bool]]:
        """Converts the PyTorch model inputs to exported ONNX model inputs format.

        Due to design differences, input/output format between PyTorch model and exported
        ONNX model are often not the same. E.g., None is allowed for PyTorch model, but are
        not supported by ONNX. Nested constructs of tensors are allowed for PyTorch model,
        but only flattened tensors are supported by ONNX, etc.

        The actual adapting steps are associated with each individual export. It
        depends on the PyTorch model, the particular set of model_args and model_kwargs
        used for the export, and export options.

        This method replays the adapting steps recorded during export.

        Args:
            model_args: The PyTorch model inputs.
            model_kwargs: The PyTorch model keyword inputs.

        Returns:
            A sequence of tensors converted from PyTorch model inputs.

        Example::

            # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)
            >>> import torch
            >>> import torch.onnx
            >>> from typing import Dict, Tuple
            >>> def func_with_nested_input_structure(
            ...     x_dict: Dict[str, torch.Tensor],
            ...     y_tuple: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            ... ):
            ...     if "a" in x_dict:
            ...         x = x_dict["a"]
            ...     elif "b" in x_dict:
            ...         x = x_dict["b"]
            ...     else:
            ...         x = torch.randn(3)
            ...
            ...     y1, (y2, y3) = y_tuple
            ...
            ...     return x + y1 + y2 + y3
            >>> x_dict = {"a": torch.tensor(1.)}
            >>> y_tuple = (torch.tensor(2.), (torch.tensor(3.), torch.tensor(4.)))
            >>> export_output = torch.onnx.dynamo_export(func_with_nested_input_structure, x_dict, y_tuple)
            >>> print(x_dict, y_tuple)
            {'a': tensor(1.)}
            (tensor(2.), (tensor(3.), tensor(4.)))
            >>> print(export_output.adapt_torch_inputs_to_onnx(x_dict, y_tuple))
            (tensor(1.), tensor(2.), tensor(3.), tensor(4.))

        .. warning::
            This API is experimental and is *NOT* backward-compatible.

        """
        return self._input_adapter.apply(*model_args, **model_kwargs)

    @_beartype.beartype
    def adapt_torch_outputs_to_onnx(
        self, model_outputs: Any
    ) -> Sequence[Union[torch.Tensor, int, float, bool]]:
        """Converts the PyTorch model outputs to exported ONNX model outputs format.

        Due to design differences, input/output format between PyTorch model and exported
        ONNX model are often not the same. E.g., None is allowed for PyTorch model, but are
        not supported by ONNX. Nested constructs of tensors are allowed for PyTorch model,
        but only flattened tensors are supported by ONNX, etc.

        The actual adapting steps are associated with each individual export. It
        depends on the PyTorch model, the particular set of model_args and model_kwargs
        used for the export, and export options.

        This method replays the adapting steps recorded during export.

        Args:
            model_outputs: The PyTorch model outputs.

        Returns:
            PyTorch model outputs in exported ONNX model outputs format.

        Example::

            # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)
            >>> import torch
            >>> import torch.onnx
            >>> def func_returning_tuples(x, y, z):
            ...     x = x + y
            ...     y = y + z
            ...     z = x + y
            ...     return (x, (y, z))
            >>> x = torch.tensor(1.)
            >>> y = torch.tensor(2.)
            >>> z = torch.tensor(3.)
            >>> export_output = torch.onnx.dynamo_export(func_returning_tuples, x, y, z)
            >>> pt_output = func_returning_tuples(x, y, z)
            >>> print(pt_output)
            (tensor(3.), (tensor(5.), tensor(8.)))
            >>> print(export_output.adapt_torch_outputs_to_onnx(pt_output))
            [tensor(3.), tensor(5.), tensor(8.)]

        .. warning::
            This API is experimental and is *NOT* backward-compatible.

        """
        return self._output_adapter.apply(model_outputs)

    @_beartype.beartype
    def save(
        self,
        destination: Union[str, io.BufferedIOBase],
        *,
        serializer: Optional[ExportOutputSerializer] = None,
    ) -> None:
        """Saves the in-memory ONNX model to ``destination`` using specified ``serializer``.
        If no ``serializer`` is specified, the model will be serialized as Protobuf."""

        if serializer is None:
            serializer = ProtobufExportOutputSerializer()
        if isinstance(destination, str):
            with open(destination, "wb") as f:
                serializer.serialize(self, f)
        else:
            serializer.serialize(self, destination)


class FXGraphExtractor(abc.ABC):
    """Abstract interface for FX graph extractor engines.
    This class isolates FX extraction logic from the rest of the export logic.
    That allows a single ONNX exporter that can leverage different FX graphs."""

    def __init__(self) -> None:
        super().__init__()
        self.input_adapter: InputAdapter = InputAdapter()
        self.output_adapter: OutputAdapter = OutputAdapter()

    @_beartype.beartype
    def _export_fx_to_onnx(
        self,
        options: ResolvedExportOptions,
        fx_module: torch.fx.GraphModule,
        fx_module_args: Sequence[Any],
    ) -> ExportOutput:
        # TODO: Import here to prevent circular dependency
        import torch.onnx._internal.fx.fx_exporter as fx_exporter
        import torch.onnx._internal.fx.passes as passes

        # Apply decomposition table to the input graph.
        module = passes.Decompose(
            fx_module,
            options.decomposition_table,
            enable_dynamic_axes=options.dynamic_shapes,
        ).run(*fx_module_args)

        # ONNX does not support views and mutations.
        # Functionalize to get a semantically equivalent graph without mutations.
        module = passes.Functionalize(
            module, enable_dynamic_axes=options.dynamic_shapes
        ).run(*fx_module_args)
        # Input mutations are detected and distilled after `Functionalize` pass.
        # Remove them since ONNX inference does not need them.
        module = passes.RemoveInputMutation(module).run(*fx_module_args)

        # Run ShapeInferenceWithFakeTensor to get static shape of nodes for op_level_debug purposes
        # The pass added nodes with static shape into original node metadata:
        # node.meta["static_shape"]: FakeTensor/int/float/SymInt/SynFloat
        if options.op_level_debug:
            module = passes.ShapeInferenceWithFakeTensor(module).run(*fx_module_args)

        # We want to pass list of ints and floats to TorchScript graph correctly
        # in _export_fx_to_ts, so we must disable FakeTensorMode. Otherwise, graph may
        # receive FakeTensor and results runtime error. In addition, TorchScript-based
        # ONNX exporter used in _ts_graph_to_onnx_model_in_protobuf is not compatible
        # with FakeTensorMode.
        with torch.utils._mode_utils.no_dispatch():
            onnxscript_graph = passes.export_fx_to_onnxscript(module, options)
            # ONNX does not support None inputs. During graph building, all None inputs
            # are removed. Here we register this step to input adapter.
            self.adapt_input(fx_exporter.RemoveNoneInputStep, fx_module_args, {})
            # TODO(titaiwang): temp workaround for https://github.com/pytorch/pytorch/issues/99534
            # Dynamo doesn't support constant inputs, but could change. We can remove this
            # once Dynamo supports constant inputs.
            self.adapt_input(fx_exporter.RemoveConstantInputStep, fx_module_args, {})
            # ONNX can't represent collection types (e.g., dictionary, tuple of tuple of
            # tensor, etc), we flatten the collection and register each element as output.
            self.output_adapter.append_step(fx_exporter.FlattenOutputStep())

        # Export TorchScript graph to ONNX ModelProto.
        onnx_model = onnxscript_graph.to_model_proto(options.opset_version)
        return torch.onnx.ExportOutput(
            onnx_model, self.input_adapter, self.output_adapter
        )

    def adapt_input(
        self,
        adapt_step_cls: Type[InputAdaptStep],
        model_args: Sequence[Any],
        model_kwargs: Mapping[str, Any],
        step_init_args: Optional[Sequence[Any]] = None,
    ) -> Tuple[Sequence[Any], Mapping[str, Any]]:
        """Apply an input adapt step to the model args and kwargs.
        An input adapt step object is initialized, applied and recorded as part of
        ``self.input_adapter`.
        Args:
            adapt_step_cls: The input adapt step class.
            model_args: The model args.
            model_kwargs: The model kwargs.
            step_init_args: The input adapt step initialization arguments.
        Returns:
            The adapted model args and kwargs.
        """
        step_init_args = step_init_args or ()
        adapt_step = adapt_step_cls(*step_init_args)
        self.input_adapter.append_step(adapt_step)
        return adapt_step.apply(model_args, model_kwargs)

    def adapt_output(
        self,
        adapt_step_cls: Type[OutputAdaptStep],
        model_outputs: Any,
        step_init_args: Optional[Sequence[Any]] = None,
    ) -> Any:
        """Apply an output adapt step to the model outputs.
        An output adapt step object is initialized, applied and recorded as part of
        ``self._output_adapter`.
        Args:
            adapt_step_cls: The output adapt step class.
            model_outputs: The model outputs.
            step_init_args: The input adapt step initialization arguments.
        Returns:
            The adapted model outputs.
        """
        step_init_args = step_init_args or ()
        adapt_step = adapt_step_cls(*step_init_args)
        self.output_adapter.append_step(adapt_step)
        return adapt_step.apply(model_outputs)

    @abc.abstractmethod
    def generate_fx(
        self,
        options: ResolvedExportOptions,
        model: Union[torch.nn.Module, Callable],
        model_args: Sequence[Any],
        model_kwargs: Mapping[str, Any],
    ) -> Tuple[torch.fx.GraphModule, Tuple[Any]]:
        """Analyzes user ``model`` and generates a FX graph.
        Args:
            options: The export options.
            model: The user model.
            model_args: The model's positional input arguments.
            model_kwargs: The model's keyword input arguments.
        Returns:
            The generated FX Graph, and the model's adapted input arguments.
        """
        # By design, only torch.fx.GraphModule is needed
        # But FXSymbolicTracer modifies model data, which will be needed
        # to produce the ONNX proto in the next layer.
        # TODO: Refactor after https://github.com/pytorch/pytorch/pull/98421
        ...


class Exporter:
    @_beartype.beartype
    def __init__(
        self,
        options: Union[ExportOptions, ResolvedExportOptions],
        model: Union[torch.nn.Module, Callable],
        model_args: Sequence[Any],
        model_kwargs: Mapping[str, Any],
    ):
        self.options = ResolvedExportOptions(options)
        assert self.options is not None

        self.model = model
        self.model_args = model_args
        self.model_kwargs = model_kwargs

    def export(self) -> ExportOutput:
        graph_module, updated_model_args = self.options.fx_tracer.generate_fx(
            self.options, self.model, self.model_args, self.model_kwargs
        )

        # Export FX graph to ONNX ModelProto.
        #
        # Note that ALL kwargs are folded into constants in graph_module, so we don't pass kwargs
        # to _export.
        return self.options.fx_tracer._export_fx_to_onnx(
            self.options, graph_module, updated_model_args
        )

    @property
    def logger(self) -> logging.Logger:
        # options.logger will always be resolved to an instance when constructing
        assert isinstance(self.options.logger, logging.Logger)
        return self.options.logger


class UnsatisfiedDependencyError(RuntimeError):
    """Raised when an ONNX exporter dependency cannot be satisfied."""

    def __init__(self, package_name: str, message: str):
        super().__init__(message)
        self.package_name = package_name


@_beartype.beartype
def _assert_dependencies(export_options: ResolvedExportOptions):
    logger = export_options.logger
    opset_version = export_options.opset_version

    def missing_package(package_name: str, exc_info: logging._ExcInfoType):
        message = (
            f"Please install the `{package_name}` package "
            f"(e.g. `python -m pip install {package_name}`)."
        )
        logger.fatal(message, exc_info=exc_info)
        return UnsatisfiedDependencyError(package_name, message)

    def missing_opset(package_name: str):
        message = (
            f"The installed `{package_name}` does not support the specified ONNX opset "
            f"version {opset_version}. Install a newer `{package_name}` package or "
            f"specify an older opset version."
        )
        logger.fatal(message)
        return UnsatisfiedDependencyError(package_name, message)

    try:
        import onnx
    except ImportError as e:
        raise missing_package("onnx", e) from e

    if onnx.defs.onnx_opset_version() < opset_version:
        raise missing_opset("onnx")

    try:
        # PyTorch runs lintrunner in CI without onnxscript installed
        import onnxscript  # type: ignore[import]
    except ImportError as e:
        raise missing_package("onnxscript", e) from e

    if not isinstance(
        onnxscript.onnx_opset.all_opsets[("", opset_version)],
        onnxscript.values.Opset,
    ):
        raise missing_opset("onnxscript")


@_beartype.beartype
def dynamo_export(
    model: Union[torch.nn.Module, Callable],
    /,
    *model_args,
    export_options: Optional[ExportOptions] = None,
    **model_kwargs,
) -> ExportOutput:
    """Export a torch.nn.Module to an ONNX graph.

    Args:
        model: The PyTorch model to be exported to ONNX.
        model_args: Positional inputs to ``model``.
        model_kwargs: Keyword inputs to ``model``.
        export_options: Options to influence the export to ONNX.

    Returns:
        An in-memory representation of the exported ONNX model.

    Example:
    ::

        import torch.onnx
        torch.onnx.dynamo_export(
            my_nn_module,
            torch.randn(2, 2, 2), # positional input 1
            torch.randn(2, 2, 2), # positional input 2
            my_nn_module_attribute="hello", # keyword input
            export_options=ExportOptions(
                opset_version=17,
            )
        ).save("my_model.onnx")
    """

    resolved_export_options = (
        export_options
        if isinstance(export_options, ResolvedExportOptions)
        else ResolvedExportOptions(export_options)
    )

    _assert_dependencies(resolved_export_options)

    return Exporter(
        options=resolved_export_options,
        model=model,
        model_args=model_args,
        model_kwargs=model_kwargs,
    ).export()


__all__ = [
    "ExportOptions",
    "ExportOutput",
    "ExportOutputSerializer",
    "UnsatisfiedDependencyError",
    "dynamo_export",
]
