from pathlib import Path
from typing import Any, Union

from nebullvm.core.models import (
    ModelParams,
    DeepLearningFramework,
    QuantizationType,
    DeviceType,
)
from nebullvm.operations.inference_learners.base import BuildInferenceLearner
from nebullvm.operations.inference_learners.deepsparse import (
    PytorchDeepSparseInferenceLearner,
)
from nebullvm.operations.inference_learners.faster_transformer import (
    FasterTransformerInferenceLearner,
)
from nebullvm.operations.inference_learners.neural_compressor import (
    PytorchNeuralCompressorInferenceLearner,
)
from nebullvm.operations.inference_learners.onnx import ONNX_INFERENCE_LEARNERS
from nebullvm.operations.inference_learners.openvino import (
    OPENVINO_INFERENCE_LEARNERS,
)
from nebullvm.operations.inference_learners.tensor_rt import (
    TENSOR_RT_INFERENCE_LEARNERS,
    PytorchTensorRTInferenceLearner,
)
from nebullvm.operations.inference_learners.tensorflow import (
    TensorflowBackendInferenceLearner,
    TFLiteBackendInferenceLearner,
)
from nebullvm.operations.inference_learners.torch_dynamo import (
    TorchDynamoInferenceLearner,
)
from nebullvm.operations.inference_learners.torch_neuron import (
    TorchNeuronInferenceLearner,
)
from nebullvm.operations.inference_learners.torch_xla import (
    TorchXLAInferenceLearner,
)
from nebullvm.operations.inference_learners.torchscript import (
    TorchScriptInferenceLearner,
)
from nebullvm.operations.inference_learners.tvm import (
    APACHE_TVM_INFERENCE_LEARNERS,
    PytorchApacheTVMInferenceLearner,
)
from nebullvm.optional_modules.tensor_rt import tensorrt as trt
from nebullvm.optional_modules.tensorflow import tensorflow as tf
from nebullvm.optional_modules.torch import (
    ScriptModule,
    Module,
    GraphModule,
    torch,
)
from nebullvm.optional_modules.tvm import tvm, ExecutorFactoryModule
from nebullvm.tools.onnx import get_input_names, get_output_names
from nebullvm.tools.transformations import (
    MultiStageTransformation,
    VerifyContiguity,
)


class TorchScriptBuildInferenceLearner(BuildInferenceLearner):
    def execute(
        self,
        model: ScriptModule,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation,
        **kwargs,
    ):
        self.inference_learner = TorchScriptInferenceLearner(
            torch_model=model,
            network_parameters=model_params,
            input_tfms=input_tfms,
            device=self.device,
        )


class TorchXLABuildInferenceLearner(BuildInferenceLearner):
    def execute(
        self,
        model: torch.nn.Module,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation,
        **kwargs,
    ):
        self.inference_learner = TorchXLAInferenceLearner(
            torch_model=model,
            network_parameters=model_params,
            input_tfms=input_tfms,
            device=self.device,
        )


class TorchNeuronBuildInferenceLearner(BuildInferenceLearner):
    def execute(
        self,
        model: ScriptModule,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation,
        **kwargs,
    ):
        self.inference_learner = TorchNeuronInferenceLearner(
            torch_model=model,
            network_parameters=model_params,
            input_tfms=input_tfms,
            device=self.device,
        )


class TorchDynamoBuildInferenceLearner(BuildInferenceLearner):
    def execute(
        self,
        model: ScriptModule,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation,
        **kwargs,
    ):
        self.inference_learner = TorchDynamoInferenceLearner(
            torch_model=model,
            network_parameters=model_params,
            input_tfms=input_tfms,
            device=self.device,
        )


class TensorflowBuildInferenceLearner(BuildInferenceLearner):
    def execute(
        self,
        model: tf.Module,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation,
        **kwargs,
    ):
        self.inference_learner = TensorflowBackendInferenceLearner(
            model,
            network_parameters=model_params,
            input_tfms=input_tfms,
            device=self.device,
        )


class TFLiteBuildInferenceLearner(BuildInferenceLearner):
    def execute(
        self,
        model: bytes,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation,
        **kwargs,
    ):
        self.inference_learner = TFLiteBackendInferenceLearner(
            model,
            network_parameters=model_params,
            input_tfms=input_tfms,
            device=self.device,
        )


class DeepSparseBuildInferenceLearner(BuildInferenceLearner):
    def execute(
        self,
        model: Union[str, Path],
        model_params: ModelParams,
        **kwargs,
    ):
        input_names = get_input_names(str(model))
        output_names = get_output_names(str(model))

        self.inference_learner = PytorchDeepSparseInferenceLearner(
            onnx_path=model,
            network_parameters=model_params,
            input_names=input_names,
            output_names=output_names,
            device=self.device,
        )


class ONNXBuildInferenceLearner(BuildInferenceLearner):
    def execute(
        self,
        model: Union[str, Path],
        model_params: ModelParams,
        input_tfms: MultiStageTransformation,
        source_dl_framework: DeepLearningFramework,
        quantization_type: QuantizationType,
        **kwargs,
    ):
        input_names = get_input_names(str(model))
        output_names = get_output_names(str(model))

        self.inference_learner = ONNX_INFERENCE_LEARNERS[source_dl_framework](
            onnx_path=model,
            network_parameters=model_params,
            input_names=input_names,
            output_names=output_names,
            input_tfms=input_tfms,
            device=self.device,
            quantization_type=quantization_type,
        )


class OpenVINOBuildInferenceLearner(BuildInferenceLearner):
    def execute(
        self,
        model: str,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation,
        source_dl_framework: DeepLearningFramework,
        **kwargs,
    ):
        self.inference_learner = OPENVINO_INFERENCE_LEARNERS[
            source_dl_framework
        ].from_model_name(
            model_name=model + ".xml",
            model_weights=model + ".bin",
            input_tfms=input_tfms,
            network_parameters=model_params,
            device=self.device,
        )


class PyTorchTensorRTBuildInferenceLearner(BuildInferenceLearner):
    def execute(
        self,
        model: ScriptModule,
        input_tfms: MultiStageTransformation,
        model_params: ModelParams,
        **kwargs,
    ):
        self.inference_learner = PytorchTensorRTInferenceLearner(
            torch_model=model,
            input_tfms=input_tfms,
            network_parameters=model_params,
            device=self.device,
        )


class ONNXTensorRTBuildInferenceLearner(BuildInferenceLearner):
    def execute(
        self,
        model: Any,
        model_orig: Union[str, Path],
        model_params: ModelParams,
        input_tfms: MultiStageTransformation,
        source_dl_framework: DeepLearningFramework,
        **kwargs,
    ):
        nvidia_logger = trt.Logger(trt.Logger.ERROR)
        input_names = get_input_names(str(model_orig))
        output_names = get_output_names(str(model_orig))

        input_tfms.append(VerifyContiguity())
        runtime = trt.Runtime(nvidia_logger)
        engine = runtime.deserialize_cuda_engine(model)

        self.inference_learner = TENSOR_RT_INFERENCE_LEARNERS[
            source_dl_framework
        ](
            engine=engine,
            input_tfms=input_tfms,
            network_parameters=model_params,
            input_names=input_names,
            output_names=output_names,
            nvidia_logger=nvidia_logger,
            device=self.device,
        )


class IntelNeuralCompressorBuildInferenceLearner(BuildInferenceLearner):
    def execute(
        self,
        model: GraphModule,
        model_orig: Module,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation,
        **kwargs,
    ):
        self.inference_learner = PytorchNeuralCompressorInferenceLearner(
            model=model_orig,
            model_quant=model,
            input_tfms=input_tfms,
            network_parameters=model_params,
            device=self.device,
        )


class PyTorchApacheTVMBuildInferenceLearner(BuildInferenceLearner):
    def execute(
        self,
        model: ExecutorFactoryModule,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation,
        **kwargs,
    ):
        target_device = (
            str(tvm.target.cuda())
            if self.device.type is DeviceType.GPU
            else "llvm"
        )
        dev = tvm.device(str(target_device), 0)

        input_names = [
            f"input_{i}" for i in range(len(model_params.input_infos))
        ]

        graph_executor_module = tvm.contrib.graph_executor.GraphModule(
            model["default"](dev)
        )
        self.inference_learner = PytorchApacheTVMInferenceLearner(
            input_tfms=input_tfms,
            network_parameters=model_params,
            graph_executor_module=graph_executor_module,
            input_names=input_names,
            lib=model,
            target=target_device,
            device=self.device,
        )


class ONNXApacheTVMBuildInferenceLearner(BuildInferenceLearner):
    def execute(
        self,
        model: ExecutorFactoryModule,
        model_orig: str,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation,
        source_dl_framework: DeepLearningFramework,
        **kwargs,
    ):
        target_device = (
            str(tvm.target.cuda())
            if self.device.type is DeviceType.GPU
            else "llvm"
        )
        dev = tvm.device(str(target_device), 0)

        input_names = (
            get_input_names(model_orig)
            if model_orig is not None
            else [f"input_{i}" for i in range(len(model_params.input_infos))]
        )

        graph_executor_module = tvm.contrib.graph_executor.GraphModule(
            model["default"](dev)
        )
        self.inference_learner = APACHE_TVM_INFERENCE_LEARNERS[
            source_dl_framework
        ](
            input_tfms=input_tfms,
            network_parameters=model_params,
            graph_executor_module=graph_executor_module,
            input_names=input_names,
            lib=model,
            target=target_device,
            device=self.device,
        )


class FasterTransformerBuildInferenceLearner(BuildInferenceLearner):
    def execute(
        self,
        model: ScriptModule,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation,
        **kwargs,
    ):
        self.inference_learner = FasterTransformerInferenceLearner(
            torch_model=model,
            network_parameters=model_params,
            input_tfms=input_tfms,
            device=self.device,
        )
