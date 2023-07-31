from typing import List, Tuple

from nebullvm.core.models import QuantizationType
from nebullvm.optional_modules.tensorflow import tensorflow as tf


def _quantize_dynamic(model: tf.Module):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()
    return tflite_quant_model


def _quantize_static(model: tf.Module, dataset: List[Tuple[tf.Tensor, ...]]):
    def representative_dataset():
        for data_tuple in dataset:
            yield list(data_tuple)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    tflite_quant_model = converter.convert()
    return tflite_quant_model


def _half_precision(model: tf.Module):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_quant_model = converter.convert()
    return tflite_quant_model


def quantize_tensorflow(
    model: tf.Module,
    quantization_type: QuantizationType,
    input_data_tensorflow: List[Tuple[tf.Tensor, ...]],
):
    if quantization_type is QuantizationType.DYNAMIC:
        quantized_model = _quantize_dynamic(model)
    elif quantization_type is QuantizationType.STATIC:
        quantized_model = _quantize_static(model, input_data_tensorflow)
    elif quantization_type is QuantizationType.HALF:
        quantized_model = _half_precision(model)
    else:
        raise NotImplementedError(
            f"Quantization not supported for type {quantization_type}"
        )

    return quantized_model
