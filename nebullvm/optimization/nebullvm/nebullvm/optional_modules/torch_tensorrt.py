from nebullvm.optional_modules.dummy import DummyClass

try:
    import torch_tensorrt
    from torch_tensorrt.ptq import DataLoaderCalibrator  # noqa F401
except ImportError:
    torch_tensorrt = DummyClass
    DataLoaderCalibrator = None
