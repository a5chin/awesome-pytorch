import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir) + "/../")

from torchnet import TorchNet


def test_model():
    torchnet = TorchNet()
    model = torchnet.create_model(layers=[5, 32, 256, 1024, 256, 32, 8, 2])
    assert model is not None
