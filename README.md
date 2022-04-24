<div align="center">

<h1>TorchNet</h1>

[![Pytest](https://github.com/a5chin/awesome-pytorch/actions/workflows/pytest.yml/badge.svg)](https://github.com/a5chin/awesome-pytorch/actions/workflows/pytest.yml) [![Linting](https://github.com/a5chin/awesome-pytorch/actions/workflows/linting.yml/badge.svg)](https://github.com/a5chin/awesome-pytorch/actions/workflows/linting.yml) [![License](https://img.shields.io/pypi/l/ansicolortags.svg)](https://img.shields.io/pypi/l/ansicolortags.svg)

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/) [![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)](examples/classification.ipynb) [![Numpy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/) [![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)

</div>

# Usage
## Installation

```sh
pip install -r requirements.txt
```
or
```sh
pip install git+https://github.com/a5chin/awesome-pytorch
```

## Classification
### Create Model
```python
torchnet = TorchNet()
model = torchnet.create_model(layers=[5, 8, 10, 12, 10, 8, 4, 1])
```
<img alt="assets/images/nn.svg" src="assets/images/nn.svg" width="100%"></img>

### Set Data
```python
df = pd.read_csv('assets/data/train.csv')
torchnet.set_data(
    data=df,
    target='Survived',
    ignore_features=['PassengerId', 'Name', 'Sex', 'Age', 'Ticket', 'Cabin', 'Embarked']
)
```
- target
   - The Value to predict in header
- ignore_features
   - Values to ignore in header

### Train Model
```python
trained_model = torchnet.train(model, total_epoch=100)
```
