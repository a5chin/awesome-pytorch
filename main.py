from torchnet import TorchNet
import pandas as pd

torchnet = TorchNet()
model = torchnet.create_model(layers=[5, 32, 256, 1024, 256, 32, 8, 2])

df = pd.read_csv('assets/data/train.csv')
torchnet.set_data(
    data=df,
    target='Survived',
    ignore_features=['PassengerId', 'Name', 'Sex', 'Age', 'Ticket', 'Cabin', 'Embarked']
)

trained_model = torchnet.train(model, total_epoch=10)