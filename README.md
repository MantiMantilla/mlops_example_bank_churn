# mlops_example_bank_churn

An example of an ML project, structured for production deployment. Includes testing and packaging configs, as well as robust software architecture principles to train and serialize the model.

## Deployment

To prepare for deployment, follow these steps:

1. Clone this project.

```bash
git clone https://github.com/MantiMantilla/mlops_example_bank_churn

# or
# git clone git@github.com:MantiMantilla/mlops_example_bank_churn.git

cd mlops_example_bank_churn
```

2. Make sure `tox` and `build` are installed.

```bash
python -m venv venv
source ./venv/bin/activate
pip install tox build
```

3. Train and test the model and prediction script.

```bash
tox run -e train
tox run -e test_package

# just for formatting:
# tox run -e checks
```

4. Build the package for distribution and install it.

```bash
python3 -m build
pip install ./dist/model_abandono-0.0.1-py3-none-any.whl
```

5. Make sure it is properly installed and executes.

```python
import pandas as pd
from model.predict import make_prediction

sample_input_data = pd.read_csv("./model/datasets/bankchurn_test.csv")
result = make_prediction(input_data=sample_input_data)
print(result)
```

## Credits

Developed by: [Juan Fernando Perez](https://juanfperez.com/)
Used in: MIAD - Despliegue de Soluciones Analíticas

