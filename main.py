from core.dataset import Dataset, generate_patient_data
from core.logistic_regression import LogisticRegression
from core.neural_network import NeuralNetwork
from pipeline.trainer import Trainer
from pipeline.evaluator import Evaluator
from app.api import diagnose_patient

generate_patient_data()                      # crée data/patient_data.csv
X_tr, X_te, y_tr, y_te = Dataset().split()

patient = {"age": 62, "temperature": 39.1, "toux": 1, "fatigue": 1,
           "saturation_o2": 92, "leucocytes": 13.0, "crp": 60}
sain = {"age": 30, "temperature": 36.8, "toux": 0, "fatigue": 0,
        "saturation_o2": 98, "leucocytes": 6.5, "crp": 2}

# Le même ClinicalPredictor fonctionne avec les deux modèles (polymorphisme)
for model in (LogisticRegression(), NeuralNetwork()):
    model, scaler = Trainer(model).train(X_tr, y_tr)
    print(type(model).__name__, Evaluator(model, scaler).evaluate(X_te, y_te))
    print("  patient 1 ->", diagnose_patient(model, scaler, patient))
    print("  patient 2 ->", diagnose_patient(model, scaler, sain))
