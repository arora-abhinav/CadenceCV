#File for purely logging experiment metrics and hyperparameters
import mlflow
import sys
sys.path.append("/Users/abhinavarora/Desktop/CadenceCV/ml/src")
from models.d_frame_model import experiment_svm
import pickle

#Keys: "Macro F1", "N F1", "Y F1", "Macro Recall", "N Recall", "Y Recall",
# "Macro Precision", "N Precision", "Y Precision", "Macro Support",
# "N Support", "Y Support", "Epochs", "Batch_size",
# "Model Architecture", "Weight Decay", "Learning Rate"

#Both experiments return the same metric keys, so logging them the same way keeps the MLP and SVM
#runs directly comparable in the MLflow UI
metric_keys = ["Macro F1", "N F1", "Y F1",
               "Macro Recall", "N Recall", "Y Recall",
               "Macro Precision", "N Precision", "Y Precision",
               "Macro Support", "N Support", "Y Support"]


def log_svm_experiment(kernel="poly", degree=3, coef0=1):
    #Each experiment gets its own MLflow run via the with statement
    with mlflow.start_run(run_name="SVM"):
        results = experiment_svm(C=15)
        # Log parameters (the SVM has its own hyperparameters instead of the MLP's)
        with open('/Users/abhinavarora/Desktop/CadenceCV/ml/weights/d_frame_svm.pkl', "wb") as file:
            pickle.dump(results["Trained Model"], file)

        mlflow.log_param("Kernel", results["Kernel"])
        mlflow.log_param("C", results["C"])
        mlflow.log_param("Gamma", results["Gamma"])
        mlflow.log_param("Sampling Num", results["Sampling Num"])
        mlflow.log_param("Degree", results["Degree"])
        mlflow.log_param("Coef0", results["Coef0"])

        # Log metrics (same schema as the MLP so the two runs are directly comparable)
        for key in metric_keys:
            mlflow.log_metric(key, results[key])


if __name__ == "__main__":
    log_svm_experiment()
