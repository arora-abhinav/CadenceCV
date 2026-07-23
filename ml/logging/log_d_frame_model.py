#File for purely logging experiment metrics and hyperparameters
import mlflow
import sys
sys.path.append("/Users/abhinavarora/Desktop/CadenceCV/ml/src")
from models.d_frame_model import experiment_MLP, experiment_svm
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


def log_mlp_experiment():
    #Each experiment gets its own MLflow run via the with statement
    with mlflow.start_run(run_name="MLP"):
        results = experiment_MLP()
        dataset = results["Training Set"]
        dataset = mlflow.data.from_pandas(dataset, name="Train Dataset", targets="Label")
        mlflow.set_tag("Samplig Num", "It refers to how much of the initial training set is kept (of non-contact frames) to fix class imbalance")
        # Log parameters

        with open("/Users/abhinavarora/Desktop/CadenceCV/ml/weights/d_frame_svm.pkl", "wb") as file:
            pickle.dump(results["Trained Model"], file)
        mlflow.log_param("Epochs", results["Epochs"])
        mlflow.log_param("Batch_size", results["Batch_size"])
        mlflow.log_param("Model Architecture", results["Model Architecture"])
        mlflow.log_param("Weight Decay", results["Weight Decay"])
        mlflow.log_param("Learning Rate", results["Learning Rate"])
        mlflow.log_param("Sampling Num", results["Sampling Num"])
        mlflow.log_param("Criterion", results["Criterion"])
        mlflow.log_param("Activation", results["Activation"])

        # Log metrics
        for key in metric_keys:
            mlflow.log_metric(key, results[key])

        mlflow.log_input(dataset, context="Training")


def log_svm_experiment(kernel="poly", degree=3, coef0=1):
    #Each experiment gets its own MLflow run via the with statement
    with mlflow.start_run(run_name="SVM"):
        results = experiment_svm(C=15)
        # Log parameters (the SVM has its own hyperparameters instead of the MLP's)
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
