#File for purely logging experiment metrics and hyperparameters
import mlflow
import sys
import pickle
sys.path.append("/Users/abhinavarora/Desktop/CadenceCV/ml/src")
from archive.legacy.gait_cycle_classifier_SVM import experiment_gait_cycle_svm

#Keys: "Macro F1", "N F1", "Y F1", "Macro Recall", "N Recall", "Y Recall",
# "Macro Precision", "N Precision", "Y Precision", "Macro Support",
# "N Support", "Y Support", "Kernel", "C", "Gamma", "Degree", "Coef0"

#Same metric keys as the verifier models so every run lines up in the MLflow UI
metric_keys = ["Macro F1", "N F1", "Y F1",
               "Macro Recall", "N Recall", "Y Recall",
               "Macro Precision", "N Precision", "Y Precision",
               "Macro Support", "N Support", "Y Support"]


def log_gait_cycle_svm_experiment():
    #Each experiment gets its own MLflow run via the with statement
    with mlflow.start_run(run_name="Gait Cycle SVM"):
        results = experiment_gait_cycle_svm(kernel="rbf", C=2)

        # Log parameters (the SVM's own hyperparameters)
        mlflow.log_param("Kernel", results["Kernel"])
        mlflow.log_param("C", results["C"])
        mlflow.log_param("Gamma", results["Gamma"])
        mlflow.log_param("Degree", results["Degree"])
        mlflow.log_param("Coef0", results["Coef0"])

        # Log metrics (same schema as the other models so the runs are directly comparable)
        for key in metric_keys:
            mlflow.log_metric(key, results[key])


if __name__ == "__main__":
    log_gait_cycle_svm_experiment()
