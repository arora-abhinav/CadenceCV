#File for purely logging experiment metrics and hyperparameters
import mlflow
import sys
sys.path.append("/Users/abhinavarora/Desktop/CadenceCV/ml/src")
from src.models.mlp_model import experiment
import mlflow

#Keys: "Macro F1", "N F1", "Y F1", "Macro Recall", "N Recall", "Y Recall", 
# "Macro Precision", "N Precision", "Y Precision", "Macro Support", 
# "N Support", "Y Support", "Epochs", "Batch_size", 
# "Model Architecture", "Weight Decay", "Learning Rate"
with mlflow.start_run():
    results = experiment()
    dataset = results["Training Set"]
    dataset = mlflow.data.pandas_dataset(dataset, name="Train Dataset")
    mlflow.set_tag("Sampling Num refers to how much of the initial training set is kept (of non-contact frames) to fix class imbalance")
    # Log parameters
    mlflow.log_param("Epochs", results["Epochs"])
    mlflow.log_param("Batch_size", results["Batch_size"])
    mlflow.log_param("Model Architecture", results["Model Architecture"])
    mlflow.log_param("Weight Decay", results["Weight Decay"])
    mlflow.log_param("Learning Rate", results["Learning Rate"])
    mlflow.log_param("Sampling Num", results["Sampling Num"])
    mlflow.log_param("Criterion", results["Criterion"])
    mlflow.log_param("Activation", results["Activation"])
    
    # Log metrics
    mlflow.log_metric("Macro F1", results["Macro F1"])
    mlflow.log_metric("N F1", results["N F1"])
    mlflow.log_metric("Y F1", results["Y F1"])
    mlflow.log_metric("Macro Recall", results["Macro Recall"])
    mlflow.log_metric("N Recall", results["N Recall"])
    mlflow.log_metric("Y Recall", results["Y Recall"])
    mlflow.log_metric("Macro Precision", results["Macro Precision"])
    mlflow.log_metric("N Precision", results["N Precision"])
    mlflow.log_metric("Y Precision", results["Y Precision"])
    mlflow.log_metric("Macro Support", results["Macro Support"])
    mlflow.log_metric("N Support", results["N Support"])
    mlflow.log_metric("Y Support", results["Y Support"])

    mlflow.log_input(dataset, context="Training")


