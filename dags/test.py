import mlflow
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging
import joblib
from datetime import datetime
from itertools import product
from airflow.decorators import dag, task
from mlflow_provider.hooks.client import MLflowClientHook
# from mlflow_provider.operators.registry import CreateRegisteredModelOperator
from airflow.providers.amazon.aws.operators.s3 import S3CreateBucketOperator
import boto3
from airflow.hooks.base import BaseHook
import uuid


MLFLOW_CONN_ID = "mlflow_default"
MINIO_CONN_ID = "minio_local"
ARTIFACT_BUCKET = "titanic_test1"
EXPERIMENT_NAME = "titanic_test1"
REGISTERED_MODEL_NAME = "titanic_test1"
AWS_ACCESS_KEY_ID = "minio"
AWS_SECRET  = "minio123"
AWS_ENDPOINT_URL = "http://localhost:9000"

# Logging configuration
logging.basicConfig(level=logging.INFO)

@dag(
    dag_id="titanic_xgb",
    default_args={
        'owner': 'airflow',
        'start_date': datetime(2025, 3, 12),
        'retries': 1,
    },
    schedule_interval='@once',
    catchup=False
)

def titanic_xgb():
    @task
    def connect_minio():    

        conn = BaseHook.get_connection(MINIO_CONN_ID)
        s3 = boto3.resource('s3',
                     endpoint_url=AWS_ENDPOINT_URL,
                     aws_access_key_id=AWS_ACCESS_KEY_ID,
                     aws_secret_access_key=AWS_SECRET,
        )
        s3client = s3.meta.client 

    def create_experiment_ids(experiment_name):
        # Check if the experiment already exists; if not, create it
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except:
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        return experiment_id
    
    @task
    def create_experiment(experiment_name, artifact_bucket, **context):

        ts = context.get("ts", "default_ts")

        mlflow_hook = MLflowClientHook(mlflow_conn_id=MLFLOW_CONN_ID)
        new_experiment_information = mlflow_hook.run(
            endpoint="api/2.0/mlflow/experiments/create",
            request_params={
                "name": ts + "_" + experiment_name,
                "artifact_location": f"s3://{artifact_bucket}/",
            },
        ).json()

        return new_experiment_information["experiment_id"]

    @task
    def preprocess_data(**kwargs):
        # read data
        data = pd.read_csv('include/test/data/train_and_test2.csv')

        # fill missing values
        if 'Embarked' in data.columns:
            data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

        joblib.dump((data), '/tmp/preprocess_data.pkl')
        logging.info("Data saved for preprocess_data")


    @task
    def split_data(**kwargs):
        data = joblib.load('/tmp/preprocess_data.pkl')
        
        features = [ col for col in data.columns if col not in ['Passengerid','2urvived']]
        X = data[features]
        y = data['2urvived']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        joblib.dump((X_train, y_train, X_test, y_test), '/tmp/data.pkl')
        logging.info("Data saved for training and testing.")

    @task
    def train_model_with_mlflow(experiment_id, **kwargs):
        X_train, y_train,  X_test, y_test = joblib.load('/tmp/data.pkl')
        categorical_features = ['Sex', 'Pclass', 'Embarked']
        for col in categorical_features:
            X_train[col] = X_train[col].astype('category')
            X_test[col] = X_test[col].astype('category')

        param_grid = {
            'objective': ['binary:logistic'],
            'eval_metric': ['logloss'],
            'booster': ['gbtree'],
            'num_estimators': [100, 200],
            'learning_rate': [0.01, 0.05],
            'max_depth': [5, 7],
            'num_leaves': [31, 63] 
        }
        # param_grid = {
        #     'objective': 'binary',
        #     'metric': 'binary_error',
        #     'boosting_type': 'gbdt',
        #     'num_estimators': [50, 100, 200],
        #     'learning_rate': [0.01, 0.05, 0.1],
        #     'max_depth': [3, 5, 7],
        #     'num_leaves': [31, 63, 127]
        # }
        # Generate all combinations of parameters
        param_combinations = list(product(*param_grid.values()))

        

        # Iterate through parameter combinations and train a model for each
        best_accuracy = 0
        best_model = None
        best_params = None
        
        for params in param_combinations:

            experiment_name = f"xgboost_titanic_experiment_{str(uuid.uuid4())[:8]}"  # Unique experiment ID
            new_experiment_id = create_experiment_ids(experiment_name)
            mlflow.set_experiment(
                                   experiment_name=experiment_name
                                   )

            # mlflow.sklearn.autolog()
            
            with mlflow.start_run(run_name=experiment_id ,experiment_id=new_experiment_id) as run:
                # Map parameter combinations to dict
                param_dict = dict(zip(param_grid.keys(), params))
                logging.info(f"Training model with parameters: {param_dict}")
                
                # Initialize LightGBM model with current parameters
                model = xgb.XGBClassifier(
                    n_estimators=param_dict['num_estimators'],
                    learning_rate=param_dict['learning_rate'],
                    max_depth=param_dict['max_depth'],
                    max_leaves=param_dict['num_leaves'],
                    enable_categorical=True,
                )
                
                # Train the model
                model.fit(X_train, y_train)
                
                # Save model to file with parameter-based filename
                model_filename = f"/tmp/lgb_model_{param_dict['num_estimators']}_{param_dict['learning_rate']}_{param_dict['max_depth']}_{param_dict['num_leaves']}.pkl"
                joblib.dump(model, model_filename)
                logging.info(f"Model trained and saved to {model_filename}.")

                # Evaluate the model and track the best one
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)
                class_report = classification_report(y_test, y_pred, output_dict=True)

                logging.info(f"Accuracy for parameters {param_dict}: {accuracy:.4f}")
                logging.info(f"Classification Report: \n{classification_report(y_test, y_pred)}")


                # Log hyperparameters and accuracy to MLflow
                mlflow.log_params(param_dict)
                mlflow.log_metric("accuracy", accuracy)
                # mlflow.log_metric("f1", accuracy)
                mlflow.log_text(str(class_report), "classification_report.txt")
                # mlflow.log_artifact("confusion_matrix.txt", str(cm))
                # Check if the problem is binary or multiclass
                if cm.shape == (2, 2):  # Binary classification
                    # Extract values
                    tn, fp, fn, tp = cm.ravel()

                    # Log the metrics
                    mlflow.log_metric("tp", tp)
                    mlflow.log_metric("tn", tn)
                    mlflow.log_metric("fp", fp)
                logging.info(f"Logged params: {param_dict} - Accuracy: {accuracy}")
                logging.info(f"Classification Report: \n{classification_report(y_test, y_pred)}")
                logging.info(f"Confusion matrix for multiclass classification: \n{cm}")
                mlflow.log_param("confusion_matrix", cm.tolist())

                print(f"Logged params: {param_dict} - Accuracy: {accuracy}")
                print(f"Classification Report: \n{classification_report(y_test, y_pred)}")

                
                # Track the best model based on accuracy
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                    best_params = param_dict
        
        # Log the best model and its parameters
        logging.info(f"Best model found with parameters: {best_params} and accuracy: {best_accuracy:.4f}")
        
        # Save the best model to file
        joblib.dump(best_model, '/tmp/best_lgb_model.pkl')
        logging.info("Best model saved to /tmp/best_lgb_model.pkl")

    
    connect_minio = connect_minio()
    experiment_created = create_experiment(EXPERIMENT_NAME, ARTIFACT_BUCKET)
    preprocess_task = preprocess_data()
    split_task = split_data()
    train_task = train_model_with_mlflow(experiment_id=experiment_created)

    # Define task dependencies
    (
        connect_minio
        # >> create_buckets_if_not_exists
        >> experiment_created
        >> preprocess_task
        >> split_task
        >> train_task
        # >> create_registered_model
    )

titanic_xgb()