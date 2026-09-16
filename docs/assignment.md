# PMLDL Assignment 1: Deployment

This assignment concerns **MLOps**. You are asked to implement a **simple automated pipeline** with three required stages: **1) data engineering**, **2) model engineering**, and **3) deployment**. The result should be a **GitHub repository** containing the code and instructions needed to run the pipeline.

The pipeline must process data, train and evaluate a machine learning model, and **deploy the model in an API** together with a **web application** that interacts with it. The application must contain input fields, a button to make a prediction, and an area displaying the prediction. The API and the application must run in separate **Docker containers**. FastAPI and Streamlit are suggested frameworks, but you may use alternatives.

The complete pipeline must **run automatically every 5 minutes**. If a run takes longer, you may increase the interval between runs. All three stages and their automation are required for the full assignment grade of **5 points**.

# Pipeline Stages

## Stage 1: Data Engineering

### Input Artifacts

* File(s) with raw data

### Output Artifacts

* File with training data
* File with testing data

### Operations

This stage involves **data loading**, **data cleaning**, and **data splitting**. Firstly, the pipeline loads the data by reading it from the file(s). Then, the pipeline should clean the data by removing/imputing missing values and removing the outliers. At the end of this stage, the pipeline should split the data into train and test datasets and save them in the corresponding files.

### Tools

The operations of the Stage 1 may be implemented using data pipelines of **DVC** or using **Airflow** tasks.

### Additional Notes

You can use any dataset except CelebFaces and patient's smoking status, which were presented in the lab. The size of the dataset is up to you.

## Stage 2: Model Engineering

### Input Artifacts

* File with training data
* File with testing data

### Output Artifacts

* File with a trained model
* Values of the testing metrics

### Operations

This stage involves **feature engineering**, **model** **training, evaluation,** and **packaging**. When the pipeline starts this stage, it should obtain the training and testing data from the previous stage. The pipeline firstly runs feature engineering routines to transform training and testing data into the features for the model. Then, a model is trained using the features of the training data. At the end, the trained model should be evaluated by calculating performance metrics on the testing features. The testing metrics should be logged. The trained model should be saved (packaged) in a file.

### Tools

This stage *may* be entirely performed in **MLflow**.

### Additional Notes

In this stage it is enough to train a single model. However, you can train multiple models if you want. The type and complexity of the model is up to you. Even a simple model will be enough for this stage. You can also do hyperparameter tuning, but it is not necessary. The choice of the extension of the file with the trained model is up to you.

## Stage 3: Deployment

### Input Artifacts

* File with a trained model

### Output Artifacts

* Running model API
* Running app

### Operations

This stage involves **model API deployment** and **app deployment**. The pipeline should create a Docker image with the model API using the trained model from Stage 2 and run it in a container. Also, the pipeline should create a Docker image with the app communicating with the model API. The application should contain the input fields, button to make the prediction. After pressing the button the predictions of the model should be shown.

### Tools

Use Docker to run the API and the app! For the API you can use FastAPI or other web framework. For the app you can use Streamlit or similar frameworks.

### Additional Notes

The API and the app must be deployed in separate Docker containers. The app may be very simple and contain the input fields to enter input data, button to run prediction, and the prediction itself.

## Expected Repository Structure

Here is the structure of the repository that you are encouraged to follow:

```yaml
├── code
│   ├── datasets
│   ├── deployment
│   │   ├── api
│   │   └── app
│   └── models
├── data
│   ├── processed
│   └── raw
├── notebooks
├── models
├── services
│   └── airflow
│       ├── dags
│       └── logs
└── requirements.txt
```

## Recommended Steps to Complete the Assignment

The following steps provide one way to implement the pipeline. You may use other tools and repository layouts, provided that all required stages are implemented and the repository is logically structured.


 1. Create a GitHub repository. Make sure that your repository is __public__. Clone the repository. The directory of the cloned repository is now your working directory.
 2. Create a new virtual environment in your working directory.
 3. Create a `requirements.txt` file with the list of Python libraries necessary for your pipeline. Always keep this file updated with a fresh list of Python libraries.
 4. Download the data that you will use for model training and validation. Create the `data/raw` folder and place the files with the data to this folder.
 5. Write code to load, clean, and split the data using DVC or Airflow. Make the code save the split data in `data/processed` folder. If you are using Airflow, save the files with the code in `services/airflow/dags`, otherwise save the code to `code/datasets` folder.
 6. Write code to create features for the model. Implement model training, evaluation, and packaging. Log the metrics and the model in MLflow. Save the code to `code/models` folder. Save the trained model in `models` directory.
 7. Implement API for the model using FastAPI. Save the code to `code/deployment/api` folder. Write Dockerfile for the API and save it in `code/deployment/api/Dockerfile`
 8. Implement the application using Streamlit. Save the code to `code/deployment/app` folder. Write Dockerfile for the application and save it in `code/deployment/app/Dockerfile`
 9. Write a docker-compose file and save it in `code/deployment/docker-compose.yml`
10. Connect data processing, model training and evaluation, and deployment into a single pipeline. The deployment stage should build and start the API and app using the Docker Compose file from step 9. If you use Airflow, save the pipeline DAG in `services/airflow/dags`. Verify that the complete pipeline works and schedule it to run every 5 minutes; increase the interval if a run takes longer.
11. Document how to run the pipeline and access the API and app in the repository README. Commit the files and push them to GitHub.

## Other Notes

* You may use any dataset except CelebFaces and patient's smoking status, which were presented in the lab. Using these datasets will lead to a **deduction of 50% of the points gained**.
* You may use tools other than those suggested and add other tools to the pipeline, but all three stages must be present. The `services/airflow` directory is only needed if you use Airflow.
* Small datasets and simple models are sufficient: the focus is on the working automated pipeline.
* **Docker is required** for the API and app deployment. Ignoring this requirement will lead to **0 points** for the assignment.
* If the trained model is too large for GitHub, you do not have to push it. Document how to obtain or generate it when running the pipeline.

## Submission

Submit your solution as a **link to a public GitHub repository**. After the submission deadline, a **meeting with TAs** will be arranged where you must demonstrate the complete automated pipeline, including predictions through the web application.

## Grading Criteria

The assignment grade is the sum of the following criteria, up to **5 points**:

* **Data engineering** stage is implemented and working — 1 point
* **Model engineering** stage is implemented and working — 1 point
* **Deployment** stage is implemented and working: the model API and web application run in separate Docker containers, and the application displays predictions obtained from the API — 1 point
* The complete pipeline is **automated** and runs on the required schedule — 1 point
* The repository is **structured** (has a logical hierarchy of files and folders) — 1 point

## Useful Links

* [DVC Get Started Guide](https://dvc.org/doc/start)
* [MLflow Get Started Guide](https://mlflow.org/docs/latest/getting-started/index.html)
* [FastAPI Website](https://fastapi.tiangolo.com/)
* [Docker Tutorial for Beginners](https://docker-curriculum.com/)
* [Apache Airflow Tutorial](https://airflow.apache.org/docs/apache-airflow/1.10.15/tutorial.html)
* [Get Started with Streamlit](https://docs.streamlit.io/get-started)
