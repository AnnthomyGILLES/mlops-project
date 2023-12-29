## Overview

In the fast-evolving field of machine learning (ML), the ability to efficiently operationalize models is crucial. This
project is an exploration into ZenML, an MLOps framework designed to simplify and streamline the process of building and
managing ML workflows. The objective is to quickly ramp up on ZenML, demonstrating its utility and effectiveness in a
practical context.

#### Project Context

Machine Learning engineers often face challenges related to the scalability, reproducibility, and deployment of ML
models. Traditional approaches can lead to cumbersome and disjointed workflows. ZenML addresses these issues, providing
an elegant and powerful solution for ML operations (MLOps). This project serves as a hands-on introduction to ZenML,
showcasing its capabilities through a concrete example.

#### Learning Objectives

The key learning objectives of this project include:

- Gaining practical experience with ZenML as an MLOps tool.
- Understanding how to transition from traditional ML workflows to those managed by ZenML. Demonstrating the ease of
  building, running, and monitoring ML pipelines with ZenML.
- Highlighting the advantages of using an MLOps framework in terms of scalability, reproducibility, and efficiency.

## ML Objective: Predicting how a customer will feel about a product before they even ordered it

**Dataset**: For a given customer's historical data, we are tasked to predict the review score for the next order or
purchase. We will be using
the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). This
dataset has information on 100,000 orders from 2016 to 2018 made at multiple marketplaces in Brazil. Its features allow
viewing charges from various dimensions: from order status, price, payment, freight performance to customer location,
product attributes and finally, reviews written by customers. The objective here is to predict the customer satisfaction
score for a given order based on features like order status, price, payment, etc. In order to achieve this in a
real-world scenario, we will be using [ZenML](https://zenml.io/) to build a production-ready pipeline to predict the
customer satisfaction score for the next order or purchase.

The purpose of this repository is to demonstrate how [ZenML](https://github.com/zenml-io/zenml) empowers your business
to build and deploy machine learning pipelines in a multitude of ways:

- By offering you a framework and template to base your own work on.
- By integrating with tools like [MLflow](https://mlflow.org/) for deployment, tracking and more
- By allowing you to build and deploy your machine learning pipelines easily

## :snake: Python Requirements

Let's jump into the Python packages you need. Within the Python environment of your choice, run:

```bash
git clone https://github.com/AnnthomyGILLES/mlops-project.git
poetry install
```

Starting with ZenML 0.20.0, ZenML comes bundled with a React-based dashboard. This dashboard allows you
to observe your stacks, stack components and pipeline DAGs in a dashboard interface. To access this, you need
to [launch the ZenML Server and Dashboard locally](https://docs.zenml.io/user-guide/starter-guide#explore-the-dashboard),
but first you must install the optional dependencies for the ZenML server:

```bash
pip install zenml["server"]zenml up
```

If you are running the `run_deployment.py` script, you will also need to install some integrations using ZenML:

```bash
zenml integration install mlflow -y
```

The project can only be executed with a ZenML stack that has an MLflow experiment tracker and model deployer as a
component. Configuring a new stack with the two components are as follows:

```bash
zenml integration install mlflow -yzenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set
```

## 📙 Resources & References

We had written a blog that explains this project
in-depth: [Predicting how a customer will feel about a product before they even ordered it](https://github.com/zenml-io/zenml-projects/tree/main/customer-satisfaction).

If you'd like to watch the video that explains the project, you can watch the [video](https://youtu.be/L3_pFTlF9EQ).

## :thumbsup: The Solution

In order to build a real-world workflow for predicting the customer satisfaction score for the next order or purchase (
which will help make better decisions), it is not enough to just train the model once.

Instead, we are building an end-to-end pipeline for continuously predicting and deploying the machine learning model,
alongside a data application that utilizes the latest deployed model for the business to consume.

This pipeline can be deployed to the cloud, scale up according to our needs, and ensure that we track the parameters and
data that flow through every pipeline that runs. It includes raw data input, features, results, the machine learning
model and model parameters, and prediction outputs. ZenML helps us to build such a pipeline in a simple, yet powerful,
way.

In this Project, we give special consideration to
the [MLflow integration](https://github.com/zenml-io/zenml/tree/main/examples) of ZenML. In particular, we utilize
MLflow tracking to track our metrics and parameters, and MLflow deployment to deploy our model. We also
use [Streamlit](https://streamlit.io/) to showcase how this model will be used in a real-world setting.

### Training Pipeline

Our standard training pipeline consists of several steps:

- `ingest_data`: This step will ingest the data and create a `DataFrame`.
- `clean_data`: This step will clean the data and remove the unwanted columns.
- `train_model`: This step will train the model and save the model
  using [MLflow autologging](https://www.mlflow.org/docs/latest/tracking.html).
- `evaluation`: This step will evaluate the model and save the metrics -- using MLflow autologging -- into the artifact
  store.

### Deployment Pipeline

WIP
