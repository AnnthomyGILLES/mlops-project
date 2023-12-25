from zenml import pipeline

from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.moden_train import train_model


@pipeline(enable_cache=True)
def train_pipeline(data_path: str):
    df = ingest_df(data_path)
    clean_df(df)
    train_model(df)
    evaluate_model(df)


if __name__ == '__main__':
    train_pipeline(data_path=r"C:\Users\gotam\PycharmProjects\mlops-project\data\olist_customers_dataset.csv")
