from zenml import pipeline

from steps.clean_data import clean_data
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model


@pipeline(enable_cache=True)
def train_pipeline(data_path: str):
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_data(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    train_pipeline(data_path=r"/data/raw/olist_customers_dataset.csv")
