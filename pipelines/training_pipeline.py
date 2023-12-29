from zenml import pipeline

from steps.dataloader_step import ingest_df
from steps.evaluator_step import evaluate_model
from steps.preprocessing_step import clean_data
from steps.trainer_step import train_model


@pipeline(enable_cache=True)
def train_pipeline(data_path: str):
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_data(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    train_pipeline(data_path=r"/data/raw/olist_customers_dataset.csv")
