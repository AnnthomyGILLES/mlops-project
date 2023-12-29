from zenml.client import Client

from pipelines.training_pipeline import train_pipeline

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(
        data_path=r"C:\Users\gotam\PycharmProjects\mlops-project\mlops_maker\data\olist_customers_dataset.csv"
    )
