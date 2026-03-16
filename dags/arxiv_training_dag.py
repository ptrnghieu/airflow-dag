"""
ArXiv Category Training DAG

This DAG trains a machine learning model to predict paper categories
based on title and abstract from MongoDB data.

Schedule: Weekly (Sunday 00:00)
"""

from datetime import datetime, timedelta

from airflow.sdk import dag, task
from airflow.providers.standard.operators.bash import BashOperator

# Default arguments
default_args = {
    "owner": "duonganh",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="arxiv_category_trainer",
    default_args=default_args,
    description="Train ML model to predict ArXiv paper categories",
    schedule="0 0 * * 0",  # Weekly on Sunday at 00:00
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["ml", "training", "arxiv"],
)
def arxiv_category_training_dag():
    """ArXiv Category Training DAG"""

    # Task 1: Create directories
    create_directories = BashOperator(
        task_id="create_directories",
        bash_command="""
        mkdir -p /opt/airflow/tmp/ml_training/data && \
        mkdir -p /opt/airflow/tmp/ml_training/models && \
        mkdir -p /opt/airflow/tmp/ml_training/logs && \
        echo "Directories created successfully"
        """,
    )

    # Task 2: Check data availability
    @task(task_id="check_data_availability")
    def check_data():
        """Check if MongoDB has enough data for training."""
        from ml.arxiv_trainer import ArxivTrainer

        trainer = ArxivTrainer()
        result = trainer.check_data_availability()
        return result

    # Task 3: Load data from MongoDB
    @task(task_id="load_data_from_mongodb")
    def load_data():
        """Load training data from MongoDB."""
        from ml.arxiv_trainer import ArxivTrainer

        trainer = ArxivTrainer()
        result = trainer.load_data_from_mongodb()
        return result

    # Task 4: Validate data
    @task(task_id="data_validation")
    def validate_data():
        """Validate data quality."""
        from ml.arxiv_trainer import ArxivTrainer

        trainer = ArxivTrainer()
        result = trainer.validate_data()
        return result

    # Task 5: Preprocess text
    @task(task_id="preprocess_text")
    def preprocess():
        """Preprocess text data."""
        from ml.arxiv_trainer import ArxivTrainer

        trainer = ArxivTrainer()
        result = trainer.preprocess_text()
        return result

    # Task 6: Split dataset
    @task(task_id="split_dataset")
    def split_data():
        """Split data into train/val/test sets."""
        from ml.arxiv_trainer import ArxivTrainer

        trainer = ArxivTrainer()
        result = trainer.split_dataset()
        return result

    # Task 7: Train model
    @task(task_id="train_model")
    def train():
        """Train the classification model."""
        from ml.arxiv_trainer import ArxivTrainer

        trainer = ArxivTrainer()
        result = trainer.train_model()
        return result

    # Task 8: Evaluate model
    @task(task_id="evaluate_model")
    def evaluate():
        """Evaluate model performance."""
        from ml.arxiv_trainer import ArxivTrainer

        trainer = ArxivTrainer()
        result = trainer.evaluate_model()
        return result

    # Task 9: Save artifacts
    @task(task_id="save_artifacts")
    def save_artifacts(**context):
        """Save model artifacts and metadata."""
        from ml.arxiv_trainer import ArxivTrainer

        trainer = ArxivTrainer()
        result = trainer.save_artifacts(**context)
        return result

    # Define task dependencies
    (
        create_directories
        >> check_data()
        >> load_data()
        >> validate_data()
        >> preprocess()
        >> split_data()
        >> train()
        >> evaluate()
        >> save_artifacts()
    )


# Instantiate the DAG
arxiv_training = arxiv_category_training_dag()

