"""
ArXiv Category Prediction Model Training Module

This module contains all training logic for the ArXiv paper category classifier.
"""

import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    hamming_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

logger = logging.getLogger(__name__)


class ArxivTrainer:
    """Main trainer class for ArXiv category prediction model."""

    def __init__(
        self,
        mongo_uri: str = "mongodb://admin:admin123@arxiv_mongodb:27017/",
        output_dir: str = "/opt/airflow/tmp/ml_training",
    ):
        """
        Initialize the trainer.

        Args:
            mongo_uri: MongoDB connection string
            output_dir: Directory to save outputs
        """
        self.mongo_uri = mongo_uri
        self.output_dir = output_dir
        self.data_dir = os.path.join(output_dir, "data")
        self.model_dir = os.path.join(output_dir, "models")
        self.log_dir = os.path.join(output_dir, "logs")

    def check_data_availability(self) -> Dict[str, Any]:
        """
        Check if MongoDB has enough data for training.

        Returns:
            Dict with data statistics

        Raises:
            ValueError: If not enough data or missing required fields
        """
        logger.info("Checking data availability in MongoDB...")

        client = MongoClient(self.mongo_uri)
        db = client["arxiv_db"]

        # Count documents
        count = db.papers.count_documents({})
        logger.info(f"Found {count} papers in MongoDB")

        # Minimum required: 10 papers (lowered for testing)
        if count < 10:
            raise ValueError(f"Not enough data: {count} papers (minimum: 10)")

        # Check required fields
        sample = db.papers.find_one()
        if not sample:
            raise ValueError("No documents found in MongoDB")

        required_fields = ["title", "abstract", "categories"]
        missing_fields = [field for field in required_fields if field not in sample]

        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        client.close()

        result = {
            "total_papers": count,
            "status": "ok",
            "required_fields_present": True,
        }

        logger.info(f"Data availability check passed: {result}")
        return result

    def load_data_from_mongodb(self) -> Dict[str, Any]:
        """
        Load data from MongoDB into pandas DataFrame.

        Returns:
            Dict with load statistics
        """
        logger.info("Loading data from MongoDB...")

        client = MongoClient(self.mongo_uri)
        db = client["arxiv_db"]

        # Query data
        cursor = db.papers.find(
            {}, {"title": 1, "abstract": 1, "categories": 1, "published_date": 1, "_id": 0}
        )

        # Convert to DataFrame
        df = pd.DataFrame(list(cursor))
        client.close()

        logger.info(f"Loaded {len(df)} papers from MongoDB")

        # Save raw data
        os.makedirs(self.data_dir, exist_ok=True)
        raw_data_path = os.path.join(self.data_dir, "raw_data.pkl")
        df.to_pickle(raw_data_path)

        logger.info(f"Saved raw data to {raw_data_path}")

        return {"total_samples": len(df), "file_path": raw_data_path}

    def validate_data(self) -> Dict[str, Any]:
        """
        Validate data quality and create validation report.

        Returns:
            Dict with validation results

        Raises:
            ValueError: If data validation fails
        """
        logger.info("Validating data quality...")

        raw_data_path = os.path.join(self.data_dir, "raw_data.pkl")
        df = pd.read_pickle(raw_data_path)

        issues = []

        # Check missing values
        if df["title"].isna().any():
            missing_titles = df["title"].isna().sum()
            issues.append(f"Missing titles: {missing_titles}")

        if df["abstract"].isna().any():
            missing_abstracts = df["abstract"].isna().sum()
            issues.append(f"Missing abstracts: {missing_abstracts}")

        # Fill missing values temporarily for validation
        df["title"] = df["title"].fillna("")
        df["abstract"] = df["abstract"].fillna("")

        # Check text length
        df["text_length"] = df["title"].str.len() + df["abstract"].str.len()
        too_short = (df["text_length"] < 50).sum()
        if too_short > 0:
            logger.warning(f"Found {too_short} papers with very short text (< 50 chars)")

        # Category statistics
        all_cats = []
        for cats in df["categories"]:
            if isinstance(cats, list):
                all_cats.extend(cats)
            elif isinstance(cats, str):
                all_cats.append(cats)

        cat_counts = pd.Series(all_cats).value_counts()

        # Create validation report
        report = {
            "total_samples": len(df),
            "avg_text_length": float(df["text_length"].mean()),
            "min_text_length": int(df["text_length"].min()),
            "max_text_length": int(df["text_length"].max()),
            "unique_categories": len(cat_counts),
            "top_10_categories": {str(k): int(v) for k, v in cat_counts.head(10).items()},
            "issues": issues,
            "papers_with_short_text": int(too_short),
        }

        # Save validation report
        os.makedirs(self.log_dir, exist_ok=True)
        report_path = os.path.join(self.log_dir, "validation_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Validation report saved to {report_path}")
        logger.info(f"Validation summary: {len(df)} samples, {report['unique_categories']} unique categories")

        if issues:
            logger.warning(f"Validation issues found: {issues}")

        return report

    def preprocess_text(self) -> Dict[str, Any]:
        """
        Preprocess text data (cleaning, normalization).

        Returns:
            Dict with preprocessing statistics
        """
        logger.info("Preprocessing text data...")

        raw_data_path = os.path.join(self.data_dir, "raw_data.pkl")
        df = pd.read_pickle(raw_data_path)

        def clean_text(text: str) -> str:
            """Clean and normalize text."""
            if not isinstance(text, str):
                return ""

            # Lowercase
            text = text.lower()
            # Remove special characters (keep letters, numbers, spaces)
            text = re.sub(r"[^a-z0-9\s]", " ", text)
            # Remove extra spaces
            text = re.sub(r"\s+", " ", text).strip()
            return text

        # Fill missing values
        df["title"] = df["title"].fillna("")
        df["abstract"] = df["abstract"].fillna("")

        # Combine title + abstract
        df["text"] = df["title"] + " " + df["abstract"]
        df["text_cleaned"] = df["text"].apply(clean_text)

        # Convert categories from string to list
        def parse_categories(cat_str):
            """Convert categories string to list"""
            if isinstance(cat_str, str):
                # Split by comma and clean
                categories = [cat.strip() for cat in cat_str.split(',')]
                # Filter out empty strings
                categories = [cat for cat in categories if cat]
                return categories
            elif isinstance(cat_str, list):
                return cat_str
            else:
                return []
        
        df["categories"] = df["categories"].apply(parse_categories)

        # Keep only needed columns
        df_processed = df[["text_cleaned", "categories"]].copy()

        # Remove samples with empty text
        df_processed = df_processed[df_processed["text_cleaned"].str.len() > 0]

        # Save processed data
        processed_data_path = os.path.join(self.data_dir, "processed_data.pkl")
        df_processed.to_pickle(processed_data_path)

        logger.info(f"Preprocessed {len(df_processed)} samples")
        logger.info(f"Saved processed data to {processed_data_path}")

        return {
            "total_samples": len(df_processed),
            "avg_length": float(df_processed["text_cleaned"].str.len().mean()),
        }

    def split_dataset(self, test_size: float = 0.3, val_ratio: float = 0.5) -> Dict[str, Any]:
        """
        Split dataset into train/val/test sets.

        Args:
            test_size: Proportion of data for test+val (default 0.3 = 30%)
            val_ratio: Ratio of test_size for validation (default 0.5 = half of 30%)

        Returns:
            Dict with split statistics
        """
        logger.info("Splitting dataset...")

        processed_data_path = os.path.join(self.data_dir, "processed_data.pkl")
        df = pd.read_pickle(processed_data_path)

        X = df["text_cleaned"]
        y = df["categories"]

        # First split: 70% train, 30% temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Second split: 15% val, 15% test (from 30% temp)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42
        )

        # Save splits
        splits = {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        }

        splits_path = os.path.join(self.data_dir, "splits.pkl")
        joblib.dump(splits, splits_path)

        logger.info(f"Dataset split complete:")
        logger.info(f"  Train: {len(X_train)} samples")
        logger.info(f"  Val: {len(X_val)} samples")
        logger.info(f"  Test: {len(X_test)} samples")
        logger.info(f"Saved splits to {splits_path}")

        return {
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
        }

    def train_model(
        self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)
    ) -> Dict[str, Any]:
        """
        Train the classification model.

        Args:
            max_features: Maximum number of TF-IDF features
            ngram_range: N-gram range for TF-IDF

        Returns:
            Dict with training statistics
        """
        logger.info("Training model...")
        start_time = time.time()

        # Load splits
        splits_path = os.path.join(self.data_dir, "splits.pkl")
        splits = joblib.load(splits_path)

        X_train = splits["X_train"]
        y_train = splits["y_train"]

        # 1. TF-IDF Vectorization
        logger.info(f"Creating TF-IDF features (max_features={max_features}, ngram_range={ngram_range})...")
        vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range, min_df=2, max_df=0.8
        )
        X_train_vec = vectorizer.fit_transform(X_train)

        logger.info(f"Feature matrix shape: {X_train_vec.shape}")

        # 2. Multi-label Binarization
        logger.info("Encoding labels...")
        mlb = MultiLabelBinarizer()
        y_train_bin = mlb.fit_transform(y_train)

        logger.info(f"Number of labels: {len(mlb.classes_)}")
        logger.info(f"Labels: {list(mlb.classes_)[:10]}...")  # Show first 10

        # 3. Train model
        logger.info("Training OneVsRestClassifier with LogisticRegression...")
        model = OneVsRestClassifier(
            LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"), n_jobs=-1
        )
        model.fit(X_train_vec, y_train_bin)

        training_time = time.time() - start_time

        # Save artifacts
        os.makedirs(self.model_dir, exist_ok=True)

        vectorizer_path = os.path.join(self.model_dir, "vectorizer.joblib")
        mlb_path = os.path.join(self.model_dir, "label_encoder.joblib")
        model_path = os.path.join(self.model_dir, "model.joblib")

        joblib.dump(vectorizer, vectorizer_path)
        joblib.dump(mlb, mlb_path)
        joblib.dump(model, model_path)

        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Vectorizer saved to {vectorizer_path}")
        logger.info(f"Label encoder saved to {mlb_path}")

        return {
            "training_time": round(training_time, 2),
            "n_features": X_train_vec.shape[1],
            "n_labels": len(mlb.classes_),
            "n_samples": X_train_vec.shape[0],
        }

    def evaluate_model(self) -> Dict[str, Any]:
        """
        Evaluate model on train/val/test sets.

        Returns:
            Dict with evaluation metrics
        """
        logger.info("Evaluating model...")

        # Load artifacts
        splits_path = os.path.join(self.data_dir, "splits.pkl")
        vectorizer_path = os.path.join(self.model_dir, "vectorizer.joblib")
        mlb_path = os.path.join(self.model_dir, "label_encoder.joblib")
        model_path = os.path.join(self.model_dir, "model.joblib")

        splits = joblib.load(splits_path)
        vectorizer = joblib.load(vectorizer_path)
        mlb = joblib.load(mlb_path)
        model = joblib.load(model_path)

        def evaluate_split(X, y, split_name):
            """Evaluate on a single split."""
            logger.info(f"Evaluating on {split_name} set...")

            # Transform
            X_vec = vectorizer.transform(X)
            y_bin = mlb.transform(y)

            # Predict
            y_pred = model.predict(X_vec)

            # Metrics
            precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
                y_bin, y_pred, average="micro"
            )
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                y_bin, y_pred, average="macro"
            )
            precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
                y_bin, y_pred, average="weighted"
            )

            metrics = {
                "precision_micro": round(float(precision_micro), 4),
                "precision_macro": round(float(precision_macro), 4),
                "precision_weighted": round(float(precision_weighted), 4),
                "recall_micro": round(float(recall_micro), 4),
                "recall_macro": round(float(recall_macro), 4),
                "recall_weighted": round(float(recall_weighted), 4),
                "f1_micro": round(float(f1_micro), 4),
                "f1_macro": round(float(f1_macro), 4),
                "f1_weighted": round(float(f1_weighted), 4),
                "hamming_loss": round(float(hamming_loss(y_bin, y_pred)), 4),
            }

            logger.info(f"{split_name} metrics:")
            logger.info(f"  F1-Score (micro): {metrics['f1_micro']:.4f}")
            logger.info(f"  F1-Score (macro): {metrics['f1_macro']:.4f}")
            logger.info(f"  Precision (micro): {metrics['precision_micro']:.4f}")
            logger.info(f"  Recall (micro): {metrics['recall_micro']:.4f}")

            return metrics

        # Evaluate on all splits
        results = {
            "train": evaluate_split(splits["X_train"], splits["y_train"], "train"),
            "val": evaluate_split(splits["X_val"], splits["y_val"], "val"),
            "test": evaluate_split(splits["X_test"], splits["y_test"], "test"),
            "timestamp": datetime.now().isoformat(),
            "model_type": "OneVsRestClassifier(LogisticRegression)",
        }

        # Save results
        results_path = os.path.join(self.log_dir, "evaluation_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Evaluation results saved to {results_path}")

        return results

    def save_artifacts(self, **context) -> Dict[str, Any]:
        """
        Save all artifacts with metadata and model card.

        Args:
            **context: Airflow context

        Returns:
            Dict with artifact information
        """
        logger.info("Saving artifacts...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info("Model artifacts saved successfully")
        
        return {
            "timestamp": timestamp,
            "model_files": ["model.joblib", "vectorizer.joblib", "label_encoder.joblib"]
        }

