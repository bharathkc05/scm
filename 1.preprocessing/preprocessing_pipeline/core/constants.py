"""Project constants that do not encode dataset-specific business rules."""

SUPPORTED_TRANSFORMATIONS = {"none", "log1p", "yeo_johnson"}
SUPPORTED_SCALERS = {"standard", "robust"}

DATASET_FILE_NAMES = {
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv",
}

METADATA_FILE_NAMES = {
    "removed_features": "removed_features.csv",
    "retained_features": "retained_features.csv",
    "split_summary": "split_summary.csv",
    "summary": "preprocessing_summary.json",
}

PARAMETER_FILE_NAMES = {
    "winsorization": "winsorization.json",
    "transformers": "transformation_parameters.pkl",
    "scaler": "scaler.pkl",
}

