from pathlib import Path

# ===========================
# Configuration
# ===========================

PROJECT_NAME = "churn_prediction"

DIRECTORIES = [
    "data/raw",
    "data/interim",
    "data/processed",
    "notebooks",
    "src",
    "src/data",
    "src/features",
    "src/models",
    "src/visualization",
    "src/utils",
    "configs",
    "models",
    "reports/figures",
    "tests",
]

FILES = {
    ".gitignore": "",
    "requirements.txt": "",
    "main.py": "",

    "configs/config.yaml": "",

    "src/__init__.py": "",
    "src/data/__init__.py": "",
    "src/data/load_data.py": "",
    "src/data/preprocess.py": "",

    "src/features/__init__.py": "",
    "src/features/feature_engineering.py": "",

    "src/models/__init__.py": "",
    "src/models/train.py": "",
    "src/models/evaluate.py": "",
    "src/models/predict.py": "",

    "src/visualization/__init__.py": "",
    "src/visualization/plots.py": "",

    "src/utils/__init__.py": "",
    "src/utils/helpers.py": "",

    "tests/__init__.py": "",
    "tests/test_data.py": "",
    "tests/test_models.py": "",
}


def create_project(project_name: str):
    root = Path(project_name)

    for directory in DIRECTORIES:
        (root / directory).mkdir(parents=True, exist_ok=True)

    for file_path, content in FILES.items():
        path = root / file_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    print(f"✅ Project '{project_name}' created successfully!")


if __name__ == "__main__":
    create_project(PROJECT_NAME)