import os
from pathlib import Path

def create_ml_template():
    root = Path(os.getcwd())
    src_dir = root / "src" 
    
    print(f"Initializing production ML structure at: {root}\n")

    # 1. Define all directories to build
    dirs_to_create = [
        root / "configs" / "model",
        root / "data" / "raw",
        root / "data" / "processed",
        root / "notebooks",
        src_dir,
        root / "tests"
    ]
    
    for folder in dirs_to_create:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {folder.relative_to(root.parent)}")

    # 2. Define empty files or simple stubs
    files_to_touch = [
        root / ".gitignore",
        root / "README.md",
        root / "pyproject.toml",
        root / "requirements.txt",
        root / "configs" / "config.yaml",
        root / "configs" / "model" / "baseline.yaml",
        root / "data" / "raw" / ".gitkeep",
        root / "data" / "processed" / ".gitkeep",
        root / "notebooks" / "01_exploration.ipynb",
        src_dir / "__init__.py",
        src_dir / "data_loader.py",
        src_dir / "modeling.py",
        src_dir / "utils.py",
        src_dir / "pipeline.py",
        root / "tests" / "test_pipeline.py"
    ]

    # 3. Create all files empty
    for path in files_to_touch:
        path.touch()
        print(f"Generated file: {path.relative_to(root.parent)}")

    print("\nClean template structure generated successfully!")

if __name__ == "__main__":
    create_ml_template()