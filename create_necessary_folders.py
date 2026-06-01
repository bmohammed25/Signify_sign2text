import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

base = config.BASE_DIR

folders = [
    # Data layers
    "data/raw/kaggle_asl",
    "data/raw/asl_citizen",
    "data/raw/wlasl",
    "data/processed/landmarks",
    "data/processed/landmarks_augmented",
    "data/processed/sequences",
    "data/augmented",

    # Model layers
    "models/saved",
    "models/config",
    "models/architecture",

    # Training pipeline
    "training",

    # Agent layers
    "agents",

    # API layer
    "api",

    # Tests
    "tests",

    # Notebooks
    "notebooks",

    # Logs
    "logs",
]

# Create folders with .gitkeep
for folder in folders:
    full_path = os.path.join(base, folder)
    os.makedirs(full_path, exist_ok=True)
    gitkeep = os.path.join(full_path, ".gitkeep")
    if not os.path.exists(gitkeep):
        with open(gitkeep, "w") as f:
            pass

# Create empty starter files if they don't exist
starter_files = [
    "main.py",
    "agents/__init__.py",
    "api/__init__.py",
    "training/__init__.py",
    "models/__init__.py",
    "models/architecture/__init__.py",
    "tests/__init__.py",
]

for file in starter_files:
    full_path = os.path.join(base, file)
    if not os.path.exists(full_path):
        with open(full_path, "w") as f:
            pass

print("✅ Signify project structure created successfully")
print(f"   Base directory: {base}")
print(f"\n   Folders created:")
for folder in folders:
    print(f"     📁 {folder}")
print(f"\n   Starter files created:")
for file in starter_files:
    print(f"     📄 {file}")
print("\n🚀 Ready to start building Signify")