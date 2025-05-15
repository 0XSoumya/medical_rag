import os

# Project structure
folders = [
    "src/data",
    "src/retrieval",
    "src/llm",
    "src/utils",
    "tests",
    "scripts",
    "docs"
]

files = [
    ".env",
    ".gitignore",
    "README.md",
    "requirements.txt",
    "src/app.py",
    "src/config.py",
    "src/data/loader.py",
    "src/data/store.py",
    "src/retrieval/sparse.py",
    "src/retrieval/dense.py",
    "src/retrieval/fusion.py",
    "src/llm/client.py",
    "src/llm/prompts.py",
    "src/pipeline.py",
    "src/utils/logging.py",
    "tests/test_data.py",
    "tests/test_retrieval.py",
    "scripts/bootstrap_kb.py",
    "docs/architecture.md"
]

gitignore_content = """
# Python virtual env
venv/

# Environment variables
.env

# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Test cache
.pytest_cache/

# VSCode settings
.vscode/
"""

def create_folders():
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}")

def create_files():
    for file in files:
        # Create file if doesn't exist
        if not os.path.exists(file):
            with open(file, 'w') as f:
                # Add .gitignore content only for that file
                if file == ".gitignore":
                    f.write(gitignore_content.strip() + "\n")
            print(f"Created file: {file}")
        else:
            print(f"File already exists: {file}")

def main():
    print("Setting up medical-rag project structure...")
    create_folders()
    create_files()
    print("Setup complete!")
    print("Next steps:")
    print("  - Create and activate a virtual environment (python -m venv venv)")
    print("  - Install requirements")
    print("  - Open this folder in your editor")

if __name__ == "__main__":
    main()
