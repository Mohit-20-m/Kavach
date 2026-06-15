from setuptools import setup, find_packages

setup(
    name="kavach-cli",
    version="2.0.0",
    description="KAVACH — Intelligent Supply Chain Security CLI (Standalone + Server modes)",
    packages=find_packages(),
    py_modules=["kavach_cli", "kavach_standalone"],
    install_requires=[
        "typer[all]>=0.9.0",
        "httpx>=0.26.0",
        "rich>=13.0.0",
        "scikit-learn>=1.4.0",
        "xgboost>=2.0.0",
        "numpy>=1.26.0",
        "joblib>=1.3.0",
        "sentence-transformers>=2.3.0",
        "torch>=2.2.0",
    ],
    entry_points={
        "console_scripts": [
            "kavach=kavach_cli:app",
            "kavach-standalone=kavach_standalone:app",
        ],
    },
    python_requires=">=3.10",
)