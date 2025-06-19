# install_libraries.py

import subprocess
import sys

# List of required packages with specific versions
required_packages = [
    "tensorflow==2.19.0",
    "scikit-learn==1.6.1",
    "numpy==2.1.3",
    "pandas==2.2.3",
    "matplotlib==3.10.1",
    "joblib==1.4.2"
]

# Function to install packages using pip
def install_packages():
    for package in required_packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    install_packages()

print("\n Packages installed successfully.")
