# sign_interpreter

## About
AI app for recognizing and intrepreting sign language

## Development
It's recommended to continue development in a virtual environment to minimize package conflicts.
This guide will cover Anaconda specifically, but you can choose to use venv if you prefer.
### Step 0: Prerequisties:
- CUDA installed to speed up TensorFlow (only if you have an NVIDIA GPU)
### Step 1: Virtual environment
Clone this repository to your directory of choice then create a Conda environment in said directory.
```
git clone https://github.com/Andrew1013-development/sign_interpreter.git
conda create -n <env_name> python=3.8
```
### Step 2: Installation
Activate your newly created Conda virtual environment and install all dependencies.
```
conda activate <env_name>
pip install -r requirements.txt
```
After everything is done, run the MainGUI.py file. 
### Directories
- `data_test`: Testing videos
- `weights`: Embeddings for the AI model