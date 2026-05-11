# CR3BP Orbital Stability Classification using Machine Learning
# Interactive demo of the application here
https://cr3bp-ml-orbital-classifier.streamlit.app/ 

# Overview
This project applies machine learning to classify orbital behavior in the **Circular Restricted Three-Body Problem (CR3BP)**. Given initial conditions, the model predicts whether a small object will:

- Remain stable (bounded)
- Escape the system
- Collide with one of the primary bodies

The dataset is generated using numerical simulation of the governing differential equations.

# Problem Description
The Circular Restricted Three-Body Problem models the motion of a small object under the gravitational influence of two larger bodies.

The goal is to predict orbital behavior from initial conditions:
- Position: (x₀, y₀)
- Velocity: (vₓ₀, vᵧ₀)
- Mass ratio μ


# Dataset Generation
The dataset is created by:
1. Randomly sampling initial conditions
2. Numerically integrating the equations of motion using SciPy
3. Labeling each trajectory based on its behavior

# Labeling Rules
- Escape (1): Distance exceeds a threshold radius
- Collision (2): Object gets too close to either primary body
- Stable (0): Remains bounded during simulation

Note: Stability is defined over a finite time horizon, not guaranteed indefinitely.

#Features
Each data sample includes:
- Mass ratio (μ)
- Initial position (x₀, y₀)
- Initial velocity (vₓ₀, vᵧ₀)
- Jacobi constant
- Distance to primary bodies

# Machine Learning Models
The following models are trained and compared:

- Logistic Regression
- Random Forest
- Multi-Layer Perceptron (Neural Network)

Evaluation metrics include:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

# Project Structure
cr3bp_ml_project/
├── data/ # Generated dataset
│ └── orbits.csv
├── models/ # Trained ML models
├── results/ # Evaluation outputs (optional)
├── src/
│ ├── cr3bp.py # Physics simulation (CR3BP equations)
│ ├── generate_dataset.py # Dataset generation script
│ ├── train_models.py # Model training and evaluation
│ └── app.py # Streamlit interactive demo
├── requirements.txt
└── README.md

# Running the Project
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the dataset
python3 src/generate_dataset.py --n_samples 3000

# 3. Train Models
python3 src/train_models.py

# 4. Run Streamlit interactive app 
streamlit run src/app.py

# Interactive app features 
An interactive web app that lets users define a three-body system, adjust initial conditions, and instantly see machine learning predictions with real-time trajectory visualization.
