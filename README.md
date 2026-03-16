# Velocity Prediction using Machine Learning

## Overview

This project predicts the **velocity of a projectile in a gas gun system** using **machine learning models**.
A graphical user interface (GUI) is provided to input projectile parameters and visualize prediction results.

The system uses a trained **Decision Tree model** to estimate projectile velocity based on physical parameters such as projectile type, shape, and other dataset features.

The application also generates **plots and visualizations** for analysis.

---

## Features

* Machine Learning based **velocity prediction**
* **GUI interface (Tkinter)** for user interaction
* Visualization of results using **Matplotlib**
* Dataset preprocessing and model utilities
* Modular project structure

---

## Project Structure

```
velocity_prediction/
│
├── main.py                  # Main GUI application
├── models/
│   └── model_utils.py        # ML model and prediction functions
│
├── preprocessing/            # Data preprocessing scripts
│
├── data/
│   └── fake_gas_gun_dataset.csv
│
├── plots/                    # Generated plots
│
├── ui/                       # UI components
│
├── graphs.png                # Sample graph output
├── ballistics_report.pdf     # Project report
│
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Prerequisites

Before running the project, make sure you have:

* Python **3.8 or higher**
* pip (Python package manager)

Install required libraries:

```
pip install -r requirements.txt
```

Required libraries include:

* pandas
* numpy
* matplotlib
* scikit-learn
* tkinter

---

## How to Run the Project

Step 1 — Clone the repository

```
git clone https://github.com/ansh-vashisht2003/velocity_prediction.git
```

Step 2 — Navigate to the project folder

```
cd velocity_prediction
```

Step 3 — Install dependencies

```
pip install -r requirements.txt
```

Step 4 — Run the application

```
python main.py
```

---

## Input Parameters

The GUI allows users to select or input parameters such as:

* Projectile Type
* Shape
* Other dataset features

The model processes these inputs and predicts the **projectile velocity**.

---

## Output

The system displays:

* Predicted projectile velocity
* Graphical visualization of results
* Generated plots saved in the `plots/` folder

---

## Technologies Used

* Python
* Scikit-learn
* Pandas
* Matplotlib
* Tkinter (GUI)

---

## Future Improvements

* Support for multiple ML models
* Improved dataset and real experimental data
* Enhanced GUI design
* Export prediction results to reports


