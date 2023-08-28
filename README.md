
# Sports Game Outcome Prediction

This project demonstrates a simple sports game outcome prediction using machine learning. It predicts the outcome of sports games (e.g., basketball) based on historical game data.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [Dependencies](#dependencies)


## Introduction

This project uses historical sports game data to predict the outcome of games. It leverages a machine learning model trained on features such as team statistics, player performance, and game conditions.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Aayush518/sports-prediction-project.git
   cd sports-prediction-project
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Place your sports game dataset (e.g., `sports_data.csv`) in the `data/` directory.

2. Run the user interface:
   ```
   python ui/ui.py
   ```

3. Use the UI to browse and select your dataset. The UI will display the accuracy of the game outcome predictions.

## Folder Structure

The project follows the following folder structure:

```
sports-prediction-project/
│
├── data/
│   ├── sports_data.csv  # Your dataset
│
├── model/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│
├── ui/
│   ├── __init__.py
│   ├── ui.py
│
├── requirements.txt
│
└── README.md
```

## Dependencies

- scikit-learn==0.24.2
- pandas==1.3.3
- numpy==1.21.2
- tk==0.1.0
