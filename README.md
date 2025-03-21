# Data Visualizer Tool

## Overview

Data Visualizer Tool is a Python application designed to assist users in performing exploratory data analysis (EDA) and preprocessing for supervised machine learning tasks. The application provides a user-friendly interface built with Tkinter, allowing users to easily load datasets, visualize data, and apply various transformations.

This project was created during my early years at university, and while it serves its purpose, I plan to redo it using more modern frameworks like Flask, Django, or FastAPI if I have the time, as Tkinter is considered an outdated tool for building GUIs.

## Features

- **Load CSV Files**: Users can select and load CSV files into the application for analysis.
- **Exploratory Data Analysis (EDA)**: The application provides detailed EDA capabilities, including:

  - Summary statistics of the dataset.
  - Visualization of missing values.
  - Detailed analysis of individual columns, including histograms and common values.

  ![Detailed EDA](Results%20Pictures/Detailed%20EDA.png)

- **Data Transformation**: Users can perform various data transformations, including:

  - Handling missing values (mean, median, or removal).
  - Encoding categorical columns using label encoding.
  - Renaming and removing columns.
  - Removing duplicates from the dataset.

- **Data Visualization**: The application includes several visualization options:

  - Histograms for numerical data.
  - Stacked bar charts for categorical data.
  - Scatter plots to visualize interactions between two columns.

  ![Histogram EDA](Results%20Pictures/Histogram%20EDA.png)

- **Correlation Analysis**: Users can visualize correlations between different columns in the dataset using heatmaps.

- **Interaction Analysis**: Users can explore interactions between different features in the dataset through scatter plots.

- **User-Friendly Interface**: The application is designed with a simple and intuitive interface, making it accessible for users with varying levels of expertise in data science.

## Getting Started

### Prerequisites

- Python 3.x
- Required libraries:
  - NumPy
  - Pandas
  - Matplotlib
  - Seaborn
  - Tkinter
  - scikit-learn
  - Pillow

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Supervised_ML_Helper.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Supervised_ML_Helper
   ```
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. You can run the application using the provided Python file for the GUI:
   ```bash
   python Supervised_ML_Classifier_python.py
   ```
2. Alternatively, there is a Jupyter Notebook available for additional analysis and exploration of the dataset.
3. Follow the on-screen instructions to load your dataset and explore the various features of the application.

## Screenshots

Here are some screenshots of the application in action:

- **Histogram EDA**:
  ![Histogram EDA](Results%20Pictures/Histogram%20EDA.png)

- **Detailed EDA**:
  ![Detailed EDA](Results%20Pictures/Detailed%20EDA.png)

- **Correlation Heatmap**:
  ![Correlation Heatmap](Results%20Pictures/Correlation.png)
