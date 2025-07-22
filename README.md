# Laptop Price Prediction

This project aims to develop a robust machine learning model capable of predicting laptop prices based on their specifications and features. The insights gained can be incredibly valuable for both consumers looking to buy and sellers wanting to price their products competitively in the dynamic electronics market.

The entire process, from gathering raw data to deploying an interactive web application, is covered. You'll see how to clean and transform data, engineer impactful features, train various machine learning models, and finally, present the predictions in a user-friendly way.

---

## Features

This project incorporates several key stages to deliver a comprehensive solution:

* **Data Collection & Cleaning:** Initial acquisition of laptop data, followed by meticulous handling of missing values, duplicate entries, and inconsistencies to ensure data quality.
* **Exploratory Data Analysis (EDA):** A deep dive into the dataset using statistical methods and visualizations to uncover patterns, understand feature distributions, and identify relationships that could impact price.
* **Feature Engineering:** Crafting new, more informative features from existing ones (e.g., extracting CPU brand from full CPU name, or breaking down storage into type and capacity) to boost model performance.
* **Model Training:** Experimenting with a variety of machine learning **regression algorithms** like Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, and XGBoost to find the optimal predictive model.
* **Model Evaluation:** Rigorously assessing the trained models using standard metrics such as **R-squared**, **Mean Absolute Error (MAE)**, and **Mean Squared Error (MSE)** to ensure accuracy and reliability.
* **Model Deployment:** Building an intuitive web application using **Streamlit** that allows users to input laptop specifications and receive instant price predictions.

---

## Technologies Used

The project is built upon a solid foundation of popular Python libraries and tools:

* **Python:** The primary programming language for all scripts and notebooks.
* **Pandas:** Essential for efficient data manipulation and analysis.
* **NumPy:** Provides robust support for numerical operations.
* **Scikit-learn:** The go-to library for machine learning models, preprocessing techniques, and evaluation metrics.
* **Matplotlib / Seaborn:** Used extensively for creating insightful data visualizations.
* **Jupyter Notebook:** An interactive environment used for exploratory data analysis and model development.
* **Streamlit:** Facilitates the rapid creation of interactive web applications for model deployment.
* **Pickle:** Utilized for saving and loading the trained machine learning model and any associated preprocessors.

---

## Dataset

The dataset central to this project typically comprises detailed information about various laptops. Key features often include:

* **Company:** The brand of the laptop (e.g., HP, Dell, Lenovo, Apple).
* **TypeName:** The specific category of the laptop (e.g., Notebook, Gaming, Ultrabook).
* **Inches:** The screen size, measured in inches.
* **ScreenResolution:** Details about the display's resolution and type (e.g., Full HD, Retina, Touchscreen).
* **Cpu:** Specifications of the processor (e.g., Intel Core i5, AMD Ryzen 7).
* **Ram:** The amount of RAM, typically in GB.
* **Memory:** Information on storage type and capacity (e.g., 256GB SSD, 1TB HDD).
* **Gpu:** Details of the graphics card (e.g., Intel Iris Xe, NVIDIA GeForce RTX).
* **OpSys:** The installed operating system (e.g., Windows 10, macOS, Linux).
* **Weight:** The weight of the laptop, usually in kg.
* **Price:** The target variable – the actual price of the laptop.

*(Note: The specific features in your dataset might vary. If you're using a particular dataset, you may want to update this section with its exact attributes or a link to its source.)*

---

## Installation

To get this project up and running on your local machine, follow these simple steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/laptop-price-prediction.git](https://github.com/your-username/laptop-price-prediction.git)
    cd laptop-price-prediction
    ```
    *(Remember to replace `your-username` with your actual GitHub username or the repository owner's username.)*

2.  **Create a virtual environment (highly recommended):**
    This helps manage project dependencies without conflicts.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    All necessary Python libraries are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

You can interact with this project in two main ways: through the Jupyter notebooks for development and analysis, or by running the web application for live predictions.

### 1. Data Analysis and Model Training

Dive into the `notebooks` directory to understand the data deeply, preprocess it, and train the machine learning models.

```bash
jupyter notebook
```
---
### 2. Running the web application
```bash
streamlit run app.py
```
