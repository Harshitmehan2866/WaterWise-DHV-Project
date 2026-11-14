# WaterWise-DHV-Project
Data-Driven Urban Water Network Analysis and Fault-Proximity Classification
# WaterWise – Data-Driven Urban Water Network Analysis

## Project Overview
WaterWise is a Data Handling and Visualization (DHV) project that analyses urban water distribution network data using data analysis, visualization, and machine learning techniques.

The project focuses on identifying patterns in historical sensor readings and classifying records based on their proximity to predefined fault events.

## Objectives
- Analyse historical urban water network sensor data.
- Perform data cleaning and preprocessing.
- Explore sensor patterns using data visualization.
- Identify relationships between different sensor features.
- Apply machine learning algorithms for fault-proximity classification.
- Present insights through an interactive Streamlit dashboard.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Jupyter Notebook
- Streamlit

## Machine Learning Algorithms
1. **Logistic Regression** – Used as a classification model to identify fault-proximity patterns.
2. **Random Forest** – Used to classify records using multiple decision trees.

## Project Workflow
1. Dataset collection and loading.
2. Data inspection and preprocessing.
3. Exploratory Data Analysis (EDA).
4. Feature engineering.
5. Model training and testing.
6. Model evaluation.
7. Interactive dashboard development.

## Dataset Information
The project uses an urban water distribution network dataset containing 8,737 records of historical sensor measurements.

The target variable is `fault_d7`:
- **1:** Record falls within the predefined seven-day fault-proximity window.
- **0:** Record falls outside the defined window.

**Note:** The classification identifies patterns associated with historical fault-proximity labels. It does not independently confirm an actual physical water leakage or fault.

## Dashboard
The Streamlit dashboard presents:
- Dataset overview and statistics.
- Sensor data trends.
- Exploratory data visualizations.
- Correlation heatmap.
- Fault-proximity classification insights.

## How to Run the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/YOUR-USERNAME/WaterWise-DHV-Project.git
   ```

2. Open the project folder:

   ```bash
   cd WaterWise-DHV-Project
   ```

3. Install the required libraries:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn streamlit
   ```

4. Run the Streamlit application:

   ```bash
   streamlit run app.py
   ```

   Replace `app.py` with the actual Streamlit application filename if it is different.

## Project Details
- **Project Name:** WaterWise
- **Subject:** Data Handling and Visualization (DHV)
- **Semester:** B.Tech – Semester 5

## Future Improvements
- Include real-time sensor data.
- Improve model performance and class-wise evaluation.
- Explore advanced anomaly detection techniques.
- Add more interactive monitoring features.

---

**WaterWise – Making Urban Water Network Analysis Smarter Through Data.**
