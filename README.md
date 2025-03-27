# ML-7-Drug-Prediction-DecisionTree
This project aims to predict the type of drug prescribed based on patient attributes using a Decision Tree Classifier. The dataset contains various features, including age, sex, blood pressure levels, cholesterol levels, and sodium-potassium ratio.

# Drug Prediction using Decision Trees

## Overview

![image](https://github.com/user-attachments/assets/84bc9ead-b64a-42af-bbff-cf561ced9811)

## Dataset
The dataset used is **`drug200.csv`**, which consists of:
- `Age`: Patient's age.
- `Sex`: Male or Female.
- `BP`: Blood Pressure levels (High, Normal, Low).
- `Cholesterol`: Cholesterol levels (High, Normal).
- `Na_to_K`: Sodium-to-Potassium ratio in blood.
- `Drug`: Target variable (Type of drug prescribed).

## Steps to Run the Project

### 1. Install Dependencies
Ensure Python is installed along with the required libraries.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 2. Load the Data
```python
import pandas as pd

# Load dataset
drug_data = pd.read_csv('drug200.csv')
drug_data.head()
```

## Exploratory Data Analysis (EDA)
### Data Summary
- The dataset consists of **categorical** and **numerical** variables.
- It provides insights into how different patient attributes influence drug prescriptions.

### Data Distribution
- **Categorical Features**: `Sex`, `BP`, and `Cholesterol` were visualized using count plots.
- **Numerical Features**: `Age` and `Na_to_K` were analyzed using histograms to understand their distributions.

### Correlation Analysis
- **Sodium-to-Potassium ratio** plays a significant role in drug classification.
- Some drugs are prescribed specifically based on blood pressure and cholesterol levels.

## Data Preprocessing
- **Label Encoding**: Categorical features were converted into numerical format using `LabelEncoder`.
- **Feature Selection**: Selected `Age`, `Sex`, `BP`, `Cholesterol`, and `Na_to_K` as independent variables, while `Drug` was the target variable.

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(drug_data['Drug'])
```

## Model Training
### Algorithm Used: Decision Tree Classifier
- A **Decision Tree Classifier** was selected due to its interpretability and efficiency in classification problems.
- The model learns patterns from the dataset and predicts drug prescriptions.

```python
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()

# Train the model
dtree.fit(X_train, y_train)
```

### Decision Tree Visualization
The trained tree structure helps understand how features contribute to decision-making.

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plot_tree(dtree, filled=True, feature_names=X.columns, class_names=le.classes_, fontsize=8)
plt.show()
```

## Insights
- **Blood pressure and cholesterol levels are key decision factors**.
- **The sodium-to-potassium ratio highly influences drug prescriptions**.
- **Decision Trees provide clear rules for classification**, making them useful for medical prediction tasks.

## Future Improvements
- Hyperparameter tuning to optimize tree depth.
- Compare with other models like Random Forest or Logistic Regression for better accuracy.
- Deploy the model as a web app for real-time predictions.

## Repository Structure
```
|-- main.ipynb  # Jupyter Notebook with analysis
|-- drug200.csv # Dataset file
|-- README.md   # Project documentation
```

## Contributing
Feel free to open issues or submit pull requests to improve the project.

## License
This project is licensed under the MIT License.
