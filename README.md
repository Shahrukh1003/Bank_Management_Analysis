# SCT_DS_03
Build a decision tree classifier to predict whether a customer will purchase a product or service based on their demographic and behavioral data. Use a dataset such as the Bank Marketing dataset from the UCI Machine Learning Repository.
# Decision Tree Classifier for Bank Marketing Dataset

## Task Description
This project involves building a **Decision Tree Classifier** to predict whether a customer will subscribe to a term deposit based on their demographic and behavioral data. The dataset used is the **Bank Marketing Dataset** from the UCI Machine Learning Repository.

## Dataset Overview
The dataset contains information related to direct marketing campaigns (phone calls) conducted by a Portuguese bank. The objective is to predict whether the client will subscribe to a term deposit (binary target variable `y`).

- **Demographic Attributes**: e.g., `age`, `job`, `marital status`, `education`.
- **Behavioral Attributes**: e.g., `duration`, `campaign`, `previous`, `poutcome` (outcome of a previous campaign).
- **Target Variable (y)**: Binary variable indicating if the client subscribed (`yes`/`no`).

| Variable Name | Role    | Type        | Description |
|---------------|---------|-------------|-------------|
| age           | Feature | Integer     | Age of the customer |
| job           | Feature | Categorical | Type of job (e.g., admin, blue-collar, student) |
| marital       | Feature | Categorical | Marital status (e.g., married, single) |
| education     | Feature | Categorical | Level of education (e.g., university degree, high school) |
| ...           | ...     | ...         | ... |
| y             | Target  | Binary      | Whether the client subscribed to a term deposit |

## Libraries Used
The following Python libraries are used in this notebook:
- `numpy` and `pandas` for data manipulation.
- `matplotlib` and `seaborn` for data visualization.
- `sklearn` for machine learning model building and evaluation.
- `%matplotlib inline` for displaying plots inline.

## Data Preprocessing
1. **Loading the Dataset**: The dataset is loaded into a Pandas DataFrame for exploration and preprocessing.
2. **Exploratory Data Analysis (EDA)**: We visualize the distribution of features and the relationship between features and the target variable using plots like histograms and pair plots.
3. **Handling Categorical Variables**: Categorical features like `job`, `marital`, and `education` are transformed into numerical form using **one-hot encoding**.
4. **Feature Scaling**: Although decision trees are not sensitive to feature scaling, we perform standardization if necessary.

## Building the Decision Tree Classifier
The decision tree model is built using `DecisionTreeClassifier` from the `sklearn.tree` module. Key concepts include:
- **Gini Impurity**: The measure of how often a randomly chosen element would be incorrectly labeled.
- **Information Gain**: Used to determine the best feature to split the dataset.
- **Pruning**: Parameters such as `max_depth` and `min_samples_split` are tuned to prevent overfitting.

## Model Training and Testing
1. **Train-Test Split**: The data is split into training and testing sets (e.g., 70% for training, 30% for testing).
2. **Training the Model**: The decision tree is trained on the training data, learning patterns from the feature set.
3. **Evaluating the Model**:
   - **Accuracy**: The proportion of correctly classified instances.
   - **Confusion Matrix**: A matrix representing true positives, false positives, true negatives, and false negatives.
   - **Precision, Recall, F1-Score**: Additional metrics to assess model performance.

## Model Interpretation
The decision tree structure is visualized to interpret the decision-making process. Visualization tools such as `plot_tree` or `graphviz` are used to display the tree.

## Conclusion
The project concludes with an analysis of the model's performance and the significance of different features. Important insights include:
- Which features played the biggest role in predicting term deposit subscription.
- The trade-offs between accuracy and model interpretability.

