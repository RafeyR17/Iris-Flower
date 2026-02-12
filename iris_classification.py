import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def main():
    # 1. Load the dataset
    print("--- 1. Loading Dataset ---")
    iris = load_iris()
    X_raw = iris.data
    y_raw = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    # 2. Exploratory Data Analysis (EDA)
    print("\n--- 2. Performing EDA ---")
    df = pd.DataFrame(X_raw, columns=feature_names)
    df['target'] = y_raw
    df['species'] = df['target'].map({i: name for i, name in enumerate(target_names)})

    # Summary Statistics
    print("\nSummary Statistics:")
    print(df.describe())

    # Correlation Matrix
    print("\nCorrelation Matrix:")
    corr_matrix = df.drop(columns=['species']).corr()
    print(corr_matrix)

    # Visualizations
    # a. Pairplot
    sns.set_theme(style="whitegrid")
    pairplot = sns.pairplot(df.drop(columns=['target']), hue='species', markers=["o", "s", "D"])
    pairplot.fig.suptitle("Iris Feature Pairplot", y=1.02)
    pairplot.savefig('eda_pairplot.png')
    print("Saved 'eda_pairplot.png'")

    # b. Boxplots
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(feature_names):
        plt.subplot(2, 2, i + 1)
        sns.boxplot(x='species', y=col, data=df)
        plt.title(f'{col} by Species')
    plt.tight_layout()
    plt.savefig('eda_boxplots.png')
    print("Saved 'eda_boxplots.png'")

    # c. Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig('eda_heatmap.png')
    print("Saved 'eda_heatmap.png'")

    """
    EDA Key Insights:
    - Petal length and petal width are highly correlated (0.96), and both are strongly correlated with the target class.
    - Setosa is easily distinguishable from the other two species based on petal features.
    - Versicolor and Virginica have some overlap, especially in sepal measurements, but remain fairly distinct in petal space.
    """

    # 3. Preprocessing
    print("\n--- 3. Preprocessing Data ---")
    X = df.drop(columns=['target', 'species'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Scaling (recommended for SVM, KNN, and Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4 & 5. Experiment with Algorithms
    print("\n--- 4 & 5. Training and Evaluating Models ---")
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200, random_state=RANDOM_STATE),
        "KNN (n=5)": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel='linear', random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    }

    results_data = []

    for name, model in models.items():
        # Train
        # Use scaled data for distance/gradient based models
        if name in ["Logistic Regression", "KNN (n=5)", "SVM"]:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\n{'='*40}")
        print(f"Algorithm: {name}")
        print(f"{'='*40}")
        print(f"Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        print("Confusion Matrix:")
        print(cm)
        
        results_data.append({
            "Model": name,
            "Accuracy": acc,
            "F1-Score (Macro)": f1
        })

    # 6. Comparison
    print("\n--- 6. Model Comparison ---")
    comparison_df = pd.DataFrame(results_data)
    print(comparison_df.to_string(index=False))

    # Visualization of comparison
    plt.figure(figsize=(10, 6))
    melted_results = comparison_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    sns.barplot(x="Score", y="Model", hue="Metric", data=melted_results)
    plt.title("Model Performance Comparison")
    plt.xlim(0.8, 1.05)  # Zoom in to see differences
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    print("Saved 'model_comparison.png'")

    # Identify best model
    best_model_name = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']
    print(f"\nBest Performing Model: {best_model_name}")

    # Save the best model (SVM) and scaler for the web app
    import joblib
    print("\n--- 7. Saving Model and Scaler ---")
    # We specifically want to save the SVM model and the scaler used for it
    svm_model = models["SVM"]
    joblib.dump(svm_model, 'iris_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    print("Saved 'iris_model.joblib' and 'scaler.joblib'")

    """
    Comment: Most models perform exceptionally well on the Iris dataset as it is linearly separable 
    for Setosa and nearly so for the others. Typically, SVM or Random Forest yield the most robust results.
    """

if __name__ == "__main__":
    main()
