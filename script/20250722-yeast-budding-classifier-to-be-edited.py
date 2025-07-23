"""
title: yeast cell budding classifier
author: Claude generated script. needs to be edited before using
date: 2025-07-22
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def load_and_explore_data(csv_file):
    """Load CSV and perform initial exploration"""
    print("=" * 50)
    print("DATA LOADING AND EXPLORATION")
    print("=" * 50)
    
    # Load data
    df = pd.read_csv(csv_file)
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumn names: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Check for missing values
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    # Basic statistics
    print(f"\nBasic statistics:")
    print(df.describe())
    
    # Check target variable distribution
    target_col = 'budded'  # Adjust this to match your CSV column name
    if target_col in df.columns:
        print(f"\nTarget variable distribution:")
        print(df[target_col].value_counts())
        print(f"Class balance: {df[target_col].value_counts(normalize=True)}")
    
    return df

def visualize_data(df, target_col='budded'):
    """Create visualizations to understand the data"""
    print("\n" + "=" * 50)
    print("DATA VISUALIZATION")
    print("=" * 50)
    
    # Identify feature columns (assuming they're numeric and not the target)
    feature_cols = [col for col in df.columns if col != target_col and df[col].dtype in ['int64', 'float64']]
    
    # Create subplots for distributions
    n_features = len(feature_cols)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, feature in enumerate(feature_cols[:4]):  # Show first 4 features
        if i < len(axes):
            # Box plots by class
            df.boxplot(column=feature, by=target_col, ax=axes[i])
            axes[i].set_title(f'{feature} by {target_col}')
            axes[i].set_xlabel(target_col)
            axes[i].set_ylabel(feature)
    
    plt.tight_layout()
    plt.show()
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[feature_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.show()
    
    return feature_cols

def prepare_data(df, feature_cols, target_col='budded'):
    """Prepare features and target for modeling"""
    print("\n" + "=" * 50)
    print("DATA PREPARATION")
    print("=" * 50)
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Convert target to binary if it's string
    if y.dtype == 'object':
        y = y.map({'budded': 1, 'unbudded': 0})  # Adjust mapping as needed
        print("Target variable mapped: budded=1, unbudded=0")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"Feature columns: {feature_cols}")
    
    return X_train, X_test, y_train, y_test

def train_decision_tree(X_train, X_test, y_train, y_test, feature_names):
    """Train and evaluate decision tree"""
    print("\n" + "=" * 50)
    print("DECISION TREE ANALYSIS")
    print("=" * 50)
    
    # Train decision tree with cross-validation for hyperparameter tuning
    param_grid = {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    dt = DecisionTreeClassifier(random_state=42)
    dt_grid = GridSearchCV(dt, param_grid, cv=5, scoring='roc_auc')
    dt_grid.fit(X_train, y_train)
    
    print(f"Best parameters: {dt_grid.best_params_}")
    print(f"Best cross-validation score: {dt_grid.best_score_:.3f}")
    
    # Get best model
    best_dt = dt_grid.best_estimator_
    
    # Make predictions
    y_pred = best_dt.predict(X_test)
    y_prob = best_dt.predict_proba(X_test)[:, 1]
    
    # Evaluate
    print(f"\nTest set performance:")
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.3f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': best_dt.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature Importance:")
    print(feature_importance)
    
    # Visualize decision tree (simplified)
    plt.figure(figsize=(20, 10))
    plot_tree(best_dt, max_depth=3, feature_names=feature_names, 
              class_names=['unbudded', 'budded'], filled=True, rounded=True)
    plt.title('Decision Tree (max_depth=3 for visualization)')
    plt.show()
    
    # Extract thresholds from the tree
    print(f"\nDecision Tree Thresholds:")
    tree = best_dt.tree_
    feature_names_array = np.array(feature_names)
    
    def get_tree_thresholds(tree, feature_names):
        thresholds = {}
        for i in range(tree.node_count):
            if tree.children_left[i] != tree.children_right[i]:  # Not a leaf
                feature_idx = tree.feature[i]
                feature_name = feature_names[feature_idx]
                threshold = tree.threshold[i]
                if feature_name not in thresholds:
                    thresholds[feature_name] = []
                thresholds[feature_name].append(threshold)
        
        # Sort thresholds for each feature
        for feature in thresholds:
            thresholds[feature] = sorted(list(set(thresholds[feature])))
        
        return thresholds
    
    thresholds = get_tree_thresholds(tree, feature_names_array)
    for feature, thresh_list in thresholds.items():
        print(f"{feature}: {thresh_list}")
    
    return best_dt, feature_importance

def train_random_forest(X_train, X_test, y_train, y_test, feature_names):
    """Train and evaluate random forest"""
    print("\n" + "=" * 50)
    print("RANDOM FOREST ANALYSIS")
    print("=" * 50)
    
    # Train random forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Cross-validation score
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"Cross-validation ROC AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Make predictions
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    # Evaluate
    print(f"Test set ROC AUC: {roc_auc_score(y_test, y_prob):.3f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    rf_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nRandom Forest Feature Importance:")
    print(rf_importance)
    
    return rf, rf_importance

def compare_models(X_train, X_test, y_train, y_test):
    """Compare multiple models"""
    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)
    
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    results = {}
    
    # Scale features for SVM and Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    for name, model in models.items():
        if name in ['SVM', 'Logistic Regression']:
            X_tr, X_te = X_train_scaled, X_test_scaled
        else:
            X_tr, X_te = X_train, X_test
        
        # Train model
        model.fit(X_tr, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring='roc_auc')
        
        # Test prediction
        y_prob = model.predict_proba(X_te)[:, 1]
        test_auc = roc_auc_score(y_test, y_prob)
        
        results[name] = {
            'CV_AUC_mean': cv_scores.mean(),
            'CV_AUC_std': cv_scores.std(),
            'Test_AUC': test_auc
        }
        
        print(f"{name}:")
        print(f"  CV ROC AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        print(f"  Test ROC AUC: {test_auc:.3f}")
        print()
    
    return results

def plot_feature_importance_comparison(dt_importance, rf_importance):
    """Plot feature importance comparison"""
    print("\n" + "=" * 50)
    print("FEATURE IMPORTANCE COMPARISON")
    print("=" * 50)
    
    # Merge importance scores
    comparison = dt_importance.merge(rf_importance, on='feature', suffixes=('_dt', '_rf'))
    
    # Plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(comparison))
    width = 0.35
    
    plt.bar(x - width/2, comparison['importance_dt'], width, label='Decision Tree', alpha=0.8)
    plt.bar(x + width/2, comparison['importance_rf'], width, label='Random Forest', alpha=0.8)
    
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance: Decision Tree vs Random Forest')
    plt.xticks(x, comparison['feature'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print("Feature importance ranking:")
    print(comparison[['feature', 'importance_dt', 'importance_rf']])

# Main execution function
def main(csv_file):
    """Main analysis pipeline"""
    try:
        # Step 1: Load and explore data
        df = load_and_explore_data(csv_file)
        
        # Step 2: Visualize data
        feature_cols = visualize_data(df)
        
        # Step 3: Prepare data
        X_train, X_test, y_train, y_test = prepare_data(df, feature_cols)
        
        # Step 4: Train decision tree
        best_dt, dt_importance = train_decision_tree(X_train, X_test, y_train, y_test, feature_cols)
        
        # Step 5: Train random forest
        best_rf, rf_importance = train_random_forest(X_train, X_test, y_train, y_test, feature_cols)
        
        # Step 6: Compare all models
        model_results = compare_models(X_train, X_test, y_train, y_test)
        
        # Step 7: Compare feature importance
        plot_feature_importance_comparison(dt_importance, rf_importance)
        
        print("\n" + "=" * 50)
        print("ANALYSIS COMPLETE!")
        print("=" * 50)
        print("Key takeaways:")
        print("1. Check the feature importance rankings to see which morphological parameters matter most")
        print("2. Decision tree thresholds show exact cutoff values for classification")
        print("3. Model comparison helps choose the best approach for your data")
        print("4. Visualizations reveal relationships between features and budding state")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your CSV file has the correct column names and format")

# Example usage:
if __name__ == "__main__":
    # Replace 'your_data.csv' with your actual file path
    csv_file = 'your_data.csv'
    
    # Note: Make sure your CSV has columns like:
    # - 'budded' (or similar) for the target variable (budded/unbudded)
    # - 'circularity', 'perimeter', 'area', 'aspect_ratio' for features
    # Adjust column names in the script as needed
    
    main(csv_file)