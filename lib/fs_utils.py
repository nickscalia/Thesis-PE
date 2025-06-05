import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from libemg.emg_predictor import EMGClassifier

def plot_label_distribution(df, labels, title='Label Distribution', palette='Set1'):
    """
    Plots the distribution of labels in a given DataFrame using Seaborn countplot.
    """
    temp = df.copy()  # Avoid modifying the original DataFrame
    ax = sns.countplot(data=temp, x='label', hue='label', order=labels, hue_order=labels, palette=palette, legend=False)
    ax.set_title(title)
    ax.set_xlabel('Label')
    ax.set_ylabel('Count')
    plt.show()

def balance_via_undersampling(df, label_col='label', random_state=42):
    """
    Performs random undersampling to balance the number of samples per class,
    using the size of the minority class.
    """
    # Find the number of samples in the least represented class
    min_count = df[label_col].value_counts().min()
    balanced_classes = []

    # Perform undersampling for each class
    for label in df[label_col].unique():
        class_subset = df[df[label_col] == label]
        class_downsampled = resample(class_subset, replace=False, n_samples=min_count, random_state=random_state)
        balanced_classes.append(class_downsampled)

    # Concatenate and shuffle the balanced DataFrame
    balanced_df = pd.concat(balanced_classes)
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return balanced_df

def plot_feature_vs_label(df, feature_col, bins, feature_labels, label_col='label', palette='Set1'):
    """
    Plot label distribution grouped by categorized feature bins.
    Categorizes a continuous feature into bins, then shows count of labels per bin category.
    """
    temp = df.copy()
    temp[f'{feature_col}_cat'] = pd.cut(temp[feature_col], bins=bins, labels=feature_labels, right=False)  # Categorize feature into bins
    label_order = sorted(temp[label_col].unique())  # Get sorted unique labels for consistent hue order

    ax = sns.countplot(data=temp, x=f'{feature_col}_cat', hue=label_col, palette=palette, order=feature_labels, hue_order=label_order)
    ax.set_title(f'Incidence of {label_col} by {feature_col} Category')
    ax.set_xlabel(f'{feature_col} Category')
    ax.set_ylabel('Count')
    plt.show()

def plot_feature_correlation(X, suffixes=None, all_features=False):
    """
    Plot correlation matrices for selected feature groups or the entire feature set.
    """
    if all_features:  # Plot full correlation matrix
        plt.figure(figsize=(30, 18))
        sns.heatmap(X.corr(), annot=True, cmap='RdYlGn', linewidths=0.3)
        plt.title('Correlation Matrix of All Features', fontsize=16)
        plt.tight_layout()
        plt.show()

    if suffixes:  # Plot correlation matrices by suffix
        for suffix in suffixes:
            cols = [col for col in X.columns if col.endswith(suffix)]
            if not cols: continue  # Skip if no matching columns
            subset = X[cols]
            plt.figure(figsize=(12, 8))
            sns.heatmap(subset.corr(), annot=True, cmap='RdYlGn', linewidths=0.2)
            plt.title(f'Correlation Matrix for Features Ending with \"{suffix}\"')
            plt.tight_layout()
            plt.show()

def drop_highly_correlated_features(X, suffixes, preferred_features, threshold=0.8):
    """
    Drop highly correlated features within each suffix group, keeping only the preferred ones.
    """
    for suffix in suffixes:
        # Find all columns that end with the current suffix
        cols = [col for col in X.columns if col.endswith(suffix)]
        subset = X[cols]

        # Compute absolute correlation matrix and extract upper triangle
        corr_matrix = subset.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Define which features should be preserved based on the preferred list
        features_to_keep = [f + suffix for f in preferred_features if f + suffix in subset.columns]

        # Initialize set to collect features marked for removal
        features_to_drop = set()

        for keep_feat in features_to_keep:
            if keep_feat in upper.columns:
                # Identify correlated features above threshold that are not in the keep list
                correlated = [
                    col for col in upper.columns 
                    if col != keep_feat and corr_matrix.loc[keep_feat, col] > threshold and col not in features_to_keep
                ]
                features_to_drop.update(correlated)

        # Drop identified redundant features
        X = X.drop(columns=list(features_to_drop))

    return X

def report_remaining_high_correlations(X, suffixes, threshold=0.8):
    """
    Report remaining correlations above a specified threshold within each suffix group.
    """
    print(f"Remaining highly correlated features (>{threshold}) after filtering:")
    all_high_corr = []
    
    for suffix in suffixes:
        cols = [col for col in X.columns if col.endswith(suffix)]
        subset = X[cols]

        # Compute absolute correlation matrix and extract upper triangle
        corr_matrix = subset.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Collect pairs of highly correlated features
        high_corr = [
            (suffix, col, row, corr_matrix.loc[row, col]) 
            for col in upper.columns 
            for row in upper.index 
            if upper.loc[row, col] > threshold
        ]

        all_high_corr.extend(high_corr)

    if not all_high_corr:
        print("No high correlations found in any subset.")
    else:
        for suffix in suffixes:
            suffix_corrs = [item for item in all_high_corr if item[0] == suffix]
            if suffix_corrs:
                print(f'\nSubset "{suffix}":')
                for _, feat1, feat2, corr_val in suffix_corrs:
                    print(f'  {feat1} and {feat2} are correlated with {corr_val:.2f}')

def prepare_cv_data(X, n_folds, random_state=42):
    """
    Prepare fold labels and data dictionary for cross-validation in feature selection.
    """
    n_samples = len(X)  
    fold_labels = np.arange(n_samples) % n_folds  # Create fold labels cycling through n_folds

    np.random.seed(random_state) 
    np.random.shuffle(fold_labels)  
    fold_labels = pd.Series(fold_labels)  # Convert fold labels to pandas Series

    crossvalidation_var = {
        "var": fold_labels,  
        "crossval_amount": n_folds  
    }
    
    # Create a dictionary mapping each feature name to its values reshaped as 2D arrays
    data_dict = {col: X[[col]].values for col in X.columns}

    return crossvalidation_var, data_dict

def plot_feature_accuracy(features, accuracies):
    """
    Plots individual feature accuracies sorted in descending order.
    """
    # Convert to numpy array if needed
    accuracies = np.array(accuracies)
    
    plt.figure(figsize=(12, 6))
    plt.bar(features, accuracies)
    plt.xlabel('Features')
    plt.ylabel('Accuracy (%)')
    plt.title('Individual Feature Accuracy [LDA]')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 100)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_sequential_selection_heatmap(metric_matrix, feature_names):
    """
    Plot a heatmap of the sequential feature selection metric matrix,
    showing only the upper triangular part with annotations.
    """
    mask = np.tril(np.ones_like(metric_matrix, dtype=bool))  # mask lower triangle
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(metric_matrix, mask=mask, annot=True, fmt=".0f", cmap="viridis", 
                xticklabels=feature_names, yticklabels=feature_names, vmin=0, vmax=100, 
                cbar_kws={"shrink": .8, "label": "Accuracy (%)"})
    plt.title('Sequential Feature Selection Metric [LDA]', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()

def plot_feature_selection_performance(results):
    """
    Plot accuracy vs number of features from a list of (k, accuracy) tuples.
    """
    # Extract the number of features and corresponding accuracy values
    k_vals = [k for k, acc in results]
    acc_vals = [acc for k, acc in results]

    # Create the plot
    plt.figure(figsize=(8, 5))
    plt.plot(k_vals, acc_vals, marker='o', linestyle='-', label='Accuracy')
    plt.title("Accuracy vs Number of Features [RF]")
    plt.xlabel("Number of Features")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.xticks(k_vals)
    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_feature_subsets(X_red, y, accuracy_fs, n_folds, model='RF', model_params={'n_estimators': 100}, random_seed=0):
    """
    Evaluate classifier performance using different feature subset sizes.
    """
    results = []  # Store (number of features, accuracy)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)  # Cross-validation setup
    
    for k in range(4, len(accuracy_fs) + 1, 2):  # Vary number of features
        features_k = accuracy_fs[:k]  # Select top-k features
        X_k = X_red[features_k].values
        y_values = y.values if hasattr(y, 'values') else y
    
        clf = EMGClassifier(model=model, model_parameters=model_params, random_seed=random_seed)  # Initialize classifier
        scores = []
    
        for train_index, test_index in kf.split(X_k):  # Cross-validation loop
            X_train, X_test = X_k[train_index], X_k[test_index]
            y_train, y_test = y_values[train_index], y_values[test_index]
    
            train_dict = {
                'training_features': X_train,
                'training_labels': y_train
            }
            test_dict = {
                'test_features': X_test
            }
    
            clf.fit(train_dict)  # Train model
            preds, _ = clf.run(test_dict)  # Predict
            acc = np.mean(preds == y_test)  # Accuracy
            scores.append(acc)
    
        mean_acc = np.mean(scores)  # Mean accuracy across folds
        results.append((k, mean_acc))
    
    return results