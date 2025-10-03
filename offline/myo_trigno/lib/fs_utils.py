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
    #plt.savefig("mio_plot_alta_risoluzione.png", dpi=300, bbox_inches='tight')
    plt.show()

def balance_via_undersampling(df, label_col='label', random_state=None):
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
    label_order = ['no weight', 'light', 'medium', 'heavy']

    ax = sns.countplot(data=temp, x=f'{feature_col}_cat', hue=label_col, palette=palette, order=feature_labels, hue_order=label_order)
    ax.set_title(f'Incidence of {label_col} by {feature_col} Category')
    ax.set_xlabel(f'{feature_col} Category')
    ax.set_ylabel('Count')
    #plt.savefig("mio_plot_alta_risoluzione.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_correlation(X, suffixes=None, all_features=False):
    """
    Plot correlation matrices for selected feature groups or the entire feature set.
    """
    if all_features:  # Plot full correlation matrix
        plt.figure(figsize=(30, 18))
        sns.heatmap(X.corr(), annot=True, cmap='RdYlGn', linewidths=0.3, fmt=".2f")
        plt.title('Correlation Matrix of All Features', fontsize=32)
        plt.tight_layout()
        plt.show()

    if suffixes:  # Plot correlation matrices by suffix
        for suffix in suffixes:
            cols = [col for col in X.columns if col.endswith(suffix)]
            if not cols: continue  # Skip if no matching columns
            subset = X[cols]
            plt.figure(figsize=(12, 8))
            sns.heatmap(subset.corr(), annot=True, cmap='RdYlGn', linewidths=0.2, fmt=".2f")
            plt.title(f'Correlation Matrix for Features Ending with \"{suffix}\"')
            plt.tight_layout()
            #plt.savefig("mio_plot_alta_risoluzione.png", dpi=300, bbox_inches='tight')
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
        ordered_preferred = [f + suffix for f in preferred_features if f + suffix in subset.columns]

        # Initialize set to collect features marked for removal
        features_to_drop = set()
        kept_features = []
        
        for i, keep_feat in enumerate(ordered_preferred):
                    if keep_feat in corr_matrix.columns:
                        # Confronta con le feature preferite successive (evita doppie eliminazioni)
                        for j in range(i + 1, len(ordered_preferred)):
                            other_feat = ordered_preferred[j]
                            if other_feat in corr_matrix.columns and corr_matrix.loc[keep_feat, other_feat] > threshold:
                                features_to_drop.add(other_feat)
        
                        # Confronta con tutte le altre (non preferite)
                        for col in corr_matrix.columns:
                            if col != keep_feat and col not in ordered_preferred and corr_matrix.loc[keep_feat, col] > threshold:
                                features_to_drop.add(col)
        
                        kept_features.append(keep_feat)

        # Drop identified redundant features
        X = X.drop(columns=list(features_to_drop))

    return X

def report_remaining_high_correlations(X, suffixes, intra_threshold=0.8, inter_threshold=0.9):
    """
    Report remaining high correlations among features with specified suffixes.
    """
    # Select only columns that end with one of the specified suffixes
    selected_cols = [col for col in X.columns if any(col.endswith(suf) for suf in suffixes)]
    
    if not selected_cols:
        print("No matching columns found with given suffixes.")
        return

    # Compute the absolute correlation matrix for the selected columns
    corr_matrix = X[selected_cols].corr().abs()

    # Keep only the upper triangle of the correlation matrix (to avoid duplicates)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    all_high_corr = []

    # Iterate over the upper triangle to find high correlations
    for col in upper.columns:
        for row in upper.index:
            corr_val = upper.loc[row, col]
            if np.isnan(corr_val):
                continue

            # Identify suffixes of the two features
            suffix1 = next((suf for suf in suffixes if col.endswith(suf)), "unknown")
            suffix2 = next((suf for suf in suffixes if row.endswith(suf)), "unknown")

            # Classify correlation as intra- or inter-channel based on suffix
            if suffix1 == suffix2 and corr_val > intra_threshold:
                all_high_corr.append((col, row, corr_val, "intra"))
            elif suffix1 != suffix2 and corr_val > inter_threshold:
                all_high_corr.append((col, row, corr_val, "inter"))

    # Report results
    if not all_high_corr:
        print("No high correlations found above thresholds.")
    else:
        print("High correlation report:")
        for feat1, feat2, corr_val, corr_type in all_high_corr:
            suf1 = next((suf for suf in suffixes if feat1.endswith(suf)), "unknown")
            suf2 = next((suf for suf in suffixes if feat2.endswith(suf)), "unknown")

            if corr_type == "intra":
                print(f'ISSUE: (Intra-channel "{suf1}") {feat1} and {feat2} correlated: {corr_val:.2f}')
            else:
                print(f'WARNING: (Inter-channel "{suf1}" vs "{suf2}") {feat1} and {feat2} correlated: {corr_val:.2f}')


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

def plot_feature_accuracy(features, accuracies, batch_size=40):
    """
    Plots individual feature accuracies in batches for readability.
    """
    accuracies = np.array(accuracies)
    n_features = len(features)
    
    for i in range(0, n_features, batch_size):
        f_batch = features[i:i+batch_size]
        a_batch = accuracies[i:i+batch_size]
        
        n = len(f_batch)
        width = max(12, min(0.5 * n, 50))
        height = max(6, min(0.25 * n, 20))

        def scale_font(base, min_val=10, max_val=32):
            return int(np.clip(width * base, min_val, max_val))

        xlabel_size = scale_font(1.2)
        ylabel_size = scale_font(1.2)
        title_size  = scale_font(1.5)
        tick_size   = scale_font(1)

        plt.figure(figsize=(width, height))
        plt.bar(f_batch, a_batch)

        plt.xlabel('Features', fontsize=xlabel_size)
        plt.ylabel('Accuracy [%]', fontsize=ylabel_size)
        plt.title(f'Individual Feature Accuracy [LDA] [{i+1}-{i+len(f_batch)}]', fontsize=title_size)

        plt.xticks(rotation=45, ha='right', fontsize=2*tick_size/3)
        plt.yticks(fontsize=tick_size)

        plt.ylim(0, 100)
        plt.grid(True)
        plt.tight_layout()
        #if i == 0: 
        #    plt.savefig("mio_plot_alta_risoluzione.png", dpi=300, bbox_inches='tight')
        plt.show()

def plot_sequential_selection_heatmap(metric_matrix, feature_names, batch_size=40):
    """
    Plot heatmap in square batches along the diagonal only with consistent color scale.
    """
    n = len(feature_names)
    vmin = np.nanmin(metric_matrix)
    vmax = np.nanmax(metric_matrix)
    
    for i in range(0, n, batch_size):
        batch_features = feature_names[i:i+batch_size]
        batch_matrix = metric_matrix[i:i+batch_size, i:i+batch_size]
        
        width = max(8, min(0.5 * len(batch_features), 30))
        height = max(8, min(0.5 * len(batch_features), 30))
        
        plt.figure(figsize=(width, height))
        sns.heatmap(batch_matrix, annot=True, fmt=".1f", cmap="viridis", 
                    xticklabels=batch_features, yticklabels=batch_features,
                    vmin=vmin, vmax=vmax,  # scala colori fissa
                    cbar_kws={"orientation": "horizontal", "label": "Accuracy [%]"})
        plt.title(f'Sequential Feature Selection Metric [LDA]\nFeatures {i+1}-{i+len(batch_features)}', fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.show()


def plot_feature_selection_performance(results, model_name='Model'):
    """
    Plot accuracy vs number of features from a list of (k, accuracy) tuples.
    """
    # Extract the number of features and corresponding accuracy values
    k_vals = np.array([k for k, acc in results])
    acc_vals = np.array([acc for k, acc in results])

    # Create the plot
    plt.figure(figsize=(12, 6))
    # Plot full line without markers
    line, = plt.plot(k_vals, acc_vals, linestyle='-', label='Accuracy')
    color = line.get_color()    

    step = 5
    indices = np.where(k_vals % step == 0)[0]
    xlabel_size = scale_font(1.2)
    ylabel_size = scale_font(1.2)
    title_size  = scale_font(1.5)

    extra_indices = [0, len(k_vals) - 1]
    all_indices = np.unique(np.concatenate([indices, extra_indices]))
    
    # Seleziona i punti corrispondenti
    marker_k = k_vals[all_indices]
    marker_acc = acc_vals[all_indices]
    
    # Plotta i marker con stesso colore
    plt.plot(marker_k, marker_acc, linestyle='None', marker='o', color=color, label='_nolegend_')
    
    plt.title(f"Accuracy vs Number of Features [{model_name}]", fontsize=title_size)
    plt.xlabel("Number of Features", fontsize=xlabel_size)
    plt.ylabel("Accuracy [%]", fontsize=ylabel_size)
    plt.grid(True)
    plt.xticks(np.arange(0,  max(k_vals) + 1, 5))
    plt.legend()
    plt.tight_layout()
    #plt.savefig("mio_plot_risoluzione.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_pca_variance(pca):
    """
    Plots the cumulative explained variance to visualize how many PCA components are needed to reach 95%
    """    
    plt.figure(figsize=(8, 4))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')  # Cumulative variance
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.title('Explained variance vs number of components')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.savefig("mio_plot_alta_risoluzione.png", dpi=300, bbox_inches='tight')
    plt.show()