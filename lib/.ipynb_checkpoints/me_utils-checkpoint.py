import matplotlib.pyplot as plt
import seaborn as sns
import time
from libemg.emg_predictor import EMGClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_emg_classifier(X_train, y_train, X_test, y_test, model_name, model_params=None, random_seed=42, set_index=None):
    """
    Train and evaluate an EMGClassifier with specified model.
    """
    if model_params is None:
        model_params = {}

    # Initialize EMGClassifier with specified model
    clf = EMGClassifier(model=model_name, model_parameters=model_params, random_seed=random_seed)
    
    # Prepare training data dictionary
    train_dict = {
        'training_features': X_train,
        'training_labels': y_train
    }
    
    # Start timing
    start_time = time.time()
    
    # Fit the model
    clf.fit(train_dict)

    # Prepare test data
    test_dict = {
        'test_features': X_test
    }

    # Predict
    preds, _ = clf.run(test_dict)

    # End timing
    end_time = time.time()
    processing_time = (end_time - start_time) * 1000

    # Plot confusion matrix
    cm = confusion_matrix(y_test, preds)
    class_names = ['no weight', 'light', 'medium', 'heavy']
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    title = f'Confusion Matrix - {model_name}'
    if set_index is not None:
        title += f' (Set {set_index})'
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()
    
    # Metrics
    acc = accuracy_score(y_test, preds) * 100
    precision = precision_score(y_test, preds, average='weighted', zero_division=0) * 100
    recall = recall_score(y_test, preds, average='weighted', zero_division=0) * 100
    f1 = f1_score(y_test, preds, average='weighted', zero_division=0) * 100
    
    return acc, precision, recall, f1, processing_time