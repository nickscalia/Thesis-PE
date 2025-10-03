import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
import joblib
from libemg.emg_predictor import EMGClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_emg_classifier(X_train, y_train, X_test, y_test, model_name, output_folder, model_params=None, random_seed=None, set_index=0):
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

    start_time = time.time() 
    clf.fit(train_dict) # Fit the model
    end_time = time.time() 
    training_time = (end_time - start_time)

    prediction_times = []
    all_preds = []

    joblib.dump(clf, f'{output_folder}/set_{set_index}.pkl')
    
    for i in range(X_test.shape[0]):
        single_test_dict = {
            'test_features': X_test[i:i+1] 
        }
        start_time = time.time()
        preds, _ = clf.run(single_test_dict)
        end_time = time.time()
        prediction_times.append(end_time - start_time)
        all_preds.append(preds)
    
    pred_time = np.mean(prediction_times) 
    preds = np.array(all_preds) 

    # Plot confusion matrix
    cm = confusion_matrix(y_test, preds)
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100  # percentuale per riga
    class_names = ['no weight', 'light', 'medium', 'heavy']
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    title = f'Confusion Matrix - {model_name}'
    #title = f'S07 Confusion Matrix'
    if set_index is not None:
        title += f' (Set {set_index})'
        #title += f' Myo + Trigno, EMG + IMU ({model_name})'
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig("mio_plot_alta_risoluzione.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Metrics
    acc = accuracy_score(y_test, preds) * 100
    precision = precision_score(y_test, preds, average='weighted', zero_division=0) * 100
    recall = recall_score(y_test, preds, average='weighted', zero_division=0) * 100
    f1 = f1_score(y_test, preds, average='weighted', zero_division=0) * 100
    
    return acc, precision, recall, f1, training_time, pred_time

def evaluate_and_store_results(feature_sets_train, feature_sets_test, y_train, y_test, model_name, results, output_folder, model_params=None):
    """
    Runs evaluation for each feature set, storing metrics in the results dictionary.
    """
    results[model_name] = {
        'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'training_time': [], 'inference_time': []
    }

    for i, (X_tr, X_te) in enumerate(zip(feature_sets_train, feature_sets_test), start=1):
        acc, precision, recall, f1, tr_time, pred_time = evaluate_emg_classifier(
            X_tr, y_train, X_te, y_test, 
            model_name,
            output_folder,
            model_params=model_params, 
            set_index=i
        )
        results[model_name]['accuracy'].append(acc)
        results[model_name]['precision'].append(precision)
        results[model_name]['recall'].append(recall)
        results[model_name]['f1'].append(f1)
        results[model_name]['training_time'].append(tr_time)
        results[model_name]['inference_time'].append(pred_time)


def print_results(model_name: str, results: dict):
    """
    Prints evaluation metrics for a model, formatting time metrics in milliseconds and others as percentages.
    """
    print(f'-------- {model_name} Classifier - Evaluation Metrics --------')
    for metric_name, values in results.items():
        if metric_name.lower() == 'training_time':
            # training time in seconds with 4 decimals
            values_str = ' | '.join([
                f'Set {i}: {v:.4f} s' for i, v in enumerate(values, start=1)
            ])
        elif metric_name.lower() == 'inference_time':
            # prediction time in milliseconds with 4 decimals
            values_str = ' | '.join([
                f'Set {i}: {v*1000:.4f} ms' for i, v in enumerate(values, start=1)
            ])
        else:
            # other metrics in percent with 2 decimals
            values_str = ' | '.join([
                f'Set {i}: {v:.2f}%' for i, v in enumerate(values, start=1)
            ])
        print(f'{metric_name.capitalize()}: {values_str}')