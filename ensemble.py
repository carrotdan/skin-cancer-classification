import os
import gc
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    roc_curve, 
    auc, 
    balanced_accuracy_score
)
import tensorflow as tf
from tensorflow.keras import mixed_precision

# --- ENVIRONMENT SETUP ---
try:
    mixed_precision.set_global_policy('mixed_float16')
except:
    pass

import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- BEAUTIFUL CONSOLE OUTPUT ---
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_step(msg):
    print(f"\n{Colors.OKCYAN}{Colors.BOLD}>>> {msg}{Colors.ENDC}")

def print_success(msg):
    print(f"{Colors.OKGREEN}✔ {msg}{Colors.ENDC}")

def print_warning(msg):
    print(f"{Colors.WARNING}⚠ {msg}{Colors.ENDC}")

# --- CONFIGURATION CLASS ---
class Config:
    def __init__(self, data_path):
        self.BASE_PATH = data_path
        self.MODEL_SAVE_DIR = "./model_weights"
        self.METRICS_DIR = "./model_metrics"
        
        self.IMG_SIZE = 320
        self.BATCH_SIZE = 16
        self.SEED = 42
        self.TEST_SIZE = 0.2
        self.VAL_SIZE = 0.15
        self.CUTOUT_SIZE = 24  

    def create_dirs(self):
        if not os.path.exists(self.METRICS_DIR):
            os.makedirs(self.METRICS_DIR)

# --- CUSTOM LOSS FUNCTION ---
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=None, from_logits=False, **kwargs):
        super(FocalLoss, self).__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        weight = tf.math.pow((1 - y_pred), self.gamma)
        if self.alpha is not None:
            weight = weight * self.alpha
            
        return tf.reduce_sum(weight * cross_entropy, axis=-1)

# --- DATASET UTILS ---
def load_metadata(cfg: Config):
    image_path = {os.path.splitext(os.path.basename(x))[0]: x
                  for x in glob(os.path.join(cfg.BASE_PATH, '*', '*.jpg'))}
    
    csv_path = os.path.join(cfg.BASE_PATH, 'HAM10000_metadata.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found at {csv_path}. Please check your dataset path.")
        
    df = pd.read_csv(csv_path)
    df['path'] = df['image_id'].map(image_path.get)
    
    # Drop rows without image path
    df = df.dropna(subset=['path'])
    
    df['cell_type_idx'] = pd.Categorical(df['dx']).codes
    idx_to_label = {i: label for i, label in enumerate(df['dx'].astype('category').cat.categories)}
    return df, idx_to_label

def create_dataset(df, cfg, num_classes=7):
    paths = df['path'].values
    labels = tf.keras.utils.to_categorical(df['cell_type_idx'].values, num_classes)
    
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    
    def load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [cfg.IMG_SIZE, cfg.IMG_SIZE])
        img = tf.cast(img, tf.float32) / 255.0
        return img, label
    
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(cfg.BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# --- PLOTTING UTILS ---
def plot_confusion_matrix_custom(y_true, y_pred_idx, classes, title="Confusion Matrix", save_path=None):
    cm = confusion_matrix(y_true, y_pred_idx)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                linewidths=.5, cbar_kws={"shrink": .75})
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_multiclass_roc(y_true, y_pred_proba, classes, title="ROC Curves", save_path=None):
    n_classes = len(classes)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=n_classes)
    
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_classes))
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, 
                 label=f'{classes[i]} (AUC = {roc_auc[i]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

# --- MAIN LOGIC ---
def main():
    parser = argparse.ArgumentParser(description="Ensemble Top 3 Models for Skin Cancer Classification")
    parser.add_argument('--data_path', type=str, default="./dataset", help="Path to the HAM10000 dataset")
    parser.add_argument('--weights_dir', type=str, default="./model_weights", help="Directory containing .keras files")
    args = parser.parse_args()

    cfg = Config(args.data_path)
    cfg.MODEL_SAVE_DIR = args.weights_dir
    cfg.create_dirs()

    print_step("1. Loading Data and Splitting")
    df, idx_to_label = load_metadata(cfg)
    num_classes = len(idx_to_label)
    classes = [idx_to_label[i] for i in range(num_classes)]

    train_df_raw, test_df = train_test_split(
        df, test_size=cfg.TEST_SIZE, stratify=df['cell_type_idx'], random_state=cfg.SEED
    )
    train_df_raw, val_df = train_test_split(
        train_df_raw, test_size=cfg.VAL_SIZE, stratify=train_df_raw['cell_type_idx'], random_state=cfg.SEED
    )

    print_success(f"Validation set size: {len(val_df)}")
    print_success(f"Test set size: {len(test_df)}")

    val_ds = create_dataset(val_df, cfg, num_classes=num_classes)
    test_ds = create_dataset(test_df, cfg, num_classes=num_classes)

    print_step("2. Scanning for Trained Models")
    model_files = glob(os.path.join(cfg.MODEL_SAVE_DIR, "*.keras"))
    if not model_files:
        print_warning(f"No .keras models found in {cfg.MODEL_SAVE_DIR}!")
        return
    print_success(f"Found {len(model_files)} models: {[os.path.basename(m) for m in model_files]}")

    print_step("3. Evaluating Models on Validation Set to Select Top 3")
    val_scores = {}
    
    for m_path in model_files:
        m_name = os.path.basename(m_path).replace("_best.keras", "").replace(".keras", "")
        print(f"\n{Colors.OKBLUE}Evaluating {m_name}...{Colors.ENDC}")
        model = tf.keras.models.load_model(m_path, custom_objects={'FocalLoss': FocalLoss})
        
        preds_val = model.predict(val_ds, verbose=0)
        y_val_pred = np.argmax(preds_val, axis=1)
        y_val_true = val_df['cell_type_idx'].values
        
        score = balanced_accuracy_score(y_val_true, y_val_pred)
        val_scores[m_name] = {'path': m_path, 'score': score}
        print(f"  --> Balanced Accuracy: {score:.4f}")
        
        del model
        tf.keras.backend.clear_session()
        gc.collect()

    # Sort models by score (descending)
    sorted_models = sorted(val_scores.items(), key=lambda x: x[1]['score'], reverse=True)
    top_3_models = sorted_models[:3]
    
    print_step("4. Top 3 Models Selected")
    for i, (m_name, info) in enumerate(top_3_models):
        print_success(f"Rank {i+1}: {m_name} (Score: {info['score']:.4f})")

    print_step("5. Running Ensemble on Test Set (Weighted Averaging)")
    # Calculate weights based on validation scores
    total_score = sum([info['score'] for _, info in top_3_models])
    ensemble_weights = {m_name: info['score'] / total_score for m_name, info in top_3_models}
    
    y_true_test = test_df['cell_type_idx'].values
    ensemble_probs = np.zeros((len(test_df), num_classes))

    for m_name, info in top_3_models:
        weight = ensemble_weights[m_name]
        print(f"\n{Colors.OKBLUE}Running inference with {m_name} (Weight: {weight:.3f})...{Colors.ENDC}")
        
        model = tf.keras.models.load_model(info['path'], custom_objects={'FocalLoss': FocalLoss})
        preds_test = model.predict(test_ds, verbose=1)
        
        ensemble_probs += preds_test * weight
        
        del model
        tf.keras.backend.clear_session()
        gc.collect()

    y_pred_ensemble = np.argmax(ensemble_probs, axis=1)

    print_step("6. Final Ensemble Evaluation")
    
    bal_acc = balanced_accuracy_score(y_true_test, y_pred_ensemble)
    auc_score = roc_auc_score(tf.keras.utils.to_categorical(y_true_test, num_classes), ensemble_probs, multi_class='ovr')
    
    print(f"{Colors.BOLD}\n=== ENSEMBLE RESULTS ==={Colors.ENDC}")
    print(f"Balanced Accuracy : {bal_acc:.4f}")
    print(f"AUC (OVR)         : {auc_score:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true_test, y_pred_ensemble, target_names=classes))

    print_step("7. Generating Beautiful Visualizations")
    cm_path = os.path.join(cfg.METRICS_DIR, "ensemble_confusion_matrix.png")
    plot_confusion_matrix_custom(y_true_test, y_pred_ensemble, classes, 
                                 title="Top-3 Ensemble Confusion Matrix", save_path=cm_path)
    
    roc_path = os.path.join(cfg.METRICS_DIR, "ensemble_roc_curves.png")
    plot_multiclass_roc(y_true_test, ensemble_probs, classes, 
                        title="Top-3 Ensemble ROC Curves", save_path=roc_path)
                        
    print_success(f"Visualizations saved to {cfg.METRICS_DIR}")
    
    # Save predictions
    submission = pd.DataFrame({
        'image_id': test_df['image_id'],
        'true_label': test_df['dx'],
        'pred_label': [idx_to_label[i] for i in y_pred_ensemble]
    })
    for i, label in idx_to_label.items():
        submission[f'prob_{label}'] = ensemble_probs[:, i]
    
    save_path = "ensemble_predictions.csv"
    submission.to_csv(save_path, index=False)
    print_success(f"Predictions saved to {save_path}")
    
    print(f"\n{Colors.OKGREEN}{Colors.BOLD}🎉 Ensemble Process Completed Successfully! 🎉{Colors.ENDC}\n")

if __name__ == "__main__":
    main()
