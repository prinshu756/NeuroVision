
"""
TumorVision: Enhanced Utilities Module
Contains custom loss functions, metrics, data generators, and augmentation pipelines
for improved brain tumor detection and segmentation.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import cv2
import tensorflow as tf
import os 
from PIL import Image
from tensorflow.keras import backend as K
from keras.saving import register_keras_serializable
import albumentations as A
from scipy import ndimage

# ============================================================================
# ENHANCED DATA AUGMENTATION PIPELINE
# ============================================================================

def get_training_augmentation():
    """
    Advanced augmentation pipeline for training data.
    Improves model generalization and robustness.
    """
    return A.Compose([
        # Spatial transformations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=30,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),
        # Elastic deformation for medical images
        A.ElasticTransform(
            alpha=120,
            sigma=120 * 0.05,
            p=0.3
        ),
        A.GridDistortion(p=0.3),
        # Intensity transformations
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        ], p=0.5),
        # Noise and blur
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.3),
    ])

def get_validation_augmentation():
    """Light augmentation for validation (only normalization)."""
    return A.Compose([])

# ============================================================================
# ENHANCED DATA GENERATOR WITH AUGMENTATION
# ============================================================================

class DataGenerator(tf.keras.utils.Sequence):
    """
    Enhanced data generator with on-the-fly augmentation.
    Supports both training and validation modes.
    """
    def __init__(self, ids, mask, image_dir='./', batch_size=16, 
                 img_h=256, img_w=256, shuffle=True, augment=True):
        self.ids = ids
        self.mask = mask
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.img_h = img_h
        self.img_w = img_w
        self.shuffle = shuffle
        self.augment = augment
        self.augmentation = get_training_augmentation() if augment else get_validation_augmentation()
        self.on_epoch_end()

    def __len__(self):
        'Get the number of batches per epoch'
        return int(np.floor(len(self.ids)) / self.batch_size)

    def __getitem__(self, index):
        'Generate a batch of data'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_ids = [self.ids[i] for i in indexes]
        list_mask = [self.mask[i] for i in indexes]
        X, y = self.__data_generation(list_ids, list_mask)
        return X, y

    def on_epoch_end(self):
        'Update indices after each epoch'
        self.indexes = np.arange(len(self.ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids, list_mask):
        'Generate data for a batch of images'
        X = np.empty((self.batch_size, self.img_h, self.img_w, 3))
        y = np.empty((self.batch_size, self.img_h, self.img_w, 1))

        for i in range(len(list_ids)):
            img_path = './' + str(list_ids[i])
            mask_path = './' + str(list_mask[i])
            
            # Read images
            img = np.array(Image.open(img_path))
            mask = np.array(Image.open(mask_path))

            # Resize
            img = cv2.resize(img, (self.img_h, self.img_w))
            mask = cv2.resize(mask, (self.img_h, self.img_w))
            
            # Apply augmentation
            if self.augment:
                augmented = self.augmentation(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']
            
            # Convert to float64
            img = np.array(img, dtype=np.float64)
            mask = np.array(mask, dtype=np.float64)

            # Standardization
            img -= img.mean()
            img /= (img.std() + 1e-8)  # Add epsilon to prevent division by zero
            
            X[i,] = img
            y[i,] = np.expand_dims(mask, axis=2)
        
        # Binary mask
        y = (y > 0).astype(np.float32)

        return X, y


class EnhancedDataGenerator(tf.keras.utils.Sequence):
    """
    Advanced data generator with mixup, cutmix, and class balancing.
    """
    def __init__(self, ids, mask, image_dir='./', batch_size=16,
                 img_h=256, img_w=256, shuffle=True, augment=True,
                 mixup_alpha=0.2, cutmix_alpha=0.0):
        self.ids = ids
        self.mask = mask
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.img_h = img_h
        self.img_w = img_w
        self.shuffle = shuffle
        self.augment = augment
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.augmentation = get_training_augmentation() if augment else None
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.ids))
        indexes = self.indexes[start_idx:end_idx]
        
        list_ids = [self.ids[i] for i in indexes]
        list_mask = [self.mask[i] for i in indexes]
        
        X, y = self.__data_generation(list_ids, list_mask)
        
        # Apply mixup augmentation
        if self.mixup_alpha > 0 and self.augment:
            X, y = self.__mixup(X, y, self.mixup_alpha)
        
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __mixup(self, X, y, alpha=0.2):
        """Apply mixup augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = X.shape[0]
        index = np.random.permutation(batch_size)
        
        mixed_X = lam * X + (1 - lam) * X[index]
        mixed_y = lam * y + (1 - lam) * y[index]
        
        return mixed_X, mixed_y

    def __data_generation(self, list_ids, list_mask):
        actual_batch_size = len(list_ids)
        X = np.empty((actual_batch_size, self.img_h, self.img_w, 3))
        y = np.empty((actual_batch_size, self.img_h, self.img_w, 1))

        for i in range(actual_batch_size):
            img_path = './' + str(list_ids[i])
            mask_path = './' + str(list_mask[i])
            
            img = np.array(Image.open(img_path))
            mask = np.array(Image.open(mask_path))

            img = cv2.resize(img, (self.img_h, self.img_w))
            mask = cv2.resize(mask, (self.img_h, self.img_w))
            
            if self.augmentation is not None:
                augmented = self.augmentation(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']
            
            img = np.array(img, dtype=np.float64)
            mask = np.array(mask, dtype=np.float64)

            img -= img.mean()
            img /= (img.std() + 1e-8)
            
            X[i,] = img
            y[i,] = np.expand_dims(mask, axis=2)
        
        y = (y > 0).astype(np.float32)

        return X, y


# ============================================================================
# ENHANCED PREDICTION FUNCTION WITH TTA
# ============================================================================

def prediction(test, model, model_seg, use_tta=False):
    """
    Enhanced prediction function with optional Test Time Augmentation (TTA).
    
    Two-stage prediction:
    1. Classification: Does the image have a tumor?
    2. Segmentation: If yes, where is the tumor located?
    
    Args:
        test: DataFrame with image paths
        model: Classification model
        model_seg: Segmentation model
        use_tta: Enable Test Time Augmentation for better accuracy
    """
    directory = "./"
    mask = []
    image_id = []
    has_mask = []

    for i in test.image_path:
        path = directory + str(i)
        
        # Read image
        img = np.array(Image.open(path))
        
        # Normalize for classification
        img_norm = img * 1./255.
        img_norm = cv2.resize(img_norm, (256, 256))
        img_norm = np.array(img_norm, dtype=np.float64)
        img_norm = np.reshape(img_norm, (1, 256, 256, 3))

        # Classification prediction (with optional TTA)
        if use_tta:
            is_defect = predict_with_tta_classification(model, img_norm)
        else:
            is_defect = model.predict(img_norm, verbose=0)

        # If no tumor detected
        if np.argmax(is_defect) == 0:
            image_id.append(i)
            has_mask.append(0)
            mask.append('No mask')
            continue

        # Prepare for segmentation
        img = np.array(Image.open(path))
        X = np.empty((1, 256, 256, 3))
        img = cv2.resize(img, (256, 256))
        img = np.array(img, dtype=np.float64)
        
        # Standardize
        img -= img.mean()
        img /= (img.std() + 1e-8)
        X[0,] = img

        # Segmentation prediction (with optional TTA)
        if use_tta:
            predict = predict_with_tta_segmentation(model_seg, X)
        else:
            predict = model_seg.predict(X, verbose=0)

        if predict.round().astype(int).sum() == 0:
            image_id.append(i)
            has_mask.append(0)
            mask.append('No mask')
        else:
            image_id.append(i)
            has_mask.append(1)
            mask.append(predict)

    return image_id, mask, has_mask


def predict_with_tta_classification(model, img, strategy='max'):
    """
    Test Time Augmentation for classification.
    
    Args:
        model: Classification model
        img: Preprocessed image
        strategy: 'mean' for average, 'max' for maximum confidence
    
    Returns:
        Prediction with higher confidence
    """
    predictions = []
    
    # Original
    predictions.append(model.predict(img, verbose=0))
    
    # Horizontal flip
    img_flip = np.flip(img, axis=2)
    predictions.append(model.predict(img_flip, verbose=0))
    
    # Vertical flip
    img_flip_v = np.flip(img, axis=1)
    predictions.append(model.predict(img_flip_v, verbose=0))
    
    if strategy == 'max':
        # Return prediction with highest confidence for detected class
        max_conf = 0
        best_pred = predictions[0]
        for pred in predictions:
            conf = np.max(pred)
            if conf > max_conf:
                max_conf = conf
                best_pred = pred
        return best_pred
    else:
        # Average predictions
        return np.mean(predictions, axis=0)


def predict_with_tta_segmentation(model, img):
    """
    Test Time Augmentation for segmentation.
    Applies geometric transforms and averages the results.
    """
    predictions = []
    
    # Original
    pred = model.predict(img, verbose=0)
    predictions.append(pred)
    
    # Horizontal flip
    img_flip = np.flip(img, axis=2)
    pred_flip = model.predict(img_flip, verbose=0)
    pred_flip = np.flip(pred_flip, axis=2)
    predictions.append(pred_flip)
    
    # Vertical flip
    img_flip_v = np.flip(img, axis=1)
    pred_flip_v = model.predict(img_flip_v, verbose=0)
    pred_flip_v = np.flip(pred_flip_v, axis=1)
    predictions.append(pred_flip_v)
    
    # 90 degree rotation
    img_rot90 = np.rot90(img, axes=(1, 2))
    pred_rot90 = model.predict(img_rot90, verbose=0)
    pred_rot90 = np.rot90(pred_rot90, k=-1, axes=(1, 2))
    predictions.append(pred_rot90)
    
    # Average all predictions
    return np.mean(predictions, axis=0)


# ============================================================================
# ENHANCED LOSS FUNCTIONS AND METRICS
# ============================================================================

'''
Custom loss functions for training ResUNet segmentation model.
Based on: https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py

@article{focal-unet,
  title={A novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation},
  author={Abraham, Nabila and Khan, Naimul Mefraz},
  journal={arXiv preprint arXiv:1810.07842},
  year={2018}
}
'''

@register_keras_serializable()
def tversky(y_true, y_pred, smooth=1e-6):
    """
    Tversky Index: Generalization of Dice coefficient.
    Allows different weights for false positives and false negatives.
    α = 0.7 penalizes false negatives more (important for medical imaging)
    """
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

@register_keras_serializable()
def tversky_loss(y_true, y_pred):
    """Tversky loss for optimization."""
    return 1 - tversky(y_true, y_pred)

@register_keras_serializable()
def focal_tversky(y_true, y_pred):
    """
    Focal Tversky Loss: Focuses training on hard examples.
    γ = 0.75 provides good balance between easy and hard examples.
    """
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1 - pt_1), gamma)

@register_keras_serializable()
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Dice Coefficient: 2 * |A ∩ B| / (|A| + |B|)
    Standard metric for segmentation tasks.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

@register_keras_serializable()
def dice_loss(y_true, y_pred):
    """Dice loss for optimization."""
    return 1 - dice_coefficient(y_true, y_pred)

@register_keras_serializable()
def bce_dice_loss(y_true, y_pred):
    """
    Combined Binary Cross-Entropy and Dice Loss.
    Provides both pixel-wise and region-based optimization.
    """
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return 0.5 * bce + 0.5 * dice

@register_keras_serializable()
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal Loss: Addresses class imbalance by down-weighting easy examples.
    Reference: Lin et al., 2017
    """
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    cross_entropy = -y_true * K.log(y_pred)
    focal_weight = alpha * K.pow(1 - y_pred, gamma)
    focal = focal_weight * cross_entropy
    return K.mean(K.sum(focal, axis=-1))

@register_keras_serializable()
def combo_loss(y_true, y_pred, alpha=0.5, ce_ratio=0.5):
    """
    Combo Loss: Combines Dice and weighted cross-entropy.
    Effective for highly imbalanced datasets.
    """
    dice = dice_loss(y_true, y_pred)
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    weighted_ce = -y_true * K.log(y_pred) * alpha - (1 - y_true) * K.log(1 - y_pred) * (1 - alpha)
    weighted_ce = K.mean(weighted_ce)
    return ce_ratio * weighted_ce + (1 - ce_ratio) * dice

@register_keras_serializable()
def iou_score(y_true, y_pred, smooth=1e-6):
    """
    Intersection over Union (IoU / Jaccard Index).
    Standard metric for segmentation evaluation.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

@register_keras_serializable()
def sensitivity(y_true, y_pred, smooth=1e-6):
    """Sensitivity / Recall / True Positive Rate."""
    true_positives = K.sum(y_true * K.round(y_pred))
    possible_positives = K.sum(y_true)
    return (true_positives + smooth) / (possible_positives + smooth)

@register_keras_serializable()
def specificity(y_true, y_pred, smooth=1e-6):
    """Specificity / True Negative Rate."""
    true_negatives = K.sum((1 - y_true) * (1 - K.round(y_pred)))
    possible_negatives = K.sum(1 - y_true)
    return (true_negatives + smooth) / (possible_negatives + smooth)

@register_keras_serializable()
def precision_metric(y_true, y_pred, smooth=1e-6):
    """Precision: TP / (TP + FP)"""
    true_positives = K.sum(y_true * K.round(y_pred))
    predicted_positives = K.sum(K.round(y_pred))
    return (true_positives + smooth) / (predicted_positives + smooth)


# ============================================================================
# LEARNING RATE SCHEDULES
# ============================================================================

def cosine_annealing_schedule(epoch, lr, epochs=100, min_lr=1e-6):
    """Cosine Annealing Learning Rate Schedule."""
    return min_lr + (lr - min_lr) * (1 + np.cos(np.pi * epoch / epochs)) / 2

def warmup_cosine_schedule(epoch, lr, warmup_epochs=5, total_epochs=100, min_lr=1e-6):
    """Warmup + Cosine Annealing Schedule."""
    if epoch < warmup_epochs:
        return lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + (lr - min_lr) * (1 + np.cos(np.pi * progress)) / 2

def get_lr_scheduler(schedule_type='reduce_on_plateau', **kwargs):
    """Factory function for learning rate scheduler callbacks."""
    if schedule_type == 'cosine':
        return tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: cosine_annealing_schedule(epoch, lr, **kwargs)
        )
    elif schedule_type == 'warmup_cosine':
        return tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: warmup_cosine_schedule(epoch, lr, **kwargs)
        )
    elif schedule_type == 'reduce_on_plateau':
        return tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


# ============================================================================
# ENHANCED CALLBACKS
# ============================================================================

def get_training_callbacks(checkpoint_path, patience=15, schedule='reduce_on_plateau'):
    """Get comprehensive training callbacks."""
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            mode='min',
            verbose=1,
            restore_best_weights=True
        ),
        get_lr_scheduler(schedule),
    ]
    return callbacks


# ============================================================================
# MODEL EVALUATION UTILITIES
# ============================================================================

def evaluate_segmentation(y_true, y_pred):
    """
    Comprehensive evaluation of segmentation predictions.
    
    Returns:
        dict: Dictionary containing all metrics
    """
    y_true = y_true.flatten()
    y_pred = (y_pred.flatten() > 0.5).astype(int)
    
    # Calculate metrics
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Compute scores
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # Dice and IoU
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'dice': dice,
        'iou': iou,
        'sensitivity': recall,
        'specificity': tn / (tn + fp + 1e-8)
    }