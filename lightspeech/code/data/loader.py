"""
Dataset loading utilities
"""

def load_dataset(dataset_name, data_dir):
    """
    Load a speech emotion recognition dataset.
    
    Args:
        dataset_name: Name of dataset (CREMA-D, RAVDESS, Emo-DB)
        data_dir: Directory containing raw dataset files
    
    Returns:
        Dataset object with audio files and labels
    """
    # TODO: Implement dataset loading
    # This is a template - students should implement based on their dataset
    raise NotImplementedError("Students should implement dataset loading based on their chosen dataset")


def split_dataset(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split dataset into train/validation/test sets.
    
    Args:
        data: Dataset to split
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    # TODO: Implement dataset splitting
    raise NotImplementedError("Students should implement dataset splitting")

