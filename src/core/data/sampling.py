import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

def compute_class_weights(y: np.ndarray | list) -> dict[int, float]:
    """
    Compute class weights for unbalanced datasets.
    
    Args:
        y: Array of labels (integers).
        
    Returns:
        Dictionary mapping class index to weight.
    """
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return dict(zip(classes, weights))

def oversample_dataframe(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Oversample the minority classes in a DataFrame to match the majority class count.
    
    Args:
        df: Input DataFrame.
        target_column: Name of the column containing class labels.
        
    Returns:
        Oversampled DataFrame.
    """
    max_size = df[target_column].value_counts().max()
    lst = [df]
    
    for class_index, group in df.groupby(target_column):
        if len(group) < max_size:
            lst.append(group.sample(max_size - len(group), replace=True))
            
    return pd.concat(lst)
