"""Data loading and preprocessing utilities."""
import os
import zipfile
import tensorflow as tf


class DataLoader:
    """Handle dataset loading and preprocessing."""
    
    DEFAULT_CLASS_NAMES = ['AbdomenCT', 'BreastMRI', 'ChestCT', 'CXR', 'Hand', 'HeadCT']
    
    def __init__(self, dataset_path, img_size=64, batch_size=64, class_names=None):
        """Initialize DataLoader.
        
        Args:
            dataset_path: Path to dataset or zip file
            img_size: Size to resize images to
            batch_size: Batch size for loading
            class_names: List of class names (default: medical classes)
        """
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = class_names or self.DEFAULT_CLASS_NAMES
        self.autotune = tf.data.AUTOTUNE
        self.datasets = {}
    
    def extract_if_needed(self):
        """Extract zip file if dataset_path points to a zip file."""
        if self.dataset_path.endswith('.zip'):
            extract_path = self.dataset_path.replace('.zip', '')
            with zipfile.ZipFile(self.dataset_path, 'r') as z:
                z.extractall(extract_path)
            return extract_path
        return self.dataset_path
    
    def preprocess(self, path):
        """Load and preprocess image.
        
        Args:
            path: Path to image file
            
        Returns:
            Preprocessed image tensor
        """
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=1)
        img = tf.image.resize(img, [self.img_size, self.img_size])
        return tf.cast(img, tf.float32) / 255.0
    
    def make_dataset(self, class_name, dataset_path):
        """Create tf.data dataset for a class.
        
        Args:
            class_name: Name of the class
            dataset_path: Base path to dataset
            
        Returns:
            tf.data.Dataset: Prepared dataset
        """
        path = os.path.join(dataset_path, class_name)
        files = tf.data.Dataset.list_files(path + '/*.jpeg', shuffle=True)
        ds = files.map(
            lambda p: (self.preprocess(p), self.preprocess(p)), 
            num_parallel_calls=self.autotune
        )
        return ds.cache().shuffle(1000).batch(self.batch_size).prefetch(self.autotune)
    
    def load_all_datasets(self):
        """Load all class datasets.
        
        Returns:
            dict: Dictionary mapping class names to datasets
        """
        dataset_path = self.extract_if_needed()
        
        for cls in self.class_names:
            self.datasets[cls] = self.make_dataset(cls, dataset_path)
        
        return self.datasets
    
    def get_dataset(self, class_name):
        """Get dataset for a specific class.
        
        Args:
            class_name: Name of the class
            
        Returns:
            tf.data.Dataset: Dataset for the class
        """
        if not self.datasets:
            self.load_all_datasets()
        return self.datasets.get(class_name)
