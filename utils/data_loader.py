import os
import zipfile
import tensorflow as tf


class DataLoader:
    
    DEFAULT_CLASS_NAMES = ['AbdomenCT', 'BreastMRI', 'ChestCT', 'CXR', 'Hand', 'HeadCT']
    
    def __init__(self, dataset_path, img_size=64, batch_size=64, class_names=None):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = class_names or self.DEFAULT_CLASS_NAMES
        self.autotune = tf.data.AUTOTUNE
        self.datasets = {}
    
    def extract_if_needed(self):
        if self.dataset_path.endswith('.zip'):
            extract_path = self.dataset_path.replace('.zip', '')
            with zipfile.ZipFile(self.dataset_path, 'r') as z:
                z.extractall(extract_path)
            return extract_path
        return self.dataset_path
    
    def preprocess(self, path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=1)
        img = tf.image.resize(img, [self.img_size, self.img_size])
        return tf.cast(img, tf.float32) / 255.0
    
    def make_dataset(self, class_name, dataset_path):
        path = os.path.join(dataset_path, class_name)
        files = tf.data.Dataset.list_files(path + '/*.jpeg', shuffle=True)
        ds = files.map(
            lambda p: (self.preprocess(p), self.preprocess(p)), 
            num_parallel_calls=self.autotune
        )
        return ds.cache().shuffle(1000).batch(self.batch_size).prefetch(self.autotune)
    
    def load_all_datasets(self):
        dataset_path = self.extract_if_needed()
        
        for cls in self.class_names:
            self.datasets[cls] = self.make_dataset(cls, dataset_path)
        
        return self.datasets
    
    def get_dataset(self, class_name):
        if not self.datasets:
            self.load_all_datasets()
        return self.datasets.get(class_name)
