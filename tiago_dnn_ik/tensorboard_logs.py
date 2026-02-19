import tensorflow as tf
import numpy as np



class TensorBoardLogging(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        super(TensorBoardLogging, self).__init__()
        self.log_dir = log_dir
        self.file_writer = tf.summary.create_file_writer(log_dir)
        self.metrics_history = {}
        print(f"✓ TensorBoardLogging инициализирован: {log_dir}")
    
    def log_metric(self, name, value, step):
        with self.file_writer.as_default():
            tf.summary.scalar(name, value, step=step)
            self.file_writer.flush()
            print(f"  → Логируем {name} = {value:.6f} на шаге {step}")  # отладка
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            print("⚠️ logs is None")
            return
        print(f"\n📊 Эпоха {epoch}: получены метрики: {list(logs.keys())}")  # отладка
        # Логируем все метрики из logs
        for name, value in logs.items():
            if isinstance(value, (int, float)):
                self.log_metric(name, value, epoch)
                # Сохраняем историю
                if name not in self.metrics_history:
                    self.metrics_history[name] = []
                self.metrics_history[name].append(value)