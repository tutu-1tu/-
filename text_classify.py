# 文本分类模型加载和推理
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

# 配置TensorFlow使用CPU，避免GPU相关错误
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

# 导入配置
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEXT_CLASSIFY_MODEL_PATH, TEXT_CLASSIFY_VOCAB_PATH, TEXT_CLASSIFY_CATEGORIES, TEXT_CLASSIFY_SEQ_LENGTH

class TextClassifier:
    def __init__(self):
        self.model = None
        self.words = None
        self.word_to_id = None
        self.categories = TEXT_CLASSIFY_CATEGORIES
        self.seq_length = TEXT_CLASSIFY_SEQ_LENGTH
        self.load_model()
    
    def open_file(self, filename, mode='r'):
        """打开文件"""
        return open(filename, mode, encoding='utf-8', errors='ignore')
    
    def read_vocab(self, vocab_dir):
        """读取词汇表"""
        with self.open_file(vocab_dir) as fp:
            words = [i.strip() for i in fp.readlines()]
        word_to_id = dict(zip(words, range(len(words))))
        return words, word_to_id
    
    def load_model(self):
        """加载模型和词汇表"""
        try:
            # 读取词汇表
            if os.path.exists(TEXT_CLASSIFY_VOCAB_PATH):
                self.words, self.word_to_id = self.read_vocab(TEXT_CLASSIFY_VOCAB_PATH)
            else:
                raise FileNotFoundError(f"词汇表文件不存在: {TEXT_CLASSIFY_VOCAB_PATH}")
            
            # 使用CPU加载模型，避免GPU相关错误
            with tf.device('/CPU:0'):
                # 优先加载最佳模型，如果不存在则加载最终模型
                best_model_path = TEXT_CLASSIFY_MODEL_PATH.replace('my_model.h5', 'best_model.h5')
                
                if os.path.exists(best_model_path):
                    self.model = load_model(best_model_path)
                    print(f"文本分类模型加载成功（最佳模型）: {best_model_path}")
                elif os.path.exists(TEXT_CLASSIFY_MODEL_PATH):
                    self.model = load_model(TEXT_CLASSIFY_MODEL_PATH)
                    print(f"文本分类模型加载成功（最终模型）: {TEXT_CLASSIFY_MODEL_PATH}")
                else:
                    # 尝试其他可能的路径
                    alt_path = TEXT_CLASSIFY_MODEL_PATH.replace('my_model.h5', 'best_validation_best.h5')
                    if os.path.exists(alt_path):
                        self.model = load_model(alt_path)
                        print(f"文本分类模型加载成功: {alt_path}")
                    else:
                        raise FileNotFoundError(f"模型文件不存在。尝试过的路径: {best_model_path}, {TEXT_CLASSIFY_MODEL_PATH}")
        except Exception as e:
            print(f"加载文本分类模型失败: {e}")
            self.model = None
    
    def preprocess_text(self, text):
        """预处理文本"""
        if not text:
            return None
        
        # 将文本转换为字符列表
        content = list(text)
        # 转换为ID序列
        data_id = [self.word_to_id.get(x, 0) for x in content if x in self.word_to_id]
        
        if not data_id:
            return None
        
        # 使用 pad_sequences 填充到固定长度（与训练代码保持一致）
        x_pad = keras.preprocessing.sequence.pad_sequences(
            [data_id], 
            maxlen=self.seq_length, 
            padding='post', 
            truncating='post'
        )
        return x_pad
    
    def predict(self, text):
        """预测文本类别"""
        if self.model is None:
            return {"error": "模型未加载"}
        
        try:
            # 预处理文本
            x_pad = self.preprocess_text(text)
            if x_pad is None:
                return {"error": "文本预处理失败"}
            
            # 使用CPU进行预测，避免GPU相关错误
            with tf.device('/CPU:0'):
                # 预测
                y_pred = self.model.predict(x_pad, verbose=0)
                predicted_class_idx = np.argmax(y_pred[0])
                confidence = float(y_pred[0][predicted_class_idx])
                predicted_class = self.categories[predicted_class_idx]
                
                # 返回所有类别的概率
                probabilities = {
                    self.categories[i]: float(y_pred[0][i]) 
                    for i in range(len(self.categories))
                }
                
                return {
                    "category": predicted_class,
                    "confidence": confidence,
                    "probabilities": probabilities
                }
        except Exception as e:
            error_msg = str(e)
            # 如果是GPU相关错误，提供更友好的提示
            if "stream" in error_msg.lower() or "gpu" in error_msg.lower():
                return {"error": "模型预测时发生设备错误，请稍后重试"}
            return {"error": f"预测失败: {error_msg}"}

# 全局实例
text_classifier = TextClassifier()

