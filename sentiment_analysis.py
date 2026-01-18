# 情感分析模型加载和推理
import os
import re
import numpy as np
import pandas as pd
import jieba
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
import sys

# 配置TensorFlow使用CPU，避免GPU相关错误
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SENTIMENT_MODEL_PATH, SENTIMENT_DICT_PATH, SENTIMENT_SEQ_LENGTH

class SentimentAnalyzer:
    def __init__(self):
        self.model = None
        self.dicts = None
        self.maxlen = SENTIMENT_SEQ_LENGTH
        self.confidence_threshold = 0.5  # 置信度阈值
        self._init_keywords()
        self.load_model()
    
    def _init_keywords(self):
        """初始化情感关键词词典"""
        # 正面情感词（更全面）
        self.positive_words = [
            '好', '棒', '赞', '喜欢', '满意', '不错', '优秀', '完美', '开心', '高兴',
            '爱', '美', '棒极了', '太好了', '推荐', '值得', '满意', '赞', '👍',
            '喜欢', '喜爱', '热爱', '赞美', '称赞', '表扬', '夸奖', '欣赏', '认可',
            '支持', '赞同', '同意', '肯定', '正面', '积极', '乐观', '愉快', '欢乐',
            '兴奋', '激动', '惊喜', '感动', '温暖', '舒适', '安心', '放心', '信任',
            '成功', '胜利', '成就', '进步', '提升', '改善', '优化', '增强', '加强',
            '美好', '精彩', '出色', '卓越', '杰出', '优秀', '优良', '优质', '上乘',
            '超值', '划算', '实惠', '便宜', '经济', '高效', '快速', '便捷', '方便'
        ]
        
        # 负面情感词（更全面）
        self.negative_words = [
            '差', '坏', '烂', '讨厌', '失望', '糟糕', '垃圾', '不好', '伤心', '难过',
            '差劲', '不行', '不推荐', '后悔', '糟糕', '差评', '👎',
            '讨厌', '厌恶', '反感', '嫌弃', '鄙视', '批评', '指责', '抱怨', '埋怨',
            '反对', '拒绝', '否定', '负面', '消极', '悲观', '沮丧', '失落', '绝望',
            '愤怒', '生气', '恼火', '烦躁', '焦虑', '担心', '忧虑', '恐惧', '害怕',
            '失败', '挫折', '困难', '问题', '麻烦', '困扰', '阻碍', '障碍', '缺陷',
            '糟糕', '恶劣', '低劣', '劣质', '次品', '残次', '破损', '损坏', '故障',
            '昂贵', '浪费', '低效', '缓慢', '麻烦', '复杂', '困难', '不便', '不实用'
        ]
        
        # 否定词
        self.negation_words = ['不', '没', '无', '非', '未', '别', '莫', '勿', '否', '没有', '不是', '不能', '不会', '不想', '不要']
        
        # 程度词（增强情感强度）
        self.intensity_words = {
            '非常': 1.5, '特别': 1.5, '极其': 1.8, '十分': 1.4, '相当': 1.3,
            '很': 1.2, '挺': 1.1, '比较': 0.9, '有点': 0.7, '稍微': 0.6,
            '超级': 1.6, '超': 1.5, '太': 1.4, '最': 1.7, '更': 1.2,
            '极其': 1.8, '极度': 1.7, '异常': 1.5, '格外': 1.4
        }
        
        # 停用词（用于文本清洗，注意：不包含否定词）
        self.stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '人', '都', '一', '一个',
            '上', '也', '到', '说', '要', '去', '你', '会', '着', '看',
            '自己', '这', '那', '他', '她', '它', '们', '个', '中', '为', '而',
            '与', '及', '或', '但', '如果', '因为', '所以', '虽然', '然而'
        }
    
    def load_model(self):
        """加载模型和词典"""
        try:
            # 使用CPU加载模型，避免GPU相关错误
            with tf.device('/CPU:0'):
                # 加载模型
                if os.path.exists(SENTIMENT_MODEL_PATH):
                    self.model = load_model(SENTIMENT_MODEL_PATH)
                    print(f"情感分析模型加载成功: {SENTIMENT_MODEL_PATH}")
                else:
                    print(f"警告: 情感分析模型文件不存在: {SENTIMENT_MODEL_PATH}")
                    self.model = None
            
            # 加载或创建词典
            if os.path.exists(SENTIMENT_DICT_PATH):
                with open(SENTIMENT_DICT_PATH, 'rb') as f:
                    self.dicts = pickle.load(f)
                print(f"情感分析词典加载成功: {SENTIMENT_DICT_PATH}")
            else:
                print(f"警告: 情感分析词典文件不存在: {SENTIMENT_DICT_PATH}")
                print("将使用简化版情感分析（基于关键词）")
                self.dicts = None
        except Exception as e:
            print(f"加载情感分析模型失败: {e}")
            self.model = None
            self.dicts = None
    
    def clean_text(self, text):
        """清洗文本：去除特殊字符、URL、数字等"""
        if not text:
            return ""
        
        # 去除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        # 去除邮箱
        text = re.sub(r'\S+@\S+', '', text)
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text)
        # 去除特殊符号（保留中文标点）
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、；：]', '', text)
        # 去除纯数字
        text = re.sub(r'\d+', '', text)
        
        return text.strip()
    
    def preprocess_text(self, text):
        """预处理文本"""
        if not text:
            return None
        
        try:
            # 清洗文本
            cleaned_text = self.clean_text(text)
            if not cleaned_text:
                return None
            
            # 分词
            words = list(jieba.cut(cleaned_text))
            
            # 去除停用词和空字符
            words = [w for w in words if w.strip() and w not in self.stop_words and len(w.strip()) > 0]
            
            if not words:
                return None
            
            if self.dicts is not None:
                # 使用训练时的词典
                word_ids = []
                for word in words:
                    if word in self.dicts.index:
                        word_ids.append(self.dicts.loc[word, 'id'])
                
                if not word_ids:
                    return None
                
                # 填充序列
                sent = sequence.pad_sequences([word_ids], maxlen=self.maxlen)
                return sent
            else:
                # 简化版：基于关键词的情感分析
                return None
        except Exception as e:
            print(f"文本预处理错误: {e}")
            return None
    
    def predict_with_keywords(self, text):
        """基于关键词的简化情感分析（考虑否定词和程度词）"""
        if not text:
            return {"sentiment": "中性", "confidence": 0.5}
        
        # 清洗文本
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return {"sentiment": "中性", "confidence": 0.5}
        
        # 分词
        words = list(jieba.cut(cleaned_text))
        words = [w for w in words if w.strip() and w not in self.stop_words]
        
        if not words:
            return {"sentiment": "中性", "confidence": 0.5}
        
        pos_score = 0.0
        neg_score = 0.0
        
        # 遍历每个词，考虑否定词和程度词的影响
        for i, word in enumerate(words):
            intensity = 1.0  # 默认强度
            negated = False  # 是否被否定
            
            # 检查前面是否有程度词（检查前1-2个词）
            for j in range(max(0, i-2), i):
                if words[j] in self.intensity_words:
                    intensity = self.intensity_words[words[j]]
                    break
            
            # 检查前面是否有否定词（检查前1-3个词，因为否定词可能距离较远）
            for j in range(max(0, i-3), i):
                if words[j] in self.negation_words:
                    negated = True
                    break
            
            # 计算情感分数
            if word in self.positive_words:
                score = 1.0 * intensity
                if negated:
                    neg_score += score  # 否定正面词 = 负面
                else:
                    pos_score += score
            
            elif word in self.negative_words:
                score = 1.0 * intensity
                if negated:
                    pos_score += score  # 否定负面词 = 正面
                else:
                    neg_score += score
        
        # 如果没有找到任何情感词，返回中性
        total_score = pos_score + neg_score
        if total_score == 0:
            return {"sentiment": "中性", "confidence": 0.5}
        
        # 计算置信度（基于分数差异和总分数）
        score_diff = abs(pos_score - neg_score)
        # 如果分数差异明显，置信度更高
        if total_score > 0:
            confidence = 0.5 + min(score_diff / total_score * 0.45, 0.45)
        else:
            confidence = 0.5
        
        # 判断情感（改进判断逻辑，降低阈值以提高准确性）
        # 如果负面分数明显大于正面分数，判定为负面
        if neg_score > pos_score * 1.2:  # 负面分数至少是正面的1.2倍
            return {"sentiment": "负面", "confidence": min(confidence, 0.95)}
        elif pos_score > neg_score * 1.2:  # 正面分数至少是负面的1.2倍
            return {"sentiment": "正面", "confidence": min(confidence, 0.95)}
        elif neg_score > 0 and pos_score == 0:
            # 只有负面词，没有正面词
            return {"sentiment": "负面", "confidence": min(confidence, 0.9)}
        elif pos_score > 0 and neg_score == 0:
            # 只有正面词，没有负面词
            return {"sentiment": "正面", "confidence": min(confidence, 0.9)}
        else:
            # 正面和负面词都存在，根据比例判断
            if neg_score > pos_score:
                return {"sentiment": "负面", "confidence": min(confidence, 0.85)}
            elif pos_score > neg_score:
                return {"sentiment": "正面", "confidence": min(confidence, 0.85)}
            else:
                return {"sentiment": "中性", "confidence": 0.5}
    
    def predict(self, text):
        """预测文本情感"""
        if not text or not text.strip():
            return {"sentiment": "中性", "confidence": 0.5, "method": "default"}
        
        # 如果模型不存在，使用关键词方法
        if self.model is None:
            result = self.predict_with_keywords(text)
            result["method"] = "keywords"
            return result
    
        try:
            # 预处理文本
            x_pad = self.preprocess_text(text)
            if x_pad is None:
                # 如果预处理失败，使用关键词方法
                result = self.predict_with_keywords(text)
                result["method"] = "keywords_fallback"
                return result
        
            # 使用模型预测
            with tf.device('/CPU:0'):
                y_pred = self.model.predict(x_pad, verbose=0)
        
            # 处理模型输出（根据训练代码，模型使用 sigmoid 输出，标签：1=正面，0=负面）
            # 模型输出形状可能是 (1, 1) 或 (1,)
            if len(y_pred.shape) == 2 and y_pred.shape[1] == 2:
                # 二分类 softmax 输出（如果模型被修改过）
                negative_prob = float(y_pred[0][0])  # 第一个类别（负面=0）
                positive_prob = float(y_pred[0][1])  # 第二个类别（正面=1）
                
                # 判断情感
                if positive_prob > negative_prob:
                    sentiment = "正面"
                    confidence = positive_prob
                else:
                    sentiment = "负面"
                    confidence = negative_prob
                
                # 获取关键词预测结果用于验证
                keyword_result = self.predict_with_keywords(text)
                
                # 如果模型置信度较低，或者模型预测与关键词预测不一致，需要谨慎处理
                model_uncertain = confidence < self.confidence_threshold or abs(positive_prob - negative_prob) < 0.15
                prediction_conflict = sentiment != keyword_result["sentiment"] and keyword_result["sentiment"] != "中性"
                
                if model_uncertain or prediction_conflict:
                    # 当模型不确定或与关键词预测冲突时，优先参考关键词结果
                    if prediction_conflict and keyword_result["confidence"] > 0.7:
                        # 如果关键词预测置信度高且与模型冲突，优先使用关键词结果
                        sentiment = keyword_result["sentiment"]
                        # 降低模型权重，提高关键词权重
                        combined_confidence = (confidence * 0.3 + keyword_result["confidence"] * 0.7)
                        confidence = combined_confidence
                        return {
                            "sentiment": sentiment,
                            "confidence": float(confidence),
                            "negative_prob": negative_prob,
                            "positive_prob": positive_prob,
                            "method": "model_keywords_combined",
                            "model_sentiment": "正面" if positive_prob > negative_prob else "负面",
                            "keyword_sentiment": keyword_result["sentiment"]
                        }
                    else:
                        # 模型不确定但无冲突，或关键词也不确定，使用加权平均
                        combined_confidence = (confidence * 0.4 + keyword_result["confidence"] * 0.6)
                        if abs(positive_prob - negative_prob) < 0.1:  # 概率接近时，参考关键词结果
                            sentiment = keyword_result["sentiment"]
                        confidence = combined_confidence
                
                return {
                    "sentiment": sentiment,
                    "confidence": float(confidence),
                    "negative_prob": negative_prob,
                    "positive_prob": positive_prob,
                    "method": "model"
                }
            else:
                # 处理 sigmoid 单值输出
                # 输出形状可能是 (1, 1) 或 (1,)
                if len(y_pred.shape) == 2:
                    sentiment_score = float(y_pred[0][0])  # 形状为 (1, 1)
                else:
                    sentiment_score = float(y_pred[0])  # 形状为 (1,)
            
            # 处理 sigmoid 输出（根据训练代码：1=正面，0=负面）
            # sentiment_score 接近 1 表示正面，接近 0 表示负面
            if sentiment_score >= 0.5:
                sentiment = "正面"
                confidence = sentiment_score
            else:
                sentiment = "负面"
                confidence = 1 - sentiment_score
            
            # 获取关键词预测结果用于验证
            keyword_result = self.predict_with_keywords(text)
            
            # 如果模型置信度较低，或者模型预测与关键词预测不一致，需要谨慎处理
            model_uncertain = confidence < self.confidence_threshold or abs(sentiment_score - 0.5) < 0.15
            prediction_conflict = sentiment != keyword_result["sentiment"] and keyword_result["sentiment"] != "中性"
            
            if model_uncertain or prediction_conflict:
                # 当模型不确定或与关键词预测冲突时，优先参考关键词结果
                # 特别是对于明显的负面词（如"伤心"、"难过"），关键词方法更可靠
                if prediction_conflict and keyword_result["confidence"] > 0.7:
                    # 如果关键词预测置信度高且与模型冲突，优先使用关键词结果
                    sentiment = keyword_result["sentiment"]
                    # 降低模型权重，提高关键词权重
                    combined_confidence = (confidence * 0.3 + keyword_result["confidence"] * 0.7)
                    confidence = combined_confidence
                    return {
                        "sentiment": sentiment,
                        "confidence": float(confidence),
                        "score": sentiment_score,
                        "method": "model_keywords_combined",
                        "model_sentiment": "正面" if sentiment_score >= 0.5 else "负面",
                        "keyword_sentiment": keyword_result["sentiment"]
                    }
                else:
                    # 模型不确定但无冲突，或关键词也不确定，使用加权平均
                    combined_confidence = (confidence * 0.4 + keyword_result["confidence"] * 0.6)
                    if abs(sentiment_score - 0.5) < 0.1:  # 接近中性时，参考关键词结果
                        sentiment = keyword_result["sentiment"]
                    confidence = combined_confidence
            
            return {
                "sentiment": sentiment,
                "confidence": float(confidence),
                "score": sentiment_score,
                "method": "model"
            }
        except Exception as e:
            print(f"模型预测错误: {e}")
            # 出错时回退到关键词方法
            result = self.predict_with_keywords(text)
            result["method"] = "keywords_error_fallback"
            return result

# 全局实例
sentiment_analyzer = SentimentAnalyzer()

