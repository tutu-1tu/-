# 创意功能模块
import re
import jieba
import jieba.analyse
from collections import Counter

class CreativeFeatures:
    @staticmethod
    def extract_keywords(text, topK=5):
        """提取关键词"""
        try:
            # 使用 TF-IDF 提取关键词
            keywords = jieba.analyse.extract_tags(text, topK=topK, withWeight=True)
            return {
                "keywords": [{"word": word, "weight": float(weight)} for word, weight in keywords],
                "top_keywords": [word for word, _ in keywords]
            }
        except Exception as e:
            return {"error": f"关键词提取失败: {str(e)}"}
    
    @staticmethod
    def text_summary(text, max_length=100):
        """文本摘要（简化版）"""
        try:
            if len(text) <= max_length:
                return {"summary": text, "original_length": len(text), "summary_length": len(text)}
            
            # 简单摘要：取前几句和后几句
            sentences = re.split(r'[。！？\n]', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) <= 2:
                summary = text[:max_length] + "..."
            else:
                # 取前2句和后1句
                summary = "。".join(sentences[:2])
                if len(sentences) > 3:
                    summary += "。" + sentences[-1]
                if len(summary) > max_length:
                    summary = summary[:max_length] + "..."
            
            return {
                "summary": summary,
                "original_length": len(text),
                "summary_length": len(summary),
                "compression_ratio": round(len(summary) / len(text), 2)
            }
        except Exception as e:
            return {"error": f"文本摘要失败: {str(e)}"}
    
    @staticmethod
    def word_frequency(text, topN=10):
        """词频统计"""
        try:
            words = list(jieba.cut(text))
            # 过滤停用词和标点
            stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', 
                         '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', 
                         '自己', '这', '，', '。', '？', '！', '、', '；', '：', '"', "'", '（', '）'}
            words = [w for w in words if w.strip() and w not in stop_words and len(w) > 1]
            
            word_freq = Counter(words)
            top_words = word_freq.most_common(topN)
            
            return {
                "word_frequency": [{"word": word, "count": count} for word, count in top_words],
                "total_words": len(words),
                "unique_words": len(word_freq)
            }
        except Exception as e:
            return {"error": f"词频统计失败: {str(e)}"}
    
    @staticmethod
    def text_statistics(text):
        """文本统计信息"""
        try:
            stats = {
                "character_count": len(text),
                "character_count_no_spaces": len(text.replace(' ', '')),
                "word_count": len(list(jieba.cut(text))),
                "sentence_count": len(re.split(r'[。！？\n]', text)),
                "paragraph_count": len([p for p in text.split('\n') if p.strip()]),
                "avg_sentence_length": 0,
                "avg_word_length": 0
            }
            
            sentences = [s for s in re.split(r'[。！？\n]', text) if s.strip()]
            if sentences:
                stats["avg_sentence_length"] = round(sum(len(s) for s in sentences) / len(sentences), 2)
            
            words = list(jieba.cut(text))
            if words:
                stats["avg_word_length"] = round(sum(len(w) for w in words) / len(words), 2)
            
            return stats
        except Exception as e:
            return {"error": f"文本统计失败: {str(e)}"}
    
    @staticmethod
    def detect_language(text):
        """检测语言（简化版）"""
        try:
            # 简单检测：中文字符比例
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
            english_chars = len(re.findall(r'[a-zA-Z]', text))
            total_chars = len(re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]', text))
            
            if total_chars == 0:
                return {"language": "未知", "confidence": 0}
            
            chinese_ratio = chinese_chars / total_chars
            english_ratio = english_chars / total_chars
            
            if chinese_ratio > 0.5:
                return {"language": "中文", "confidence": chinese_ratio}
            elif english_ratio > 0.5:
                return {"language": "英文", "confidence": english_ratio}
            else:
                return {"language": "混合", "confidence": 0.5}
        except Exception as e:
            return {"error": f"语言检测失败: {str(e)}"}

# 全局实例
creative_features = CreativeFeatures()

