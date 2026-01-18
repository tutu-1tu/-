# Flask 主应用
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.doubao_api import doubao_api
from models.text_classify import text_classifier
from models.sentiment_analysis import sentiment_analyzer
from models.translator import translator
from utils.creative_features import creative_features
from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG

app = Flask(__name__)
CORS(app)  # 允许跨域请求

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """豆包API对话接口"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        system_prompt = data.get('system_prompt', 'You are a helpful assistant.')
        conversation_history = data.get('history', None)
        
        if not message:
            return jsonify({"success": False, "error": "消息不能为空"}), 400
        
        result = doubao_api.chat(message, system_prompt, conversation_history)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/classify', methods=['POST'])
def classify():
    """文本分类接口"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "文本不能为空"}), 400
        
        result = text_classifier.predict(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/sentiment', methods=['POST'])
def sentiment():
    """情感分析接口"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "文本不能为空"}), 400
        
        result = sentiment_analyzer.predict(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/translate', methods=['POST'])
def translate():
    """机器翻译接口"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        direction = data.get('direction', 'zh2en')  # zh2en 或 en2zh
        
        if not text:
            return jsonify({"error": "文本不能为空"}), 400
        
        result = translator.translate(text, direction)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/keywords', methods=['POST'])
def keywords():
    """关键词提取接口"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        topK = data.get('topK', 5)
        
        if not text:
            return jsonify({"error": "文本不能为空"}), 400
        
        result = creative_features.extract_keywords(text, topK)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/summary', methods=['POST'])
def summary():
    """文本摘要接口"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        max_length = data.get('max_length', 100)
        
        if not text:
            return jsonify({"error": "文本不能为空"}), 400
        
        result = creative_features.text_summary(text, max_length)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/wordfreq', methods=['POST'])
def wordfreq():
    """词频统计接口"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        topN = data.get('topN', 10)
        
        if not text:
            return jsonify({"error": "文本不能为空"}), 400
        
        result = creative_features.word_frequency(text, topN)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/statistics', methods=['POST'])
def statistics():
    """文本统计接口"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "文本不能为空"}), 400
        
        result = creative_features.text_statistics(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/detect_language', methods=['POST'])
def detect_language():
    """语言检测接口"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "文本不能为空"}), 400
        
        result = creative_features.detect_language(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """健康检查接口"""
    # 检查翻译服务状态（支持双向翻译）
    translation_status = "unavailable"
    if translator.model_loaded_zh2en or translator.model_loaded_en2zh:
        translation_status = "available"
    elif translator.encoder_zh2en or translator.encoder_en2zh:
        translation_status = "available"
    else:
        translation_status = "simplified"
    
    return jsonify({
        "status": "healthy",
        "services": {
            "doubao_api": "available",
            "text_classify": "available" if text_classifier.model else "unavailable",
            "sentiment_analysis": "available" if sentiment_analyzer.model else "unavailable",
            "translation": translation_status,
            "translation_zh2en": "available" if translator.model_loaded_zh2en else "unavailable",
            "translation_en2zh": "available" if translator.model_loaded_en2zh else "unavailable"
        }
    })

if __name__ == '__main__':
    import socket
    
    def is_port_available(port, host='0.0.0.0'):
        """检查端口是否可用"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((host, port))
                return True
        except OSError:
            return False
    
    def find_available_port(start_port, max_attempts=10):
        """查找可用端口"""
        for i in range(max_attempts):
            port = start_port + i
            if is_port_available(port):
                return port
        return None
    
    # 检查端口是否可用
    port = FLASK_PORT
    if not is_port_available(port):
        print(f"⚠️  警告: 端口 {port} 已被占用")
        print("正在查找可用端口...")
        
        # 尝试查找可用端口
        available_port = find_available_port(port)
        if available_port:
            port = available_port
            print(f"✓ 找到可用端口: {port}")
        else:
            print(f"❌ 错误: 无法找到可用端口（尝试了 {port} 到 {port + 9}）")
            print("请手动停止占用端口的进程，或修改配置文件中的端口号")
            print("\n提示：可以使用以下命令查找并停止占用端口的进程：")
            print(f"  ps aux | grep app.py | grep -v grep")
            print(f"  kill <进程ID>")
            exit(1)
    
    print(f"\n{'='*60}")
    print(f"启动 Flask 服务器...")
    print(f"访问地址: http://{FLASK_HOST}:{port}")
    print(f"内网访问: http://localhost:{port}")
    print(f"{'='*60}\n")
    
    try:
        app.run(host=FLASK_HOST, port=port, debug=FLASK_DEBUG)
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"\n❌ 错误: 端口 {port} 仍然被占用")
            print("请使用以下命令停止占用端口的进程：")
            print(f"  ps aux | grep app.py | grep -v grep")
            print(f"  kill <进程ID>")
        else:
            print(f"\n❌ 启动失败: {e}")
        exit(1)

