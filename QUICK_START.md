# 快速开始指南

## 一、环境准备

### 1. 检查 Python 版本
```bash
python3 --version  # 需要 Python 3.8+
```

### 2. 安装依赖
```bash
cd /root/autodl-tmp/NLP/qa_system
pip install -r requirements.txt
```

## 二、配置检查

### 1. 检查模型文件
```bash
# 检查文本分类模型
ls -lh ../nlp_deeplearn/tmp/my_model.h5
# 或
ls -lh ../nlp_deeplearn/tmp/best_validation_best.h5

# 检查情感分析模型
ls -lh ../nlp_deeplearn/tmp/sentiment_best_model.h5

# 检查词汇表
ls -lh ../nlp_deeplearn/data/cnews.vocab.txt
```

### 2. 检查豆包API配置
编辑 `config.py`，确认：
- `DOUBAO_API_KEY` 是否正确
- `DOUBAO_MODEL` 是否正确

## 三、启动服务

### 方式一：使用启动脚本（推荐）
```bash
./start.sh
```

### 方式二：直接运行
```bash
python3 app.py
```

### 方式三：后台运行
```bash
nohup python3 app.py > app.log 2>&1 &
```

## 四、访问系统

### 本地访问
打开浏览器访问：`http://localhost:5000`

### 外网访问
1. 获取服务器IP地址：
   ```bash
   hostname -I
   ```

2. 访问：`http://YOUR_IP:5000`

3. 如果无法访问，请参考 `PORT_MAPPING.md` 配置端口映射

## 五、测试接口

### 使用测试脚本
```bash
python3 test_api.py
```

### 使用 curl 测试
```bash
# 健康检查
curl http://localhost:5000/api/health

# 智能问答
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "你好"}'

# 文本分类
curl -X POST http://localhost:5000/api/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "今天股市大涨"}'

# 情感分析
curl -X POST http://localhost:5000/api/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "这个产品很好用"}'
```

## 六、功能使用

### 1. 智能问答
- 在"智能问答"标签页输入问题
- 点击"发送"或按 Enter 键
- 系统会调用豆包API返回答案

### 2. 文本分类
- 切换到"文本分类"标签页
- 输入要分类的文本
- 点击"分类"按钮
- 查看分类结果和概率分布

### 3. 情感分析
- 切换到"情感分析"标签页
- 输入要分析的文本
- 点击"分析"按钮
- 查看情感倾向和置信度

### 4. 机器翻译
- 切换到"机器翻译"标签页
- 选择翻译方向（中文→英文 或 英文→中文）
- 输入要翻译的文本
- 点击"翻译"按钮

### 5. 创意功能
- 切换到"创意功能"标签页
- 输入文本
- 点击相应功能按钮：
  - 🔑 提取关键词
  - 📄 文本摘要
  - 📊 词频统计
  - 📈 文本统计
  - 🌍 语言检测

## 七、常见问题

### Q1: 模型加载失败
**A:** 检查模型文件路径是否正确，参考 `config.py` 中的路径配置

### Q2: 豆包API调用失败
**A:** 
- 检查网络连接
- 验证API密钥是否正确
- 查看控制台错误信息

### Q3: 端口被占用
**A:** 
```bash
# 查看端口占用
sudo lsof -i :5000

# 修改 config.py 中的端口号
```

### Q4: 无法从外网访问
**A:** 
- 检查防火墙设置
- 检查云服务器安全组
- 参考 `PORT_MAPPING.md` 配置端口映射

### Q5: 前端页面无法加载
**A:** 
- 检查 Flask 是否正常运行
- 查看浏览器控制台错误
- 确认静态文件路径正确

## 八、性能优化建议

1. **使用 Gunicorn**（生产环境）
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

2. **启用缓存**（可选）
   ```bash
   pip install flask-caching
   ```

3. **使用 Nginx 反向代理**（生产环境）
   参考 `PORT_MAPPING.md`

## 九、日志查看

### 查看应用日志
```bash
# 如果使用 nohup 启动
tail -f app.log

# 如果使用 systemd
sudo journalctl -u qa_system -f
```

### 查看错误信息
检查控制台输出的错误信息，或查看日志文件

## 十、停止服务

### 前台运行
按 `Ctrl+C` 停止

### 后台运行
```bash
# 查找进程
ps aux | grep app.py

# 停止进程
kill <PID>
```

### systemd 服务
```bash
sudo systemctl stop qa_system
```

## 十一、更新系统

1. 备份当前配置
2. 更新代码
3. 重新安装依赖（如有新依赖）
4. 重启服务

## 十二、技术支持

如遇问题，请检查：
1. Python 版本是否兼容
2. 依赖包是否完整安装
3. 模型文件是否存在
4. 配置文件是否正确
5. 网络连接是否正常

更多详细信息请参考 `README.md`

