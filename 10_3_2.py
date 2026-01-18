# 10.3.2 情感分析
# 代码10-7 读取语料数据
import pandas as pd
from tensorflow.keras.preprocessing import sequence
import jieba
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras import Input
import tensorflow as tf
import time
from sklearn import metrics
import pickle
import os

# 配置 GPU 和 cuDNN 支持
# 检查 GPU 是否可用
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 启用 GPU 内存增长，避免一次性分配所有内存
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"检测到 {len(gpus)} 个 GPU，已启用 GPU 加速和 cuDNN 支持")
        print(f"GPU 设备: {gpus}")
    except RuntimeError as e:
        print(f"GPU 配置错误: {e}")
else:
    print("未检测到 GPU，将使用 CPU 运行")

# 定义路径（与配置文件保持一致）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NLP_DEEPLEARN_DIR = os.path.join(os.path.dirname(BASE_DIR), 'nlp_deeplearn')

SENTIMENT_MODEL_PATH = os.path.join(NLP_DEEPLEARN_DIR, 'tmp', 'sentiment_model.h5')
SENTIMENT_DICT_PATH = os.path.join(NLP_DEEPLEARN_DIR, 'tmp', 'sentiment_dict.pkl')

# 确保保存目录存在
os.makedirs(os.path.dirname(SENTIMENT_MODEL_PATH), exist_ok=True)

# 读取正负情感语料
neg = pd.read_excel('/root/autodl-tmp/NLP/nlp_deeplearn/data/neg.xls', header=None, index_col=None)
pos = pd.read_excel('/root/autodl-tmp/NLP/nlp_deeplearn/data/pos.xls', header=None, index_col=None)

# 给训练语料贴标签
pos['mark'] = 1
neg['mark'] = 0

# 代码10-8 词语向量化


# 分词
cut_word = lambda x: list(jieba.cut(x))  # 定义分词函数
pn_all = pd.concat([pos, neg], ignore_index=True)  # 合并正负情感语料
# 先确保所有输入都是字符串类型
pn_all[0] = pn_all[0].astype(str)

# 然后应用分词
cut_word = lambda x: list(jieba.cut(x))  # 定义分词函数
pn_all['words'] = pn_all[0].apply(cut_word)  # 对情感语料分词
comment = pd.read_excel('/root/autodl-tmp/NLP/nlp_deeplearn/data/sum.xls')  # 读入评论内容,增加语料
comment = comment[comment['rateContent'].notnull()]  # 仅读取非空评论
comment['words'] = comment['rateContent'].apply(cut_word)  # 对评论语料分词
pn_comment = pd.concat([pn_all['words'], comment['words']], ignore_index=True)  # 合并所有的数据

# 正负情感评论词语向量化
w = [] 
for i in pn_comment:
    w.extend(i)    
dicts = pd.DataFrame(pd.Series(w).value_counts())  # 建立统计词典
del w, pn_comment  # 删除临时文件 w，d2v_train
dicts['id'] = list(range(1, len(dicts)+1))
get_sent = lambda x: list(dicts['id'][x])
pn_all['sent'] = pn_all['words'].apply(get_sent)

# 评论词语向量标准化，对样本进行padding填充和truncating修剪
maxlen = 50  # 设置评论词语最大长度
pn_all['sent'] = list(sequence.pad_sequences(pn_all['sent'], maxlen=maxlen))  # 正负情感评论词语向量化

# 训练集、测试集
x_all = np.array(list(pn_all['sent']))
y_all = np.array(list(pn_all['mark']))
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.25)

print('训练集的特征数据形状为：', x_train.shape)
print('训练集的标签数据形状为：', y_train.shape)
print('测试集的特征数据形状为：', x_test.shape)
print('测试集的标签数据形状为：', y_test.shape)
print('训练集的特征数据为：\n', x_train)

# 代码10-9 模型构建


# 搭建简化的LSTM模型（单层双向LSTM）
model = Sequential()
model.add(Input(shape=(50,)))
# 嵌入层（降低维度）
model.add(Embedding(len(dicts)+1, 128, mask_zero=True))
model.add(Dropout(0.2))

# 单层双向LSTM（优化以使用 cuDNN 内核）
# 注意：cuDNN 内核要求：
# 1. 使用默认的 activation='tanh' 和 recurrent_activation='sigmoid'
# 2. 不使用 unroll=True
# 3. 不使用 go_backwards=True（在 Bidirectional 中会自动处理）
# 4. dropout 和 recurrent_dropout 在 cuDNN 中会被正确处理
model.add(tf.keras.layers.Bidirectional(
    LSTM(64, dropout=0.2, recurrent_dropout=0.2, 
         # 确保使用 cuDNN 兼容的参数
         activation='tanh', recurrent_activation='sigmoid')
))

# 简化的全连接层
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

# 输出层
model.add(Dense(1))
model.add(Activation('sigmoid'))

# 打印模型摘要
model.summary()

# 验证 cuDNN 支持
print("\n=== cuDNN 支持检查 ===")
if gpus:
    print("✓ GPU 已启用")
    # 检查 TensorFlow 是否编译了 cuDNN 支持
    if tf.test.is_built_with_cuda():
        print("✓ TensorFlow 已编译 CUDA 支持")
        # 检查 cuDNN 版本
        try:
            cudnn_version = tf.sysconfig.get_build_info().get('cudnn_version', '未知')
            print(f"✓ cuDNN 版本: {cudnn_version}")
        except:
            print("⚠ 无法获取 cuDNN 版本信息")
        print("✓ LSTM 层将自动使用 cuDNN 内核加速（如果可用）")
    else:
        print("⚠ TensorFlow 未编译 CUDA 支持，将使用 CPU")
else:
    print("⚠ 未检测到 GPU，将使用 CPU 运行")
print("=" * 30 + "\n")

# 代码10-10 模型训练（优化版 - 使用 cuDNN，无早停机制）

# 导入回调函数
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# 设置超参（使用Adam优化器，学习率自适应）
model.compile(
    loss='binary_crossentropy', 
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# 回调函数（移除早停机制，只保留模型检查点和学习率调整）
callbacks = [
    ModelCheckpoint(
        filepath=SENTIMENT_MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1,
        save_weights_only=False
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]

# 训练模型（简化训练轮次）
from sklearn.model_selection import train_test_split
x_train_fit, x_val_fit, y_train_fit, y_val_fit = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)

timeA = time.time()
# 训练模型（无早停机制，训练完整轮数）
# 由于使用 cuDNN 加速，可以适当增加训练轮数
history = model.fit(
    x_train_fit, y_train_fit, 
    batch_size=64,  # 增大batch size以加快训练（cuDNN 对大批次更高效）
    epochs=20,  # 增加训练轮数（因为移除了早停机制）
    validation_data=(x_val_fit, y_val_fit),
    callbacks=callbacks,
    verbose=1
) 
timeB = time.time()
print('训练耗时: ', int(timeB-timeA), '秒')

# 代码10-11 模型测试

# 使用测试数据进行模型测试
y_pred = model.predict(x_test).round().astype(int)
# 模型评价
acc = metrics.accuracy_score(y_test, y_pred)
print('测试集的准确率为：', acc)
print('精确率，召回率，F1值分别为：')
print(metrics.classification_report(y_test, y_pred))
print('混淆矩阵为：')
cm = metrics.confusion_matrix(y_test, y_pred)  # 混淆矩阵
print(cm)

# ===== 保存训练结果 =====
# 模型已在回调函数中自动保存最佳版本
# 这里再保存一次最终模型作为备份
final_model_path = SENTIMENT_MODEL_PATH.replace('.h5', '_final.h5')
print(f"\n正在保存最终模型到：{final_model_path}")
model.save(final_model_path)
print("最终模型保存成功！")

print(f"\n正在保存词典到：{SENTIMENT_DICT_PATH}")
with open(SENTIMENT_DICT_PATH, 'wb') as f:
    pickle.dump(dicts, f)
print("词典保存成功！")

print("\n训练结果已保存，现在可以在预测端加载使用了。")