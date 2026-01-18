# 10.3.1 文本分类
# 代码10-1 自定义语料预处理函数
import tensorflow as tf
from collections import Counter
from tensorflow import keras
import numpy as np
import seaborn as sns
from keras.models import load_model
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import os
# 打开文件
def open_file(filename, mode='r'):
    '''
    filename：表示读取/写入的文件路径
    mode：'r' or 'w'表示读取/写入文件
    '''
    return open(filename, mode, encoding='utf-8', errors='ignore')
# 读取文件数据
def read_file(filename):
    '''
    filename：表示文件路径
    '''
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')  # 按照制表符分割字符串
                if content:
                    contents.append(list(content))
                    labels.append(label)
            except:
                pass
    return contents, labels
# 构建词汇表
def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    '''
    train_dir：训练集文件的存放路径
    vocab_dir：词汇表的存放路径
    vocab_size：词汇表的大小
    '''
    data_train, lab = read_file(train_dir)
    all_data = []
    for content in data_train:
        all_data.extend(content)
    counter = Counter(all_data)  # 词袋
    count_pairs = counter.most_common(vocab_size - 1)  # top n
    words, temp = list(zip(*count_pairs))  # 获取key
    words = ['<PAD>'] + list(words)  # 添加一个<PAD>将所有文本pad为同一长度
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')
# 读取词汇表
def read_vocab(vocab_dir):
    '''
    vocab_dir：词汇表的存放路径
    '''
    with open_file(vocab_dir) as fp:
        words = [i.strip() for i in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id
# 读取分类目录
def read_category():
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    # 得到类别与编号相对应的字典，分别为0-9
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id
# 将id表示的内容转换为文字
def to_words(content, words):
    '''
    content：id表示的内容
    words：文本内容
    '''
    return ''.join(words[x] for x in content)
# 将文件转换为id表示
def process_file(filename, word_to_id, cat_to_id, max_length=600):
    '''
    filename：文件路径
    word_to_id：词汇表
    cat_to_id：类别对应的编号
    max_length：词向量的最大长度
    '''
    contents, labels = read_file(filename)
    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])
    # 使用Keras提供的pad_sequences将文本pad为固定长度
    x_pad = keras.preprocessing.sequence.pad_sequences(data_id, max_length)
    # 将标签转为独热编码（one-hot）表示
    y_pad = keras.utils.to_categorical(label_id, num_classes=len(cat_to_id))
    return x_pad, y_pad


# 代码10-2 加载数据并进行预处理

# 设置数据读取、模型、结果保存路径
base_dir = '/root/autodl-tmp/NLP/nlp_deeplearn/data/'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')
save_dir = '/root/autodl-tmp/NLP/nlp_deeplearn/tmp/'
save_path = os.path.join(save_dir, 'best_validation')

# 若不存在词汇表，则重新建立词汇表
vocab_size = 5000
if not os.path.exists(vocab_dir):
    build_vocab(train_dir, vocab_dir, vocab_size)

# 读取分类目录
categories, cat_to_id = read_category()
# 读取词汇表
words, word_to_id = read_vocab(vocab_dir)
# 词汇表大小
vocab_size = len(words)

# 数据加载
seq_length = 600  # 序列长度

# 获取训练数据
x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, seq_length)
# 获取验证数据
x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, seq_length)
# 获取测试数据
x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, seq_length)

# 代码10-3 设置模型参数并构建模型


# 搭建简化的LSTM模型（单层双向LSTM）
def TextRNN():
    model = tf.keras.Sequential()
    # 嵌入层（降低维度以加快训练）
    model.add(tf.keras.layers.Embedding(vocab_size+1, 128, input_length=600, mask_zero=True))
    model.add(tf.keras.layers.Dropout(0.2))
    
    # 单层双向LSTM（简化结构）
    model.add(tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2)
    ))
    
    # 简化的全连接层
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    
    # 输出层
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model

# 代码10-4 模型训练（优化版）

# 使用回调函数保存最佳模型
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# 创建保存目录
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 简化的回调函数
callbacks = [
    # 保存最佳验证准确率的模型
    ModelCheckpoint(
        filepath=os.path.join(save_dir, 'best_model.h5'),
        monitor='val_categorical_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    # 早停机制（减少patience以加快训练）
    EarlyStopping(
        monitor='val_categorical_accuracy',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
]

# 训练参数设置（使用Adam优化器，性能更好）
try:
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    with strategy.scope():
        model = TextRNN()
        # 使用Adam优化器，学习率衰减
        model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=['categorical_accuracy']
        )
except:
    # 如果多GPU策略失败，使用单GPU或CPU
    model = TextRNN()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['categorical_accuracy']
    )

# 模型训练（简化训练轮次）
history = model.fit(
    x_train, y_train, 
    batch_size=128,  # 增大batch size以加快训练
    epochs=10,  # 减少训练轮数
    validation_data=(x_val, y_val),
    callbacks=callbacks,
    verbose=1
)
# 设置绘图的字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SIMHEI']
# 绘制训练过程
def plot_acc_loss(history):
    '''
    history：模型训练的返回值
    '''
    plt.subplot(121)
    plt.title('准确率趋势图')
    epochs_trained = len(history.history['categorical_accuracy'])
    plt.plot(range(1, epochs_trained+1), history.history['categorical_accuracy'], linestyle='-', color='g', label='训练集')
    plt.plot(range(1, epochs_trained+1), history.history['val_categorical_accuracy'], linestyle='-.', color='b', label='验证集')
    plt.legend(loc='best')  # 设置图例
    # x轴按1刻度显示
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)  
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.xlabel('迭代次数')
    plt.ylabel('准确率')
    plt.subplot(122)
    plt.title('损失趋势图')
    epochs_trained = len(history.history['loss'])
    plt.plot(range(1, epochs_trained+1), history.history['loss'], linestyle='-', color='g', label='训练集')
    plt.plot(range(1, epochs_trained+1), history.history['val_loss'], linestyle='-.', color='b', label='验证集')
    plt.legend(loc='best')
    # x轴按1刻度显示
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)  
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.tight_layout()
    plt.show()
    plt.savefig("3.png")
plot_acc_loss(history)

# 代码10-5 查看模型架构并保存模型
model.summary()
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 保存最终模型
final_model_path = os.path.join(save_dir, 'my_model.h5')
model.save(final_model_path)
print(f"最终模型已保存到: {final_model_path}")

# 如果存在最佳模型，也加载它用于测试
best_model_path = os.path.join(save_dir, 'best_model.h5')
if os.path.exists(best_model_path):
    print(f"使用最佳模型进行测试: {best_model_path}")
    model1 = load_model(best_model_path)
else:
    print(f"使用最终模型进行测试: {final_model_path}")
    model1 = model

# 代码10-6 模型测试

# 对测试集进行预测
y_pre = model1.predict(x_test)
# 计算混淆矩阵
confm = confusion_matrix(np.argmax(y_pre, axis=1), np.argmax(y_test, axis=1))
# 打印模型评价
print(classification_report(np.argmax(y_pre, axis=1), np.argmax(y_test, axis=1)))

# 混淆矩阵可视化
plt.figure(figsize=(8, 8), dpi=600)
# 设置绘图的字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SIMHEI']
sns.heatmap(confm.T, square=True, annot=True,
            fmt='d', cbar=False, linewidths=.8,
            cmap='YlGnBu')
plt.xlabel('真实标签', size=14)
plt.ylabel('预测标签', size=14)
plt.xticks(np.arange(10)+0.5, categories, size=12)
plt.yticks(np.arange(10)+0.3, categories, size=12)
plt.show()
plt.savefig("1.png")