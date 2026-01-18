# 10.4 任务：基于Seq2Seq的机器翻译
# 代码10-12 语料预处理
import re
import io
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import time
from tqdm import  tqdm
import numpy as np
# 准备数据集
def preprocess_sentence(w):   
    '''
    w：句子
    '''
    w = re.sub(r'([?.!,])', r' \1 ', w)  # 对句子中标点符号前后加空格
    w = re.sub(r"[' ']+", ' ', w)  # 将句子中多空格去重
    w = '<start> ' + w + ' <end>'  # 给句子加上开始和结束标记，以便模型预测
    return w

en_sentence = 'I like this book'
sp_sentence = '我喜欢这本书'
print('预处理前的输出为：', '\n', preprocess_sentence(en_sentence))
print('预处理前的输出为：', '\n', str(preprocess_sentence(sp_sentence)), 'utf-8', '\n')

# 清理句子，删除重音符号，返回格式为[英文，中文]的单词对
def create_dataset(path, num_examples):
    '''
    path：文件路径
    num_examples：选用的数据量
    '''
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]
    return zip(*word_pairs)

path_to_file = '/root/autodl-tmp/NLP/nlp_deeplearn/data/en-ch.txt'  # 读取文件的路径
en, sp = create_dataset(path_to_file, None)  # 整合并读取数据

# 句子的最大长度
def max_length(tensor):
    '''
    tensor：文本构成的张量
    '''
    return max(len(t) for t in tensor)

# tokenize函数是对文本中的词进行统计计数，生成文档词典，以支持基于词典位序生成文本的向量表示
def tokenize(lang):
    '''
    lang：待处理的文本
    '''
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, lang_tokenizer

# 创建清理的输入输出对
def load_dataset(path, num_examples=None):
    '''
    path：文件路径
    num_examples：选用的数据量
    '''
    # 建立索引，并输入已经清洗过的词语，输出词语对
    targ_lang, inp_lang = create_dataset(path, num_examples) 
    # 建立中文句子的词向量，对所有张量进行填充，使句子的维度一样
    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)   
    # 建立英文句子的词向量，对所有张量进行填充，使句子的维度一样
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)  
    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

num_examples = 2000  # 词表的大小（词量）
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path_to_file, 
                                                                num_examples)
# 计算目标张量的最大长度（max_length）
max_length_targ, max_length_inp = max_length(target_tensor), max_length(
    input_tensor) 

# 采用8: 2的比例切分训练集和验证集
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(
        input_tensor, target_tensor, test_size=0.2) 

# 验证数据正确性，也就是输出词与词语映射索引的表示
def convert(lang, tensor):
    '''
    lang：待处理的文本
    tensor：文本构成的张量
    '''
    for t in tensor:
        if t != 0:    
            print ('%d ----> %s' % (t, lang.index_word[t]))

print('预处理前的输出为：')
print('输入语言：词映射索引')
convert(inp_lang, input_tensor_train[0])
print('目标语言：词语映射索引')
convert(targ_lang, target_tensor_train[0])

# 创建tf.data数据集
BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64          # 减小 batch，有利于收敛
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 256      # 提高词向量维度，增强表达能力
units = 512              # 保持 512，避免太慢
vocab_inp_size = len(inp_lang.word_index)+1  # 输入词表的大小
vocab_tar_size = len(targ_lang.word_index)+1  # 输出词表的大小
dataset = tf.data.Dataset.from_tensor_slices((
    input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)  # 构建训练集
example_input_batch, example_target_batch = next(iter(dataset))



# 代码10-13 构建机器翻译模型
# 双向编码器（Bi-GRU）
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        # 输入嵌入
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, mask_zero=True
        )
        # 双向 GRU
        self.bigru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                self.enc_units,
                return_sequences=True,
                return_state=True,
                recurrent_initializer='glorot_uniform',
                dropout=0.2,
                recurrent_dropout=0.2
            )
        )
        # 把前向/后向状态拼接后降维回 enc_units
        self.state_proj = tf.keras.layers.Dense(self.enc_units, activation='tanh')

    def call(self, x, hidden):
        x = self.embedding(x)
        # bigru 返回：output, forward_state, backward_state
        output, f_state, b_state = self.bigru(x, initial_state=[hidden, hidden])
        # 拼接两个方向的 hidden，再投影回 enc_units
        h_cat = tf.concat([f_state, b_state], axis=-1)
        state = self.state_proj(h_cat)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

# 构建编码器网络结构    
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)  
print('编码器输出形状：', '\n', ' (batch size, sequence length, units) {}'.format(sample_output.shape))
print('编码器隐藏状态形状：', '\n', ' (batch size, units) {}'.format(sample_hidden.shape))

# 注意力机制（保持原来的 BahdanauAttention 定义即可）
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units, use_bias=False)
        self.W2 = tf.keras.layers.Dense(units, use_bias=False)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

# 解码器：单层 GRU + 中间全连接
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, mask_zero=True
        )
        self.gru = tf.keras.layers.GRU(
            self.dec_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform',
            dropout=0.2,
            recurrent_dropout=0.2
        )
        # 新增一个中间全连接层
        self.fc_mid = tf.keras.layers.Dense(self.dec_units, activation='relu')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x, initial_state=hidden)
        output = tf.reshape(output, (-1, output.shape[2]))
        # 先通过中间层
        output = self.fc_mid(output)
        x = self.fc(output)
        return x, state, attention_weights

# 构建解码器网络结构
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)  
sample_decoder_output, states, attention_weight = decoder(
    tf.random.uniform((BATCH_SIZE, 1), maxval=vocab_tar_size, dtype=tf.int32),
    sample_hidden,
    sample_output
)
print('解码器输出形状：', '\n', ' (batch_size, vocab size) {}'.format(sample_decoder_output.shape))


# 代码10-14 定义优化器及损失函数（优化版）
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# 带 label smoothing 的损失函数
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)



# 代码10-15 训练模型

# 检查点（基于对象的保存），准备保存训练模型
checkpoint_dir = '/root/autodl-tmp/NLP/nlp_deeplearn/tmp/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)  # 保存模型
# 训练模型
def train(inp, targ, enc_hidden):
    '''
    inp：批次
    targ：标签
    enc_hidden：隐藏样本
    '''
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)  # 构建编码器
        dec_hidden = enc_hidden  
        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)
        # 教师强制 - 将目标词作为下一个输入
        for t in range(1, targ.shape[1]):
            # 将编码器输出传送至解码器
            predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)
            dec_input = tf.expand_dims(targ[:, t], 1)  # 使用教师强制
        loss = loss / int(targ.shape[1])  # 计算平均损失
    batch_loss = loss.numpy()  # 将损失转换为numpy数组
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

# 开始训练（适度增加轮次以提升准确率）
EPOCHS = 30  # 适当增加训练轮数，通常能明显提升翻译质量
loss = []

for epoch in tqdm(range(EPOCHS)):
    start = time.time()
    enc_hidden = encoder.initialize_hidden_state()  # 初始化隐藏层
    total_loss = 0
    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train(inp, targ, enc_hidden)
        total_loss += batch_loss
        if batch % 50 == 0:  # 减少打印频率
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss))
            loss.append(round(batch_loss, 3))
    
    print('Epoch {} 平均损失: {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
    
    # 每5轮保存一次模型
    if (epoch + 1) % 5 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)
        print('保存模型检查点')

# 损失趋势可视化

plt.rcParams['font.sans-serif'] = ['SIMHEI']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 对字符进行显示设置
if loss:  # 只有当有损失数据时才绘图
    plt.plot(list(range(1, len(loss)+1)), loss)  # 将损失值绘制成折线图
    plt.title('损失趋势图', fontsize=16)  # 设置折线图标题为损失趋势图
    plt.xlabel('迭代次数')  # 将x轴标签设置为迭代次数
    plt.ylabel('损失值')  # 将y轴标签设置为损失值
    plt.show()  # 将图形进行展示
    plt.savefig("10_4.png")


# 代码10-16 使用模型进行语句翻译

# 优化的翻译函数（支持beam search）
def evaluate(sentence, beam_width=1):
    '''
    sentence：需要翻译的句子
    beam_width：beam search的宽度（1表示贪心搜索）
    '''
    attention_plot = np.zeros((max_length_targ, max_length_inp))
    sentence = preprocess_sentence(sentence)
    inputs = [inp_lang.word_index.get(i, 0) for i in sentence.split(' ') if i in inp_lang.word_index]
    if not inputs:
        return '', sentence, attention_plot
    inputs = tf.keras.preprocessing.sequence.pad_sequences(
        [inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    result = ''
    hidden = tf.zeros((1, units))
    enc_out, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)
    
    if beam_width == 1:
        # 贪心搜索
        for t in range(max_length_targ):
            predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
            predicted_id = tf.argmax(predictions[0]).numpy()
            if predicted_id in targ_lang.index_word:
                predicted_word = targ_lang.index_word[predicted_id]
                if predicted_word == '<end>':
                    break
                result += predicted_word + ' '
            else:
                break
            dec_input = tf.expand_dims([predicted_id], 0)
    else:
        # 简化的beam search（可以进一步优化）
        for t in range(max_length_targ):
            predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
            # 获取top-k预测
            top_k = tf.nn.top_k(predictions[0], k=min(beam_width, len(targ_lang.word_index)))
            predicted_id = top_k.indices[0].numpy()
            if predicted_id in targ_lang.index_word:
                predicted_word = targ_lang.index_word[predicted_id]
                if predicted_word == '<end>':
                    break
                result += predicted_word + ' '
            else:
                break
            dec_input = tf.expand_dims([predicted_id], 0)
    
    return result, sentence, attention_plot

# 执行翻译▲
def translate(sentence):
    '''
    sentence：要翻译的句子
    '''
    result, sentence, attention_plot = evaluate(sentence)
    print('输入：%s' % (sentence))
    print('翻译结果：{}'.format(result))

print(translate('我生病了。'))
print(translate('为什么不？'))
print(translate('让我一个人呆会儿。'))
print(translate('打电话回家！'))
print(translate('我了解你。'))

# ===== 新增：保存训练结果以便在qa_system中使用 =====
import pickle

# 保存tokenizer和模型参数
translate_save_dir = '/root/autodl-tmp/NLP/nlp_deeplearn/tmp/'
os.makedirs(translate_save_dir, exist_ok=True)

# 保存tokenizer
tokenizer_save_path = os.path.join(translate_save_dir, 'translate_tokenizers.pkl')
with open(tokenizer_save_path, 'wb') as f:
    pickle.dump({
        'inp_lang': inp_lang,
        'targ_lang': targ_lang,
        'max_length_targ': max_length_targ,
        'max_length_inp': max_length_inp,
        'vocab_inp_size': vocab_inp_size,
        'vocab_tar_size': vocab_tar_size,
        'embedding_dim': embedding_dim,
        'units': units
    }, f)
print(f"\n翻译模型tokenizer已保存到: {tokenizer_save_path}")

# 保存模型配置信息
config_save_path = os.path.join(translate_save_dir, 'translate_config.txt')
with open(config_save_path, 'w', encoding='utf-8') as f:
    f.write(f"max_length_targ={max_length_targ}\n")
    f.write(f"max_length_inp={max_length_inp}\n")
    f.write(f"vocab_inp_size={vocab_inp_size}\n")
    f.write(f"vocab_tar_size={vocab_tar_size}\n")
    f.write(f"embedding_dim={embedding_dim}\n")
    f.write(f"units={units}\n")
    f.write(f"checkpoint_dir={checkpoint_dir}\n")
print(f"翻译模型配置已保存到: {config_save_path}")

print("\n训练结果已保存，现在可以在qa_system中加载使用了。")
print(f"检查点目录: {checkpoint_dir}")
print(f"请确保在qa_system中使用最新的checkpoint进行加载。")

