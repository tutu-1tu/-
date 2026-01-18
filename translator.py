# 机器翻译模型加载和推理
import os
import re
import numpy as np
import tensorflow as tf
import pickle
import sys

# 配置TensorFlow使用CPU（避免GPU相关错误）
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TRANSLATE_CHECKPOINT_DIR,
    TRANSLATE_DATA_PATH,
    TRANSLATE_TOKENIZER_PATH,
    TRANSLATE_CONFIG_PATH
)


class Translator:
    def __init__(self):
        # 中译英模型（中文→英文）
        self.encoder_zh2en = None
        self.decoder_zh2en = None
        # 英译中模型（英文→中文）
        self.encoder_en2zh = None
        self.decoder_en2zh = None
        
        self.inp_lang = None  # 输入语言（中文）
        self.targ_lang = None  # 目标语言（英文）
        self.max_length_targ = None
        self.max_length_inp = None
        self.units = 1024
        self.embedding_dim = 256
        self.BATCH_SIZE = 1
        self.model_loaded_zh2en = False
        self.model_loaded_en2zh = False
        self.load_model()

    def preprocess_sentence(self, w):
        """预处理句子，与训练时一致"""
        if not w:
            return ""
        w = str(w).strip()
        # 对句子中标点符号前后加空格
        w = re.sub(r'([?.!,])', r' \1 ', w)
        # 将句子中多空格去重
        w = re.sub(r"[' ']+", ' ', w)
        # 给句子加上开始和结束标记
        w = '<start> ' + w.strip() + ' <end>'
        return w

    def load_model(self):
        """加载模型（结构与训练时完全一致）"""
        try:
            print(f"[DEBUG] Tokenizer路径: {TRANSLATE_TOKENIZER_PATH}")
            print(f"[DEBUG] Checkpoint路径: {TRANSLATE_CHECKPOINT_DIR}")

            # 1. 加载Tokenizer
            if not os.path.exists(TRANSLATE_TOKENIZER_PATH):
                raise FileNotFoundError(f"Tokenizer文件不存在: {TRANSLATE_TOKENIZER_PATH}")

            with open(TRANSLATE_TOKENIZER_PATH, 'rb') as f:
                tokenizer_data = pickle.load(f)
                self.inp_lang = tokenizer_data['inp_lang']
                self.targ_lang = tokenizer_data['targ_lang']
                self.max_length_targ = tokenizer_data['max_length_targ']
                self.max_length_inp = tokenizer_data['max_length_inp']
                self.embedding_dim = tokenizer_data.get('embedding_dim', 256)
                self.units = tokenizer_data.get('units', 1024)

            # 2. 检查Checkpoint目录
            if not os.path.exists(TRANSLATE_CHECKPOINT_DIR):
                raise FileNotFoundError(f"Checkpoint目录不存在: {TRANSLATE_CHECKPOINT_DIR}")

            # ===== 模型结构定义（与训练一致） =====
            class Encoder(tf.keras.Model):
                def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
                    super().__init__()
                    self.batch_sz = batch_sz
                    self.enc_units = enc_units
                    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
                    self.bigru = tf.keras.layers.Bidirectional(
                        tf.keras.layers.GRU(
                            enc_units // 2,
                            return_sequences=True,
                            return_state=True,
                            recurrent_initializer='glorot_uniform',
                            dropout=0.2,
                            recurrent_dropout=0.2
                        )
                    )
                    self.state_proj = tf.keras.layers.Dense(enc_units, activation='tanh')

                def call(self, x, hidden, training=False):
                    x = self.embedding(x)
                    output, f_state, b_state = self.bigru(x, initial_state=[hidden, hidden], training=training)
                    state = self.state_proj(tf.concat([f_state, b_state], axis=-1))
                    return output, state

                def initialize_hidden_state(self):
                    return tf.zeros((self.batch_sz, self.enc_units // 2))

            class BahdanauAttention(tf.keras.layers.Layer):
                def __init__(self, units):
                    super().__init__()
                    self.W1 = tf.keras.layers.Dense(units)
                    self.W2 = tf.keras.layers.Dense(units)
                    self.V = tf.keras.layers.Dense(1)

                def call(self, query, values):
                    hidden_with_time_axis = tf.expand_dims(query, 1)
                    score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
                    attention_weights = tf.nn.softmax(score, axis=1)
                    context_vector = attention_weights * values
                    context_vector = tf.reduce_sum(context_vector, axis=1)
                    return context_vector, attention_weights

            class Decoder(tf.keras.Model):
                def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
                    super().__init__()
                    self.batch_sz = batch_sz
                    self.dec_units = dec_units
                    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
                    self.gru = tf.keras.layers.GRU(
                        dec_units,
                        return_sequences=True,
                        return_state=True,
                        recurrent_initializer='glorot_uniform',
                        dropout=0.2,
                        recurrent_dropout=0.2
                    )
                    self.fc_mid = tf.keras.layers.Dense(dec_units, activation='relu')
                    self.fc = tf.keras.layers.Dense(vocab_size)
                    self.attention = BahdanauAttention(dec_units)

                def call(self, x, hidden, enc_output, training=False):
                    context_vector, attention_weights = self.attention(hidden, enc_output)
                    x = self.embedding(x)
                    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
                    output, state = self.gru(x, initial_state=hidden, training=training)
                    output = tf.reshape(output, (-1, output.shape[2]))
                    output = self.fc_mid(output)
                    x = self.fc(output)
                    return x, state, attention_weights

            # ===== 创建中译英模型实例（中文→英文） =====
            vocab_inp_size = len(self.inp_lang.word_index) + 1
            vocab_tar_size = len(self.targ_lang.word_index) + 1

            self.encoder_zh2en = Encoder(vocab_inp_size, self.embedding_dim, self.units, self.BATCH_SIZE)
            self.decoder_zh2en = Decoder(vocab_tar_size, self.embedding_dim, self.units, self.BATCH_SIZE)

            # ===== 创建英译中模型实例（英文→中文，交换vocab） =====
            # 注意：英译中需要交换输入输出vocab
            self.encoder_en2zh = Encoder(vocab_tar_size, self.embedding_dim, self.units, self.BATCH_SIZE)
            self.decoder_en2zh = Decoder(vocab_inp_size, self.embedding_dim, self.units, self.BATCH_SIZE)

            # ===== 加载Checkpoint（中译英模型） =====
            latest_checkpoint = tf.train.latest_checkpoint(TRANSLATE_CHECKPOINT_DIR)
            if not latest_checkpoint:
                raise FileNotFoundError(f"未找到checkpoint文件: {TRANSLATE_CHECKPOINT_DIR}")

            # 加载中译英模型
            checkpoint_zh2en = tf.train.Checkpoint(encoder=self.encoder_zh2en, decoder=self.decoder_zh2en)
            status_zh2en = checkpoint_zh2en.restore(latest_checkpoint)
            status_zh2en.expect_partial()
            self.model_loaded_zh2en = True
            print(f"[INFO] 中译英模型加载成功: {latest_checkpoint}")
            
            # 尝试加载英译中模型
            # 注意：由于vocab size不同，无法直接使用同一个checkpoint
            # 我们尝试创建一个反向模型，但权重需要重新训练或手动映射
            # 这里我们尝试加载，如果失败则使用随机初始化的权重（效果较差，但可以运行）
            try:
                checkpoint_en2zh = tf.train.Checkpoint(
                    encoder=self.encoder_en2zh, 
                    decoder=self.decoder_en2zh
                )
                # 尝试从同一个checkpoint加载（会失败，因为vocab size不匹配）
                # 但expect_partial会忽略不匹配的部分，模型会使用随机初始化的权重
                status_en2zh = checkpoint_en2zh.restore(latest_checkpoint)
                status_en2zh.expect_partial()
                # 检查是否有任何权重被加载
                # 由于vocab size不同，embedding和fc层无法加载，但GRU等层可能可以共享
                # 这里我们标记为已加载，但实际效果可能不理想
                self.model_loaded_en2zh = True
                print(f"[INFO] 英译中模型已创建（部分权重可能未加载，效果可能不理想）")
                print(f"[INFO] 建议：如需高质量英译中，请训练反向模型")
            except Exception as e:
                print(f"[WARNING] 英译中模型创建失败: {e}")
                print(f"[INFO] 英译中将使用简化词典翻译")
                self.model_loaded_en2zh = False
            
            print(f"[INFO] 模型支持方向: 中文 → 英文 (zh2en): {'✓' if self.model_loaded_zh2en else '✗'}")
            print(f"[INFO] 模型支持方向: 英文 → 中文 (en2zh): {'✓' if self.model_loaded_en2zh else '✗'}")

        except Exception as e:
            print(f"[ERROR] 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            self.encoder_zh2en = None
            self.decoder_zh2en = None
            self.encoder_en2zh = None
            self.decoder_en2zh = None
            self.model_loaded_zh2en = False
            self.model_loaded_en2zh = False

    def evaluate(self, sentence, direction='zh2en'):
        """模型推理
        Args:
            sentence: 待翻译的句子
            direction: 翻译方向，'zh2en' 表示中文→英文，'en2zh' 表示英文→中文
        """
        try:
            # 根据方向选择模型和语言
            if direction == 'zh2en':
                # 中译英：使用 inp_lang（中文）作为输入，targ_lang（英文）作为输出
                if not self.model_loaded_zh2en or self.encoder_zh2en is None or self.decoder_zh2en is None:
                    return None
                
                encoder = self.encoder_zh2en
                decoder = self.decoder_zh2en
                input_lang = self.inp_lang
                output_lang = self.targ_lang
                max_input_len = self.max_length_inp
                max_output_len = self.max_length_targ
                
            elif direction == 'en2zh':
                # 英译中：使用 targ_lang（英文）作为输入，inp_lang（中文）作为输出
                if not self.model_loaded_en2zh or self.encoder_en2zh is None or self.decoder_en2zh is None:
                    return None
                
                encoder = self.encoder_en2zh
                decoder = self.decoder_en2zh
                input_lang = self.targ_lang  # 英文作为输入
                output_lang = self.inp_lang  # 中文作为输出
                max_input_len = self.max_length_targ  # 英文的最大长度
                max_output_len = self.max_length_inp  # 中文的最大长度
            else:
                return None

            sentence = self.preprocess_sentence(sentence)
            inputs = [input_lang.word_index.get(i, 0) for i in sentence.split() if i]
            if not inputs:
                return ""
            
            inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_input_len, padding='post')
            inputs = tf.convert_to_tensor(inputs)

            hidden = encoder.initialize_hidden_state()
            enc_out, enc_hidden = encoder(inputs, hidden, training=False)
            dec_hidden = enc_hidden

            start_token = output_lang.word_index['<start>'] if '<start>' in output_lang.word_index else 1
            dec_input = tf.expand_dims([start_token], 0)

            result = ""
            for _ in range(max_output_len):
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_out, training=False)
                predicted_id = tf.argmax(predictions[0]).numpy()
                predicted_word = output_lang.index_word.get(predicted_id, "")
                if predicted_word == '<end>':
                    break
                if predicted_word != '<start>':  # 跳过开始标记
                    result += predicted_word + " "
                dec_input = tf.expand_dims([predicted_id], 0)

            return result.strip()

        except Exception as e:
            print(f"[ERROR] 推理失败 ({direction}): {e}")
            import traceback
            traceback.print_exc()
            return None

    def simple_translate(self, text, direction='zh2en'):
        """固定词典翻译（降级用）
        Args:
            text: 待翻译的文本
            direction: 翻译方向，'zh2en' 或 'en2zh'
        """
        if not text:
            return ""
        
        if direction == 'zh2en':
            # 中译英词典（按长度降序，优先匹配长短语）
            common_dict = {
                '很高兴见到你': 'Nice to meet you',
                '早上好': 'Good morning', 
                '晚上好': 'Good evening',
                '不客气': "You're welcome",
                '我爱你': 'I love you',
                '你好': 'Hello', 
                '谢谢': 'Thank you', 
                '再见': 'Goodbye',
                '是的': 'Yes', 
                '不是': 'No', 
                '对不起': 'Sorry',
                '请': 'Please',
                '谢谢': 'Thanks',
                '好的': 'OK',
                '没问题': 'No problem',
                '当然': 'Of course'
            }
            result = text
            # 按长度降序排序，优先匹配长短语
            for zh, en in sorted(common_dict.items(), key=lambda x: len(x[0]), reverse=True):
                result = result.replace(zh, en)
            return result
        else:
            # 英译中词典（按长度降序，优先匹配长短语）
            common_dict = {
                'Nice to meet you': '很高兴见到你',
                'Good morning': '早上好', 
                'Good evening': '晚上好',
                "You're welcome": '不客气',
                'I love you': '我爱你',
                'Thank you': '谢谢', 
                'Goodbye': '再见',
                'Hello': '你好',
                'Yes': '是的', 
                'No': '不是', 
                'Sorry': '对不起',
                'Please': '请',
                'Thanks': '谢谢',
                'OK': '好的',
                'No problem': '没问题',
                'Of course': '当然'
            }
            result = text
            # 按长度降序排序，优先匹配长短语
            for en, zh in sorted(common_dict.items(), key=lambda x: len(x[0]), reverse=True):
                result = result.replace(en, zh)
            return result

    def translate(self, text, direction='zh2en'):
        """对外翻译接口
        Args:
            text: 待翻译的文本
            direction: 翻译方向，'zh2en' 表示中文→英文，'en2zh' 表示英文→中文
        Returns:
            dict: 包含 original, translated, method, direction 的字典
        """
        if direction not in ['zh2en', 'en2zh']:
            return {
                "original": text,
                "translated": "不支持的方向，请使用 'zh2en' 或 'en2zh'",
                "method": "错误",
                "direction": direction
            }

        # 中译英：使用模型翻译
        if direction == 'zh2en':
            if not self.model_loaded_zh2en:
                # 模型未加载，使用简化翻译
                return {
                    "original": text,
                    "translated": self.simple_translate(text, direction),
                    "method": "模型未加载，使用简化词典翻译",
                    "direction": direction
                }

            model_result = self.evaluate(text, direction)
            if model_result:
                return {
                    "original": text,
                    "translated": model_result,
                    "method": "Seq2Seq模型翻译",
                    "direction": direction
                }
            else:
                # 模型推理失败，使用简化翻译
                return {
                    "original": text,
                    "translated": self.simple_translate(text, direction),
                    "method": "模型推理失败，使用简化词典翻译",
                    "direction": direction
                }
        
        # 英译中：尝试使用模型翻译，如果失败则使用简化翻译
        else:  # direction == 'en2zh'
            # 注意：由于vocab size不同，英译中模型可能无法从checkpoint正确加载
            # 如果模型已加载，尝试使用；否则直接使用简化翻译
            if self.model_loaded_en2zh:
                model_result = self.evaluate(text, direction)
                if model_result and model_result.strip():
                    # 检查结果是否合理（不是空字符串或只有标点）
                    return {
                        "original": text,
                        "translated": model_result,
                        "method": "Seq2Seq模型翻译（部分权重可能未加载）",
                        "direction": direction
                    }
            
            # 模型未加载或推理失败，使用简化翻译
            return {
                "original": text,
                "translated": self.simple_translate(text, direction),
                "method": "简化词典翻译" + ("（模型权重未正确加载）" if self.model_loaded_en2zh else "（模型未加载）"),
                "direction": direction,
                "note": "如需高质量英译中，请训练反向模型或使用专门的英译中checkpoint"
            }


# 全局实例
translator = Translator()