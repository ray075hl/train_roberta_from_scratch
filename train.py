from pathlib import Path 
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import os
import json

import torch

from transformers import RobertaConfig, RobertaTokenizer, RobertaModel, RobertaForMaskedLM
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, pipeline 

from utils import * 


print(torch.cuda.is_available())

# 查找目录下的txt文件
texts = [str(x) for x in Path("./data").glob("*.txt")]
print(texts)


# 初始化一个tokenizer
tokenizer = ByteLevelBPETokenizer()
print(tokenizer)


# 训练一个tokenzier
tokenizer.train(files=texts,  # 训练语料
                 vocab_size=52000,  # 词汇量
                 min_frequency=2,  # 单词最小频率
                 special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])


# 保存tokenizer到本地
token_dir = './roberaModel/tokenzier'

if not os.path.exists(token_dir):
    os.makedirs(token_dir)

tokenizer.save_model(token_dir)


# test tokenizer 
result = tokenizer.encode('girls help girls!')
print(result.tokens)
print(result.ids)
print(result.type_ids)
print(result.attention_mask)
print(result.special_tokens_mask)

# 定义后处理， 即加上<s>, </s>
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>"))
)

# 最大长度定义
tokenizer.enable_truncation(max_length=512)
tokenizer.save('roberaModel/config.json')



# 定义roberta的超参数
config = RobertaConfig(
    vocab_size = 52000,  # 词汇表大小
    max_position_embeddings = 514, # position的大小
    num_attention_heads = 12,  # attention heads 数量
    num_hidden_layers = 6,  # 6层
    type_vocab_size = 1  # token_type_ids 的类别
)

print(config)



# 创建 Roberta tokenizer
roberta_tokenizer = RobertaTokenizer.from_pretrained('./roberaModel/tokenzier', max_length=512)

# 初始化model
roberta_model = RobertaForMaskedLM(config=config)
print(roberta_model)


print(roberta_model.num_parameters())
params = list(roberta_model.parameters())

# 打印层数
params_total = 0
for layer_index in range(len(params)):
    num_, shape = get_list_item_number(params[layer_index])
    params_total += num_
    print(f"第{layer_index}层：形状{shape}")

print("参数总数：", params_total)


# 加载训练数据
dataset = LineByLineTextDataset(tokenizer=roberta_tokenizer,
                                file_path='./data/shakespeare.txt',
                                block_size=128  # 每批次读128行
                                )

data_collector = DataCollatorForLanguageModeling(tokenizer=roberta_tokenizer,
                                                 mlm=True,
                                                 mlm_probability=0.15)
# 训练参数定义
trainArgs = TrainingArguments(output_dir='./roberta_output/',
                              overwrite_output_dir=True,
                              do_train=True,
                              num_train_epochs=1)

# trainer定义
trainer = Trainer(model=roberta_model,  # 模型对象
                  args=trainArgs,  # 训练参数
                  data_collator=data_collector,  #  数据生成器
                  train_dataset=dataset,  # 数据集
                  )

# 预训练开始
trainer.train()

# 模型保存
model_saved_path = './saved_model'
if os.path.exists(model_saved_path) is False:
    os.makedirs(model_saved_path)

trainer.save_model(model_saved_path)

