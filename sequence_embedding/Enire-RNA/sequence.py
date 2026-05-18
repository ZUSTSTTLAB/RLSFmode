from multimolecule import RnaTokenizer, ErnieRnaModel
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

model_path = "D:\\temp_model"
tokenizer = RnaTokenizer.from_pretrained(model_path)
model = ErnieRnaModel.from_pretrained(model_path)

text = ""
input = tokenizer(text, return_tensors="pt")

output = model(**input)

print(output)
vector = output.pooler_output.detach().numpy()[0]  # 转为numpy数组并取第一个样本（批处理中的唯一样本）

# 保存到文本文件（每行一个维度值）
with open("_sequence.txt", "w") as f:
    # 写入768个维度值，每行一个
    for value in vector:
        f.write(f"{value}\n")  # 换行分隔

print("RNA序列向量已保存")