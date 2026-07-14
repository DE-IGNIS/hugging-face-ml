# FILE CONTENTS
# -> Loading a Model from HuggingFace using a pipeline & Directly
# -> Using tokenizer from gpt2 understand what & how tokenization works (Tokenization process)

# Using a Pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="openai-community/gpt2")

prompt = "What is Machine Learning?"
output = pipe(prompt)
print(output[0]["generated_text"])

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
model

# Load models using AutoTokenizer
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("gpt2")

sentence = "unsure"
input_ids = tokenizer(sentence, return_tensors="pt")["input_ids"] # pt = pyTorch
input_ids

tokenizer.decode(input_ids[0])

sentence = "unbelivable"
input_ids = tokenizer(sentence, return_tensors="pt").input_ids
input_ids

for token_id in input_ids[0]:
  print(tokenizer.decode(token_id))

word = "homoscedasticity"
my_ids = tokenizer(word ,return_tensors="pt").input_ids
my_ids

tokenizer.decode(my_ids.squeeze())

sentence = "pneumonoultramicroscopicsilicovolcanoconiosis"
token_id = tokenizer(sentence, return_tensors="pt").input_ids
len(token_id[0])

for t_id in token_id[0]:
  print(tokenizer.decode(t_id))

# When calling a model always use .from_pretrained()
from transformers import AutoModelForCausalLM
gpt2 = AutoModelForCausalLM.from_pretrained("gpt2");

gpt2

sentence = "I like machine learning to be able to predcit the future."
# Data Processing technique
token_ids = tokenizer(sentence , return_tensors="pt").input_ids
outputs = gpt2(token_ids).logits[0,-1]
tokenizer.decode(outputs.argmax())

sentence = "I learn machine learning to enhance"
token_ids = tokenizer(sentence, return_tensors="pt").input_ids
outputs = gpt2(token_ids).logits[0,-1]
final_logits = torch.topk(outputs,3)
final_logits

for index in final_logits.indices:
  print(tokenizer.decode(index))

torch.softmax(final_logits.values , dim=0)

def greedy_decode(logits):
  """Return token index with maximum probability"""
  return torch.argmax(logits,dim=-1)

# tokenizer.decode(greedy_decode(outputs))

# Top K Sampling
def top_k_sampling(logits,k=50):
  """Keeps only top k logits, normalized them into probability then sample one token from the filtered distribution."""
  values, indices = torch.topk(logits,k)
  probs = F.softmax(values,dim=-1)
  sampled = torch.multinomial(probs,1)
  return indices[sampled]

# Top-p (Nuecles) Sampling
def top_p_sampling(logits,p=0.9):
  """
  Sort tokens by probability, keep smallet number whose culumative
  probability exceeds threshold p, then sample one token.
  """

  sorted_logits, sorted_indices = torch.sort(logits, descending=True)
  sorted_probs = F.softmax(sorted_logits, dim=-1)
  cumulative_probs = sorted_probs.cumsum(dim=-1)

  # Mask tokens outside Nuclues
  mask = cumulative_probs > p
  sorted_logits[mask] = float("-inf")

  # Sample from filtered logits
  filtered_probs = F.softmax(sorted_logits, dim=-1)
  sampled = torch.multinomial(filtered_probs,1)

  # Return token index in original vocabulary
  return sorted_indices[sampled]

## Temperature Sampling ##
def temperature_sampling(logits, temperature=1.0):
  """
  Scale logits by temperature before sampling.
  Lower temperature => sharper distribution
  """
  scaled = logits / temperature
  probs = F.softmax(scaled , dim=-1)
  return torch.multinomial(probs , 1)

# tokenizer.decode(top_k_sampling(outputs))
# tokenizer.decode(top_p_sampling(outputs , p=0.9))
# tokenizer.decode(top_p_sampling(outputs , p=0.9))
tokenizer.decode(temperature_sampling(outputs , temperature=0.1))

sentence = "Today I decided to go to the local library and find out if there was anything I could do to help."
inputs = tokenizer(sentence , return_tensors="pt")
output = gpt2(**inputs)
logits = output.logits[0,-1]

# Different types of sampling
print(f"Greedy Decode: ",tokenizer.decode([greedy_decode(logits)]))
print(f"Top-K Sampling: ",tokenizer.decode(top_k_sampling(logits,k=10)))
print(f"Top-P Sampling: ",tokenizer.decode(top_p_sampling(logits,p=0.9)))
print(f"Temp: ",tokenizer.decode(temperature_sampling(logits,temperature=0.7)))

sentence = "I learn machine learning to enhance our data"
token_ids = tokenizer(sentence , return_tensors="pt").input_ids
outputs = gpt2(token_ids).logits # Raw Unnormalized Score - Values
outputs = torch.softmax(outputs[0,-1] , dim=-1)

top10 = torch.topk(outputs,k=10)

for index , value in zip(top10.indices , top10.values):
  print(f"{tokenizer.decode(index)} -- {value:.1%}")

"""### Sentiment Analysis"""

from datasets import load_dataset

ds = load_dataset("stanfordnlp/imdb")

type(ds)
ds

import pandas as pd
df = ds["train"].to_pandas()
df["text"]

from transformers import pipeline
device = 0
classifier = pipeline(
    "sentiment-analysis" ,
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device=device
    )

texts = df["text"].str.slice(0,500).tolist()
results = classifier(texts,batch_size=64,truncation=True)

labels = []
for r in results:
  labels.append(r["label"])

df["model_prediction"] = labels

# classifier("The day is great!")[0]["label"]
# df[["label","model_prediction"]][:20]

# 1:46:35
review = df.iloc[0]["text"]
classifier(review)[0]["label"]

from transformers import pipeline

finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")

sentence = "The company reported a strong increase in quaterly revenue exceeding market expectations"
sentence1 = "Shares fell after the firm reported insider trading scandal"
# Also
sentence2 = ["Strong consumer demand drove record sales across all regions" ,
             "Supply chain disruptions severly affected production output"]
# finbert(sentence)
# finbert(sentence1)
finbert(sentence2)

"""### Named Entity Recognition"""

sentence = "Apple announced record earnings in the United States on Monday"
ner = pipeline("ner")   # default model

sentence = ["Apple announced record earnings in the United States on Monday",
            "I worked at Facebook after graduating from Harvard"]
ner(sentence)

"""### Question Answering"""

from transformers import AutoTokenizer , AutoModelForQuestionAnswering
import torch

model_name = "distilbert-base-cased-distilled-squad"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# question = "Why is financial sentiment analysis a challenging task?"
question = "Who all agreed to a ceasefire?"

# context = """
# Financial sentiment analysis is a challenging task due to the specialized language and lack of labeled data in that domain. General-purpose models are not effective enough because of the specialized language used in a financial context. We hypothesize that pre-trained language models can help with this problem because they require fewer labeled examples and they can be further trained on domain-specific corpora. We introduce FinBERT, a language model based on BERT, to tackle NLP tasks in the financial domain. Our results show improvement in every measured metric on current state-of-the-art results for two financial sentiment analysis datasets. We find that even with a smaller training set and fine-tuning only a part of the model, FinBERT outperforms state-of-the-art machine learning methods.
# """
context = "Israel, Hezbollah agree to ceasefire starting on Friday"

inputs = tokenizer(
    question,
    context,
    return_tensors="pt",
    truncation=True
)

with torch.no_grad():
  outputs = model(**inputs)

start_idx = torch.argmax(outputs.start_logits)
end_idx = torch.argmax(outputs.end_logits) + 1

input_ids = inputs["input_ids"][0]

answer_tokens = input_ids[start_idx:end_idx]
answer = tokenizer.decode(answer_tokens)

print(answer)

"""### Machine Translation"""

from transformers import pipeline
translator = pipeline("text-generation", model="google/flan-t5-small")

result = translator("Translate English to French: Hello, how are you?")
print(result[0]["generated_text"])

from transformers import AutoTokenizer , AutoModelForSeq2SeqLM
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def translate_sentence(text):
  tokenizer.src_lang = "eng_Latn"
  inputs = tokenizer(text,return_tensors="pt")
  translated_tokens = model.generate(
      **inputs,
      forced_bos_token_id=tokenizer.convert_tokens_to_ids("fra_Latn")
  )
  output = tokenizer.batch_decode(translated_tokens,skip_special_tokens=True)

  return output[0]

# text = "Hello, how are you?"
# text = "I am good, myself Prince"
text = "Thanks"
result = translate_sentence(text)
print(result)