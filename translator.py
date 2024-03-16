from transformers import MarianMTModel, MarianTokenizer


tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

def Translate(input):
    translated = model.generate(**tokenizer(input, return_tensors="pt"))
    res = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return res

