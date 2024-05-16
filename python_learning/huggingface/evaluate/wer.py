import evaluate

wer = evaluate.load("cer")
wer_res = wer.compute(predictions=["hello there"], references=["hello there, i am"])
print(wer_res)
