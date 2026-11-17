import json, random, re, csv, sys
from pathlib import Path
import numpy as np, torch, pandas as pd
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
root = Path("test/fake"); root.mkdir(parents=True, exist_ok=True)
hans = pd.read_csv("hans_data/heuristics_evaluation_set.txt", sep="\t", quoting=csv.QUOTE_NONE, dtype=str)
words = set()
for s in pd.concat([hans.sentence1, hans.sentence2]):
    words.update(re.findall(r"\w+|[^\w\s]", s.lower()))
words = sorted(words)
vocab = ["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]"] + words
(root/"vocab.txt").write_text("\n".join(vocab))
tok = BertTokenizerFast(vocab_file=str(root/"vocab.txt"), do_lower_case=True)
# fake MNLI dev (GLUE tsv format, mixed case so lower-casing matters)
rng = random.Random(0)
rows=[]
for k in range(600):
    p = " ".join(rng.choice(words) for _ in range(rng.randint(4,15))).capitalize()+" ."
    h = " ".join(rng.choice(words) for _ in range(rng.randint(3,8))).capitalize()+" ."
    rows.append(dict(sentence1=p, sentence2=h, gold_label=rng.choice(["entailment","neutral","contradiction","-"])))
with open(root/"dev_matched.tsv","w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=["index","sentence1","sentence2","gold_label"], delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
    w.writeheader()
    for i,r in enumerate(rows): w.writerow(dict(index=i, **r))
cfg = BertConfig(vocab_size=len(vocab), hidden_size=64, num_hidden_layers=2, num_attention_heads=4, intermediate_size=128, num_labels=3)
for i in range(12):
    torch.manual_seed(i)
    m = BertForSequenceClassification(cfg).eval()
    with torch.no_grad():  # give each model a different label bias so models differ in a stable way
        m.classifier.bias.copy_(torch.randn(3)*1.5)
    d = root/f"feather_berts_{i}"; m.save_pretrained(d, safe_serialization=False); tok.save_pretrained(d)
    # stored eval file in the repo format: label_dict + samples with lower-cased "[CLS] p [SEP] h [SEP]"
    kept=[r for r in rows if r["gold_label"]!="-"]
    lab={"contradiction":0,"entailment":1,"neutral":2}
    enc = tok([r["sentence1"] for r in kept],[r["sentence2"] for r in kept], padding=True, return_tensors="pt")
    with torch.no_grad(): lg = m(**enc).logits.numpy()
    samples=[dict(probabilities=torch.softmax(torch.tensor(l),0).tolist(), logits=l.tolist(), label=lab[r["gold_label"]],
                  sentence=f"[CLS] {r['sentence1'].lower()} [SEP] {r['sentence2'].lower()} [SEP]") for l,r in zip(lg,kept)]
    json.dump(dict(label_dict=lab, samples=samples), open(d/f"mnli_eval_bert_{i}.json","w"))
print("made", len(vocab), "vocab;", sum(r['gold_label']!='-' for r in rows), "labelled MNLI rows")
