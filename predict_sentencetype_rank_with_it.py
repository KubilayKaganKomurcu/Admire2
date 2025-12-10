
# !pip install torch torchvision transformers pillow pandas deep-translator langdetect sentence-transformers scikit-learn

import torch
import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
import torch
import pickle
import zipfile
from IPython.display import display
from IPython.display import Image as IPImage
import os
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
import glob
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer

device = "cuda"


# =========================
# LOAD MODELS
# =========================
text_model = SentenceTransformer("clip-ViT-B-32-multilingual-v1").to(device)
img_model = SentenceTransformer("clip-ViT-B-32").to(device)

# image encoder frozen
for p in img_model.parameters():
    p.requires_grad = False

# Load dataset (auto-detect tsv/csv)
def load_dataset(path):
    if path.endswith(".tsv"):
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)
    return df

def no_transpose_collate(batch):
    # batch = list of items returned by __getitem__
    sentences = [item[0] for item in batch]
    compounds = [item[1] for item in batch]
    labels = torch.stack([item[2] for item in batch])
    img_paths = [item[3] for item in batch]   # DO NOT TRANSPOSE
    captions = [item[4] for item in batch]
    exp_order = [item[5] for item in batch]

    return sentences, compounds, labels, img_paths, captions, exp_order



# =========================
# CLASSIFIER (idiom / literal)
# =========================
classifier = nn.Linear(512, 2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-4)



import os

import torch
from torch.utils.data import Dataset

class IdiomDataset(Dataset):
    def __init__(self, df,
                 img_cols=["image1_name","image2_name","image3_name","image4_name","image5_name"],
                 img_caption_cols=["image1_caption","image2_caption","image3_caption","image4_caption","image5_caption"],
                 expected_order_col="expected_order",
                 folder="train",
                ):
        self.sentences = df["sentence"].tolist()
        self.labels = (df["sentence_type"].apply(lambda x: 0 if x=="literal" else 1)).tolist()
        self.compound = df["compound"].tolist()
        self.img_caption_cols = img_caption_cols
        self.expected_order_col = df[expected_order_col]
        self.df = df
        self.img_cols = df[img_cols].values.tolist()
        self.folder = folder


    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        language = row["language"]
        # Sentence, label, compound
        if self.folder=="train":
          img_base_path = "/content/drive/MyDrive/Subtask A copy/{}/train/".format(language)
        if self.folder=="test":
          img_base_path = "/content/drive/MyDrive/Subtask A copy/{}/test/".format(language)
        if self.folder=="eval":
          img_base_path = "/content/drive/MyDrive/Subtask A copy/{}/xeval/".format(language)
        sentence = self.sentences[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        compound = self.compound[idx]
      # -> already a list of 5 image names
        filenames = self.img_cols[idx]

        img_paths = [f"{img_base_path}{compound}/{fname}"for fname in filenames]

        # Image captions
        img_captions = [row[col] for col in self.img_caption_cols]

        # Expected order
        expected_order = self.expected_order_col

        return sentence, compound, label, img_paths,  img_captions, expected_order







train_df_en = load_dataset("/content/drive/MyDrive/Subtask A copy/EN/train/subtask_a_train.tsv")
train_df_en["language"]="EN"
train_df_pt = load_dataset("/content/drive/MyDrive/Subtask A copy/PT/train/subtask_a_train.tsv")
train_df_pt["language"]="PT"

val_df_en = load_dataset("/content/drive/MyDrive/Subtask A copy/EN/xeval/subtask_a_xe.tsv")
val_df_en["language"]="EN"
val_df_pt = load_dataset("/content/drive/MyDrive/Subtask A copy/PT/xeval/subtask_a_xp.tsv")
val_df_pt["language"]="PT"

test_df_en = load_dataset("/content/drive/MyDrive/Subtask A copy/EN/test/subtask_a_test.tsv")
test_df_en["language"]="EN"
test_df_pt = load_dataset("/content/drive/MyDrive/Subtask A copy/PT/test/subtask_a_test.tsv")
test_df_pt["language"]="PT"

import pandas as pd

train_df = pd.concat([train_df_en, train_df_pt], ignore_index=True)
val_df = pd.concat([val_df_en, val_df_pt], ignore_index=True)
test_df = pd.concat([test_df_en, test_df_pt], ignore_index=True)

train_loader_en = DataLoader(IdiomDataset(df =train_df, folder="train" ), batch_size=16, shuffle=True, collate_fn=no_transpose_collate)
val_loader_en = DataLoader(IdiomDataset(val_df, folder="eval" ), batch_size=16, shuffle=False,  collate_fn=no_transpose_collate)
test_loader_en = DataLoader(IdiomDataset(test_df,folder="test"), batch_size=16, shuffle=False,  collate_fn=no_transpose_collate)


# ===========================
# Training Setup
# ===========================
text_dim = text_model.get_sentence_embedding_dimension()
classifier = nn.Linear(text_dim, 2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    list(text_model.parameters()) + list(classifier.parameters()),
    lr=1e-5
)
best_val_acc = 0

best_val_acc = 0

for epoch in range(30):
    classifier.train()
    running_loss = 0
    print("epoch:", epoch)

    for sentences, compound, labels, _, captions,_  in train_loader_en:
        print(sentences)
        labels = labels.to(device)
        sentence_with_extra=[]
        for l in range(len(sentences)):
          caps = ", ".join(captions[l])   # list → string
          combined = f"{sentences[l]}. imagescaptions: {caps}"
          sentence_with_extra.append(combined)

        # 1) TOKENIZE
        batch = text_model.tokenize(sentence_with_extra)
        batch = {k: v.to(device) for k, v in batch.items()}

        # 2) FORWARD (training modunda)
        output = text_model(batch)
        text_emb = output['sentence_embedding']     # (batch,512)

        # 3) CLASSIFIER
        logits = classifier(text_emb)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # ========== VALIDATION =============
    classifier.eval()
    preds_list, label_list = [], []

    with torch.no_grad():
        for sentences, compound, labels, _, _, _ in val_loader_en:
            labels = labels.to(device)

            batch = text_model.tokenize(sentences)
            batch = {k: v.to(device) for k, v in batch.items()}

            output = text_model(batch)
            text_emb = output['sentence_embedding']

            logits = classifier(text_emb)
            preds = logits.argmax(dim=1)

            preds_list.extend(preds.cpu().tolist())
            label_list.extend(labels.cpu().tolist())

    val_acc = accuracy_score(label_list, preds_list)
    print(f"Epoch {epoch+1} | Loss: {running_loss:.3f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        print(" Saving BEST checkpoint...")

        os.makedirs("best_model", exist_ok=True)
        torch.save(classifier.state_dict(), "best_model/classifier.pt")
        text_model.save("best_model/text_encoder")

print("\n=== LOADING BEST CHECKPOINT FOR TEST ===")

classifier.load_state_dict(torch.load("best_model/classifier.pt", map_location=device))
classifier.to(device)
text_encoder = text_model
text_encoder.eval()
classifier.eval()

preds_list = []
label_list = []

with torch.no_grad():
    for sentences, compound, labels, _, _, _ in test_loader_en:
        labels = labels.to(device)


        batch = text_model.tokenize(sentences)
        batch = {k: v.to(device) for k, v in batch.items()}

        output = text_model(batch)
        text_emb = output['sentence_embedding']

        logits = classifier(text_emb)
        preds = logits.argmax(dim=1)

        preds_list.extend(preds.cpu().tolist())
        label_list.extend(labels.cpu().tolist())

test_acc = accuracy_score(label_list, preds_list)
text_encoder.save("best_model/text_encoder")
print(f"🎉 TEST Accuracy: {test_acc:.4f}")


def predicter(sentences):

  batch = text_model.tokenize(sentences)
  batch = {k: v.to(device) for k, v in batch.items()}

  output = text_model(batch)
  text_emb = output['sentence_embedding']

  logits = classifier(text_emb)
  preds = logits.argmax(dim=1)


  return preds

from PIL import Image

total_img_embedding = []
labels = []

for sentence, compound, label, img_paths, img_captions, expected_order in test_loader_en:

    # text labels
    pred_label = predicter(sentence)
    labels.extend(pred_label)

    # flatten image paths
    flat_imgs = [Image.open(p).convert("RGB")
                 for sample_paths in img_paths
                 for p in sample_paths]

    # Encode all images
    img_emb = img_model.encode(
        flat_imgs,
        batch_size=16,
        convert_to_tensor=True,
        show_progress_bar=False,
        device=device
    )
    img_emb = img_emb.view(len(sentence), 5, -1). # reshape images acording to batch size


    total_img_embedding.append(img_emb)

print(total_img_embedding[0].shape)

################# RANKING ##########################

correct_top1 = 0
total = 0
import ast
for i,(sentences, compounds, labels, img_paths_batch, captions, expected_orders) in enumerate(test_loader_en):

    # TEXT embedding

    preds = predicter(sentences)   # text_emb → (B,512)
    print(labels)
    pred_labels = ["literal" if p == 0 else "idiomatic"
               for p in preds]

    sentences_with_predtype = [
        f"[{label}] {sentence}"
         for sentence, label in zip(sentences, pred_labels)
    ]
    print(sentences_with_predtype)
    sentence_with_extra = []
    for l in range(len(sentences)):
        caps = ", ".join(captions[l])   # list → string
        combined = f"{pred_labels[l]}: {sentences[l]}. Captions: {caps}"
        sentence_with_extra.append(combined)


    text_emb = text_model.encode(
    sentence_with_extra,
    batch_size=len(sentences),
    convert_to_tensor=True,
    device=device)
    print(preds,sentences)
    print(expected_orders[0][0])

    img_emb_batch = total_img_embedding[i]  # (B,5,512)
    B = img_emb_batch.size(0)

    for b in range(B):
        print(compounds[b])
        img_emb = img_emb_batch[b]          # (5,512)
        txt = text_emb[b]                  # (512)

        cos_sim = F.cosine_similarity(img_emb, txt.unsqueeze(0).repeat(5,1), dim=1)
        ranking = torch.argsort(cos_sim, descending=True)
        top1 = ranking[0].item()
        top2  = ranking[2].item()
        result = img_paths_batch[b][top1].split("/")[-1]
        print("result:",result)

        # Expected index
        print("full:", ast.literal_eval(expected_orders[0][b]))

        expected = ast.literal_eval(expected_orders[0][b])[0]   # first element
        print("expected:",expected)
        if result == expected:
            correct_top1 += 1

        total += 1

print("TOP-1 Accuracy =", correct_top1 / total)

