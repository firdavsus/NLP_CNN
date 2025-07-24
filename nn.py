import torch, csv
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from functools import partial
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
smooth = SmoothingFunction().method1

# ─── 1) Model ────────────────────────────────────────────────────────────────

class NLP(nn.Module):
    def __init__(self, stoi: dict, embed_dim=16, max_seq_len=100):
        super().__init__()
        self.stoi = dict(stoi)
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Embedding layers
        self.embed = nn.Embedding(len(self.stoi), embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

        # Convolutional stack
        self.conv1 = nn.Conv1d(embed_dim, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1, dilation=1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, padding=8, dilation=4)
        self.bn3 = nn.BatchNorm1d(256)

        # self.conv4 = nn.Conv1d(256, 256, kernel_size=7, padding=18, dilation=6)
        # self.bn4 = nn.BatchNorm1d(256)

        # self.conv5 = nn.Conv1d(256, 256, kernel_size=5, padding=8, dilation=2)
        # self.bn5 = nn.BatchNorm1d(256)

        self.fc1 = nn.Linear(512, 128)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, len(self.stoi))

    def forward(self, x: torch.LongTensor):
        B, T = x.shape
        device = x.device

        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        x_embed = self.embed(x) + self.pos_embed(pos) 
        x = x_embed.transpose(1, 2) 

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # x = F.relu(self.bn4(self.conv4(x)))
        # x = F.relu(self.bn5(self.conv5(x)))

        x_max = F.max_pool1d(x, x.size(2)).squeeze(2)
        x_avg = F.avg_pool1d(x, x.size(2)).squeeze(2)
        x = torch.cat([x_max, x_avg], dim=1)

        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)
    
def collate_fn(batch, max_len=100):
    x_batch, y_batch = zip(*batch)
    def pad_to_fixed(tensor_list):
        padded = []
        for t in tensor_list:
            t = t[:max_len]
            pad_len = max_len - len(t)
            if pad_len > 0:
                t = F.pad(t, (0, pad_len), value=0)
            padded.append(t)
        return torch.stack(padded)

    x_pad = pad_to_fixed(x_batch)
    y_pad = pad_to_fixed(y_batch)

    return x_pad, y_pad

def load_data(path):
    input = []
    output = []
    with open(path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            input.append(row['\ufeffinstruction'] + row['input'])
            output.append(row['output'])

    to_pop = []
    for i in range(len(input)):
        if len(input[i])+len(output[i])>100:
            to_pop.append(i)

    new_input = []
    for i in range(len(input)):
        if i not in to_pop:
            new_input.append([input[i], output[i]+'<eos>'])
        else:
            new_input.append([input[i][:int(100/2)], output[i][:int(100/2)]+'<eos>'])
    data = new_input
    vocab = set()
    for text in data:
        for ch in text:
            vocab.update(ch)
    stoi = {token: i+3 for i, token in enumerate(sorted(vocab))}
    stoi["<pad>"] = 0
    stoi["<unk>"] = 1
    stoi["<eos>"] = 2
    return new_input, stoi


class CharDataset(Dataset):
    def __init__(self, data, stoi, max_len=100):
        self.data = data
        self.stoi = stoi
        self.max_len = max_len

    def encode(self, text):
        return [self.stoi.get(c, self.stoi["<unk>"]) for c in text]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_str, y_str = self.data[idx]
        x_ids = self.encode(x_str)
        y_ids = self.encode(y_str)
        return torch.tensor(x_ids), torch.tensor(y_ids)

def generate_sequence(model, prefix, max_len=20, device="cpu", eos_token_id=None):
    model.eval()
    with torch.no_grad():
        generated = prefix[:, :1].clone().to(device)  # Start with first token of prefix (e.g. BOS)
        for _ in range(max_len):
            logits = model(generated)  # logits for entire sequence
            next_token = torch.argmax(logits, dim=-1).unsqueeze(1) # last token
            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
    return generated.squeeze(0).tolist()


def train_model(
    model, train_loader, optimizer, loss_fn,
    stoi, itos, epochs=5, device="cpu",
    n_steps=1000, save_path="best_model.pth"
):
    model.train()
    best_bleu = 0.0
    step = 0
    eos_token_id = stoi.get("<eos>", None)
    pad_token_id = stoi.get("<pad>", 0)  # fallback if needed

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_tokens = 0
        bleu_scores = []

        for xb, yb in train_loader:
            step += 1
            xb, yb = xb.to(device), yb.to(device)

            # Predict the full sequence of outputs
            logits = model(xb)  # [B, T, V]
            vocab_size = logits.size(-1)

            # Flatten logits and targets for CrossEntropyLoss
            logits = logits.view(-1, vocab_size)  # [B*T, V]
            targets = yb.view(-1)  # [B*T]

            loss = loss_fn(logits, yb[:, 0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * targets.size(0)
            total_tokens += targets.size(0)

            if step % n_steps == 0 or step == len(train_loader):
                model.eval()
                with torch.no_grad():
                    for x_sample, y_sample in zip(xb[:30], yb[:30]):  # limit BLEU to 30 examples
                        pred_ids = generate_sequence(
                            model, x_sample.unsqueeze(0),
                            max_len=yb.size(1),
                            device=device,
                            eos_token_id=eos_token_id
                        )
                        pred_chars = [itos.get(i, "<unk>") for i in pred_ids if i not in {0, 1}]
                        ref_chars = [itos.get(i.item(), "<unk>") for i in y_sample if i.item() not in {0, 1}]

                        if len(ref_chars) > 1 and len(pred_chars) > 1:
                            bleu = sentence_bleu([ref_chars], pred_chars, smoothing_function=smooth)
                            bleu_scores.append(bleu)
                model.train()

                avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
                if avg_bleu > best_bleu:
                    best_bleu = avg_bleu
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "stoi": stoi,
                    }, save_path)
                    print(f"✅ Step {step}: New Best BLEU {best_bleu:.4f} — model saved to {save_path}")

        avg_loss = total_loss / total_tokens
        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
        print(f"📘 Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | BLEU-1: {avg_bleu:.4f} | Best BLEU: {best_bleu:.4f}")


def load_model_and_vocab(path, embed_dim, max_seq_len, device):
    checkpoint = torch.load(path, map_location=device)
    stoi = checkpoint["stoi"]
    model = NLP(stoi, embed_dim, max_seq_len)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, stoi




if __name__=="__main__":
    data, stoi = load_data("alpaca.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    context_vindow = 100
    emb_dim = 16

    load = 0
    if load:
        model, stoi = load_model_and_vocab("best_model.pth", emb_dim, context_vindow, device)
    else:
        model = NLP(stoi, embed_dim=emb_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    train_dataset = CharDataset(data, stoi)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                          collate_fn=partial(collate_fn, max_len=context_vindow))
    
    itos = {i: ch for ch, i in stoi.items()}
    train_model(model, train_loader, optimizer, loss_fn, stoi, itos, epochs=30, device=device)