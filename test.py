from nn import NLP, load_model_and_vocab
import torch


def generate_sequence_from_text(model, prompt_text, stoi, itos,device,  max_len=20, eos_token="<eos>"):
    model.eval()
    eos_token_id = stoi.get(eos_token, None)

    # Encode string into tensor of token ids
    input_ids = [stoi.get(ch, stoi.get("<unk>", 0)) for ch in prompt_text]
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)  # shape: (1, seq_len)

    with torch.no_grad():
        generated = input_tensor[:, :1].clone()  # Start with first token (typically BOS)
        for _ in range(max_len):
            logits = model(generated)  # (1, T, V)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)  # (1, 1)
            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

    # Decode predicted token IDs into text
    output_ids = generated[0].tolist()[len(input_ids):]  # Only the generated part
    output_text = "".join([itos.get(idx, "<unk>") for idx in output_ids])
    return output_text.replace(eos_token, "")




if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, stoi = load_model_and_vocab("best_model.pth", 16, 100, device)
    itos = {i: ch for ch, i in stoi.items()}
    while True:
        print(generate_sequence_from_text(model, input("Prompt: "), stoi, itos, device))