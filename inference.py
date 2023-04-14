import torch

import modeling

tokenizer_ckpt = "/research/nfs_su_809/workspace/shared/llama/raw/tokenizer.model"
model_ckpt = "/research/nfs_su_809/workspace/shared/llama/raw/7B/consolidated.00.pth"
device = "cuda:0"


def main():
    tokenizer = modeling.load_pretrained_tokenizer("llama-7b", tokenizer_ckpt)

    prompts = [
        "Earlier this week, hedge fund billionaire Ken Griffin accomplished an embarrassingly impressive feat: he donated $300 million in such as a way as to do as close as possible to zero good with it.\n\n",
        "French: Ce n'est pas une pipe\n\nEnglish: It's not a pipe\n\nFrench: Bonjour, mon ami\n\nEnglish:",
        "German: Wo ist mein Handy?\n\nEnglish: Where is my phone?\n\nGerman: Ich bin der grosster Mann den Welt.\n\nEnglish: I am the biggest man in the world.\n\nGerman: Wo ist die Bibliotheque?\n\nEnglish:",
    ]
    tokens = tokenizer.encode_batch(prompts, bos=True, eos=False)
    tokens = tokens.to(device)
    print(tokens)

    model = modeling.load_pretrained_llama("llama-7b", model_ckpt)
    model.to(device)
    print("Loaded pretrained model.")

    model.eval()
    with torch.no_grad():
        tokens = model.generate(tokens, 256, temp=0.85, top_k=40)

    for toks in tokens:
        print(tokenizer.decode(toks.tolist()))
        print("---")


if __name__ == "__main__":
    main()
