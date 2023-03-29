import torch

import modeling

tokenizer_ckpt = "/research/nfs_su_809/workspace/stevens.994/llama/raw/tokenizer.model"
model_ckpt = (
    "/research/nfs_su_809/workspace/stevens.994/llama/raw/7B/consolidated.00.pth"
)

device = "cuda:0"
# device = "cpu"


def main():
    tokenizer = modeling.Tokenizer(tokenizer_ckpt)
    model = modeling.Llama.from_pretrained("llama-7b", model_ckpt)
    model.to(device)
    print("Loaded pretrained model.")

    prompts = [
        "Look, if you had one shot or one opportunity\nTo seize everything you ever wanted in one moment\nWould you capture it, or just let it slip? Yo\n"
    ]
    tokens = tokenizer.encode_batch(prompts, bos=True, eos=False)
    tokens = torch.tensor(tokens).to(device)
    print(tokens)
    model.eval()
    with torch.no_grad():
        tokens = model.generate(tokens, 1024, temperature=0.9, top_k=40)

    print(tokenizer.decode(tokens[0].tolist()))


if __name__ == "__main__":
    main()
