# run_gpt2.py

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

def main():
    # Inicjalizacja tokenizera i modelu
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Ustawienie urządzenia (GPU jeśli dostępne, w przeciwnym razie CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Tekst początkowy (prompt)
    text = "To jest przykład tekstu, który chcemy kontynuować za pomocą GPT-2."
    # Tokenizacja tekstu
    inputs = tokenizer.encode(text, return_tensors='pt').to(device)

    # Generowanie tekstu
    with torch.no_grad():
        output = model.generate(inputs, max_length=100, num_return_sequences=1)

    # Dekodowanie wygenerowanego tekstu
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)

if __name__ == "__main__":
    main()
