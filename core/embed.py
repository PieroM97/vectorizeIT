import torch




def vectorize_text(text, tokenizer, model, quantize=False):
    # Move tokenizer and model to CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)  # Ensure the model is moved to the correct device

    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    if isinstance(outputs, torch.Tensor):
        hidden_states = outputs
    elif isinstance(outputs, dict):
        if 'last_hidden_state' in outputs:
            hidden_states = outputs['last_hidden_state']
        elif 'pooler_output' in outputs:
            hidden_states = outputs['pooler_output']
        else:
            raise ValueError("Unsupported model output format. Unable to find 'last_hidden_state' or 'pooler_output'.")

    embeddings = hidden_states.mean(dim=1).squeeze()

    if quantize:
        min_val = embeddings.min()
        max_val = embeddings.max()
        normalized_embeddings = (embeddings - min_val) / (max_val - min_val)
        scaled_embeddings = normalized_embeddings * 255.0
        int8_embeddings = scaled_embeddings.to(torch.int8)
        return int8_embeddings.cpu().numpy() if torch.cuda.is_available() else int8_embeddings.numpy()

    else:
        return embeddings.cpu().numpy() if torch.cuda.is_available() else embeddings.numpy()