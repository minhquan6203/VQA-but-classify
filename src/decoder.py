import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class Decoder(torch.nn.Module):
    def __init__(self, pretrained_model_name_or_path):
        super(Decoder, self).__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path)
    
    def forward(self, fused_output):
        input_ids = torch.tensor([self.tokenizer.bos_token_id]).unsqueeze(0).to(fused_output.device)
        past = None
        while True:
            logits, past = self.gpt2(input_ids=input_ids, past=past)
            logits = logits[:, -1, :]
            next_token_id = torch.argmax(logits, dim=-1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            if next_token_id == self.tokenizer.eos_token_id or input_ids.shape[1] >= self.gpt2.config.max_position_embeddings:
                break
        return logits
