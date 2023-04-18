import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class VQAGPT2Decoder(torch.nn.Module):
    def __init__(self, pretrained_model_name_or_path):
        super().__init__()
        self.gpt2_model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path)
        self.fusion = torch.nn.Linear(in_features=768*2, out_features=768, bias=True)

    def forward(self, fused_output, max_length=50):
        # Đưa đầu vào fused_output qua lớp linear để giảm kích thước
        fused_output = self.fusion(fused_output)

        # Sử dụng mô hình GPT-2 để dự đoán câu trả lời
        input_ids = self.tokenizer.encode("<|startoftext|>", add_special_tokens=False, return_tensors="pt")
        input_ids = input_ids.to(fused_output.device)
        outputs = self.gpt2_model(inputs_embeds=fused_output, past_key_values=None, input_ids=input_ids)

        # Lấy logits của ký tự tiếp theo để tạo câu trả lời
        logits = outputs.logits[:, -1, :]
        for i in range(max_length):
            # Tính toán xác suất cho từ tiếp theo
            prediction_scores = logits / 0.7 # 0.7 là một hằng số được sử dụng để tăng độ đa dạng của câu trả lời
            probabilities = torch.softmax(prediction_scores, dim=-1)
            
            # Lấy mã token của từ tiếp theo dựa trên phân phối xác suất
            next_token = torch.multinomial(probabilities, num_samples=1)
            
            # Thêm mã token của từ tiếp theo vào đầu vào của mô hình
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Lấy logits cho từ tiếp theo dựa trên đầu vào mới
            outputs = self.gpt2_model(inputs_embeds=fused_output, past_key_values=None, input_ids=input_ids)
            logits = outputs.logits[:, -1, :]
            # Kiểm tra xem liệu câu trả lời đã kết thúc chưa
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return logits

