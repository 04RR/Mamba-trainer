import torch
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


model_path = "TinyPhi"
model = MambaLMHeadModel.from_pretrained(
    pretrained_model_name=model_path, dtype=torch.bfloat16, device="cuda"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

prompt_template = """### Instruction:
{prompt}

### Response:
"""

prompt = "How to derive the gradient of a function?"
prompt_ = prompt_template.format(prompt=prompt)

prompt_tokenized = tokenizer(prompt_, return_tensors="pt").to("cuda")

output_tokenized = model.generate(
    input_ids=prompt_tokenized["input_ids"],
    max_length=512,
    enable_timing=True,
    temperature=0.7,
    top_p=0.1,
    top_k=40,
)

output = tokenizer.decode(output_tokenized[0])
print("\n", prompt)
print("\n", output.split("</s>")[0].replace(prompt_, "").strip())
