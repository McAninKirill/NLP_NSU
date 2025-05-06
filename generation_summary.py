import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import Qwen3ForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Загрузка модели
model_name = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = Qwen3ForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# Размер контекстного окна модели
context_len = 120000



class Request(BaseModel):
    text: str
    max_length: int = 1024


def split_text(text: str, chunk_size: int = context_len) -> list[str]:
    """Разбивает текст на части по заданному количеству токенов"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return [
        tokenizer.decode(tokens[i:i + chunk_size])
        for i in range(0, len(tokens), chunk_size)
    ]


def create_prompt(text: str) -> str:
    """Генерирует промпт для суммаризации"""
    return f"""Суммаризируй следующий текст максимально подробно:
{text}
Краткое содержание:"""


def summarize_chunk(text: str, max_length: int) -> str:
    """Генерирует суммаризацию для одного фрагмента текста"""
    prompt = create_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)

    attention_mask = inputs['attention_mask'].to(model.device)

    output = model.generate(
        inputs.input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_length,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    return tokenizer.decode(output[0], skip_special_tokens=True).replace(prompt, "").strip()


def recursive_summarize(text: str, max_length: int) -> str:
    """Рекурсивная суммаризация с обработкой длинных текстов"""
    tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) > context_len:
        chunks = split_text(text)
        summaries = [summarize_chunk(chunk, max_length) for chunk in chunks]
        combined = ' '.join(summaries)
        return recursive_summarize(combined, max_length)

    return summarize_chunk(text, max_length)


@app.post("/v1/summarize")
async def summarize(request: Request):
    summary = recursive_summarize(request.text, request.max_length)
    return {"text": summary}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)