import torch
import transformers
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

MODEL_PATH = "./model_checkpoints"

def fine_tune_model():
    """ Fine-tunes a model using QLoRA and saves the trained model """
    
    model_name = "facebook/opt-1.3b"  # Replace with your LLM
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    # LoRA config
    lora_config = LoraConfig(
        r=8, lora_alpha=32, lora_dropout=0.05, task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    # Tokenize data
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Fine-tuning arguments
    training_args = transformers.TrainingArguments(
        output_dir=MODEL_PATH,
        per_device_train_batch_size=2,
        num_train_epochs=1,
        save_steps=500,
        logging_dir="./logs",
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )

    trainer.train()

    # Save fine-tuned model
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    print("✅ Fine-tuned model saved!")

if __name__ == "__main__":
    fine_tune_model()