from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset

def download_and_prepare_model():
    # Nama model LLaMA yang akan diunduh
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    print(f"Downloading model {model_name}...")
    
    # Mengunduh model dan tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tambahkan pad_token jika belum ada
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model downloaded successfully.")
    
    return model, tokenizer

def perform_soft_tuning(model, tokenizer):
    # Memuat dataset
    dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")
    
    # Tokenisasi berdasarkan kolom 'input' atau 'output' tergantung kebutuhan
    def tokenize_function(examples):
        inputs = tokenizer(examples["input"], truncation=True, padding="max_length", max_length=128)
        inputs["labels"] = inputs["input_ids"].copy()  # Salin input_ids sebagai labels
        return inputs

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Gunakan DataCollatorWithPadding untuk dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Bekukan semua parameter kecuali lapisan terakhir
    for param in model.parameters():
        param.requires_grad = False
    for param in model.lm_head.parameters():
        param.requires_grad = True

    # Konfigurasi soft tuning
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=3e-5,
        per_device_train_batch_size=10,
        num_train_epochs=1,  # Sesuaikan jumlah epoch sesuai kebutuhan Anda
        save_steps=50000,
        logging_dir='./logs',
        logging_steps=500,  # Menampilkan progress setiap 500 langkah
        report_to="none",
        fp16=True
    )

    # Setup Trainer dengan dynamic padding
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator  # Dynamic padding with collator
    )

    # Lakukan tuning
    trainer.train()

    # Simpan model yang sudah dituning
    model.save_pretrained("./tuned_model")
    tokenizer.save_pretrained("./tuned_model")
    
    print("Model soft tuning completed and saved.")

if __name__ == "__main__":
    # Unduh model dan tokenizer
    model, tokenizer = download_and_prepare_model()
    
    # Lakukan soft tuning
    perform_soft_tuning(model, tokenizer)