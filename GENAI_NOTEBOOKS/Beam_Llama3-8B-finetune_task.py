from beam import App, Runtime, Image, Volume, Output
from datasets import load_dataset
from datasets import DatasetDict
from transformers import AutoTokenizer
from multiprocessing import cpu_count
from transformers import BitsAndBytesConfig
import torch
from trl import SFTTrainer
from peft import LoraConfig
from transformers import TrainingArguments
import pickle
import subprocess
import os

TOKEN_READ = os.environ['HUGGINGFACE_READ']
TOKEN_WRITE = os.environ['HUGGINGFACE_WRITE']

def apply_chat_template(example, tokenizer):
    messages = example["messages"]
    # We add an empty system message if there is none
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})
    example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)

    return example

# The runtime definition
app = App(
    "fine-tune-llama",
    runtime=Runtime(
        cpu=4,
        memory="32Gi",
        gpu='A10G',#"L4",#A10G",
        image=Image(
            python_version="python3.10",
            python_packages="requirements.txt",
        ),
    ),
    volumes=[
        # checkpoints is used to save fine-tuned models
        Volume(name="checkpoints", path="./checkpoints"),
        # pretrained-models is used to cache model weights
        Volume(name="pretrained-models", path="./pretrained-models"),
    ],
)

# Training
@app.run(timeout=7200, outputs=[Output(path="train_result.pkl")])
def train_model():
    # Trained models will be saved to this path
    beam_volume_path = "./checkpoints"

    subprocess.run(["huggingface-cli", "login", "--token", TOKEN_READ])
    subprocess.run(["huggingface-cli", "login", "--token", TOKEN_WRITE])

    # We use the vicgalle/alpaca-gpt4 dataset hosted on Huggingface:
    # https://huggingface.co/datasets/vicgalle/alpaca-gpt4
    raw_datasets = load_dataset("HuggingFaceH4/ultrachat_200k")

    indices = range(0,100)

    dataset_dict = {"train": raw_datasets["train_sft"].select(indices),
                "test": raw_datasets["test_sft"].select(indices)}

    raw_datasets = DatasetDict(dataset_dict)

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    column_names = list(raw_datasets["train"].features)
    raw_datasets = raw_datasets.map(apply_chat_template,
                                num_proc=cpu_count(),
                                fn_kwargs={"tokenizer": tokenizer},
                                remove_columns=column_names,
                                desc="Applying chat template",)
    

    # set pad_token_id equal to the eos_token_id if not set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set reasonable default for models without max length
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048
   
    # create the splits
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]


    quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
    )
    device_map = "auto"#{"": torch.cuda.current_device()} if torch.cuda.is_available() else None

    model_kwargs = dict(
    #     attn_implementation=False,#"flash_attention_2", # set this to True if your GPU supports it (Flash Attention drastically speeds up model computations)
        torch_dtype="auto",
        use_cache=False, # set to False as we're going to use gradient checkpointing
        device_map=device_map,
        quantization_config=quantization_config,
    )

    # path where the Trainer will save its checkpoints and logs
    trained_model_id = "Llama-3-8B-sft-lora-ultrachat"
    output_dir = 'saved_model/' + trained_model_id

    # based on config
    training_args = TrainingArguments(
        fp16=False, # specify bf16=True instead when training on GPUs that support bf16 else fp16
        bf16=False,
        do_eval=True,
        evaluation_strategy="epoch",
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=2.0e-05,
        log_level="info",
        logging_steps=5,
        logging_strategy="steps",
        lr_scheduler_type="cosine",
        max_steps=-1,
        num_train_epochs=1,
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_eval_batch_size=1, # originally set to 8
        per_device_train_batch_size=1, # originally set to 8
        push_to_hub=True,
        hub_model_id=trained_model_id,
        # hub_strategy="every_save",
        # report_to="tensorboard",
        save_strategy="no",
        save_total_limit=None,
        seed=42,
        report_to="none", # To turn off wandb reporting
        )

        # based on config
    peft_config = LoraConfig(
                r=64,
                lora_alpha=16,
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], ## can also add th other layers  ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
        )

    trainer = SFTTrainer(
        model=model_id,
        model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        tokenizer=tokenizer,
        packing=True,
        peft_config=peft_config,
        max_seq_length=tokenizer.model_max_length,
    )

    train_result = trainer.train()

    trainer.push_to_hub()

    with open("train_result.pkl", "wb") as f:
        pickle.dump(train_result, f)

    print('Finished')
