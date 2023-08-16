import click
from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator
import os, glob


@click.command()
@click.option(
    "--model-dir",
    default="/mnt/str/models/llama-13b-4bit-128g/",
    help="Directory containing model, tokenizer, generator",
)
@click.option(
    "--batch-size",
    default=4,
    help="Batch size",
)
def batch(model_dir, batch_size):
    # Directory containing model, tokenizer, generator

    # Locate files we need within that directory

    tokenizer_path = os.path.join(model_dir, "tokenizer.model")
    model_config_path = os.path.join(model_dir, "config.json")
    st_pattern = os.path.join(model_dir, "*.safetensors")
    model_path = glob.glob(st_pattern)[0]

    # Batched prompts

    prompts = [
        "Once upon a time,",
        "I don't like to",
        "A turbo encabulator is a",
        "In the words of Mark Twain,",
    ]

    # Create config, model, tokenizer and generator

    config = ExLlamaConfig(model_config_path)  # create config from config.json
    config.model_path = model_path  # supply path to model weights file

    model = ExLlama(config)  # create ExLlama instance and load the weights
    tokenizer = ExLlamaTokenizer(
        tokenizer_path
    )  # create tokenizer from tokenizer model file

    cache = ExLlamaCache(model, batch_size=batch_size)  # create cache for inference
    generator = ExLlamaGenerator(model, tokenizer, cache)  # create generator

    # Configure generator

    generator.disallow_tokens([tokenizer.eos_token_id])

    generator.settings.token_repetition_penalty_max = 1.2
    generator.settings.temperature = 0.95
    generator.settings.top_p = 0.65
    generator.settings.top_k = 100
    generator.settings.typical = 0.5

    # Generate, batched

    for line in prompts:
        print(line)

    for batch_idx in range(0, len(prompts), batch_size):
        output = generator.generate_simple(
            prompts[batch_idx : batch_idx + batch_size], max_new_tokens=300
        )

        print(output)
        print("----------------")


if __name__ == "__main__":
    batch()
