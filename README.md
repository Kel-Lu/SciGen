# SciGen

SciGen is a generation model trained on scientific articles based on GPT2 and the code is based heavily on HuggingFace's GPT2 transformers examples. For more information see our paper [Explaining Relationships Between Scientific Documents]()

# Downloading Trained Models

[`SciGEN`](https://drive.google.com/file/d/1WQEd8skg7JzJzYLki-04dglkfFC_EN2P/view?usp=sharing)
[`SciGPT2`](https://drive.google.com/file/d/1AoNYnhvI6tensnrpQVc09KL1NWJ5MvFU/view?usp=sharing)
[`SciGPT2_Clean`](https://drive.google.com/file/d/10AnTcF7c-yQwQAl4QAYy_UfSeiJ-r5HU/view?usp=sharing)

We note that `SciGPT2_Clean` was trained on a reduced set of papers to prevent leakage in our experiments and is released for reproducibility. In general, we recommend using the full veresion of `SciGPT2`.

# Running our Scripts

## Data Processing

Please follow the steps under `data processing`.

## Training
```python ft.py --model_type=gpt2  --do_eval --max_eval_steps 100000 --num_train_epochs=1 --save_steps=5000 --eval_all_checkpoints  --tokenizer_path=$MODEL_PATH --output_dir=$OUTPUT_PATH --eval_data_file=$EVAL_FILE --model_name_or_path=$MODEL_PATH```


## Generation
```python val_generation.py  --model_type=gpt2  --length 50 --stop_token='. ' --tokenizer_path=$TOKENPATH --prompt=$TEST_FILE  --output_file $OUTPUT_FILE --model_name_or_path=$MODEL_PATH```
