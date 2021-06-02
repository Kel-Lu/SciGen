# SciGen

SciGen is a generation model trained on scientific articles based on GPT2 and the code is based heavily on HuggingFace's GPT2 transformers examples. For more information see our paper [Explaining Relationships Between Scientific Documents]()

# Downloading Trained Models

[`SciGPT2`](tbd)
[`SciGen`](https://drive.google.com/file/d/1GUwtsW0hc7pR5c59h2Xekqpjnijh6vhk/view)

# Running our Scripts

## Training
`python ft.py --output_dir=$OUTPUT_DIR  --model_type=gpt2 --model_name_or_path=$MODEL_PATH  --do_train --train_data_file=$TRAIN_FILE --max_eval_steps 10000 --save_steps=5000`


## Generation
`python val_generation.py  --model_type=gpt2  --length 50 --stop_token='. ' --tokenizer_path=$TOKENPATH --prompt=$TEST_FILE  --output_file $OUTPUT_FILE --model_name_or_path=$MODEL_PATH`
