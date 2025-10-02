# Daily LLM

This repository contains the code of the paper [DailyLLM: Context-Aware Activity Log Generation Using Multi-Modal Sensors and LLMs](https://arxiv.org/pdf/2507.13737). This paper proposes a lightweight LLM-based framework that integrates structured prompting with efficient feature extraction to enable high-level activity understanding. Extensive experiments demonstrate that outperforms state-of-the-art (SOTA) log generation methods and can be efficiently deployed on personal computers and Raspberry Pi. Utilizing only a 1.5B-parameter LLM model, achieves a 17% improvement in log generation BERTScore precision compared to the 70B-parameter SOTA baseline, while delivering nearly 10 times faster inference speed.

The training and testing dataset is also available on Hugging Face: https://huggingface.co/datasets/YeTianCS/DailyLLMDataset

## Deployment and finetuning deepseek-R1 model

Our deployment and finetuning relies on [Llama-factory](https://github.com/hiyouga/LLaMA-Factory). 

### Install from Source

```shell
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```

For alternative installation methods, please refer to the Getting Started section in [Llama-factory](https://github.com/hiyouga/LLaMA-Factory)

### Prepare data

We followed the instructions in [LLaMA-Factory/data/README](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md) to prepare our dataset. 

Taking the uci dataset as an example, we added the following code in the dataset_info.json.

```json
"uci_train": {
  "file_name": "train_all_uci.jsonl",
  "formatting": "sharegpt",
  "columns": {
    "messages": "messages"
  },
  "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system"
    }
  },
```

All other datasets are similar to this. 

### Finetuning with Llama-factory

After preparing the data, we used the webui with default settings of llama-factory to finetune our model, the webui can be opened by 

```shell
llamafactory-cli webui
```

Alternatively, you could use the following instruction to train the uci dataset example

```shell
llamafactory-cli train `
    --stage sft `
    --do_train True `
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B `
    --preprocessing_num_workers 16 `
    --finetuning_type lora `
    --template deepseek3 `
    --flash_attn auto `
    --dataset_dir data `
    --dataset uci_train `
    --cutoff_len 2048 `
    --learning_rate 5e-05 `
    --num_train_epochs 3.0 `
    --max_samples 100000 `
    --per_device_train_batch_size 2 `
    --gradient_accumulation_steps 8 `
    --lr_scheduler_type cosine `
    --max_grad_norm 1.0 `
    --logging_steps 5 `
    --save_steps 100 `
    --warmup_steps 0 `
    --packing False `
    --report_to none `
    --output_dir saves\DeepSeek-R1-1.5B-Distill\lora\train_uci`
    --bf16 True `
    --plot_loss True `
    --trust_remote_code True `
    --ddp_timeout 180000000 `
    --include_num_input_tokens_seen True `
    --optim adamw_torch `
    --lora_rank 8 `
    --lora_alpha 16 `
    --lora_dropout 0 `
    --lora_target all
```

## Deployment on Raspberry Pi with llama.cpp

### Prepare quantified model

Due to unknown issues with quantization using llama.cpp, we used the quantization api of [huggingface](https://huggingface.co/spaces/ggml-org/gguf-my-repo) described on Quick start section of [llama.cpp](https://github.com/ggml-org/llama.cpp) to quantize trained models. 

We chose the Q6_K quantization method because of the Raspberry Pi's hardware problems with int8 matrix multiplication.

* First, merge deepseek original weights and trained lora weights. `merge_weights.py` is a simplest example. 

* Then upload the merged model to a huggingface repository. 

* Finally use the [quantization api](https://huggingface.co/spaces/ggml-org/gguf-my-repo) to quantize your model. You should get a file named \*.gguf

### Deployment quantized model on Raspberry Pi

To run model on Raspberry Pi, we relies on `llama.cpp`.

An example of running batch inference is at `inference_on_pi.py`, 

An example of a line from the jsonl file used in the example is as follows:

```json
{"messages": [{"role": "system", "content": "You are an expert in signal analysis. We collected audio from the user's phone to analyze the scene they were in. We extracted 120-dimensional features from the original sound signal. The first 60 features are means of 20 MFCC static coefficients (including 0th) + 20 delta MFCC coefficients + 20 acceleration MFCC coefficients, the last 60 features are standard deviations of the same coefficients. Analyze these characteristics and combine your knowledge to predict the user's environment. Choose and output one of the following labels: ['beach', 'cafe/restaurant', 'city_center', 'forest_path', 'metro_station', 'tram', 'park', 'residential_area', 'home','bus','grocery_store','car','train','office','library']."}, {"role": "user", "content": "Here are features extracted from the user's phone audio:\n\nfeatures: [-198.6126, 14.0087, 13.7941, 13.4466, 12.9809, 12.4156, 11.7721, 11.0731, 10.3405, 9.595, 8.8545, 8.1334, 7.4429, 6.7905, 6.1807, 5.6154, 5.0944, 4.6162, 4.1783, 3.7777, 0.0044, 0.0062, 0.0061, 0.006, 0.0057, 0.0055, 0.0052, 0.0048, 0.0044, 0.004, 0.0035, 0.0031, 0.0026, 0.0021, 0.0017, 0.0012, 0.0008, 0.0004, 0.0, -0.0003, -0.0026, -0.0037, -0.0036, -0.0036, -0.0035, -0.0033, -0.0032, -0.003, -0.0028, -0.0026, -0.0024, -0.0021, -0.0019, -0.0017, -0.0015, -0.0012, -0.001, -0.0008, -0.0007, -0.0005, 6.9328, 9.7015, 9.4007, 8.9254, 8.311, 7.6003, 6.8382, 6.0671, 5.323, 4.6336, 4.0179, 3.4863, 3.0423, 2.6835, 2.403, 2.1904, 2.0339, 1.9209, 1.8396, 1.78, 0.4406, 0.6142, 0.5889, 0.5504, 0.5038, 0.4548, 0.4089, 0.37, 0.3396, 0.3168, 0.2994, 0.285, 0.2721, 0.2601, 0.2488, 0.2386, 0.2295, 0.2217, 0.2154, 0.2105, 0.2918, 0.4061, 0.3874, 0.3596, 0.3268, 0.294, 0.2658, 0.2447, 0.2309, 0.2222, 0.2159, 0.2097, 0.2027, 0.1947, 0.1863, 0.1782, 0.1712, 0.1656, 0.1617, 0.1594]\nPlease analyze these features and combine your professional knowledge to determine the user's environment.Choose and output one of the following labels: ['beach', 'cafe/restaurant', 'city_center', 'forest_path', 'metro_station', 'tram', 'park', 'residential_area', 'home','bus','grocery_store','car','train','office','library']."}, {"role": "assistant", "content": "beach"}]}
```

After running `inference_on_pi.py`, you could get the evaluation results.

## Data preparation

We have uploaded the finalized dataset on Hugging Face, and you can directly access it by: https://huggingface.co/datasets/YeTianCS/DailyLLMDataset

Alternatively, you can build the dataset from the original databases by following the steps in this section: 

### IMU

All original IMU datasets can be accessed from ([UCI](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones), [MotionSense]([mmalekzadeh/motion-sense: MotionSense Dataset for Human Activity and Attribute Recognition ( time-series data generated by smartphone's sensors: accelerometer and gyroscope) (PMC Journal) (IoTDI'19)](https://github.com/mmalekzadeh/motion-sense)), [HHAR](https://archive.ics.uci.edu/dataset/344/heterogeneity+activity+recognition), [Shoaib](https://www.utwente.nl/en/eemcs/ps/research/dataset/) [PAMAP2](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)).

Except UCI dataset, we extract the original data from QA pairs at [OpenSQA v2](https://drive.google.com/drive/folders/1Dbiro41CY6f086f72Vl9MGysFUOcA0Mp). `preprocess/imu/get_original_data.py` is our code of generating csv files of original data from QA pairs. 

For UCI dataset, we get the data directly from the [link](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) above and use `preprocess/imu/extract_uci_feature_all.py` to generate csv file. The feature extraction of all other datasets are similar to the UCI one.

Then, we generate QA pairs with required format. An example of generating MotionSense QA data is at `preprocess/imu/gen_MotionSense_jsonl.py`. 

<!-- Data of activity log and summary is at [here](https://ucsdcloud-my.sharepoint.com/:f:/r/personal/yet002_ucsd_edu/Documents/SensorLLM_dataset/Activity_logs?csf=1&web=1&e=a6TNKd). And we use `preprocess/transfer_summary.py` to format the summary data. -->

### Audio

All original audio datasets (2016–2019) can be accessed from the official DCASE challenge websites, e.g.,  
https://dcase.community/challenge2016/task-acoustic-scene-classification#audio-dataset for the 2016 dataset.  

For all audio datasets, we use `preprocess/audio/organize_feature.py` to extract features and generate corresponding CSV files.  
After that, we apply `preprocess/audio/generation_jsonl_status.py` to create the QA pairs in JSONL format.

### Location

The original location dataset can be accessed by https://studentlife.cs.dartmouth.edu/datasets.html

We use `preprocess/location/extract_location.py` to create the QA pairs in JSONL format. 

### Summary

To create the summary dataset, we use `preprocess/summary/formatted_logs.py` to format the overall data and `preprocess/summary/get_QA.py` to generate the QA pairs in JSONL format.

## ML and DL Baselines

We tested IMU baselines with both original data and data with extracted features.

`baselines/baseline_origin.py` will test the ML baseline using features and the DL baseline with raw data. 

`baselines/baseline_feature.py` will test both ML and DL baseline using features. 

Also, baseline of audios is at `baselines/baseline_audio.py`. 

The code will report accuracy, precision, recall and F1 in results_accuracy.csv and write confusion matrix to results_summary.txt

## Citation

If you use this repository or find our paper/code/dataset helpful, please cite:

```bibtex
@article{tian2025dailyllm,
  title={DailyLLM: Context-Aware Activity Log Generation Using Multi-Modal Sensors and LLMs},
  author={Tian, Ye and Ren, Xiaoyuan and Wang, Zihao and Gungor, Onat and Yu, Xiaofan and Rosing, Tajana},
  journal={arXiv preprint arXiv:2507.13737},
  year={2025}
}
```