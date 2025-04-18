# Evaluation for Finance challenge

We build a sentiment classification pipeline on finance-related text to evaluate our fine-tuned LLMs.
Three datasets have been selected for this evaluation: [FPB](https://huggingface.co/datasets/takala/financial_phrasebank), [FIQA](https://huggingface.co/datasets/pauri32/fiqa-2018), and [TFNS](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment). 


## Environment Setup

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/benchmarks/flowertune-llm/evaluation/finance ./flowertune-eval-finance && rm -rf flower && cd flowertune-eval-finance
```

Create a new Python environment (we recommend Python 3.11), activate it, then install dependencies with:

```shell
# From a new python environment, run:
pip install -r requirements.txt

# Log in HuggingFace account
huggingface-cli login
```

## Generate model decision & calculate accuracy

> [!NOTE]
> Please ensure that you use `quantization=4` to run the evaluation if you wish to participate in the LLM Leaderboard.

Download fine-tuned model from [Google Drive](https://drive.google.com/drive/folders/1BCUcVeRJiIr-xYFfNwe-pma8USO7Sqq2?usp=sharing), put everything under `results` directory

```bash
cd /your_project_path/NI-FlowerTune-LLM-Leaderboard-Finance-SmolLM2-Series
bash script.sh
```

### Benchmark

| Challenges                      | FPB        |   FIQA     |  TFNS       |   Avg      |
| :--------:                      | :--------: | :--------: | :--------:  | :--------: |
|SmolLM2-135M-Instruct (50Rounds) | 28.63      |   59.21    |  21.60      |   36.48    |
|SmolLM2-135M (50Rounds)          | 29.86      |   59.53    |  21.06      |   36.81    |
|SmolLM2-360M-Instruct (50Rounds) | 52.64      |   66.11    |  37.93      |   52.22    |
|SmolLM2-360M (50Rounds)          | 53.79      |   21.71    |  52.42      |   42.64    |

The model answers and accuracy values will be saved to `benchmarks/generation_{dataset_name}_{run_name}.jsonl` and `benchmarks/acc_{dataset_name}_{run_name}.txt`, respectively.

> [!NOTE]
> Please ensure that you provide all **three accuracy values (FPB, FIQA, TFNS)** for three evaluation datasets when submitting to the LLM Leaderboard (see the [`Make Submission`](https://github.com/adap/flower/tree/main/benchmarks/flowertune-llm/evaluation#make-submission-on-flowertune-llm-leaderboard) section).
