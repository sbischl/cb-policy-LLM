# Overview
This repository contains replication codes for ["Introducing a Textual Measure of Central Bank Policy Interactions Using ChatGPT"](https://osf.io/preprints/socarxiv/78wnp).
It contains all the codes to pre-process the dataset, run ChatGPT on two million sentences, and finally produce our indicator and aggregated results.
Moreover, we provide our manually classified validation sample `inputdata/validation_sample_all.xlsx` and the codes to conduct prompt engineering experiments, fine-tune GPT-3.5, and assess the classification quality of various ChatGPT models and Gemini Pro against this validation set.
We share a yearly aggregation of our index `dominance_coordination_dataset.csv`. This file is sufficient to produce all charts inside the appendix and main part of the paper. Importantly, we don't include any speeches or sentence-level results. The output files are more than a gigabyte in size and too large for this repository. To rerun the full analysis, the speech data would need to be scraped with the python code [here](https://github.com/HanssonMagnus/scrape_bis"). We do, however, provide the sentence-level classification of our prompt engineering results, validation exercise, and model comparisons. These are stored as Pandas DataFrames in `.pkl` format inside the `outputdata` folder.

# Instructions to run codes
- To rerun any of our analyses, an API key for ChatGPT and/or Gemini needs to be set inside the `llm_functions.py` file. Also note that these LLMs, even at a temperature set to zero, are non-deterministic. Exact results vary with each run, although with ChatGPT, usually 97%-99% of sentences are identically classified across two runs. In addition, changes to the model on OpenAI's/Google's side can impact results.
- To run R codes, the working directory should be set to the root of the project.
- Python codes expect to be run from the folder they are in.
- Validation, prompt engineering, and model comparison codes are self-contained and can be run with the inputs provided inside this repository, provided that an API key is set.

# Included files
The codes folder contains the following files:
- `0_text_preprocessing.py` This file runs the preprocessing steps described in the appendix.
- `1_chat_gpt_main_analysis.py` This code consists of the code required to run the full dataset. It requires the output produced by `0_text_preprocessing.py`.
- `2_validation_and_robustness.py` This file contains the code for the robustness checks, prompt engineering results, and different GPT versions. It requires only our validation set as input `validation_sample_all.xlsx`.
- `3_fine_tuning_and_few_shot.py` This file constructs a training dataset from our validation set, trains a fine-tuned GPT 3.5 model, and evaluates it with the remaining sample. Moreover, it contains code to run Gemini Pro using (i) the same prompts as ChatGPT and (ii) a few-shot prompting strategy.
- `llm_functions.py` Functions that are shared by the python codes are in this file. Most notably, it contains the function that takes a dataframe as input and calls either the Gemini or ChatGPT API with our prompt design. This function allows for parallel API queries to maximize rate limits.
- `merge_datasets.R` This R code calculates our relative indicator of dominance and coordination. It requires the outputs saved by `1_chat_gpt_main_analysis.py`. It also sketches how our shared dataset `dominance_coordination_dataset.csv` is produced (without including the third-party data sources).
- `run_all_charts.R` Produces all of the charts.

# Replication of Charts:
All our charts can be replicated with the R codes inside the `codes/figures` folder. Run `run_all_charts.R` to produce all charts. The R files read from the ChatGPT results provided inside the `outputdata` and the yearly aggregation of the full dataset `dominance_coordination_dataset.csv`. No access to ChatGPT is required to produce the charts. These are the files to produce the charts:
- `bin_scatter.R` Scatter charts presenting the development of dominance/coordination over time.
- `correlation.R` Pooled regressions.
- `crisis.R` Differences in fiscal dominance in crisis vs. non-crisis years.
- `levels_over_time.R` Shows the development of all three classification levels over time.
- `sentence_count_charts.R` Prompt engineering regarding the number of sentences.
- `stability.R` Stability of ChatGPT vs. Uncertainty in human coding.
- `temperature_charts.R` Prompt engineering regarding the temperature setting.
Common functions and settings to change the size of the charts are inside `functions_and_settings.R`.

# Prompts
The instructions part of our prompts are stored in the `prompts` folder. The sentences/excerpts are automatically appended to the prompt. We use a `.yaml` format to store the prompts. Our final instructions for level 1, level 2 and level 3 are in the `l1`, `l2`, `l3` subfolders. To change the prompts either modify the prompt file or modify the python code to load a different prompt.
