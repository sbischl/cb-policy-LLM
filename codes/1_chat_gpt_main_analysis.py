# Load llm functions
from llm_functions import *

# Comment in if running from root directory of the git project
#os.chdir("codes")

# This file requires the pre-processed sentences from text_processing.py to be stored in the inputdata folder. Furthermore, ChatGPT/Gemini API keys need to set in llm_functions.py
if __name__ == '__main__':
   # Read in the data
   sentence_level_data = pd.read_pickle("../inputdata/sentences.pkl").assign(
      sentence_id = lambda df: df.index
   )

   # Output data will be stored in a subfolder with this name. This is to avoid data loss by rerunning parts of the code. When rerunning something that is expensive increment this version parameter.
   version = "v2"

   # ──────────────────────────────────────────────────────────────────────
   #The following code runs all 3 levels of the classification. Each block is only enabled if run_LX_classification is set to true

   # This was run on 2023-08-05 22-36-04
   prompt_l1 = yaml.safe_load(Path("../prompts/l1/2_topics.yaml").read_text(encoding = "UTF-8"))
   
   run_L1_classification = False
   if run_L1_classification:

      level1_classification = run_classifcation(prompt_l1, sentence_level_data, parallel_prompts=12)

      level1_classification = sentence_level_data.merge(level1_classification, on = "sentence_id")
      level1_classification = fix_labels(level1_classification, mapping = label_mapping_l1, regex_mapping = label_mapping_l1_regex, drop_unknown = False)
      level1_classification_NAs = level1_classification.query("classification == 'NA'")
      level1_classification = level1_classification.query("classification != 'NA'")

      # Run second attampt on  level1_classification_NAs
      second_attempt_level1_classification = run_classifcation(prompt_l1, level1_classification_NAs, parallel_prompts=12)
      second_attempt_level1_classification = sentence_level_data.merge(second_attempt_level1_classification, on = "sentence_id")
      second_attempt_level1_classification = fix_labels(second_attempt_level1_classification, mapping = label_mapping_l1, regex_mapping = label_mapping_l1_regex, drop_unknown = False)

      level1_classification = pd.concat(
         [level1_classification,
         second_attempt_level1_classification]
      ).sort_values(by=['sentence_id'])

      level1_classification.to_pickle(f"../outputdata/{version}/complete_l1_classification.pkl")

      level1_classification[["sentence_id", "classification"]].to_parquet(f"../outputdata/{version}/l1_classification.parquet", index = False)

   else:
      level1_classification = pd.read_pickle(f"../outputdata/{version}/complete_l1_classification.pkl")

   run_L2_classification = False
   if run_L2_classification:

      prompt_l2 = yaml.safe_load(Path("../prompts/l2/2_normative_descriptive.yaml").read_text(encoding = "UTF-8"))

      level2_classification = run_classifcation(prompt_l2, sentence_level_data, continue_run = "2023-11-12 21-28-23", parallel_prompts=12)

      level2_classification = sentence_level_data.merge(level2_classification, on = "sentence_id")
      level2_classification = fix_labels(level2_classification, mapping = label_mapping_l2, drop_unknown = False)
      level2_classification_NAs = level2_classification.query("classification == 'NA'")
      level2_classification = level2_classification.query("classification != 'NA'")

      # Run second attampt on  level2_classification_NAs
      second_attempt_level2_classification = run_classifcation(prompt_l2, level2_classification_NAs, parallel_prompts=12)
      second_attempt_level2_classification = sentence_level_data.merge(second_attempt_level2_classification, on = "sentence_id")
      second_attempt_level2_classification = fix_labels(second_attempt_level2_classification, mapping = label_mapping_l2, drop_unknown = False)

      level2_classification = pd.concat(
         [level2_classification,
         second_attempt_level2_classification]
      ).sort_values(by=['sentence_id'])

      level2_classification.to_pickle(f"../outputdata/{version}/complete_l2_classification.pkl")

      level2_classification[["sentence_id", "classification"]].to_parquet(f"../outputdata/{version}/l2_classification.parquet", index = False)

   else:
      level2_classification = pd.read_pickle(f"../outputdata/{version}/complete_l2_classification.pkl")

   run_L3_classification = False
   if run_L3_classification:

      prompt_l3 = yaml.safe_load(Path("../prompts/l3/2_dominance_cooperation.yaml").read_text(encoding = "UTF-8"))

      sentence_level_data_with_context = sentence_level_data.assign(
         context_1_1 = sentence_level_data.groupby("speech_identifier",  sort = False, group_keys = False).apply(lambda df: add_neighbouring_sentences(df)))
      
      sentence_level_data_with_context.to_parquet(f"../outputdata/{version}/sentences_and_context.parquet", index = False)
      
      sentence_level_data_with_context["sentence"] = sentence_level_data_with_context["context_1_1"]

      sentence_level_data_with_context = sentence_level_data_with_context.sample(frac=1, random_state = 42)

      level3_classification = run_classifcation(prompt_l3, sentence_level_data_with_context, parallel_prompts=12)


      level3_classification = sentence_level_data_with_context.merge(level3_classification, on = "sentence_id")

      level3_classification = fix_labels(level3_classification, mapping = label_mapping_l3, regex_mapping = label_mapping_l3_regex, drop_unknown = False)

      level3_classification_NAs = level3_classification.query("classification == 'NA'").sample(frac=1, random_state = 4242)
      level3_classification = level3_classification.query("classification != 'NA'")

      # Run second attempt on level3_classification_NAs
      second_attempt_level3_classification = run_classifcation(prompt_l3, level3_classification_NAs, parallel_prompts=12)
      second_attempt_level3_classification = sentence_level_data_with_context.merge(second_attempt_level3_classification, on = "sentence_id")
      second_attempt_level3_classification = fix_labels(second_attempt_level3_classification, mapping = label_mapping_l3, regex_mapping = label_mapping_l3_regex, drop_unknown = False)

      level3_classification = pd.concat(
         [level3_classification,
         second_attempt_level3_classification]
      ).sort_values(by=['sentence_id'])

      level3_classification.to_pickle(f"../outputdata/{version}/complete_l3_classification.pkl")

      level3_classification[["sentence_id", "classification"]].to_csv(f"../outputdata/{version}/l3_classification.csv", index = False)

      level3_classification[["sentence_id", "classification"]].to_parquet(f"../outputdata/{version}/l3_classification.parquet", index = False)

   else:
      level3_classification = pd.read_pickle(f"../outputdata/{version}/complete_l3_classification.pkl")