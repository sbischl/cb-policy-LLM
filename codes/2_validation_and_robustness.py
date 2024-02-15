# Load llm functions
from llm_functions import *

# Comment in if running from root directory of the git project
# os.chdir("codes")

# This file requires the pre-processed sentences from text_processing.py to be stored in the inputdata folder. Furthermore, ChatGPT/Gemini API keys need to set in llm_functions.py
if __name__ == '__main__':
   # The following run the validation set anaylsis. How well can chatGPT match the manually classified sentences?
   # Set this flag to run ChatGPT otherwise previous results are loaded.
   run_chat_gpt = False

   validation_sample = pd.read_excel("../inputdata/validation_sample_all.xlsx")

   # Specify the label mappings to be used for all 3 stages.
   label_mapping_topic_val = {
      "financial": "financial",
      "monetary": "monetary",
      "fiscal": "fiscal",
      "macro": "macro",
      "climate": "climate",
      "international": "international",
      "other": "other",
      "inflation": "monetary",
      "Monetary (in French)": "monetary",
   }

   label_mapping_dominance_cooperation = {
      "monetary dominance": "monetary dominance",
      "financial dominance": "financial dominance",
      "fiscal dominance": "fiscal dominance",
      "monetary-fiscal coordination": "monetary-fiscal coordination",
      "fiscal-monetary coordination": "monetary-fiscal coordination",
      "Monetary coordination": "monetary-fiscal coordination",
      "monetary-financial coordination": "monetary-financial coordination",
      "Monetary-prudential coordination": "monetary-financial coordination",
      "none": "none",
      "monetary policy": "monetary dominance",
      "monetary": "monetary dominance",
      "financial": "financial dominance",
      "Financial regulation dominance": "monetary dominance"
   }

   label_normative_descriptive = {
      "normative": "normative",
      "descriptive": "descriptive",
      "normative \\(value judgement\\)": "normative"
   }

   # Create new columns that contain the agreed value. _agree is the number of coders agreeing. 3 if all agree, 2 if two agree,
   # 1 if there is complete disagreement. In the latter case, the "agreed" classification is chosen randomly. Hence
   # the set.seed.
   np.random.seed(3)
   validation_sample = validation_sample.assign(
      A_level_1=validation_sample[["S_level_1", "M_level_1", "L_level_1"]].mode(axis=1).apply(
         lambda x: x.dropna().sample(1).item(), axis=1),
      A_level_1_agree=4 - validation_sample[["S_level_1", "M_level_1", "L_level_1"]].nunique(axis=1),
      A_level_2=validation_sample[["S_level_2", "M_level_2", "L_level_2"]].mode(axis=1).apply(
         lambda x: x.dropna().sample(1).item(), axis=1),
      A_level_2_agree=4 - validation_sample[["S_level_2", "M_level_2", "L_level_2"]].nunique(axis=1),
      A_level_3=validation_sample[["S_level_3", "M_level_3", "L_level_3"]].mode(axis=1).apply(
         lambda x: x.dropna().sample(1).item(), axis=1),
      A_level_3_agree=4 - validation_sample[["S_level_3", "M_level_3", "L_level_3"]].nunique(axis=1),
   )

   # This runs chatGPT on all 3 levels of the classification set. If run_chat_gpt is true. Otherwise it is loaded from
   # from the outputdata folder
   if run_chat_gpt:
      # This produces accuracy numbers
      topic_prompt = yaml.safe_load(Path("../prompts/l1/2_topics.yaml").read_text(encoding="UTF-8"))
      topic_classification = run_classifcation(topic_prompt, validation_sample, parallel_prompts=12)
      topic_classification = fix_labels(topic_classification, mapping=label_mapping_topic_val, drop_unknown=False)

      normative_descriptive_prompt = yaml.safe_load(
         Path("../prompts/l2/2_normative_descriptive.yaml").read_text(encoding="UTF-8"))
      normative_descriptive_classification = run_classifcation(normative_descriptive_prompt, validation_sample,
                                                               parallel_prompts=12)
      normative_descriptive_classification = fix_labels(normative_descriptive_classification,
                                                         mapping=label_normative_descriptive, drop_unknown=False)

      dominance_cooperation_prompt = yaml.safe_load(
         Path("../prompts/l3/2_dominance_cooperation.yaml").read_text(encoding="UTF-8"))
      dominance_cooperation_classification = run_classifcation(dominance_cooperation_prompt, validation_sample.assign(
         sentence=lambda df: df["three-sentence"]
      ), parallel_prompts=12)
      dominance_cooperation_classification = fix_labels(dominance_cooperation_classification,
                                                         mapping=label_mapping_dominance_cooperation,
                                                         drop_unknown=False, unkown_category="none")

      validation_sample_with_chatgpt = validation_sample.merge(
         topic_classification.rename({"classification": "topic_chatGPT"}, axis=1)[["topic_chatGPT", "sentence_id"]],
         on="sentence_id").merge(
         normative_descriptive_classification.rename({"classification": "normative_descriptive_chatGPT"}, axis=1)[
               ["normative_descriptive_chatGPT", "sentence_id"]],
         on="sentence_id").merge(
         dominance_cooperation_classification.rename({"classification": "dominance_cooperation_chatGPT"}, axis=1)[
               ["dominance_cooperation_chatGPT", "sentence_id"]],
         on="sentence_id"
      )
      validation_sample_with_chatgpt.to_pickle("../outputdata/validation/validation_set_plus_chatgpt.pkl")
   else:
      validation_sample_with_chatgpt = pd.read_pickle("../outputdata/validation/validation_set_plus_chatgpt.pkl")


   def overlap_matrix(df):
      """Produces a overlap matrix between columns. Useful to see how much different coders agree"""

      def pairwise_comp(series_1, series_2):
         return ((series_1 == series_2).mean())

      return (df.stack().rank(method='dense').unstack().corr(method=pairwise_comp))


   # Calculation of accuracy scores uses either the "agreed" sample, where all three coders agree, or the entire 1000 sentences
   validation_sample_first_400 = validation_sample_with_chatgpt.loc[0:399, :]
   validation_sample_with_chatgpt_aggreement = validation_sample_with_chatgpt.query("A_level_3_agree >= 3").loc[0:399, : ]

   # Calculate overlap agreement
   overlap_matrix(validation_sample_with_chatgpt_aggreement[["S_level_1", "M_level_1", "L_level_1", "topic_chatGPT"]])
   overlap_matrix(validation_sample_with_chatgpt_aggreement[
                     ["S_level_2", "M_level_2", "L_level_2", "normative_descriptive_chatGPT"]])
   overlap_matrix(validation_sample_with_chatgpt_aggreement[
                     ["S_level_3", "M_level_3", "L_level_3", "dominance_cooperation_chatGPT"]])

   overlap_matrix(validation_sample_first_400[["S_level_1", "M_level_1", "L_level_1", "topic_chatGPT"]])
   overlap_matrix(
      validation_sample_first_400[["S_level_2", "M_level_2", "L_level_2", "normative_descriptive_chatGPT"]])
   overlap_matrix(
      validation_sample_first_400[["S_level_3", "M_level_3", "L_level_3", "dominance_cooperation_chatGPT"]])

   # Scores L1
   calculate_scores(validation_sample_with_chatgpt["A_level_1"], validation_sample_with_chatgpt["topic_chatGPT"])
   calculate_scores(validation_sample_with_chatgpt_aggreement["A_level_1"],
                  validation_sample_with_chatgpt_aggreement["topic_chatGPT"])

   calculate_confusion(validation_sample_with_chatgpt["A_level_1"], validation_sample_with_chatgpt["topic_chatGPT"])
   calculate_confusion(validation_sample_with_chatgpt_aggreement["A_level_1"],
                     validation_sample_with_chatgpt_aggreement["topic_chatGPT"])

   category_specific_accuracy(validation_sample_with_chatgpt["A_level_1"],
                              validation_sample_with_chatgpt["topic_chatGPT"])
   category_specific_accuracy(validation_sample_with_chatgpt_aggreement["A_level_1"],
                              validation_sample_with_chatgpt_aggreement["topic_chatGPT"])

   # Scores L2
   calculate_scores(validation_sample_with_chatgpt["A_level_2"],
                  validation_sample_with_chatgpt["normative_descriptive_chatGPT"])
   calculate_scores(validation_sample_with_chatgpt_aggreement["A_level_2"],
                  validation_sample_with_chatgpt_aggreement["normative_descriptive_chatGPT"])

   calculate_confusion(validation_sample_with_chatgpt["A_level_2"],
                     validation_sample_with_chatgpt["normative_descriptive_chatGPT"])
   calculate_confusion(validation_sample_with_chatgpt_aggreement["A_level_2"],
                     validation_sample_with_chatgpt_aggreement["normative_descriptive_chatGPT"])

   category_specific_accuracy(validation_sample_with_chatgpt["A_level_2"],
                              validation_sample_with_chatgpt["normative_descriptive_chatGPT"])
   category_specific_accuracy(validation_sample_with_chatgpt_aggreement["A_level_2"],
                              validation_sample_with_chatgpt_aggreement["normative_descriptive_chatGPT"])

   # Scores L3
   calculate_scores(validation_sample_with_chatgpt["A_level_3"],
                  validation_sample_with_chatgpt["dominance_cooperation_chatGPT"])
   calculate_scores(validation_sample_with_chatgpt_aggreement["A_level_3"],
                  validation_sample_with_chatgpt_aggreement["dominance_cooperation_chatGPT"])

   calculate_confusion(validation_sample_with_chatgpt["A_level_3"],
                     validation_sample_with_chatgpt["dominance_cooperation_chatGPT"])
   calculate_confusion(validation_sample_with_chatgpt_aggreement["A_level_3"],
                     validation_sample_with_chatgpt_aggreement["dominance_cooperation_chatGPT"])

   category_specific_accuracy(validation_sample_with_chatgpt["A_level_3"],
                              validation_sample_with_chatgpt["dominance_cooperation_chatGPT"])
   category_specific_accuracy(validation_sample_with_chatgpt_aggreement["A_level_3"],
                              validation_sample_with_chatgpt_aggreement["dominance_cooperation_chatGPT"])

   # ─── Calculate And Plot Confusion Matrix Using Seaborn ────────
   sns.set(font_scale=1.65)
   plt.rcParams["font.family"] = "Palatino Linotype"
   fig = plt.figure(figsize=(15, 15), constrained_layout=False)
   gs = gridspec.GridSpec(2, 4, figure=fig)
   gs.update(wspace=0.25, hspace=0.6)
   ax1 = plt.subplot(gs[0, :2], )
   ax2 = plt.subplot(gs[0, 2:])
   ax3 = plt.subplot(gs[1, 1:3])
   ax1.set_title("Level 1", fontsize=25, pad=15)
   ax2.set_title("Level 2", fontsize=25, pad=15)
   ax3.set_title("Level 3", fontsize=25, pad=15)
   #sns.set(font="Arial")
   plt.rcParams.update({'font.size': 16})


   # fig, axs = plt.subplots(ncols=2, nrows = 2, figsize=(15, 15))
   sns.heatmap(calculate_confusion(validation_sample_with_chatgpt["A_level_1"],
                                 validation_sample_with_chatgpt["topic_chatGPT"]), annot=True, cmap='Reds',
               cbar=True, ax=ax1, fmt=".2f").set(xlabel='Predicted labels (share)', ylabel='True label')
   sns.heatmap(calculate_confusion(validation_sample_with_chatgpt["A_level_2"], validation_sample_with_chatgpt
   ["normative_descriptive_chatGPT"]), cbar=True, ax=ax2, fmt=".2f", annot=True, cmap='Greens').set(
      xlabel='Predicted labels (share)', ylabel='True label')
   sns.heatmap(calculate_confusion(validation_sample_with_chatgpt["A_level_3"],
                                 validation_sample_with_chatgpt["dominance_cooperation_chatGPT"]), cbar=True, ax=ax3,
               fmt=".2f", annot=True, cmap='Blues').set(xlabel='Predicted labels (share)', ylabel='True label')
   
   ax1.figure.axes[-1].tick_params(labelsize=16)
   ax1.figure.axes[-2].tick_params(labelsize=16)
   ax1.figure.axes[-3].tick_params(labelsize=16)

   fig.savefig('../figures/confusion_matrices.pdf', bbox_inches="tight")

   # Scores
   pd.DataFrame(
      [calculate_scores(validation_sample_with_chatgpt["A_level_1"], validation_sample_with_chatgpt["topic_chatGPT"]),
      calculate_scores(validation_sample_with_chatgpt["A_level_2"],
                        validation_sample_with_chatgpt["normative_descriptive_chatGPT"]),
      calculate_scores(validation_sample_with_chatgpt["A_level_3"],
                        validation_sample_with_chatgpt["dominance_cooperation_chatGPT"])]
   ).transpose().set_axis(['Level 1', 'Level 2', 'Level 3'], axis=1).to_excel(
      "../outputdata/validation/accuracy_scores_fullsample.xlsx")

   # Scores Agreement
   pd.DataFrame(
      [calculate_scores(validation_sample_with_chatgpt_aggreement["A_level_1"],
                        validation_sample_with_chatgpt_aggreement["topic_chatGPT"]),
      calculate_scores(validation_sample_with_chatgpt_aggreement["A_level_2"],
                        validation_sample_with_chatgpt_aggreement["normative_descriptive_chatGPT"]),
      calculate_scores(validation_sample_with_chatgpt_aggreement["A_level_3"],
                        validation_sample_with_chatgpt_aggreement["dominance_cooperation_chatGPT"])]
   ).transpose().set_axis(['Level 1', 'Level 2', 'Level 3'], axis=1).to_excel(
      "../outputdata/validation/accuracy_scores_agreesample.xlsx")

   # Combined table of scores
   pd.concat([
      pd.DataFrame(
         [calculate_scores(validation_sample_with_chatgpt_aggreement["A_level_1"],
                           validation_sample_with_chatgpt_aggreement["topic_chatGPT"]),
            calculate_scores(validation_sample_with_chatgpt_aggreement["A_level_2"],
                           validation_sample_with_chatgpt_aggreement["normative_descriptive_chatGPT"]),
            calculate_scores(validation_sample_with_chatgpt_aggreement["A_level_3"],
                           validation_sample_with_chatgpt_aggreement["dominance_cooperation_chatGPT"])]
      ).set_axis(['Level 1', 'Level 2', 'Level 3'], axis=0),
      pd.DataFrame(
         [calculate_scores(validation_sample_with_chatgpt["A_level_1"],
                           validation_sample_with_chatgpt["topic_chatGPT"]),
            calculate_scores(validation_sample_with_chatgpt["A_level_2"],
                           validation_sample_with_chatgpt["normative_descriptive_chatGPT"]),
            calculate_scores(validation_sample_with_chatgpt["A_level_3"],
                           validation_sample_with_chatgpt["dominance_cooperation_chatGPT"])]
      ).set_axis(['Level 1', 'Level 2', 'Level 3'], axis=0),
   ])


   # Calculate standardized Overlap figures using nltk's AnnotationTask.
   def gen_tuples(coder, level):
      """Generates the tuples required by AnnotationTask"""
      if coder == "C":
         column = {
               "level_1": "topic_chatGPT",
               "level_2": "normative_descriptive_chatGPT",
               "level_3": "dominance_cooperation_chatGPT",
         }[level]
      else:
         column = f"{coder}_{level}"

      return_list = []
      for index, row in validation_sample_first_400.iterrows():
         return_list.append(
               (coder, index, row[column])
         )
      return (return_list)


   # How does overlap change when replacing one coder with chatGPT:
   def replace_one_with_C_and_average(level, metric="alpha"):
      """Replace one of the coders with ChatGPT and average all possible combinations"""
      if metric == "alpha":
         a1 = AnnotationTask([*gen_tuples("S", level), *gen_tuples("M", level), *gen_tuples("C", level)]).alpha()
         a2 = AnnotationTask([*gen_tuples("S", level), *gen_tuples("C", level), *gen_tuples("L", level)]).alpha()
         a3 = AnnotationTask([*gen_tuples("C", level), *gen_tuples("M", level), *gen_tuples("L", level)]).alpha()  #
      elif metric == "kappa":
         a1 = AnnotationTask([*gen_tuples("S", level), *gen_tuples("M", level), *gen_tuples("C", level)]).kappa()
         a2 = AnnotationTask([*gen_tuples("S", level), *gen_tuples("C", level), *gen_tuples("L", level)]).kappa()
         a3 = AnnotationTask([*gen_tuples("C", level), *gen_tuples("M", level), *gen_tuples("L", level)]).kappa()
      return ((a1 + a2 + a3) / 3)


   # The DataFrame contains two metrics. First, the agreement of the human coders, and the average of the agreement when replacing each of the coders with chatGPT and then taking the average of all combinations
   pd.DataFrame(
      {
         "level_1": [AnnotationTask(
               [*gen_tuples("S", "level_1"), *gen_tuples("M", "level_1"), *gen_tuples("L", "level_1")]).alpha(),
                     replace_one_with_C_and_average("level_1", metric="alpha"),
                     AnnotationTask(
                           [*gen_tuples("S", "level_1"), *gen_tuples("M", "level_1"), *gen_tuples("L", "level_1"),
                           *gen_tuples("C", "level_1")]).alpha(),
                     AnnotationTask([*gen_tuples("S", "level_1"), *gen_tuples("M", "level_1"),
                                       *gen_tuples("L", "level_1")]).kappa(),
                     replace_one_with_C_and_average("level_1", metric="kappa"),
                     AnnotationTask(
                           [*gen_tuples("S", "level_1"), *gen_tuples("M", "level_1"), *gen_tuples("L", "level_1"),
                           *gen_tuples("C", "level_1")]).kappa()],
         "level_2": [AnnotationTask(
               [*gen_tuples("S", "level_2"), *gen_tuples("M", "level_2"), *gen_tuples("L", "level_2")]).alpha(),
                     replace_one_with_C_and_average("level_2", metric="alpha"),
                     AnnotationTask(
                           [*gen_tuples("S", "level_2"), *gen_tuples("M", "level_2"), *gen_tuples("L", "level_2"),
                           *gen_tuples("C", "level_2")]).alpha(),
                     AnnotationTask([*gen_tuples("S", "level_2"), *gen_tuples("M", "level_2"),
                                       *gen_tuples("L", "level_2")]).kappa(),
                     replace_one_with_C_and_average("level_2", metric="kappa"),
                     AnnotationTask(
                           [*gen_tuples("S", "level_2"), *gen_tuples("M", "level_2"), *gen_tuples("L", "level_2"),
                           *gen_tuples("C", "level_2")]).kappa()],
         "level_3": [AnnotationTask(
               [*gen_tuples("S", "level_3"), *gen_tuples("M", "level_3"), *gen_tuples("L", "level_3")]).alpha(),
                     replace_one_with_C_and_average("level_3", metric="alpha"),
                     AnnotationTask(
                           [*gen_tuples("S", "level_3"), *gen_tuples("M", "level_3"), *gen_tuples("L", "level_3"),
                           *gen_tuples("C", "level_3")]).alpha(),
                     AnnotationTask([*gen_tuples("S", "level_3"), *gen_tuples("M", "level_3"),
                                       *gen_tuples("L", "level_3")]).kappa(),
                     replace_one_with_C_and_average("level_3", metric="kappa"),
                     AnnotationTask(
                           [*gen_tuples("S", "level_3"), *gen_tuples("M", "level_3"), *gen_tuples("L", "level_3"),
                           *gen_tuples("C", "level_3")]).kappa()]
      }
   ).set_axis(["Humans (alpha)", "Human replaced by ChatGPT (alpha)", "Human + ChatGPT (alpha)", "Humans (kappa)",
               "Human replaced by ChatGPT (kappa)", "Human + ChatGPT (kappa)"], axis="index").round(2)


   # -------------------------------------------------------------------#
   # Experiment with different models
   # -------------------------------------------------------------------#
   run_different_models = False
   additional_models = ["gpt-3.5-turbo-0125", "gpt-4-0125-preview"]
   if run_different_models:
      results = {}

      for model in additional_models:
         l1 = run_classifcation(topic_prompt, validation_sample, chatgpt_model=model, parallel_prompts=5)
         l2 = run_classifcation(normative_descriptive_prompt, validation_sample, chatgpt_model=model,
                                 parallel_prompts=12)
         l3 = run_classifcation(dominance_cooperation_prompt, validation_sample.assign(
               sentence=lambda df: df["three-sentence"]
         ), chatgpt_model=model, parallel_prompts=12)

         results[model] = {
               "l1": l1,
               "l2": l2,
               "l3": l3
         }

      pd.to_pickle(results, "../outputdata/validation/additional_models.pkl")
   else:
      results = pd.read_pickle("../outputdata/validation/additional_models.pkl")

   results["gpt-3.5-turbo-301"] = {
      "l1": pd.read_pickle("../outputdata/validation/validation_set_plus_chatgpt.pkl").rename(
         {"topic_chatGPT": "classification"}, axis=1),
      "l2": pd.read_pickle("../outputdata/validation/validation_set_plus_chatgpt.pkl").rename(
         {"normative_descriptive_chatGPT": "classification"}, axis=1),
      "l3": pd.read_pickle("../outputdata/validation/validation_set_plus_chatgpt.pkl").rename(
         {"dominance_cooperation_chatGPT": "classification"}, axis=1)
   }
   scores_full = {}
   for model in ["gpt-3.5-turbo-301", *additional_models]:
      scores_full[f"l1-{model}"] = calculate_scores(validation_sample["A_level_1"],
                                                   fix_labels(results[model]["l1"], mapping=label_mapping_topic_val,
                                                               drop_unknown=False)["classification"])
      scores_full[f"l2-{model}"] = calculate_scores(validation_sample["A_level_2"],
                                                   fix_labels(results[model]["l2"], mapping=label_normative_descriptive,
                                                               drop_unknown=False)["classification"])
      scores_full[f"l3-{model}"] = calculate_scores(validation_sample["A_level_3"], fix_labels(results[model]["l3"],
                                                                                                mapping=label_mapping_dominance_cooperation,
                                                                                                drop_unknown=False)[
         "classification"])

   scores_agree = {}
   for model in ["gpt-3.5-turbo-301", *additional_models]:
      scores_agree[f"l1-{model}"] = calculate_scores(
         validation_sample["A_level_1"][validation_sample_with_chatgpt_aggreement.index],
         fix_labels(results[model]["l1"], mapping=label_mapping_topic_val, drop_unknown=False)["classification"][
               validation_sample_with_chatgpt_aggreement.index])
      scores_agree[f"l2-{model}"] = calculate_scores(
         validation_sample["A_level_2"][validation_sample_with_chatgpt_aggreement.index],
         fix_labels(results[model]["l2"], mapping=label_normative_descriptive, drop_unknown=False)["classification"][
               validation_sample_with_chatgpt_aggreement.index])
      scores_agree[f"l3-{model}"] = calculate_scores(
         validation_sample["A_level_3"][validation_sample_with_chatgpt_aggreement.index],
         fix_labels(results[model]["l3"], mapping=label_mapping_dominance_cooperation, drop_unknown=False)[
               "classification"][validation_sample_with_chatgpt_aggreement.index])

   pd.concat(
      [
         pd.DataFrame(scores_agree).sort_index(axis=1),
         pd.DataFrame(scores_full).sort_index(axis=1)
      ]
   ).round(2)

   # -------------------------------------------------------------------#
   # Robustness checks to study the effects of temperature, sentence count, system message etc.
   # -------------------------------------------------------------------#

   robustness_checks = False
   if robustness_checks:
      # -------------------------------------------------------------------#
      # Vary temperature setting
      # -------------------------------------------------------------------#

      vary_temp_l1 = cross_validation(
         input_data=validation_sample,
         ytrue=validation_sample["A_level_1"],
         mapping=label_mapping_l1,
         regex_mapping=label_mapping_l1_regex,
         prompt=[yaml.safe_load(Path("../prompts/l1/2_topics.yaml").read_text(encoding="UTF-8"))],
         temperature=[0, 0.25, 0.5, 0.75, 1] * 5
      )

      vary_temp_l2 = cross_validation(
         input_data=validation_sample,
         ytrue=validation_sample["A_level_2"],
         mapping=label_mapping_l2,
         prompt=[yaml.safe_load(Path("../prompts/l2/2_normative_descriptive.yaml").read_text(encoding="UTF-8"))],
         temperature=[0, 0.25, 0.5, 0.75, 1] * 5
      )

      vary_temp_l3 = cross_validation(
         input_data=validation_sample.assign(
               sentence=lambda df: df["three-sentence"]),
         ytrue=validation_sample["A_level_3"],
         mapping=label_mapping_l3,
         regex_mapping=label_mapping_l3_regex,
         prompt=[yaml.safe_load(Path("../prompts/l3/2_dominance_cooperation.yaml").read_text(encoding="UTF-8"))],
         temperature=[0, 0.25, 0.5, 0.75, 1] * 5
      )

      vary_temp_l1.to_pickle("../outputdata/robustness/temperature/temperature_l1.pkl")
      vary_temp_l1[["accuracy", "f1_weighted", "temperature"]].to_stata(
         "../outputdata/robustness/temperature/temperature_l1.dta")

      vary_temp_l2.to_pickle("../outputdata/robustness/temperature/temperature_l2.pkl")
      vary_temp_l2[["accuracy", "f1_weighted", "temperature"]].to_stata(
         "../outputdata/robustness/temperature/temperature_l2.dta")

      vary_temp_l3.to_pickle("../outputdata/robustness/temperature/temperature_l3.pkl")
      vary_temp_l3[["accuracy", "f1_weighted", "temperature"]].to_stata(
         "../outputdata/robustness/temperature/temperature_l3.dta")
      
      # -------------------------------------------------------------------#
      # Vary sentences included in a single prompt
      # -------------------------------------------------------------------#

      vary_sentence_counts_l1 = cross_validation(
         input_data=validation_sample,
         ytrue=validation_sample["A_level_1"],
         mapping=label_mapping_l1,
         regex_mapping=label_mapping_l1_regex,
         prompt=[yaml.safe_load(Path("../prompts/l1/2_topics.yaml").read_text(encoding="UTF-8"))],
         maximum_sentences=[3, 5, 10, 25, 50, 75]
      )

      vary_sentence_counts_l2 = cross_validation(
         input_data=validation_sample,
         ytrue=validation_sample["A_level_2"],
         mapping=label_mapping_l2,
         prompt=[yaml.safe_load(Path("../prompts/l2/2_normative_descriptive.yaml").read_text(encoding="UTF-8"))],
         maximum_sentences=[3, 5, 10, 25, 50, 75]
      )

      vary_sentence_counts_l3 = cross_validation(
         input_data=validation_sample.assign(
               sentence=lambda df: df["three-sentence"]),
         ytrue=validation_sample["A_level_3"],
         mapping=label_mapping_l3,
         regex_mapping=label_mapping_l3_regex,
         prompt=[yaml.safe_load(Path("../prompts/l3/2_dominance_cooperation.yaml").read_text(encoding="UTF-8"))],
         maximum_sentences=[3, 5, 10, 25, 50, 75]
      )

      vary_sentence_counts_l1.to_pickle("../outputdata/robustness/sentence_count/sentence_count_l1.pkl")
      vary_sentence_counts_l1[["tokens_used", "accuracy", "f1_weighted", "maximum_sentences"]].to_stata(
         "../outputdata/robustness/sentence_count/sentence_count_l1.dta")

      vary_sentence_counts_l2.to_pickle("../outputdata/robustness/sentence_count/sentence_count_l2.pkl")
      vary_sentence_counts_l2[["tokens_used", "accuracy", "f1_weighted", "maximum_sentences"]].to_stata(
         "../outputdata/robustness/sentence_count/sentence_count_l2.dta")

      vary_sentence_counts_l3.to_pickle("../outputdata/robustness/sentence_count/sentence_count_l3.pkl")
      vary_sentence_counts_l3[["tokens_used", "accuracy", "f1_weighted", "maximum_sentences"]].to_stata(
         "../outputdata/robustness/sentence_count/sentence_count_l3.dta")

      # -------------------------------------------------------------------#
      # Test stability of prompts
      # -------------------------------------------------------------------#

      stability_l1 = cross_validation(
         input_data=validation_sample.loc[0:399, :],
         ytrue=validation_sample.loc[0:399, :]["A_level_1"],
         mapping=label_mapping_l1,
         regex_mapping=label_mapping_l1_regex,
         prompt=[yaml.safe_load(Path("../prompts/l1/2_topics.yaml").read_text(encoding="UTF-8"))],
         maximum_sentences=[10] * 25,
         temperature=[0.25]
      )
      stability_l1.to_pickle("../outputdata/robustness/stability/stability_l1.pkl")

      classifications_l1 = pd.DataFrame()
      for run in range(stability_l1.shape[0]):
         classifications_l1[f"run_{run}"] = stability_l1.data[run].classification

      stability_vs_agreement_l1 = validation_sample.loc[0:399, :].assign(
         sentence_stability=classifications_l1.apply(lambda x: x.value_counts().tolist()[0] / len(x), axis=1)
      )[["A_level_1_agree", "sentence_stability"]].groupby("A_level_1_agree").mean()

      stability_l2 = cross_validation(
         input_data=validation_sample.loc[0:399, :],
         ytrue=validation_sample.loc[0:399, :]["A_level_2"],
         mapping=label_mapping_l2,
         prompt=[yaml.safe_load(Path("../prompts/l2/2_normative_descriptive.yaml").read_text(encoding="UTF-8"))],
         maximum_sentences=[10] * 25,
         temperature=[0.25]
      )

      stability_l2.to_pickle("../outputdata/robustness/stability/stability_l2.pkl")

      classifications_l2 = pd.DataFrame()
      for run in range(stability_l2.shape[0]):
         classifications_l2[f"run_{run}"] = stability_l2.data[run].classification

      stability_vs_agreement_l2 = validation_sample.loc[0:399, :].assign(
         sentence_stability=classifications_l2.apply(lambda x: x.value_counts().tolist()[0] / len(x), axis=1)
      )[["A_level_2_agree", "sentence_stability"]].groupby("A_level_2_agree").mean()

      stability_l3 = cross_validation(
         input_data=validation_sample.loc[0:399, :].assign(
               sentence=lambda df: df["three-sentence"]
         ),
         ytrue=validation_sample.loc[0:399, :]["A_level_3"],
         mapping=label_mapping_l3,
         regex_mapping=label_mapping_l3_regex,
         prompt=[yaml.safe_load(Path("../prompts/l3/2_dominance_cooperation.yaml").read_text(encoding="UTF-8"))],
         maximum_sentences=[10] * 25,
         temperature=[0.25]
      )

      stability_l3.to_pickle("../outputdata/robustness/stability/stability_l3.pkl")

      classifications_l3 = pd.DataFrame()
      for run in range(stability_l3.shape[0]):
         classifications_l3[f"run_{run}"] = stability_l3.data[run].classification

      stability_vs_agreement_l3 = validation_sample.loc[0:399, :].assign(
         sentence_stability=classifications_l3.apply(lambda x: x.value_counts().tolist()[0] / len(x), axis=1)
      )[["A_level_3_agree", "sentence_stability"]].groupby("A_level_3_agree").mean()

      # Save everything in one stata file to produce charts
      stability_vs_agreement_l1.merge(
         stability_vs_agreement_l2,
         how="outer",
         right_index=True, left_index=True
      ).merge(
         stability_vs_agreement_l3,
         how="outer",
         right_index=True, left_index=True
      ).set_axis(['l1', 'l2', 'l3'], axis=1).reset_index().rename({"index": "agree"}, axis=1).to_stata(
         "../outputdata/robustness/stability/stability.dta")
      

      # -------------------------------------------------------------------#
      # Compare different prompts only L3:
      # -------------------------------------------------------------------#
      
      simplified_prompt_context = yaml.safe_load(
         Path("../prompts/prompt_variations/shorter_l3_context.yaml").read_text(encoding="UTF-8"))
      complex_prompt_context = yaml.safe_load(
         Path("../prompts/prompt_variations/longer_l3_context.yaml").read_text(encoding="UTF-8"))
      final_prompt_context = yaml.safe_load(
         Path("../prompts/l3/2_dominance_cooperation.yaml").read_text(encoding="UTF-8"))

      simplfied_context = run_classifcation(simplified_prompt_context, validation_sample.assign(
         sentence=lambda df: df["three-sentence"]
      ), parallel_prompts=12)

      simplfied_context = fix_labels(simplfied_context, mapping={
         # This prompt hallucinates categories. Which break the F1 macro score; we assign these to one of the provided
         **label_mapping_l3,
         "fiscal coordination": "monetary-fiscal coordination",
         "financial coordination": "monetary-financial coordination"
      }, unkown_category="none", drop_unknown=False)

      complex_context = run_classifcation(complex_prompt_context, validation_sample.assign(
         sentence=lambda df: df["three-sentence"]
      ), parallel_prompts=12)

      complex_context = fix_labels(complex_context, mapping=label_mapping_l3, drop_unknown=False)

      # Load the final L3 from the valdiation set (this is to ensure full consistency between tables)
      final_context = pd.read_pickle("../outputdata/validation/validation_set_plus_chatgpt.pkl")
      # final_context_rerun = run_classifcation(final_prompt_context, validation_sample.assign(
      #     sentence = lambda df: df["three-sentence"]
      # ), parallel_prompts=12)
      # final_context_rerun = fix_labels(final_context_rerun, mapping = label_mapping_l3, drop_unknown = False)
      #
      #

      # Run without context
      simplified_prompt_nocontext = yaml.safe_load(
         Path("../prompts/prompt_variations/shorter_l3_nocontext.yaml").read_text(encoding="UTF-8"))
      complex_prompt_nocontext = yaml.safe_load(
         Path("../prompts/prompt_variations/longer_l3_nocontext.yaml").read_text(encoding="UTF-8"))
      final_prompt_nocontext = yaml.safe_load(
         Path("../prompts/prompt_variations/final_l3_nocontext.yaml").read_text(encoding="UTF-8"))

      simplfied_nocontext = run_classifcation(simplified_prompt_nocontext, validation_sample, parallel_prompts=12)
      simplfied_nocontext = fix_labels(simplfied_nocontext, mapping={
         # This prompt hallucinates categories. Which break the F1 macro score; we assign these to one of the provided
         **label_mapping_l3,
         "fiscal coordination": "monetary-fiscal coordination",
         "financial coordination": "monetary-financial coordination",
      }, drop_unknown=False, unkown_category="none")

      complex_nocontext = run_classifcation(complex_prompt_nocontext, validation_sample, parallel_prompts=12)
      complex_nocontext = fix_labels(complex_nocontext, mapping={
         **label_mapping_l3,
         "monetary-prudential coordination": "monetary-financial coordination"
      }, drop_unknown=False, unkown_category="none")

      final_nocontext = run_classifcation(final_prompt_nocontext, validation_sample, parallel_prompts=12)
      final_nocontext = fix_labels(final_nocontext, mapping={
         **label_mapping_l3
      }, regex_mapping={
         r"(?i).*monetary dominance$": "monetary dominance",
         r"(?i).*financial dominance$": "financial dominance",
         r"(?i).*fiscal dominance$": "fiscal dominance",
         r"(?i).*monetary-fiscal coordination$": "monetary-fiscal coordination",
         r"(?i).*monetary-financial coordination$": "monetary-financial coordination",
         r"(?i).*none$": "none"
      }, unkown_category="none", drop_unknown=False)

      prompt_variations_classifications = {
         "simplfied_nocontext": simplfied_nocontext,
         "final_nocontext": final_nocontext,
         "complex_nocontext": complex_nocontext,
         "simplfied_context": simplfied_context,
         "final_context": final_context,
         "complex_context": complex_context
      }

      # Save dataframes
      pd.to_pickle(prompt_variations_classifications, "../outputdata/validation/prompt_variations.pkl")

      simplfied_context = pd.read_pickle("../outputdata/validation/prompt_variations.pkl")["simplfied_context"]
      final_context = pd.read_pickle("../outputdata/validation/prompt_variations.pkl")["final_context"]
      complex_context = pd.read_pickle("../outputdata/validation/prompt_variations.pkl")["complex_context"]

      # Produce Table
      pd.concat(
         [
               pd.DataFrame(
                  {
                     "simplified_context": calculate_scores(validation_sample["A_level_3"],
                                                            simplfied_context.classification),
                     "final_context": calculate_scores(validation_sample["A_level_3"],
                                                         final_context.dominance_cooperation_chatGPT),
                     "complex_context": calculate_scores(validation_sample["A_level_3"], complex_context.classification)
                  }
               ),
               pd.DataFrame(
                  {
                     "simplified_context": [simplfied_context.groupby("prompt_id").head(1).total_tokens.sum()],
                     "final_context": 137193,
                     # [final_context_rerun.groupby("prompt_id").head(1).total_tokens.sum()], For consistency the same classification is always used for the final model
                     "complex_context": [complex_context.groupby("prompt_id").head(1).total_tokens.sum()]
                  }
               ),
               pd.DataFrame(
                  {
                     "simplified_context": [len(simplified_prompt_context)],
                     "final_context": [len(final_prompt_context)],
                     "complex_context": [len(complex_prompt_context)]
                  }
               )
         ]
      ).round(2)

      # -------------------------------------------------------------------#
      # Robustness Check: Change System Message:
      # -------------------------------------------------------------------#

      system_message_default = "You are a helpful assistant."
      system_message_cb = "You are a research assistant at a central bank."
      system_message_long = """You are a distinguished expert on central bank communication. 
      Through your thorough studies, having read countless speeches and other central bank documents, you are familiar with the language central bankers use and know how to interpret their statements.
      This expertise enables you to understand nuanced differences in central bank communications and accurately decode the sometimes hard to grasp messages conveyed inside their communication."""

      system_message_test = cross_validation(
         input_data=validation_sample.assign(
               sentence=lambda df: df["three-sentence"]),
         ytrue=validation_sample["A_level_3"],
         mapping=label_mapping_l3,
         regex_mapping=label_mapping_l3_regex,
         prompt=[yaml.safe_load(Path("../prompts/l3/2_dominance_cooperation.yaml").read_text(encoding="UTF-8"))],
         system_message=[system_message_default, system_message_cb, system_message_long]
      )

      system_message_test[
         ["system_message", "f1_micro", "f1_weighted", "f1_macro", "accuracy", "balanced_accuracy"]].to_excel(
         "../outputdata/validation/system_message.xlsx")
