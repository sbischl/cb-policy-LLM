# Query chat gpt

import pandas as pd
import datetime as pt
import tiktoken
import datetime
import json
import ruamel.yaml as yaml
import time
from pathlib import Path
import os
import openai
from openai import AzureOpenAI
import re
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from itertools import product
from multiprocessing.pool import ThreadPool as Pool
import numpy as np
from nltk.metrics.agreement import AnnotationTask
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.pyplot as plt
import google.generativeai as genai


# Set this to true if chatGPT should run on an Azure environment. In this case, further settigns regarding proxies and certificated need to be set.
run_on_azure = False

if run_on_azure == True:

   # Configure proxies if necessary
   openai.proxies = "<PROXY URL>"
   os.environ['SSL_CERT_FILE'] = '<SSL CERTIFICATE>'
   os.environ['HTTPS_PROXY'] = '<PROXY URL>' 
   os.environ['HTTP_PROXY'] = '<PROXY URL>' 

   client = AzureOpenAI(
      api_key="<PUT API KEY HERE>", # key for identification for Azure OpenAI Service, 
      azure_endpoint = "<AZURE ENDPOINT>",
      api_version = "2023-03-15-preview",
   )
   
   # Some of the code counts tokens. The library that is provided by OpenAI (tiktoken) downloads the encoding from the internet. However, on the some python environments
   # it cannot access the internet. Therefore we are required to manually download the encoding and place it in the tiktoken_cache_dir
   tiktoken_cache_dir = "D:/temp/"
   os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
   enc = tiktoken.get_encoding("cl100k_base")
   # set up Certificates, proxies etc. for succesfull SSL handshake 
   chatgpt_model = "LLM-test"
else:
   
   enc = tiktoken.get_encoding("cl100k_base")
   
   # API Keys
   client = openai.OpenAI(
      api_key = "<PUT API KEY HERE>"
   )

   genai.configure(api_key="<PUT API KEY HERE>")

# Some helper functions
def count_tokens(text):
    """Count tokens. Requires tiktoken and a enc loaded"""
    try:
      length = len(enc.encode(text))
    except:
      length = 0
    return(length)

def set_high_value_if_none(number):
     """Returns large number if None"""
     if number is None:
        return(1_000_000_000)
     
     else:
        return (number)

def expand_grid(dictionary):
   """Provide a dictionary where keys are variables and values are possible values. It will return a pd.DataFrame with all possible combinations of the values of the provided variables. Useful for running grid search in CV"""
   return pd.DataFrame([row for row in product(*dictionary.values())], 
                       columns=dictionary.keys())

def conditional_classification(prompt, input_data, conditional, **kwargs):
   """Run different prompts based on the value in another column. Prompt should be a dictionary containing as key all possible values of the the column specified by conditional"""
   conditional_classifiaction = []
   for label, subset in input_data.groupby(conditional):
      conditional_classifiaction.append(
         run_classifcation(prompt = prompt[label], input_data = subset, **kwargs)
      )
   
   return(pd.concat(conditional_classifiaction).sort_values("sentence_id"))

# This function has a external dependency. It requires a openai client called 'client' to be loaded.
def run_classifcation(
      prompt,
      input_data,
      model = "chatgpt", # Either chatgpt or gemini
      maximum_sentences = 10, # Use this to set a limit on the number of sentences included in a prompt. None means unlimited.
      tokens_max = None, # Use this to set a limit on the number of tokens included in a single port. None means unlimited. 
      max_queries = None, # This limits the number of queries made to the openAI api
      few_shot = False,
      continue_run = None, # This is to deal with failing runs. Put the timestamp as string and it will not rerun prompts if already run
      temperature = 0,
      top_p = 1,
      parallel_prompts = 12,
      # ChatGPT specific
      chatgpt_model = "gpt-3.5-turbo-0301",
      system_message = "You are a research assistant at a central bank.",
      few_shot_sentences = None, # This should be a pandas dataframe with the columns sentence, and classification
      few_shot_follow_up = None, # If a full prompt and answer are provided as few shot learning, it might make sense to not restate the full prompt. This could be something like: "Now classify these sentences:"
      seed = None,
      # Gemini specific
      prompt_history = None
      ):
  """This is the main function used to classify sentences. It has sensible defaults. See comments on what options do"""
  # Handle the options that specify maximums. To deal with the Nones we just set a very high value that none of the paramters can reasonably reach.
  maximum_sentences = set_high_value_if_none(maximum_sentences)
  tokens_max = set_high_value_if_none(tokens_max)
  max_queries = set_high_value_if_none(max_queries)
  
  # Initalize counters
  current_query = ''
  sentences_current_query = 0
  number_of_queries = 0
  # Sentence ids (This is a list that keeps track of the sentence ids that have been added to the current.query)
  sentence_ids = []

  # Store the return values. These are only 'Promises', i.e. async
  prompt_results = []

  # Create a pool for multiprocessing:
  pool = Pool(parallel_prompts)
  
  # Save a timestamp that identifiers this run. A folder with this timestamp as name will be created in the /logs/ folder
  if continue_run is not None:
     timestamp_run = continue_run
  else:
    timestamp_run = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
     
  # Define a function that is used to add to the current query.
  def append_to_query(current_query, sentence, sentence_number):
     """This takes the current query as input and appends the next sentence"""

     return(current_query + f"{sentence_number}. {sentence}\n\n")
  
  # Few shot learning codes goes here:
  # If we are running a few shot prompt we need to sample sentences
  if few_shot:
   few_shot_prompt = prompt
   few_shot_response = ""
   for index, row in few_shot_sentences.reset_index(drop = True).iterrows():
       few_shot_prompt = append_to_query(few_shot_prompt, row.sentence, index+1)
       few_shot_response = few_shot_response + f"{index + 1}. {row.classification}\n"
  else:
     few_shot_prompt = None
     few_shot_response = None

  def run_query(
        query,
        sentence_ids,
        query_number,
        # ChatGPT specific
        few_shot_query = None,
        few_shot_response = None,
        system_message = None,
        # Gemini specific
        prompt_history = None
        ):
    """Nested function which executes the request itself."""
    start_time = time.time()

    # This variable is incremented when a query iteration fails
    required_attemps = 1
    max_attemps = 10

    if model == "chatgpt":
      if (few_shot_query is not None) and (few_shot_response is not None):
         messages = [
                     {"role": "system", "content": system_message},
                     {"role": "user", "content": few_shot_query},
                     {"role": "assistant", "content": few_shot_response},
                     {"role" : "user", "content": query}
                  ]
      else:
         messages = [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": query}
                     ]
    elif model == "gemini":
       if prompt_history is not None:   
         prompt_parts = [
            *prompt_history,
            f"input: {query}",
            "output: "
         ]
       else:
          prompt_parts = [query]      

    if Path(f"../logs/{timestamp_run}/{query_number}.json").is_file():
      api_response = json.loads(Path(f"../logs/{timestamp_run}/{query_number}.json").read_text())
    else:
      while required_attemps <= max_attemps:
         try:
            if model == "chatgpt":
               api_response = client.chat.completions.create(
               model = chatgpt_model,
               temperature = temperature,
               seed = seed,
               top_p = top_p,
               messages= messages).model_dump()
            elif model == "gemini":
               gemini_client = genai.GenerativeModel(model_name="gemini-pro",
                              generation_config= {
                                "temperature": temperature,
                                "top_p": top_p,
                                "top_k": 1,
                                "max_output_tokens": 2048,
                                })
               api_response = gemini_client.generate_content(prompt_parts)
               # Do something with the response
               api_response = api_response.text
            break
         except:
            # If there is an error try again up to max_retries times. Throw exception if it fails in the last try
            time.sleep(10)
            if (required_attemps == max_attemps):
               print(f"Running the prompt failed {max_attemps} times.")
               raise
            
         required_attemps += 1
            
      # Store the returned values from the prompt
      log_file = Path(f"../logs/{timestamp_run}/{query_number}.json")
      log_file.parent.mkdir(exist_ok=True, parents=True)
      log_file.write_text(json.dumps(api_response), encoding='UTF-8')

      # This extracts the chat completition, i.e. the reply we are insterested in
      if model == "chatgpt":
         chat_completion = api_response['choices'][0]['message']['content']
         prompt_tokens = api_response["usage"]["prompt_tokens"]
         completion_tokens = api_response["usage"]["completion_tokens"]
         total_tokens = api_response["usage"]["total_tokens"]
      elif model == "gemini":
         chat_completion = api_response
         prompt_tokens = None # Gemini does not return these
         completion_tokens = None # Gemini does not return these
         total_tokens = None # Gemini does not return these
      try:
         sentence_classification = [x[1] for x in re.findall(r"(\d+\.\s)(.*)", chat_completion)]
         assert len(sentence_classification) == len(sentence_ids)
      except:
         print(f"Something went wrong on prompt {query_number}. The returned labels are not in the specified format")
         sentence_classification = ["NA"] * len(sentence_ids)
      
      results = pd.DataFrame(
         {
         "prompt_id" : query_number,
         "time_to_run" : time.time() - start_time,
         "required_attemps" : required_attemps,
         "classification": sentence_classification,
         "sentence_id" : sentence_ids,
         "prompt_tokens" : prompt_tokens,
         "completion_tokens" : completion_tokens,
         "total_tokens" : total_tokens,
         "estimated_prompt_tokens" : count_tokens(prompt)
         })

      return(results)

  # Iterate over all the rows of the provided dataframe
  with tqdm(total=input_data.shape[0]) as pbar:
     def updateprogress(_):
        pbar.update(maximum_sentences)

     for index, row in input_data.reset_index(drop = True).iterrows():
      if current_query == '':
          current_query = few_shot_follow_up if few_shot else prompt

      sentences_current_query += 1
      appended_query = append_to_query(current_query, row.sentence, sentences_current_query)
      appended_sentence_ids = sentence_ids + [row.sentence_id] # Here we copy the list
      
      tokens = count_tokens(appended_query)

      if (tokens > tokens_max) | (sentences_current_query > maximum_sentences):

        if (number_of_queries < max_queries):
          
          prompt_results.append(pool.apply_async(run_query, callback = updateprogress, kwds = {
             "query" : current_query,
             "sentence_ids" : sentence_ids,
             "query_number" : number_of_queries,
             "few_shot_query" : few_shot_prompt,
             "few_shot_response" : few_shot_response,
             "system_message" : system_message,
             "prompt_history" : prompt_history
          }))

          number_of_queries += 1
          
          # Finally we need to reset the current_query
          sentences_current_query = 1
          current_query = append_to_query(few_shot_follow_up if few_shot else prompt, row.sentence, sentences_current_query)
          sentence_ids = [row.sentence_id]
      else:
        # Keep adding to the query:
        current_query = appended_query
        sentence_ids = appended_sentence_ids

      if (index + 1 == input_data.shape[0]) & (number_of_queries < max_queries):
         prompt_results.append(pool.apply_async(run_query, callback = updateprogress, kwds = {
             "query" : current_query,
             "sentence_ids" : sentence_ids,
             "query_number" : number_of_queries,
             "few_shot_query" : few_shot_prompt,
             "few_shot_response" : few_shot_response,
             "system_message" : system_message,
             "prompt_history" : prompt_history
          }))
         
      if(number_of_queries == max_queries):
         break
         
      # Wait for everytihng to finish       
     pool.close()
     pool.join()

  prompt_results_df = [prompt_result.get() for prompt_result in prompt_results]
  return(pd.concat(prompt_results_df, ignore_index=True))



def fix_labels(classified_sentences, #DataFrame returned from run_classifcation
               mapping = None, # Simple mapping containing labels as keys and the label mapped to as value
               regex_mapping = None, # Same as mapping except that regular expressions are used as keys
               ignore_case = True, # Should case be ignored. Only applies to mapping
               unkown_category = 'NA', # What value to set if the label in classified_sentences does not appear as key in either of the mappings
               drop_unknown = False # Should these be dropped?
               ):
   """This function is supposed to run after run_classifcation to repair labels. It takes the dataframe and modified the classification column. You can either provide lists of w"""

   # If ignore_case is set to true regex with ignore case is used
   if mapping is not None:
      if ignore_case:
         ignorecaseregex = {f"(?i)^{k}$" : v for k,v in mapping.items()} 
         classified_sentences = classified_sentences.assign(
            classification = lambda x: x["classification"].replace(ignorecaseregex, regex = True)
         )
      else:
         classified_sentences = classified_sentences.assign(
            classification = lambda x: x["classification"].replace(mapping)
         )

   if regex_mapping is not None:
      # Also apply the regular expressions
      classified_sentences = classified_sentences.assign(
            classification = lambda x: x["classification"].replace(regex_mapping, regex = True)
      )
      
   # We assigning all values that are not part of the mapping to "NA". For that  we use a regex, but we need to escape some of the characters like (
   if (regex_mapping is not None) and (mapping is not None):
      allowed_values = list(mapping.values()) + list(regex_mapping.values())
   elif mapping is not None:
      allowed_values = list(mapping.values())
   elif regex_mapping is not None:
      allowed_values = list(regex_mapping.values())

   safe_matches = [re.escape(m) for m in allowed_values]
   classified_sentences.loc[~classified_sentences["classification"].str.contains("^(" + "|".join(safe_matches) + ")$", case = False), "classification"] = unkown_category

   if drop_unknown:
      classified_sentences = classified_sentences.loc[~(classified_sentences.classification == unkown_category)]

   return(classified_sentences)

def calculate_confusion(y_true, y_pred):
  """Calculates confusion matrix based on true and predicted vectors"""
  return(pd.DataFrame(
     {
        "y_true" : y_true,
        "y_pred" : y_pred
     }
  ).groupby("y_true")["y_pred"].value_counts(normalize=True).unstack(fill_value=0))


def cross_validation(input_data, ytrue, mapping = None, regex_mapping = None, specific_prompts = None, **kwargs):
   """Cross validation like function to run different settings and compare results. Specify the settings passed to run_classifcation as **kwargs"""
   if "prompt" not in kwargs:
      print("need to specify a prompt!")
      return(None)
   
   grid = expand_grid(kwargs)
   print(grid)
   results = []

   for index, row in tqdm(grid.iterrows(), total=grid.shape[0]):
      settings_current_iteration = row.to_dict()

      results.append(
         {
            **cv_trial(mapping = mapping, regex_mapping=regex_mapping, input_data = input_data,  ytrue = ytrue,  specific_prompts = specific_prompts, **settings_current_iteration),
            **settings_current_iteration
         }        
      )

   return(
         pd.DataFrame(results)
   )

# Needs to provide the prompt in the kwargs
def cv_trial(prompt, input_data, ytrue, mapping = None, regex_mapping = None,  specific_prompts = None, **kwargs):
   """Companion function to cross_validation. This runs a single specification and returns scores, as well as token usage and the exact classification returned by the run"""
   if specific_prompts == None:
      classificationofsentences = run_classifcation(prompt = prompt, input_data = input_data, parallel_prompts= 6, **kwargs)
   else:
      classificationofsentences_by_label = []
      for label, subset in input_data.groupby(specific_prompts):
         classificationofsentences_by_label.append(
            run_classifcation(prompt = prompt[label], input_data = subset, parallel_prompts= 6, **kwargs)
         )
      classificationofsentences = pd.concat(classificationofsentences_by_label)

   # The fix labels function may need adjusting
   classificationofsentences = fix_labels(classificationofsentences, mapping = mapping, regex_mapping= regex_mapping, unkown_category= "none")

   # Need to restore the original ordering of the sentences
   merged_data = input_data[["sentence_id"]].merge(classificationofsentences, how = "left", on = "sentence_id")

   ypred = merged_data["classification"]

   ypred.loc[~ytrue.isna().values]

   return(
      {
        **calculate_scores(ytrue, ypred),
        "data" : merged_data,
        "tokens_used" : merged_data.groupby("prompt_id").head(1)["total_tokens"].sum()    
      }
   )

def calculate_scores(ytrue, ypred):
   """This function defines the scores to evaluate the model. Add to the returned dictionary to report more scores whenever scores are calculated."""
   ytrue_freq = ytrue.sort_values().value_counts(normalize= True)
   ypred_freq = ypred.sort_values().value_counts(normalize= True)
   freq_deviation = (ytrue_freq - ypred_freq).abs().sum()

   return(
      {
        "accuracy" : accuracy_score(ytrue, ypred),
        "f1_weighted" : f1_score(ytrue, ypred, average = 'weighted'),
        "f1_macro" : f1_score(ytrue, ypred, average = 'macro'),
        "precision_macro" : precision_score(ytrue, ypred, average = "macro"),
        "recall_macro" : recall_score(ytrue, ypred, average = "macro"),
        "balanced_accuracy" : balanced_accuracy_score(ytrue, ypred),
        "freq_deviation" :  freq_deviation
      }
   )

def category_specific_accuracy(ytrue, ypred):
   """Calculate category specific, i.e. how accurate is the prediction by "true" tabulated by each value found in ytrue"""
   classification_report_dict = classification_report(ytrue, ypred, output_dict= True) 
   scores = []
   for label in ytrue.unique():
      scores.append(
         {
            "category" : label,  
            **classification_report_dict[label]
         })
      
   return(
      pd.DataFrame(scores)
   )

def extract_errors(df,true_column, pred_column):
   """Helper function to extract the rows where errors were made. This can be helpful look in more detail which kind of errors chatGPT makes"""
   errors = []
   # False negatives
   for label, label_data in df.groupby(true_column):
      incorrectly_classified = label_data.query(f"{true_column} != {pred_column}")
      incorrectly_classified = incorrectly_classified.assign(
         error_type = "False Negative",
         error_category = label
      )
      errors.append(incorrectly_classified)
   
   # False positive
   for label, label_data in df.groupby(pred_column):
      incorrectly_classified = label_data.query(f"{true_column} != {pred_column}")
      incorrectly_classified = incorrectly_classified.assign(
         error_type = "False Positive",
         error_category = label,
      )
      errors.append(incorrectly_classified)

   return(
      pd.concat(errors).sort_values("error_category")[[true_column, pred_column, "error_type", "error_category", "sentence"]]
   )

def add_neighbouring_sentences(df, before = 1, after = 1):
      """Adds the sentences before and each sentence to the sentence. This requires that sentences that are passed in df are in the correct order"""
      added_context_column = ""
      for before_index in range(before, 0, -1):
         added_context_column = added_context_column + df["sentence"].shift(before_index).fillna('') + " "


      added_context_column = added_context_column +  df["sentence"]

      for after_index in range(1, after + 1, 1):
         added_context_column = added_context_column + " " + df["sentence"].shift(-after_index).fillna('')

      added_context_column = added_context_column.str.replace(r"^\s+", "", regex = True)

      added_context_column = added_context_column.str.replace(r"\s+$", "", regex = True)

      return(added_context_column)

def entire_speech(df, sentence_id):
   """Returns the entire speech belonging to a sentence_id"""
   speech = df.query(
      f"sentence_id == @sentence_id"
   ).iloc[0]["speech_identifier"]

   return(" ".join(df[
      df.speech_identifier == speech
   ]["sentence"]))


# Label mappings. These are used to assign categories when chatgpt does not assign with the specified format.
label_mapping_l1 = {
         "financial" : "financial",
         "monetary" : "monetary",
         "fiscal" : "fiscal",
         "macro" : "macro",
         "climate" : "climate",
         "international" : "international",
         "other" : "other",
         "inflation" : "monetary",
         "fiscal and monetary" : "fiscal",
         "monetary and fiscal" : "fiscal",
      }

label_mapping_l1_regex = {
         r".*\(Other\)$" : "other",
         r".*\(Fiscal\)$" : "fiscal",
         r".*\(Monetary\)$" : "monetary",
         r".*\(Financial\)$" : "financial",
         r".*\(Macro\)$" : "macro",
         r".*\(Climate\)$" : "climate",
         r".*\(International\)$" : "international",
}

label_mapping_l2 = {
         "descriptive" : "descriptive",
         "normative" : "normative",
         "normative (value judgement)" : "normative",
}

label_mapping_l3 = {
         "monetary dominance" : "monetary dominance",
         "financial dominance" : "financial dominance",
         "fiscal dominance" : "fiscal dominance",
         "monetary-fiscal coordination" : "monetary-fiscal coordination",
         "monetary-financial coordination" : "monetary-financial coordination",
         "fiscal-monetary coordination" : "monetary-fiscal coordination",
         "financial-monetary coordination" : "monetary-financial coordination",
         "none" : "none",
         "\"none\"" : "none",
         "\"monetary dominance\"" : "monetary dominance",
         "\"financial dominance\"" : "financial dominance",
         "\"fiscal dominance\"" : "fiscal dominance",
         "\"monetary-fiscal coordination\"" : "monetary-fiscal coordination",
         "\"monetary-financial coordination\"" : "monetary-financial coordination",
      }

label_mapping_l3_regex = {
         r"(?i).*\(monetary dominance\)$" : "monetary dominance",
         r"(?i).*\(financial dominance\)$" : "financial dominance",
         r"(?i).*\(fiscal dominance\)$" : "fiscal dominance",
         r"(?i).*\(monetary-fiscal coordination\)$" : "monetary-fiscal coordination",
         r"(?i).*\(monetary-financial coordination\)$" : "monetary-financial coordination",
         r"(?i).*\(none\)$" : "none"
}