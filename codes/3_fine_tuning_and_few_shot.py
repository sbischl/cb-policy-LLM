from chat_gpt_functions import *

# Settings:
system_message = "You are a research assistant at a central bank."
number_of_excerpts_per_prompt = 5
fine_tuning_fraq = 0.3
prompt = yaml.safe_load(Path("../prompts/l3/2_dominance_cooperation.yaml").read_text(encoding = "UTF-8"))

# Load the validation data
# For now we are fine-tuning level 3 classifcation only.

validation_sample = pd.read_excel("../inputdata/validation_sample_all.xlsx")

np.random.seed(3)

validation_sample = validation_sample.assign(
         A_level_1 = validation_sample[["S_level_1", "M_level_1", "L_level_1"]].mode(axis = 1).apply(lambda x: x.dropna().sample(1).item(), axis = 1),
         A_level_1_agree = 4 - validation_sample[["S_level_1", "M_level_1", "L_level_1"]].nunique(axis = 1),
         A_level_2 = validation_sample[["S_level_2", "M_level_2", "L_level_2"]].mode(axis = 1).apply(lambda x: x.dropna().sample(1).item(), axis = 1),
         A_level_2_agree = 4 - validation_sample[["S_level_2", "M_level_2", "L_level_2"]].nunique(axis = 1),
         A_level_3 = validation_sample[["S_level_3", "M_level_3", "L_level_3"]].mode(axis = 1).apply(lambda x: x.dropna().sample(1).item(), axis = 1),
         A_level_3_agree = 4 - validation_sample[["S_level_3", "M_level_3", "L_level_3"]].nunique(axis = 1),
)

# Split into training and validation_sample
fine_tuning = validation_sample.sample(frac = fine_tuning_fraq,random_state=444)
holdout = validation_sample.drop(fine_tuning.index)


def prompt_and_answer_list(data, number_of_excerpts_per_prompt):
    # List all finetuning examples:
    additional_training_list = []

    # Create prompt reply messages in a while loop
    index_prompt = 0
    while index_prompt < data.shape[0] / number_of_excerpts_per_prompt:
        sentences = data.iloc[index_prompt * number_of_excerpts_per_prompt:(index_prompt * number_of_excerpts_per_prompt + number_of_excerpts_per_prompt)]
        answer = ""
        for index_sentence, sentence in sentences.reset_index().iterrows():
            answer = answer + f"{index_sentence + 1}. {sentence['A_level_3']}\n"

        question = ""

        for index_sentence, sentence in sentences.reset_index().iterrows():
            question = question + f"{index_sentence + 1}. {sentence['three-sentence']}\n\n"

        question = prompt + question

        index_prompt = index_prompt + 1

        additional_training_list.append(
            {
                "messages": [
                    {
                        "role" : "system",
                        "content" : system_message
                    },
                    {
                        "role" : "user",
                        "content" : question
                    },
                    {
                        "role" : "assistant",
                        "content" : answer
                    }
                ]
            }
        )

    return(additional_training_list)

def prompt_and_answer_table(data, number_of_excerpts_per_prompt):
    questions = []
    answers = []
    for entry in prompt_and_answer_list(data, number_of_excerpts_per_prompt):
        print(entry)
        questions.append(entry["messages"][1]["content"]) # This assumes that the first message is always system, second the prompt, and three the expected answer
        answers.append(entry["messages"][2]["content"])

    return(pd.DataFrame({
        "questions" : questions,
        "answers" : answers
    }))

def prompt_and_answer_list_gemini(data, number_of_excerpts_per_prompt, format = "conversation"):
    model_and_user_list = []
    if format == "conversation": # This format seems to make most sense at first given the api documentation
        for entry in prompt_and_answer_list(data, number_of_excerpts_per_prompt):
            model_and_user_list.append({
                "role": "user",
                "parts": [entry["messages"][1]["content"]]
            })

            model_and_user_list.append({
                "role": "model",
                "parts": [entry["messages"][2]["content"]]
            })
    elif format == "input/output": # This format is suggested when exporting code from Google AI studio
        for entry in prompt_and_answer_list(data, number_of_excerpts_per_prompt):
            model_and_user_list.append(f"input: {entry['messages'][1]['content']}")
            model_and_user_list.append(f"output: {entry['messages'][2]['content']}")
    
    return(model_and_user_list)


with jsonlines.open('fine_tuning.jsonl', 'w') as outfile:
    outfile.write_all(prompt_and_answer_list(fine_tuning, number_of_excerpts_per_prompt= number_of_excerpts_per_prompt))

with jsonlines.open('validation.jsonl', 'w') as outfile:
    outfile.write_all(prompt_and_answer_list(holdout, number_of_excerpts_per_prompt= number_of_excerpts_per_prompt))


# Upload files to OpenAI server. This returns a file id, which needs to be passed to the finetuning call. You can also check the files here:
# https://platform.openai.com/files and copy the id
fine_tuning_file = client.files.create(
  file=open("fine_tuning.jsonl", "rb"),
  purpose="fine-tune"
)

validation_file = client.files.create(
  file=open("validation.jsonl", "rb"),
  purpose="fine-tune"
)

# Test if the file looks correct. This is code is copied from:
# https://cookbook.openai.com/examples/chat_finetuning_data_prep
text_correctness_of_finetuning_file  = False

if text_correctness_of_finetuning_file:
    data_path = "fine_tuning.jsonl"

    # Load the dataset
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]

    # Initial dataset stats
    print("Num examples:", len(dataset))
    print("First example:")
    for message in dataset[0]["messages"]:
        print(message)


    format_errors = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue
            
        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue
            
        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1
            
            if any(k not in ("role", "content", "name", "function_call") for k in message):
                format_errors["message_unrecognized_key"] += 1
            
            if message.get("role", None) not in ("system", "user", "assistant", "function"):
                format_errors["unrecognized_role"] += 1
                
            content = message.get("content", None)
            function_call = message.get("function_call", None)
            
            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1
        
        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
    else:
        print("No errors found")
        

# This can be retrieved from the web interface or from the validation_file and fine_tuning_file object
validation_file_id = "file-x0tAR84t7xSlzzhlcXSB76vz"
fine_tuning_file_id = "file-qwPTXmRYV38GcIGG5Yl6JfdK"

fine_tuned_model = client.fine_tuning.jobs.create(
  training_file= fine_tuning_file_id,
  validation_file= validation_file_id,
  model="gpt-3.5-turbo-1106"
)

# This produced:
fine_tuned_model = "ft:gpt-3.5-turbo-1106:personal::8o7DpBUs"

run_models = False
if run_models:
    fine_tuned_results = run_classifcation(prompt, holdout.assign(sentence = lambda df: df["three-sentence"]), chatgpt_model= fine_tuned_model, parallel_prompts=5)
    unmodified_0301 = run_classifcation(prompt, holdout.assign(sentence = lambda df: df["three-sentence"]), chatgpt_model= "gpt-3.5-turbo-0301", parallel_prompts=5)
    unmodified_1106 = run_classifcation(prompt, holdout.assign(sentence = lambda df: df["three-sentence"]), chatgpt_model= "gpt-3.5-turbo-1106", parallel_prompts=5)
    unmodified_4_1106 = run_classifcation(prompt, holdout.assign(sentence = lambda df: df["three-sentence"]), chatgpt_model= "gpt-4-1106-preview", parallel_prompts=5)
    gemini_pro = run_classifcation(prompt, holdout.assign(sentence = lambda df: df["three-sentence"]), model = "gemini", temperature = 0, parallel_prompts=5)
    gemini_pro_few_shot = run_classifcation(prompt, holdout.assign(sentence = lambda df: df["three-sentence"]), model = "gemini", temperature = 0, prompt_history = prompt_and_answer_list_gemini(fine_tuning, number_of_excerpts_per_prompt= 300, format = "input/output")[0:2],  parallel_prompts=5)

    fine_tuned_results.to_pickle("../outputdata/fine_tuning/gpt-3.5-turbo-1106_8o7DpBUs.pkl")
    unmodified_0301.to_pickle("../outputdata/fine_tuning/gpt-3.5-turbo-0301.pkl")
    unmodified_4_1106.to_pickle("../outputdata/fine_tuning/gpt-4-turbo-1106.pkl")
    gemini_pro.to_pickle("../outputdata/fine_tuning/gemini_pro.pkl")
    gemini_pro_few_shot.to_pickle("../outputdata/fine_tuning/gemini_pro_few_shot.pkl")
else:
    fine_tuned_results = pd.read_pickle("../outputdata/fine_tuning/gpt-3.5-turbo-1106_8o7DpBUs.pkl")
    unmodified_0301 = pd.read_pickle("../outputdata/fine_tuning/gpt-3.5-turbo-0301.pkl")
    unmodified_4_1106 = pd.read_pickle("../outputdata/fine_tuning/gpt-4-turbo-1106.pkl")
    gemini_pro = pd.read_pickle("../outputdata/fine_tuning/gemini_pro.pkl")
    gemini_pro_few_shot = pd.read_pickle("../outputdata/fine_tuning/gemini_pro_few_shot.pkl")

label_mapping_dominance_cooperation = {
            "monetary dominance" : "monetary dominance",
            "financial dominance" : "financial dominance",
            "fiscal dominance" : "fiscal dominance",
            "monetary-fiscal coordination" : "monetary-fiscal coordination",
            "fiscal-monetary coordination" : "monetary-fiscal coordination",
            "Monetary coordination" : "monetary-fiscal coordination",
            "monetary-financial coordination" : "monetary-financial coordination",
            "Monetary-prudential coordination" : "monetary-financial coordination",
            "none" : "none",
            "monetary" : "monetary dominance",
            "financial" : "financial dominance",
            "Financial regulation dominance" : "monetary dominance"
}

unmodified_0301 = fix_labels(unmodified_0301, mapping = label_mapping_dominance_cooperation, drop_unknown = False, unkown_category= "none")
fine_tuned_results = fix_labels(fine_tuned_results, mapping = label_mapping_dominance_cooperation, drop_unknown = False, unkown_category= "none") # Not really required
unmodified_4_1106 = fix_labels(unmodified_4_1106, mapping = label_mapping_dominance_cooperation, drop_unknown = False, unkown_category= "none") # Not really required
gemini_pro = fix_labels(gemini_pro, mapping = label_mapping_dominance_cooperation, drop_unknown = False, unkown_category= "none") # Not really required
gemini_pro_few_shot = fix_labels(gemini_pro_few_shot, mapping = label_mapping_dominance_cooperation, drop_unknown = False, unkown_category= "none") # Not really required

pd.DataFrame([calculate_scores(holdout["A_level_3"], unmodified_0301["classification"]),
calculate_scores(holdout["A_level_3"], fine_tuned_results["classification"]),
calculate_scores(holdout["A_level_3"], unmodified_4_1106["classification"]),
calculate_scores(holdout["A_level_3"], gemini_pro["classification"]),
calculate_scores(holdout["A_level_3"], gemini_pro_few_shot["classification"])]).set_axis(["3.5-301", "3.5-1106-fine-tune", "4-1106", "Gemini Pro", "Gemini Pro Few Shot"], axis = "index").transpose().round(2)


category_specific_accuracy(holdout["A_level_3"], unmodified_0301["classification"])
category_specific_accuracy(holdout["A_level_3"], fine_tuned_results["classification"])
category_specific_accuracy(holdout["A_level_3"], gemini_pro["classification"])
