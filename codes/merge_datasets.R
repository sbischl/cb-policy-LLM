# This file contains the calculation of our relative index and aggregates the file.
# It requires the files produced by chat_gpt_main_analysis.py as input
library(tidyverse)
library(readxl)
library(haven)
library(lubridate)
library(arrow)
library(countrycode)

aggreation_unit <- "year"

clean_column_names <- function(column_name) {
    column_name %>%
        tolower() %>%
        str_replace("-", "_") %>%
        str_replace("\\s+", "_")
}

one_hot_encoding <- function(df, column_name, prefix = "") {
    df %>%
    mutate(value = 1) %>%
    {
        if(prefix != "") {
            mutate(., "{column_name}" := paste(prefix, !!sym(column_name), sep =  "_")) 
        } else {.}
    } %>%
    pivot_wider(names_from = column_name, values_from =  value, values_fill = 0) %>%
    {
        if(prefix != "") {
            select(., -one_of(str_glue("{column_name}_NA")))
        } else {
            select(., -one_of(str_glue("NA")))
        }
    }
}

# Speeches data:
sentences_and_context <- read_parquet(str_glue("./outputdata/{version}/sentences_and_context.parquet"))

l1_classification <- read_parquet(str_glue("./outputdata/{version}/l1_classification.parquet"))
l1_levels <- clean_column_names(setdiff(unique(l1_classification$classification), "NA"))

l2_classification <- read_parquet(str_glue("./outputdata/{version}/l2_classification.parquet"))
l2_levels <- clean_column_names(setdiff(unique(l2_classification$classification), "NA"))

l3_classification <- read_parquet(str_glue("./outputdata/{version}/l3_classification.parquet"))
l3_levels <- clean_column_names(setdiff(unique(l3_classification$classification), "NA"))

full_sentence_level_classification <- sentences_and_context %>% mutate(
        date = as.Date(str_extract(speech_identifier, "\\d{4}-\\d{2}-\\d{2}"))
    ) %>%
    relocate(speech_identifier, date, sentence, context_1_1) %>%
    left_join(
        l1_classification %>% rename( "l1_classification" = classification),
        by = join_by(sentence_id)
    ) %>%
    left_join(
        l2_classification %>% rename( "l2_classification" = classification),
        by = join_by(sentence_id)
    ) %>%
    left_join(
        l3_classification %>% rename( "l3_classification" = classification),
        by = join_by(sentence_id)
    ) %>%
    rename_with(
        ~ str_glue("word_count_{.x}"),
        .cols = all_of(c("monetary", "fiscal", "financial"))
    ) %>%
    rename_with(
        clean_column_names
    ) %>%
    mutate(
        across(c(l1_classification, l2_classification, l3_classification), ~ na_if(.x, "NA"))
    )

aggregation_data <- full_sentence_level_classification %>%
    mutate(
        date_agg_unit = floor_date(date, aggreation_unit)
    ) %>%
    one_hot_encoding("l1_classification") %>%
    one_hot_encoding("l2_classification") %>%
    one_hot_encoding("l3_classification") %>%
    rename_with(
        clean_column_names
    ) %>%
    group_by(
        date_agg_unit, central_bank
    ) %>%
    summarise(
        across(
            all_of(c(l1_levels, l2_levels, l3_levels)),
            ~ sum(.x, na.rm = TRUE)
        ),
        number_of_sentences = n(),
        number_of_speeches = length(unique(speech_identifier)),
        sum_of_token = sum(token_count),
        sum_of_characters = sum(len),
        across(all_of(c("word_count_monetary", "word_count_financial", "word_count_fiscal")), ~ sum(.x))
    ) %>%
    ungroup() %>%
    complete(central_bank, date_agg_unit)

    
aggregation_data_more_vars <- aggregation_data %>%
    mutate(
        fiscal_dom_or_corp = fiscal_dominance + monetary_fiscal_coordination,
        financial_dom_or_corp = financial_dominance + monetary_financial_coordination,
        share_normative = normative / number_of_sentences,
        share_descriptive = descriptive / number_of_sentences,
        across(
          all_of(l1_levels),
          ~ .x / number_of_sentences,
          .names = "topic_{.col}"
        ),
        across(
            all_of(c(l3_levels, "fiscal_dom_or_corp", "financial_dom_or_corp")),
            ~ .x / number_of_sentences,
            .names = "{.col}_rel_sentence"
        ),
        across(
             all_of(c("monetary_dominance", "financial_dominance", "fiscal_dominance")),
            ~ .x / (monetary_dominance + financial_dominance + fiscal_dominance),
            .names = "{.col}_rel_dom"
        ),
        across(
             all_of(c("monetary_dominance", "financial_dominance", "fiscal_dominance", "monetary_fiscal_coordination", "monetary_financial_coordination", "financial_dom_or_corp", "fiscal_dom_or_corp")),
            ~ .x / (monetary_dominance + financial_dominance + fiscal_dominance + monetary_financial_coordination + monetary_fiscal_coordination),
            .names = "{.col}_rel_domcorp"
        )
    )

# # The rest of the file to procude the our shared dataset 'dominance_coordination_dataset.csv' is only outlined as it would
# # require third party data sources like VDEM, the IMF fiscal crisis dataset and bond spreads.
#  aggregation_data_with_metadata <- aggregation_data_more_vars %>%
#     left_join(....) %>% # Join other datasets
#     relocate(central_bank, country, date_agg_unit, year) %>%
#     rename_with(
#         clean_column_names
#     ) %>%
#     arrange(central_bank, date_agg_unit)
#
# # Export the shared dataset:
# shared_dataset <- aggregation_data_with_metadata %>%
#   select(central_bank,country, year, currency_code,
#          starts_with("topic_"), share_descriptive, share_normative,
#          ends_with("_rel_domcorp"),
#          number_of_speeches, number_of_sentences,
#          inflation, democracy_ind, polarization_ind, spread, advanced, gdp_real_ppp_capita, fiscal_crisis) %>%
#   rename_with(~ str_replace(.x, "_rel_domcorp", ""))
#
# shared_dataset %>% write_csv("dominance_coordination_dataset.csv", na = "")