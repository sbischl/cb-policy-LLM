source("codes/figures/functions_and_settings.R")

# Load robustness check data
level_1_sentence <- read_stata("outputdata/robustness/sentence_count/sentence_count_l1.dta") %>%
  mutate(tokens_used = tokens_used/ 500000 + 0.75) %>%
  pivot_longer(c(accuracy, tokens_used), names_to = "variable", values_to = "value") %>%
  ggplot(aes(y = value, x = maximum_sentences, color = variable)) +
  scale_y_continuous(
    limits = c(0.75, 0.9),
    breaks = seq(0.75, 1, 0.05),
    sec.axis = sec_axis(~ .*500000 - 375000, name="Tokens used")
  ) +
  geom_line() + geom_point() + theme_ecb_replica() + scale_x_continuous(breaks = c(3,5,10,25,50,75)) +
  scale_color_manual(labels=c("Accuracy","Tokens used"), values = c(ecb_colors[1], ecb_colors[2])) +
  labs(
    color = "",
    y = "Accuracy",
    x = "Sentences"
  )

level_2_sentence <- read_stata("outputdata/robustness/sentence_count/sentence_count_l2.dta") %>%
  mutate(tokens_used = tokens_used/ 150000 + 0.6) %>%
  pivot_longer(c(accuracy, tokens_used), names_to = "variable", values_to = "value") %>%
  ggplot(aes(y = value, x = maximum_sentences, color = variable)) +
  scale_y_continuous(
    limits = c(0.75, 0.9),
    breaks = seq(0.75, 1, 0.05),
    sec.axis = sec_axis(~ .*150000 - 90000, name="Tokens used")
  ) +
  geom_line() + geom_point() + theme_ecb_replica() + scale_x_continuous(breaks = c(3,5,10,25,50,75)) +
  scale_color_manual(labels=c("Accuracy","Tokens used"), values = c(ecb_colors[1], ecb_colors[2])) +
  labs(
    color = "",
    y = "Accuracy",
    x = "Sentences"
  )

level_3_sentence <- read_stata("outputdata/robustness/sentence_count/sentence_count_l3.dta") %>%
  mutate(tokens_used = tokens_used / 500000 + 0.45) %>%
  pivot_longer(c(accuracy, tokens_used), names_to = "variable", values_to = "value") %>%
  ggplot(aes(y = value, x = maximum_sentences, color = variable)) +
  scale_y_continuous(
    sec.axis = sec_axis(~ . * 500000 - 225000, name="Tokens used")
  ) +
  geom_line() + geom_point() + theme_ecb_replica() + scale_x_continuous(breaks = c(3,5,10,25,50,75)) +
  scale_color_manual(labels=c("Accuracy","Tokens used"), values = c(ecb_colors[1], ecb_colors[2])) +
  labs(
    color = "",
    y = "Accuracy",
    x = "Sentences"
  )

level_3_sentence %>% ggsave(
  filename = "figures/sentence_count/sentence_count_l3_r.pdf",
  width = default_charts_width,
  height = default_chart_height,
  device = cairo_pdf,
  units = "cm"
)

ggsave(
  wrap_plots(level_1_sentence + ggtitle("Level 1"), level_2_sentence + ggtitle("Level 2"), ncol = 2, guides = "collect") & theme(legend.position = "bottom"),
  filename = "figures/sentence_count/sentence_count_l1_l2_r.pdf",
  width = panel_chart_width, # 17
  height = default_chart_height,
  device = cairo_pdf,
  units = "cm"
)