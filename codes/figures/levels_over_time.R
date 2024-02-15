source("codes/figures/functions_and_settings.R")

over_time_level_1 <- read_csv("dominance_coordination_dataset.csv") %>%
  summarise(
    across(all_of(topics), ~ mean(.x, na.rm = T)), .by = "year"
  ) %>% pivot_longer(-year, names_to = "topic", values_to = "share") %>%
  mutate(topic = factor(topic, levels = topics)) %>%
  ggplot(aes(x = year, y = share, color = topic)) + geom_line() + theme_ecb_replica() +
  guides(color=guide_legend(nrow=2,byrow=TRUE, keyheight = 0.75)) +
  scale_color_discrete(labels = c("Monetary", "Financial", "Macro", "International", "Fiscal", "Climate", "Other")) +
  labs(
    color = "", x = "Year", y = "Share topic"
  )

over_time_level_1 %>% ggsave(
  filename = "figures/levels_over_time/topics.pdf",
  width = default_charts_width,
  height = default_chart_height,
  device = cairo_pdf,
  units = "cm"
)

median_inflation <- median(read_csv("dominance_coordination_dataset.csv")$inflation,na.rm = T)
over_time_level_2_groups <- read_csv("dominance_coordination_dataset.csv") %>%
  mutate(
    high_inflation = median(inflation, na.rm = T) > median_inflation,
    .by = "central_bank"
  )

over_time_level_2_groups <-  rbind(
  over_time_level_2_groups %>% yearly_group_averages(
    "share_normative", weight = "number_of_speeches", group_by = "advanced", group_levels = c("1" = "Advanced", "0" = "Non-Advanced")),
  over_time_level_2_groups %>% yearly_group_averages(
    "share_normative", weight = "number_of_speeches", group_by = "high_inflation", group_levels = c("FALSE" = "Low Inflation", "TRUE" = "High Inflation")
  ),
  over_time_level_2_groups %>% yearly_group_averages(
    "share_normative", weight = "number_of_speeches", group_by = "democracy_ind", group_levels = c("1" = "Democracy", "0" = "Autocracy")
  )) %>% ggplot(
  aes(y = value, x = year, color = group)
) +
  geom_line() +
  theme_ecb_replica() +
  guides(color=guide_legend(nrow=2,byrow=TRUE, keyheight = 0.75))  +
  labs(color = "", x = "Year", y = "Share normative") +
  scale_y_continuous(breaks = seq(0.22, 0.34, 0.03), limits = c(0.22, 0.34))

over_time_level_2_groups %>% ggsave(
  filename = "figures/levels_over_time/normative_by_group.pdf",
  width = default_charts_width,
  height = default_chart_height,
  device = cairo_pdf,
  units = "cm"
)

over_time_level_2_topics <- read_csv("dominance_coordination_dataset.csv") %>%
  group_by(year) %>%
  summarise(across(starts_with("normative"), ~mean(weighted.mean(.x, w = number_of_speeches, na.rm = T)))) %>%
  pivot_longer(-year, names_to = "topic", values_to = "share_normative") %>%
  filter(!topic %in% c("normative_climate", "normative_other")) %>%
  mutate(topic = str_to_title(str_extract(topic,"(?<=_).*"))) %>%
  ggplot(aes(x = year, y = share_normative, color = topic)) + geom_line() + theme_ecb_replica() + labs(color = "", x = "Year", y = "Share normative")

over_time_level_2_topics  %>%
  ggsave(
    filename = "figures/levels_over_time/normative_by_topic.pdf",
    width = default_charts_width,
    height = default_chart_height,
    device = cairo_pdf,
    units = "cm"
)


over_time_level_3 <- read_csv("dominance_coordination_dataset.csv") %>%
  summarise(
    across(all_of(dom_corp), ~ mean(.x, na.rm = T)), .by = "year"
  ) %>% pivot_longer(-year, names_to = "dominance", values_to = "share") %>%
  mutate(dominance = factor(dominance, levels = dom_corp)) %>%
  ggplot(aes(x = year, y = share, color = dominance)) + geom_line() + theme_ecb_replica() +
  guides(color=guide_legend(nrow=3,byrow=TRUE, keyheight = 0.75)) +
  scale_color_discrete(labels = c("Monetary dominance", "Financial dominance", "Fiscal dominance", "Monetary-fiscal coordination", "Monetary-financial coordination")) +
  labs(
    color = "", x = "Year", y = "Share"
  ) +
  annotate("text", x = (1997 + 2007) / 2, y = 0.47, label = "(i)", size= 4, family="Garamond") +
  geom_vline(xintercept = 2008, linetype = "dashed", size = 0.3) +
  annotate("text", x = (2008 + 2020) / 2, y = 0.47, label = "(ii)", size= 4, family="Garamond") +
  geom_vline(xintercept = 2020, linetype = "dashed", size = 0.3) +
  annotate("text", x = (2021 + 2023) / 2, y = 0.47, label = "(iii)", size= 4, family="Garamond") + coord_cartesian(ylim = c(0, 0.47))


over_time_level_3 %>% ggsave(
  filename = "figures/levels_over_time/dominance_and_corporation.pdf",
  width = default_charts_width,
  height = default_chart_height + 1, # Tall legend
  device = cairo_pdf,
  units = "cm"
)