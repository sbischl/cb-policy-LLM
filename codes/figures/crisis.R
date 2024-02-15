source("codes/figures/functions_and_settings.R")

# Beeswarm charts comparing crisis and non-crisis times
(wrap_plots(
  read_csv("dominance_coordination_dataset.csv") %>%
  filter(!is.na(fiscal_crisis)) %>%
  mutate(`Fiscal crisis` = ifelse(fiscal_crisis == 1, "Crisis", "No crisis"),
         `Fiscal crisis` = factor(`Fiscal crisis` , levels = c("No crisis", "Crisis"))
  ) %>%
  ggplot(aes(x = `Fiscal crisis`, y = fiscal_dominance, color = as.factor(advanced), group = advanced)) +
  geom_quasirandom(dodge.width=0.8, alpha = 0.3, width = 0.2, size = 0.6) +
  #stat_summary(position = position_dodge(0.8), width = 0.5, geom = "line", linetype = "dashed") + # Adds a line
  theme_ecb_replica()+ labs(y = "", color = "Advanced") + ggtitle("Fiscal Dominance") +
  stat_summary(position = position_dodge(0.8), width = 0.5, geom = "crossbar") +
  scale_y_continuous(limits = c(0,1)),
  read_csv("dominance_coordination_dataset.csv") %>%
  filter(!is.na(fiscal_crisis)) %>%
  mutate(`Fiscal crisis` = ifelse(fiscal_crisis == 1, "Crisis", "No crisis"),
         `Fiscal crisis` = factor(`Fiscal crisis` , levels = c("No crisis", "Crisis"))
  ) %>%
  ggplot(aes(x = `Fiscal crisis`, y = monetary_fiscal_coordination, color = as.factor(advanced), group = advanced)) +
  geom_quasirandom(dodge.width=0.8, alpha = 0.3, width = 0.2, size = 0.6) +
  #stat_summary(position = position_dodge(0.8), width = 0.5, geom = "line", linetype = "dashed") + # Adds a line
  theme_ecb_replica()+ labs(y = "", color = "Advanced") + ggtitle("Monetary-fiscal coordination") +
  stat_summary(position = position_dodge(0.8), width = 0.5, geom = "crossbar") +
  scale_y_continuous(limits = c(0,1)),
  guides = "collect"
) & theme(legend.position = 'bottom'))  %>%
  ggsave(filename = "./figures/crisis/crisis.pdf",
         width = panel_chart_width,
         height = default_chart_height,
         device = cairo_pdf,
         units = "cm"
  )