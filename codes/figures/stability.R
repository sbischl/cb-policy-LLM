source("codes/figures/functions_and_settings.R")

(read_stata("outputdata/robustness/stability/stability.dta") %>%
  pivot_longer(
    starts_with("l"), names_to = "classification_level", values_to = "stability"
  ) %>% ggplot(
  aes(x = agree, y = stability, color = classification_level)
) + geom_point() + geom_line() + theme_ecb_replica() +
  scale_x_continuous(breaks = c(1,2,3)) +
  scale_color_discrete(name = "", labels = c("Level 1", "Level 2", "Level 3")) +
  labs(
  x = "Coders agreeing",
  y = "Stability")) %>% ggsave(
  filename = "figures/stability/stability_r.pdf",
  width = default_charts_width,
  height = default_chart_height,
  device = cairo_pdf,
  units = "cm"
)