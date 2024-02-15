source("codes/figures/functions_and_settings.R")

level_1_temperature <- read_stata("outputdata/robustness/temperature/temperature_l1.dta") %>%
  ggplot(aes(x = temperature, y = accuracy)) +
  geom_point(aes(color = "Individual runs"), position = position_jitter(w = 0.01, h = 0)) +
  scale_y_continuous(limits = c(0.6, 0.85), breaks = seq(0.6, 0.85, 0.05)) +
  stat_summary(aes(color = "Mean"), fun = mean,show.legend = F) +
  stat_summary(aes(color = "Mean"), fun = mean, geom = "line", show.legend = F) +
  theme_ecb_replica() +
  labs(
    x = "Temperature",
    y = "Accuracy",
    color = ""
  )

level_2_temperature <- read_stata("outputdata/robustness/temperature/temperature_l2.dta") %>%
  ggplot(aes(x = temperature, y = accuracy)) +
  geom_point(aes(color = "Individual runs"), position = position_jitter(w = 0.01, h = 0)) +
  scale_y_continuous(limits = c(0.6, 0.85), breaks = seq(0.6, 0.85, 0.05)) +
  stat_summary(aes(color = "Mean"), fun = mean,show.legend = F) +
  stat_summary(aes(color = "Mean"), fun = mean, geom = "line", show.legend = F) +
  theme_ecb_replica() +
  labs(
    x = "Temperature",
    y = "Accuracy",
    color = ""
  )

level_3_temperature <- read_stata("outputdata/robustness/temperature/temperature_l3.dta") %>%
  ggplot(aes(x = temperature, y = accuracy)) +
  geom_point(aes(color = "Individual runs"), position = position_jitter(w = 0.01, h = 0)) +
  stat_summary(aes(color = "Mean"), fun = mean,show.legend = F) +
  stat_summary(aes(color = "Mean"), fun = mean, geom = "line", show.legend = F) +
  theme_ecb_replica() +
  labs(
    x = "Temperature",
    y = "Accuracy",
    color = ""
  )

level_3_temperature %>% ggsave(
  filename = "figures/temperature/temperature_l3_r.pdf",
  width = default_charts_width,
  height = default_chart_height,
  device = cairo_pdf,
  units = "cm"
)

ggsave(wrap_plots(level_1_temperature + ggtitle("Level 1"), level_2_temperature + ggtitle("Level 2"), ncol = 2, guides = "collect") & theme(legend.position = "bottom"),
  filename = "figures/temperature/temperature_l1_l2_r.pdf",
  width = panel_chart_width, # 17
  height = default_chart_height,
  device = cairo_pdf,
  units = "cm")