source("codes/figures/functions_and_settings.R")

# Correlation with inflation
(wrap_plots(
  named_dom_corp %>% imap(
    function(name, label) {
      read_csv("dominance_coordination_dataset.csv") %>%
        filter(!is.na(advanced)) %>%
        filter(inflation < 0.2) %>%
        ggplot(aes(y = get(name), x = inflation, color = as.factor(advanced))) +
        geom_point(alpha = 0.25, size = 1) +
        scale_y_continuous(limits = c(0,1)) +
        geom_smooth(method = lm, se = TRUE, show.legend = F) +
        labs(
          title = label,
          y = "",
          x = "HICP inflation",
          color = "Advanced"
        ) +
        theme_ecb_replica()
    }
  ), guides = "collect", ncol = 2) & theme(legend.position = 'bottom', axis.text = element_text(size = 9.5), title = element_text(size = 9.5))) %>%
  ggsave(filename = "./figures/correlation/inflation.pdf",
         width = panel_chart_width,
         height = panel_chart_height,
         device = cairo_pdf,
         units = "cm"
  )

# Correlation with spreads
(wrap_plots(
  named_dom_corp %>% imap(
    function(name, label) {
      read_csv("dominance_coordination_dataset.csv") %>%
        filter(!is.na(advanced)) %>%
        ggplot(aes(y = get(name), x = spread, color = as.factor(advanced))) +
        geom_point(alpha = 0.25, size = 1) +
        scale_y_continuous(limits = c(0,1)) +
        geom_smooth(method = lm, se = TRUE, show.legend = F) +
        labs(
          title = label,
          y = "",
          x = "10Y bond spread vis-a-vis Bund 10Y",
          color = "Advanced"
        ) +
        theme_ecb_replica()
    }
  ), guides = "collect", ncol = 2) & theme(legend.position = 'bottom', axis.text = element_text(size = 9.5), title = element_text(size = 9.5))) %>%
  ggsave(filename = "./figures/correlation/spreads.pdf",
         width = panel_chart_width,
         height = panel_chart_height,
         device = cairo_pdf,
         units = "cm")

# Correlation with GDP PPP
(wrap_plots(
  named_dom_corp %>% imap(
    function(name, label) {
      read_csv("dominance_coordination_dataset.csv") %>%
        filter(!is.na(advanced)) %>%
        ggplot(aes(y = get(name), x = gdp_real_ppp_capita, color = as.factor(advanced))) +
        geom_point(alpha = 0.25, size = 1) +
        scale_y_continuous(limits = c(0,1)) +
        geom_smooth(method = lm, se = TRUE, show.legend = F) +
        labs(
          title = label,
          y = "",
          x = "GDP PPP per capita",
          color = "Advanced"
        ) +
        theme_ecb_replica()
    }
  ), guides = "collect", ncol = 2) & theme(legend.position = 'bottom', axis.text = element_text(size = 9.5), title = element_text(size = 9.5))) %>%
  ggsave(filename = "./figures/correlation/gdpppp.pdf",
         width = panel_chart_width,
         height = panel_chart_height,
         device = cairo_pdf,
         units = "cm")
