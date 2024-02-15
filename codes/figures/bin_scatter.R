source("codes/figures/functions_and_settings.R")

# Dominance by democracy
(wrap_plots(
  named_dom_corp %>% imap(
    function(name, label) {
      binned_scatter(data = read_csv("dominance_coordination_dataset.csv"), name, "democracy_ind", y_axis_label = label, group_label = "Democracy", put_y_label_as_title = T)
    }
  ), guides = "collect", ncol = 2) & theme(legend.position = 'bottom', axis.text = element_text(size = 9.5), title = element_text(size = 9.5))) %>%
  ggsave(filename = "./figures/bin_scatter/democracy.pdf",
         width = panel_chart_width,
         height = panel_chart_height,
         device = cairo_pdf,
         units = "cm"
  )

# Dominance by Advanced
(wrap_plots(
  named_dom_corp %>% imap(
    function(name, label) {
      binned_scatter(data = read_csv("dominance_coordination_dataset.csv"), name, "advanced", y_axis_label = label, group_label = "Advanced", put_y_label_as_title = T)
    }
  ), guides = "collect", ncol = 2) & theme(legend.position = 'bottom', axis.text = element_text(size = 9.5), title = element_text(size = 9.5))) %>%
  ggsave(filename = "./figures/bin_scatter/advanced.pdf",
         width = panel_chart_width,
         height = panel_chart_height,
         device = cairo_pdf,
         units = "cm"
  )

# Polarization
(wrap_plots(
  named_dom_corp %>% imap(
    function(name, label) {
      binned_scatter(data = read_csv("dominance_coordination_dataset.csv"), name, "polarization_ind", y_axis_label = label, group_label = "Polarization", put_y_label_as_title = T)
    }
  ), guides = "collect", ncol = 2) & theme(legend.position = 'bottom', axis.text = element_text(size = 9.5), title = element_text(size = 9.5))) %>%
  ggsave(filename = "./figures/bin_scatter/polarization.pdf",
         width = panel_chart_width,
         height = panel_chart_height,
         device = cairo_pdf,
         units = "cm")
