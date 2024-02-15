# Check if packages are installed
required_packages <- c("ggnewscale",
                       "Hmisc",
                       "tidyverse",
                       "ggbeeswarm",
                       "ggpubr",
                       "patchwork",
                       "haven")

not_installed_packages <- required_packages[!required_packages %in% installed.packages()[, 1]]

# Ask for user confirmation before installing any packages
if (length(not_installed_packages) > 0) {
  cat("The following required packages are not installed: \n")
  cat(paste(not_installed_packages, collapse = "\n"), "\n")

  while (TRUE) {
    confirmation <- readline("Do you want to install these packages now? (y/n)")
    if (confirmation == "y") {
      install.packages(not_installed_packages)
      break
    }
    else if (confirmation == "n") {
      stop("Need to have the required packages installed. Exiting...")
    }
  }
}

# Load all libraries specified above in 'required_packages'
invisible(lapply(required_packages, FUN = library, character.only = TRUE))

# Settings:
default_charts_width <-  12 # In cm
default_chart_height <-  9 # in cm
panel_chart_height <- 19.5 #cm
panel_chart_width  <- 15 #cm

dom_corp <- c("monetary_dominance", "financial_dominance", "fiscal_dominance", "monetary_fiscal_coordination", "monetary_financial_coordination")
dom_corp_label <- c("Monetary dominance", "Financial dominance", "Fiscal dominance", "Monetary-fiscal coordination", "Monetary-financial coordination")
named_dom_corp <- set_names(dom_corp, dom_corp_label)
topics <- c("topic_monetary", "topic_financial", "topic_macro", "topic_international", "topic_fiscal", "topic_climate", "topic_other")

# Set the ECB theme
theme_ecb_replica <- function() {
   theme_minimal() %+replace%
     theme(
       text = element_text(size=rel(1), family="Garamond", color = "black"),
       axis.text = element_text(colour="black"),
       axis.title = element_text(size = rel(1), color = "black"),
       title = element_text(face = "bold"),
       legend.title = element_text(face="bold"),
       legend.text = element_text(size = rel(0.9)),
       axis.text.x = element_text(size = rel(0.9), margin = margin(t = 1, r = 0, b = 0, l = 0)),
       axis.text.y = element_text(size = rel(0.9), margin = margin(t = 0, r = 1, b = 0, l = 0)),
       #axis.line = element_line(colour = "grey75", linewidth = 0.3),
       axis.ticks = element_line(colour = "grey70", linewidth = 0.45),
       panel.grid.major = element_line(colour = "grey75", linetype = "dotted", linewidth =0.3),
       panel.grid.minor = element_blank(),
       axis.ticks.length=unit(.1, "cm"),
       panel.border = element_rect(colour = "grey70", fill=NA, linewidth =0.5, linetype="solid"),
       legend.position = "top",
       legend.spacing.y = unit(0, 'cm')
     )
}
windowsFonts("Garamond" = windowsFont("Garamond"))
ecb_colors <- c("#003299", "#FFB400", "#FF4B00", "#65B800", "#00B1EA", "#007816", "#8139C6", "#5C5C5C", "#98A1D0")
options(ggplot2.discrete.colour= ecb_colors)

# Helper functions
yearly_group_averages <- function(data, variable,
                                  group_by = "advanced",
                                  weight = NULL,
                                  group_levels = c("1" = "Advanced",  "0" = "Non-Advanced")) {
  data %>% {
    if (!is.null((weight))) {
      summarise(., value = weighted.mean(get(variable), w = get(weight), na.rm = T),
                .by = c("year", group_by))
    } else {
      summarise(., value = mean(get(variable), na.rm = T),
                .by = c("year", group_by))
    }
  } %>% setNames(c("year", "group", "value")) %>% filter(!is.na(group)) %>%
    mutate(
      group = as.character(group),
      group = recode(group, !!!group_levels)
    )
}

# Helper functions for bin scatter
wtd.stderror <- function(x, weights, na.rm = FALSE){
    if (na.rm == TRUE) {
      non_missing <- (!is.na(x)) & (!is.na(weights))
      x <-  x[non_missing]
      weights <-  weights[non_missing]
    }
    var <- Hmisc::wtd.var(x, weights)
    weights <- sum( (weights / sum(weights))^2 )
    sqrt(var*weights)
  }

# This function produces the bin scatter plots
binned_scatter <- function(data,
                           y_axis_variable,
                           grouping_var,
                           group_label = grouping_var,
                           y_axis_label = y_axis_variable,
                           weight = "number_of_speeches",
                           put_y_label_as_title = FALSE,
                           transparency_is_weight = TRUE,
                           limits = c(0,1)
                           ) {
  binned_data <- data %>%
  filter(!is.na(country) & country != "") %>%
    mutate(groups = as.factor(get(grouping_var))) %>%
  {
    if (!is.null(weight)) {
      mutate(., cb_weight = get(weight) / sum(get(weight), na.rm = T), .by = "year")
    } else .
  } %>% filter(!is.na(groups))

  if (is.null(weight)) {
    averages <- binned_data %>%  group_by(year, groups) %>% summarise(
      mean_se(get(y_axis_variable))
    )
  } else {
    averages <-  binned_data %>%  group_by(year, groups) %>% summarise(
      y = weighted.mean(x = get(y_axis_variable), w = get(weight), na.rm = T),
      ymin = y - 1.96 * wtd.stderror(get(y_axis_variable), get(weight), na.rm = T),
      ymax = y + 1.96 * wtd.stderror(get(y_axis_variable), get(weight), na.rm = T)
    )
  }

  post <- position_jitterdodge(jitter.width = 0.05, dodge.width = 0.4)

  plot <- ggplot(binned_data, aes(y = get(y_axis_variable), x = year, color = groups)) +
    geom_point(aes(alpha = cb_weight), size = 1, position = post) +
    scale_alpha_continuous(range = c(0.1, 0.5), guide = NULL) +
    geom_line(data = averages, aes(y = y), position = position_dodge(0.4)) +
    geom_point(data = averages, aes(y = y), size = 1.75, position = position_dodge(0.4)) +
    geom_errorbar(data = averages, aes(ymin = ymin, ymax = ymax, x = year, color = groups), inherit.aes = FALSE, position = position_dodge(0.4))


  if (put_y_label_as_title) {
    plot <-  plot + scale_y_continuous(limits = limits, oob = scales::oob_keep) + labs(title = y_axis_label, y ="", x = "Year", color = group_label) + theme_ecb_replica()
  } else {
    plot <-  plot + scale_y_continuous(limits = limits, oob = scales::oob_keep) + labs(y = y_axis_label,x = "Year", color = group_label) + theme_ecb_replica()
  }
  plot
}