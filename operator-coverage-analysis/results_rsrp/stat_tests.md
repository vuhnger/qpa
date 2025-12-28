# Statistical Test Results

## violin_quality_by_env.png / boxplot_quality_by_env.png

- Groups: inside (n=2747281), outdoor_driving (n=1572259), outside_walking (n=113707)
- Kruskal-Wallis H: statistic=159059.055, p=0
- Post-hoc (Bonferroni correction):
  - inside vs outdoor_driving: z=-398.704, p_adj=0 (significant)
  - inside vs outside_walking: z=-55.262, p_adj=0 (significant)
  - outdoor_driving vs outside_walking: z=75.374, p_adj=0 (significant)

## violin_quality_by_operator.png / boxplot_quality_by_operator.png / bar_quality_mean_std_by_operator.png

- Groups: Iliad/Wind (n=1028045), TIM/Vodafone (n=3405202)
- Mann-Whitney U: statistic=1836598421699.000, p=0

## boxplot_quality_by_operator_env.png [inside]

- Groups: Iliad/Wind (n=580288), TIM/Vodafone (n=2166993)
- Mann-Whitney U: statistic=654543909035.000, p=0

## boxplot_quality_by_operator_env.png [outdoor_driving]

- Groups: Iliad/Wind (n=447757), TIM/Vodafone (n=1124502)
- Mann-Whitney U: statistic=254543978191.000, p=1.557e-27
