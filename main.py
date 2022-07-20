import matplotlib.pyplot as plt

from source.bilevel_optimization import Algorithm

# some settings
generate_plots_thesis = True
generate_additional_plots = True

s1 = Algorithm("configs/masterthesis.json")
s1.calculate()

if generate_plots_thesis:
    s1.gen_plot_timeseries()
    s1.gen_plot_violin()
    s1.gen_plot_regression()

if generate_additional_plots:
    s1.gen_plot_energypv_battery()

plt.show()
