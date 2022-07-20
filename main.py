import matplotlib.pyplot as plt

from source.bilevel_optimization import Algorithm

# some settings
generate_plots = True

s1 = Algorithm("configs/masterthesis.json")
s1.calculate()

if generate_plots:
    s1.gen_plot_timeseries()
    s1.gen_plot_violin()
    s1.gen_plot_regression()
    plt.show()
