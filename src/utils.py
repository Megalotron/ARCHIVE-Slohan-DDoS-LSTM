import matplotlib.pyplot as plt

def plot_single_value(value, label, title, xlabel, ylabel) -> None:
    """
    Plot a single value with a given label.

    :param value: The value to plot.
    :param label: The label of the value.
    :param title: The title of the plot.
    :param xlabel: The x-axis label.
    :param ylabel: The y-axis label.

    :return: None
    """

    plt.plot(value, label=label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.legend()

    plt.show()


def plot_mulitple_values(values, labels, title, xlabel, ylabel):
    for value, label in zip(values, labels):
        plt.plot(value, label=label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.legend()

    plt.show()