import matplotlib.pyplot as plt

def plot_single_value(value, label, title, xlabel, ylabel):
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