import matplotlib.pyplot as plt


def plot(x, y, title, xlabel, ylabel):
    plt.plot(x, y, linewidth=2.0)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


def multiplot(list_of_x, list_of_y, title, xlabel, ylabel, list_of_legends=None):
    list_plots = []
    for i in range(len(list_of_x)):
        p, = plt.plot(list_of_x[i], list_of_y[i], linewidth=2.0)
        list_plots.append(p)

    if list_of_legends is not None:
        plt.legend(list_plots, list_of_legends)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


def histogram(data, title, xlabel, ylabel, bar_labels=None):
    y = data
    N = len(y)
    x = range(N)
    width = 1 / 1.5

    if bar_labels is not None:
        x = bar_labels

    plt.bar(x, y, width, color="blue")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
