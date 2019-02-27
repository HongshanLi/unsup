import matplotlib.pyplot as plt

def adv_performance(report_path):
    f = open(report_path, "r")
    next(f)
    D_losses = []
    G_losses = []
    for i,line in enumerate(f):
        if i % 1 == 0:
            line = line.rstrip().split(",")
            D_losses.append(float(line[2]))
            G_losses.append(float(line[3]))

    f.close()
    iterations = range(len(D_losses))
    plt.plot(iterations, D_losses, "b",
    iterations, G_losses, "r")
    plt.savefig("adv_performance.png")


def compare_training(fresh_net, adv_net):
    f = open(fresh_net, "r")
    next(f)
    F_losses = []
    for i, line in enumerate(f):
        if i % 1 == 0:
            line = line.rstrip().split(",")
            F_losses.append(float(line[2]))

    f.close()

    f = open(adv_net, "r")
    next(f)
    Adv_losses = []
    for i, line in enumerate(f):
        if i % 1 == 0:
            line = line.rstrip().split(",")
            Adv_losses.append(float(line[2]))

    f.close()

    iterations = range(len(F_losses))
    plt.plot(iterations, F_losses, "b",
    iterations, Adv_losses, "r")
    plt.savefig("compare_training.png")

adv_performance("training_report.txt")

compare_training(
"clf_report_fresh.txt",
"clf_report_adTrained.txt" )
