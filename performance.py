import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument("--log-dir", type=str,
help="directory to training logs")
parser.add_argument("--plot-frequency", type=int,
help="show one dot per --plot-frequency batch")
opt = parser.parse_args()


def adv_performance(plot_frequency):
    report_path = os.path.join(opt.log_dir, "training_report.txt")

    f = open(report_path, "r")
    next(f)
    D_losses = []
    G_losses = []
    for i,line in enumerate(f):
        if i % plot_frequency == 0:
            line = line.rstrip().split(",")
            D_losses.append(float(line[2]))
            G_losses.append(float(line[3]))

    f.close()
    iterations = range(len(D_losses))

    d_losses, = plt.plot(iterations, D_losses, "b")
    g_losses, = plt.plot(iterations, G_losses, "r")
    plt.axis([0, len(D_losses), 0, 5])
    

    plt.legend([d_losses, g_losses], ["D loss", "G loss"])
    outf_path = os.path.join(opt.log_dir, "adv_performance.png")
    plt.savefig(outf_path)
    


def compare_training(plot_frequency):
    # log for an untrained discriminator
    log_path = os.path.join(opt.log_dir, "clf_report_fresh.txt")

    f = open(log_path, "r")
    next(f)
    F_losses = []
    for i, line in enumerate(f):
        if i % plot_frequency == 0:
            line = line.rstrip().split(",")
            F_losses.append(float(line[2]))

    f.close()

    # log for an adversarially trained net
    log_path = os.path.join(opt.log_dir, "clf_report_adTrained.txt")
    f = open(log_path, "r")
    next(f)
    Adv_losses = []
    for i, line in enumerate(f):
        if i % plot_frequency == 0:
            line = line.rstrip().split(",")
            Adv_losses.append(float(line[2]))

    f.close()

    iterations = range(len(F_losses))


    fresh_net, = plt.plot(iterations, F_losses, "b+")
    adv_net, = plt.plot(iterations, Adv_losses, "r+")
    plt.axis([0, len(F_losses), 0, 5])
    
    #fresh_legend = plt.legend(handles=[fresh_net], loc=1)
    #adv_legend = plt.legend(handles=[adv_net], loc=1)

    plt.legend([fresh_net, adv_net], ["Fresh net", "Adv net"])
    outf_path = os.path.join(opt.log_dir, "compare_training.png")
    plt.savefig(outf_path)



adv_performance(opt.plot_frequency)

# compare_training(opt.plot_frequency)
