import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from utils import (
    profile,
    cdf,
    make_corner,
    covariance_matching,
    compute_mmd,
    calculate_fgd,
    c2st,
    areas_between_rocs,
)
import matplotlib.pyplot as plt
import numpy as np
import corner
from sklearn.metrics import roc_curve, auc
from scipy import stats
from scipy.stats import ks_2samp
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from scipy.stats import wasserstein_distance


def validate(samples, X, Y, save_dir, epoch, writer):
    root_save_dir = save_dir

    if writer is not None:
        save_dir = os.path.join(save_dir, f"./figures/validation@epoch-{epoch}")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
    else:
        save_dir = os.path.join(save_dir, f"./dump/validation@epoch-{epoch}")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    columns = [
        "btag",
        "recoPt",
        "recoPhi",
        "recoEta",
        "recoNConstituents",
        "nef",
        "nhf",
        "cef",
        "chf",
        "qgl",
        "jetId",
        "ncharged",
        "nneutral",
        "ctag",
        "nSV",
        "recoMass",
    ]

    saturated = []
    ws_dists = []

    for i in range(samples.shape[1]):
        reco = X[:, i]
        flash = samples[:, i]
        ws = wasserstein_distance(reco, flash)
        ws_dists.append(ws)

        fig = plt.figure(figsize=(18, 9))
        gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax3 = plt.subplot(gs[2])
        ax4 = plt.subplot(gs[3], sharex=ax3)

        fig.suptitle(f"{columns[i]} comparison")

        if i != 4 or i != 11 or i != 12:
            bins = 100
        elif i == 14:
            bins = np.arange(-0.5, 9.5, 1)
        else:
            bins = np.arange(-0.5, 80.5, 1)

        # Linear scale plot
        _, rangeR, _ = ax1.hist(reco, histtype="step", lw=1, bins=bins, label="FullSim")

        saturated_samples = np.where(flash < np.min(rangeR), np.min(rangeR), flash)
        saturated_samples = np.where(
            saturated_samples > np.max(rangeR), np.max(rangeR), saturated_samples
        )
        saturated.append(saturated_samples)
        ax1.hist(
            saturated_samples,
            histtype="step",
            lw=1,
            range=[np.min(rangeR), np.max(rangeR)],
            bins=bins,
            label=f"FlashSim, ws={round(ws, 4)}",
        )
        ax1.legend(frameon=False, loc="upper right")

        # Ratio plot for linear scale
        hist_reco, bins_reco = np.histogram(
            reco, bins=bins, range=[np.min(rangeR), np.max(rangeR)]
        )
        hist_flash, bins_flash = np.histogram(
            saturated_samples, bins=bins, range=[np.min(rangeR), np.max(rangeR)]
        )

        ratio = np.where(hist_reco > 0, hist_flash / hist_reco, 0)
        ax3.scatter(bins_reco[:-1], ratio, marker=".", color="b")
        ax3.set_ylabel("Flash/Reco")
        # horizontal line at 1
        ax3.axhline(y=1, color="r", linestyle="--", alpha=0.5)
        ax3.set_ylim(0, 2)

        # Log scale plot
        ax2.set_yscale("log")
        ax2.hist(reco, histtype="step", lw=1, bins=bins)
        ax2.hist(
            saturated_samples,
            histtype="step",
            lw=1,
            range=[np.min(rangeR), np.max(rangeR)],
            bins=bins,
        )

        # Ratio plot for log scale
        ax4.scatter(bins_reco[:-1], ratio, marker=".", color="b")
        ax4.set_ylabel("Flash/Reco")
        # horizontal line at 1
        ax4.axhline(y=1, color="r", linestyle="--", alpha=0.5)
        ax4.set_ylim(0, 2)

        plt.savefig(os.path.join(save_dir, f"{columns[i]}.png"))
        plt.savefig(os.path.join(save_dir, f"{columns[i]}.pdf"))
        if writer is not None:
            writer.add_figure(f"{columns[i]}", fig, global_step=epoch)
            writer.add_scalar(
                f"ws/{columns[i]}_wasserstein_distance", ws, global_step=epoch
            )
        plt.close()

    plt.title("iteration {}".format(epoch))
    #        ranges=[(0,100),(0,100),(0,100)]
    ranges = [
        (0, 1.1),
        (-0.5, 3),
        (0, 2 * np.pi),
        (-5, 5),
        (0, 50),
        (0, 1),
        (0, 1),
        (0, 0.5),
        (0, 1),
        (0, 1),
        (0, 5),
        (0, 50),
        (0, 50),
        (0, 1),
        (0, 10),
        (0, 3),
        (0, 200),
        (-5, 5),
        (0, 2 * np.pi),
        (0, 800),
        (-10, 10),
        (0, 50),
    ]
    labels = [
        "btag",
        "recoPt",
        "recoPhi",
        "recoEta",
        "recoNConstituents",
        "nef",
        "nhf",
        "cef",
        "chf",
        "qgl",
        "jetId",
        "ncharged",
        "nneutral",
        "ctag",
        "nSV",
        "recoMass",  # end reco variables
        "pt",
        "eta",
        "phi",
        "e",
        "flavour",
        "muonsInJet",
    ]

    # compute covariance matching
    covariance_match = covariance_matching(samples, X)
    print(f"covariance matching = {covariance_match}")
    if writer is not None:
        writer.add_scalar("covariance_matching", covariance_match, epoch)

    # compute fgd
    fgd = calculate_fgd(samples, X)
    print(f"FGD = {fgd}")
    if writer is not None:
        writer.add_scalar("fgd", fgd, epoch)
    # compute c2st
    # concatenate X and Y
    X_c2st = np.concatenate((X, Y), axis=1)
    samples_c2st = np.concatenate((samples, Y), axis=1)
    emp_loss, random_losses, pvalue_c2st = 1, 1, 1  # too slow
    if epoch > 900:
        emp_loss = c2st(X_c2st, samples_c2st, clf="hist_gradient_boosting")
    print(f"C2ST = {emp_loss}")
    if writer is not None:
        writer.add_scalar("c2st", emp_loss, epoch)
        writer.add_scalar("c2st_pvalue", pvalue_c2st, epoch)

    flash = np.concatenate((samples, Y), axis=1)
    real = np.concatenate((X, Y), axis=1)

    fig = make_corner(
        real,
        flash,
        labels=labels,
        title="corner_%s.png" % epoch,
        ranges=ranges,
    )
    if writer is not None:
        writer.add_figure("corner", fig, epoch)
    fig.savefig(save_dir + "/corner_%s.png" % epoch)

    plt.close()

    # Separate events based on category
    bjet_indices_real = np.where(np.abs(real[:, 20]) == 5)
    light_indices_real = np.where(real[:, 20] == 0)

    bjet_indices_flash = np.where(np.abs(flash[:, 20]) == 5)
    light_indices_flash = np.where(flash[:, 20] == 0)

    # Get discriminator values
    discriminator_real_bjet = real[bjet_indices_real, 0].flatten()
    discriminator_real_light = real[light_indices_real, 0].flatten()

    discriminator_flash_bjet = flash[bjet_indices_flash, 0].flatten()
    discriminator_flash_light = flash[light_indices_flash, 0].flatten()

    # same for ctag
    cjet_indices_real = np.where(np.abs(real[:, 20]) == 4)
    cjet_indices_flash = np.where(np.abs(flash[:, 20]) == 4)

    discriminator_real_cjet = real[cjet_indices_real, 13].flatten()
    discriminator_flash_cjet = flash[cjet_indices_flash, 13].flatten()

    c_discriminator_real_light = real[light_indices_real, 13].flatten()
    c_discriminator_flash_light = flash[light_indices_flash, 13].flatten()

    # Calculate ROC curves # ADD KS TEST
    BvL_fpr_real, BvL_tpr_real, _ = roc_curve(
        np.concatenate(
            (
                np.ones(len(discriminator_real_bjet)),
                np.zeros(len(discriminator_real_light)),
            )
        ),
        np.concatenate((discriminator_real_bjet, discriminator_real_light)),
    )
    BvL_fpr_flash, BvL_tpr_flash, _ = roc_curve(
        np.concatenate(
            (
                np.ones(len(discriminator_flash_bjet)),
                np.zeros(len(discriminator_flash_light)),
            )
        ),
        np.concatenate((discriminator_flash_bjet, discriminator_flash_light)),
    )

    # Same for CvL

    CvL_fpr_real, CvL_tpr_real, _ = roc_curve(
        np.concatenate(
            (
                np.ones(len(discriminator_real_cjet)),
                np.zeros(len(c_discriminator_real_light)),
            )
        ),
        np.concatenate((discriminator_real_cjet, c_discriminator_real_light)),
    )
    CvL_fpr_flash, CvL_tpr_flash, _ = roc_curve(
        np.concatenate(
            (
                np.ones(len(discriminator_flash_cjet)),
                np.zeros(len(c_discriminator_flash_light)),
            )
        ),
        np.concatenate((discriminator_flash_cjet, c_discriminator_flash_light)),
    )

    # KS TEST HERE
    ks_bjet = ks_2samp(discriminator_real_bjet, discriminator_flash_bjet)
    ks_light = ks_2samp(discriminator_real_light, discriminator_flash_light)
    ks_ctag = ks_2samp(discriminator_real_cjet, discriminator_flash_cjet)
    ks_fpr_BvL = ks_2samp(BvL_fpr_real, BvL_fpr_flash)
    ks_tpr_BvL = ks_2samp(BvL_tpr_real, BvL_tpr_flash)
    ks_fpr_CvL = ks_2samp(CvL_fpr_real, CvL_fpr_flash)
    ks_tpr_CvL = ks_2samp(CvL_tpr_real, CvL_tpr_flash)
    # compute mse between fpr and tpr

    # make 1000 bins of fpr_log
    roc_diff_integrated_BvL = areas_between_rocs(
        BvL_tpr_real, BvL_fpr_real, BvL_tpr_flash, BvL_fpr_flash
    )
    roc_diff_integrated_CvL = areas_between_rocs(
        CvL_tpr_real, CvL_fpr_real, CvL_tpr_flash, CvL_fpr_flash
    )
    print(f"Roc diff BvL= {roc_diff_integrated_BvL}")
    print(f"Roc diff CvL= {roc_diff_integrated_CvL}")
    print(
        "KS test: bjet = %.3f, light = %.3f, ctag = %.3f"
        % (ks_bjet[0], ks_light[0], ks_ctag[0])
    )
    if writer is not None:
        writer.add_scalar("ks_bjet", ks_bjet[0], epoch)
        writer.add_scalar("ks_light", ks_light[0], epoch)
        writer.add_scalar("ks_ctag", ks_ctag[0], epoch)
        writer.add_scalar("ks_fpr_BvL", ks_fpr_BvL[0], epoch)
        writer.add_scalar("ks_tpr_BvL", ks_tpr_BvL[0], epoch)
        writer.add_scalar("ks_fpr_CvL", ks_fpr_CvL[0], epoch)
        writer.add_scalar("ks_tpr_CvL", ks_tpr_CvL[0], epoch)
        writer.add_scalar("roc_diff_integrated_BvL", roc_diff_integrated_BvL, epoch)
        writer.add_scalar("roc_diff_integrated_CvL", roc_diff_integrated_CvL, epoch)
        # store pvalue as well
        writer.add_scalar("ks_bjet_pvalue", ks_bjet[1], epoch)
        writer.add_scalar("ks_light_pvalue", ks_light[1], epoch)
        writer.add_scalar("ks_ctag_pvalue", ks_ctag[1], epoch)
        writer.add_scalar("ks_fpr_BvL_pvalue", ks_fpr_BvL[1], epoch)
        writer.add_scalar("ks_tpr_BvL_pvalue", ks_tpr_BvL[1], epoch)
        writer.add_scalar("ks_fpr_CvL_pvalue", ks_fpr_CvL[1], epoch)
        writer.add_scalar("ks_tpr_CvL_pvalue", ks_tpr_CvL[1], epoch)

    # Calculate AUC
    BvL_auc_real = auc(BvL_fpr_real, BvL_tpr_real)
    BvL_auc_flash = auc(BvL_fpr_flash, BvL_tpr_flash)

    CvL_auc_real = auc(CvL_fpr_real, CvL_tpr_real)
    CvL_auc_flash = auc(CvL_fpr_flash, CvL_tpr_flash)

    # Plot ROC curves
    fig, ax = plt.subplots()  # Changed this line from plt.figure() to plt.subplots()

    ax.plot(
        BvL_tpr_real, BvL_fpr_real, label=f"Real (AUC = {BvL_auc_real:.3f})"
    )  # Changed plt.plot to ax.plot
    ax.plot(
        BvL_tpr_flash, BvL_fpr_flash, label=f"Flash (AUC = {BvL_auc_flash:.3f})"
    )  # Changed plt.plot to ax.plot
    ax.plot([0, 1], [0, 1], "k--")  # Changed plt.plot to ax.plot

    ax.set_xlim([0.0, 1.0])  # Changed plt.xlim to ax.set_xlim
    ax.set_ylim([0.0001, 1.05])  # Changed plt.ylim to ax.set_ylim
    ax.set_yscale("log")  # Changed plt.yscale to ax.set_yscale

    ax.set_ylabel("False Positive Rate")  # Changed plt.xlabel to ax.set_xlabel
    ax.set_xlabel("True Positive Rate")  # Changed plt.ylabel to ax.set_ylabel
    ax.set_title(
        "ROC curves for Real and Flash events (BvsL)"
    )  # Changed plt.title to ax.set_title

    labels = [
        f"Real (AUC = {BvL_auc_real:.2f})",
        f"Flash (AUC = {BvL_auc_flash:.2f})",
    ]
    ax.legend(loc="lower right", labels=labels)  # Changed plt.legend to ax.legend

    ax.text(
        0.05,
        0.95,
        f"KS test_bjet: {ks_bjet[0]:.3f}",
        transform=ax.transAxes,  # Now you can use ax.transAxes here without any problem
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.text(
        0.05,
        0.85,
        f"KS test_light: {ks_light[0]:.3f}",
        transform=ax.transAxes,  # Same here, ax.transAxes will work fine
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.savefig(save_dir + "/roc_%s.png" % epoch)
    if writer is not None:
        writer.add_figure("roc", fig, epoch)  # Changed plt.gcf() to fig
    plt.close()

    # Same for CvL
    fig, ax = plt.subplots()  # Changed this line from plt.figure() to plt.subplots()

    ax.plot(
        CvL_tpr_real, CvL_fpr_real, label=f"Real (AUC = {CvL_auc_real:.3f})"
    )  # Changed plt.plot to ax.plot
    ax.plot(
        CvL_tpr_flash, CvL_fpr_flash, label=f"Flash (AUC = {CvL_auc_flash:.3f})"
    )  # Changed plt.plot to ax.plot
    ax.plot([0, 1], [0, 1], "k--")  # Changed plt.plot to ax.plot

    ax.set_xlim([0.0, 1.0])  # Changed plt.xlim to ax.set_xlim
    ax.set_ylim([0.0001, 1.05])  # Changed plt.ylim to ax.set_ylim
    ax.set_yscale("log")  # Changed plt.yscale to ax.set_yscale

    ax.set_ylabel("False Positive Rate")  # Changed plt.xlabel to ax.set_xlabel
    ax.set_xlabel("True Positive Rate")  # Changed plt.ylabel to ax.set_ylabel
    ax.set_title(
        "ROC curves for Real and Flash events (CvsL)"
    )  # Changed plt.title to ax.set_title

    labels = [
        f"Real (AUC = {CvL_auc_real:.2f})",
        f"Flash (AUC = {CvL_auc_flash:.2f})",
    ]
    ax.legend(loc="lower right", labels=labels)  # Changed plt.legend to ax.legend

    ax.text(
        0.05,
        0.95,
        f"KS test_cjet: {ks_ctag[0]:.3f}",
        transform=ax.transAxes,  # Now you can use ax.transAxes here without any problem
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.savefig(save_dir + "/roc_ctag_%s.png" % epoch)
    if writer is not None:
        writer.add_figure("roc_ctag", fig, epoch)  # Changed plt.gcf() to fig
    plt.close()

    print("ROC done")

    # NOW PROFILE RETURNS KS TEST of mean and std
    fig_prof, ks_vspt_mean, ks_vspt_std = profile(
        [flash, real],
        [
            "flash",
            "real",
        ],
        1,
        16,
        30,
        save_dir,
        "responseVsPt_%s.png" % epoch,
        logScale=True,
        ymax=500,
    )
    fig_prof1, _, _ = profile(
        [
            flash[bjet_indices_flash],
            real[bjet_indices_real],
            flash[light_indices_flash],
            real[light_indices_real],
        ],
        ["flashB", "realB", "flashLight", "realLight"],
        1,
        16,
        30,
        save_dir,
        "responseVsPtFlavour_%s.png" % epoch,
        logScale=True,
        ymax=500,
    )
    fig_prof2, ks_vseta_mean, ks_vseta_std = profile(
        [flash, real],
        [
            "flash",
            "real",
        ],
        1,
        16,
        30,
        save_dir,
        "responseVsEta_%s.png" % epoch,
        logScale=False,
    )
    fig_prof3, ks_prof_bjet_mean, ks_prof_bjet_std = profile(
        [flash[bjet_indices_flash], real[bjet_indices_real]],
        [
            "flashB",
            "realB",
        ],
        1,
        16,
        30,
        save_dir,
        "responseVsPtFlavour_bjet_%s.png" % epoch,
        logScale=True,
        ymax=500,
    )
    fig_prof4, ks_prof_light_mean, ks_prof_light_std = profile(
        [flash[light_indices_flash], real[light_indices_real]],
        [
            "flashLight",
            "realLight",
        ],
        1,
        16,
        30,
        save_dir,
        "responseVsPtFlavour_light_%s.png" % epoch,
        logScale=True,
        ymax=500,
    )
    fig_prof5, ks_prof_cjet_mean, ks_prof_cjet_std = profile(
        [flash[cjet_indices_flash], real[cjet_indices_real]],
        [
            "flashC",
            "realC",
        ],
        1,
        16,
        30,
        save_dir,
        "responseVsPtFlavour_cjet_%s.png" % epoch,
        logScale=True,
        ymax=500,
    )
    # same as fig_prof1 but for cjet
    fig_prof6, _, _ = profile(
        [
            flash[cjet_indices_flash],
            real[cjet_indices_real],
            flash[light_indices_flash],
            real[light_indices_real],
        ],
        ["flashC", "realC", "flashLight", "realLight"],
        1,
        16,
        30,
        save_dir,
        "responseVsPtFlavour_cjet_%s.png" % epoch,
        logScale=True,
        ymax=500,
    )
    if writer is not None:
        writer.add_figure("responseVsPt", fig_prof, epoch)
        writer.add_figure("responseVsPtFlavour", fig_prof1, epoch)
        writer.add_figure("responseVsEta", fig_prof2, epoch)
        writer.add_figure("responseVsPtFlavour_bjet", fig_prof3, epoch)
        writer.add_figure("responseVsPtFlavour_light", fig_prof4, epoch)
        writer.add_figure("responseVsPtFlavour_cjet", fig_prof5, epoch)
        writer.add_figure("responseVsPtFlavour(cjet)", fig_prof6, epoch)
        writer.add_scalar("ks_vspt", ks_vspt_mean[0], epoch)
        writer.add_scalar("ks_vseta", ks_vseta_mean[0], epoch)
        writer.add_scalar("ks_prof_bjet", ks_prof_bjet_mean[0], epoch)
        writer.add_scalar("ks_prof_light", ks_prof_light_mean[0], epoch)
        writer.add_scalar("ks_prof_cjet", ks_prof_cjet_mean[0], epoch)
        writer.add_scalar("ks_vspt_pvalue", ks_vspt_mean[1], epoch)
        writer.add_scalar("ks_vseta_pvalue", ks_vseta_mean[1], epoch)
        writer.add_scalar("ks_prof_bjet_pvalue", ks_prof_bjet_mean[1], epoch)
        writer.add_scalar("ks_prof_light_pvalue", ks_prof_light_mean[1], epoch)
        writer.add_scalar("ks_prof_cjet_pvalue", ks_prof_cjet_mean[1], epoch)

        # same for std
        writer.add_scalar("ks_vspt_std", ks_vspt_std[0], epoch)
        writer.add_scalar("ks_vseta_std", ks_vseta_std[0], epoch)
        writer.add_scalar("ks_prof_bjet_std", ks_prof_bjet_std[0], epoch)
        writer.add_scalar("ks_prof_light_std", ks_prof_light_std[0], epoch)
        writer.add_scalar("ks_prof_cjet_std", ks_prof_cjet_std[0], epoch)
        writer.add_scalar("ks_vspt_std_pvalue", ks_vspt_std[1], epoch)
        writer.add_scalar("ks_vseta_std_pvalue", ks_vseta_std[1], epoch)
        writer.add_scalar("ks_prof_bjet_std_pvalue", ks_prof_bjet_std[1], epoch)
        writer.add_scalar("ks_prof_light_std_pvalue", ks_prof_light_std[1], epoch)
        writer.add_scalar("ks_prof_cjet_std_pvalue", ks_prof_cjet_std[1], epoch)
    plt.close()

    csv_file_path = os.path.join(root_save_dir, "validation_results.csv")

    # Check if file exists to decide between write and append mode
    mode = "a" if os.path.exists(csv_file_path) else "w"

    with open(csv_file_path, mode) as f:
        if mode == "w":
            f.write("epoch,")
            for i in range(len(columns)):
                f.write(f"{columns[i]}_wasserstein_distance,")
            f.write(
                "covariance_matching, FGD, ks_vspt_mean, ks_vspt_std, ks_vspt_mean_pvalue, ks_vspt_std_pvalue, ks_vseta_mean, ks_vseta_std, ks_vseta_mean_pvalue, ks_vseta_std_pvalue, ks_prof_bjet_mean, ks_prof_bjet_std, ks_prof_bjet_mean_pvalue, ks_prof_bjet_std_pvalue, ks_prof_light_mean, ks_prof_light_std, ks_prof_light_mean_pvalue, ks_prof_light_std_pvalue, ks_prof_cjet_mean, ks_prof_cjet_std, ks_prof_cjet_mean_pvalue, ks_prof_cjet_std_pvalue, ks_bjet, ks_bjet_pvalue, ks_light, ks_light_pvalue, ks_ctag, ks_ctag_pvalue, ks_fpr_BvL, ks_fpr_BvL_pvalue, ks_tpr_BvL, ks_tpr_BvL_pvalue, ks_fpr_CvL, ks_fpr_CvL_pvalue, ks_tpr_CvL, ks_tpr_CvL_pvalue, roc_diff_integrated_BvL, roc_diff_integrated_CvL, pvalue_c2st, c2st\n"
            )
        f.write(f"{epoch},")
        for i in range(len(columns)):
            f.write(f"{ws_dists[i]},")
        f.write(
            f"{covariance_match}, {fgd}, {ks_vspt_mean[0]}, {ks_vspt_std[0]}, {ks_vspt_mean[1]}, {ks_vspt_std[1]}, {ks_vseta_mean[0]}, {ks_vseta_std[0]}, {ks_vseta_mean[1]}, {ks_vseta_std[1]}, {ks_prof_bjet_mean[0]}, {ks_prof_bjet_std[0]}, {ks_prof_bjet_mean[1]}, {ks_prof_bjet_std[1]}, {ks_prof_light_mean[0]}, {ks_prof_light_std[0]}, {ks_prof_light_mean[1]}, {ks_prof_light_std[1]}, {ks_prof_cjet_mean[0]}, {ks_prof_cjet_std[0]}, {ks_prof_cjet_mean[1]}, {ks_prof_cjet_std[1]}, {ks_bjet[0]}, {ks_bjet[1]}, {ks_light[0]}, {ks_light[1]}, {ks_ctag[0]}, {ks_ctag[1]}, {ks_fpr_BvL[0]}, {ks_fpr_BvL[1]}, {ks_tpr_BvL[0]}, {ks_tpr_BvL[1]}, {ks_fpr_CvL[0]}, {ks_fpr_CvL[1]}, {ks_tpr_CvL[0]}, {ks_tpr_CvL[1]}, {roc_diff_integrated_BvL}, {roc_diff_integrated_CvL}, {pvalue_c2st}, {emp_loss}\n"
        )
