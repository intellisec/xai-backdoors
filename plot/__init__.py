# System
import os
import pathlib

# Lib
import copy
import numpy as np
import matplotlib.pyplot as plt

# Our source
import explain
from explain import *
import utils

def get_grid(rows, cols, double_rows):
    """
    TODO describe
    :param rows:
    :param cols:
    :param double_rows: ?
    """
    fig = plt.figure()
    gs = fig.add_gridspec(rows, cols)
    double_rows = set(double_rows)
    axs = [None for _ in range(rows)]
    for row in range(rows):
        axs[row] = [None for _ in range(cols)]
        if row in double_rows:
            continue
        for col in range(cols):
            axs[row][col] = fig.add_subplot(gs[row, col])
    return fig, axs

def add_left_text(axs, text):
    """
    TODO describe
    :param axs:
    :param text:
    """
    if text is not None:
        pos = 0
        i = 0
        while pos < len(axs):
            a = axs[pos][0]
            if a is None:
                pos+=1
                continue
            a.text(-1, .5, text[i], transform=a.transAxes, fontsize=8)
            i += 1
            pos += 1

def plot_heatmaps(outdir : pathlib.Path, epoch : int, original_model, manipulated_model, x_test : torch.Tensor, label_test : torch.Tensor, run, agg='max', save=True, show=False):
    """
    Creates an overview plot with clean samples, their version with trigger, explanations and predictions
    and explanations and predictions of the original models, as well as from the manipulated model. The number of
    samples depends on the number of attacks included in the run.

    :param outdir:
    :param epoch:
    :param original_model:
    :param manipulated_model:
    :param x_test:
    :param label_test:
    :param run:
    :param save:
    :param show:
    """
    num_samples = 6
    # Choose samples
    samples = copy.deepcopy(x_test[:num_samples].detach().clone())
    ground_truth = label_test[:num_samples].detach().clone()
    if os.getenv("DATASET") == 'cifar10':
        ground_truth_str = [utils.cifar_classes[x] for x in ground_truth]
    elif os.getenv("DATASET") == 'gtsrb':
        ground_truth_str = [utils.gtsrb_classes[x] for x in ground_truth]
    else:
        ground_truth_str = f"no labels for {os.getenv('DATASET')}"


    manipulators = run.get_manipulators()
    num_explanation_methods = len(run.explanation_methodStrs)

    def postprocess_expls(expls):
        return utils.aggregate_explanations(agg, expls)

    trg_samples = []
    for manipulator in manipulators:
        ts = manipulator(copy.deepcopy(samples.detach().clone()))
        trg_samples.append(ts)

    trg_samples = torch.stack(trg_samples)

    expls = []
    expls_man = []

    trg_expls = []
    trg_preds = []
    trg_ys = []

    trg_expls_man = []
    trg_preds_man = []
    trg_ys_man = []
    for i in range(len(run.explanation_methodStrs)):
        explanation_method = run.get_explanation_method(i)
        # Generate the explanations of clean samples on the original model
        tmp, preds, ys = explain.explain_multiple(original_model, samples, explanation_method=explanation_method, create_graph=False)
        # preds and ys does not change for different explanations methods
        tmp = postprocess_expls(tmp)
        expls.append(tmp.detach())


        # Generate the explanations of clean samples on the manipulated model
        tmp, preds_man, ys_man = explain.explain_multiple(manipulated_model, samples, explanation_method=explanation_method, create_graph=False)
        # preds_man and ys_man does not change for different explanation methods
        tmp = postprocess_expls(tmp)
        expls_man.append(tmp.detach())

        # Generate explanation for the trigger samples in the original model
        tmp_expls = []
        tmp_preds = []
        tmp_ys = []
        for man_id in range(run.num_of_attacks):
            e, p, y = explain.explain_multiple(original_model, trg_samples[man_id], explanation_method=explanation_method, create_graph=False)
            e = postprocess_expls(e)
            tmp_expls.append(e)
            tmp_preds.append(p)
            tmp_ys.append(y)

        tmp_expls = torch.stack(tmp_expls).detach()
        tmp_preds = torch.stack(tmp_preds)
        tmp_ys = torch.stack(tmp_ys)

        trg_expls.append(tmp_expls)
        trg_preds.append(tmp_preds)
        trg_ys.append(tmp_ys)


        # Generate explanation the trigger samples in the manipulated model
        tmp_expls_man = []
        tmp_preds_man = []
        tmp_ys_man = []
        for man_id in range(run.num_of_attacks):
            e, p, y = explain.explain_multiple(manipulated_model, trg_samples[man_id], explanation_method=explanation_method, create_graph=False)
            e = postprocess_expls(e)
            tmp_expls_man.append(e)
            tmp_preds_man.append(p)
            tmp_ys_man.append(y)

        tmp_expls_man = torch.stack(tmp_expls_man).detach()
        tmp_preds_man = torch.stack(tmp_preds_man)
        tmp_ys_man = torch.stack(tmp_ys_man)

        trg_expls_man.append(tmp_expls_man)
        trg_preds_man.append(tmp_preds_man)
        trg_ys_man.append(tmp_ys_man)

    num_images_per_sample = run.num_of_attacks + 1
    num_columns = num_samples * num_images_per_sample
    num_rows = 2 + (2 + (2*num_explanation_methods))

    fig, axs = get_grid(num_rows, num_columns, [1, 2+num_explanation_methods, 3+(2*num_explanation_methods)])
    axs = np.array(axs)
    fig.set_size_inches(2 + num_columns, num_rows)
    for i in range(num_samples):
        plot_single_sample(samples[i].cpu(), axs[0,i * num_images_per_sample], normalize=True)
    for i in range(num_samples):
        for man_id in range(run.num_of_attacks):
            plot_single_sample(trg_samples[man_id,i].cpu(), axs[0,i * num_images_per_sample + man_id + 1], normalize=True)

    alpha = 0.3

    # Write prediction on plot
    for i in range(num_samples):
        # Predictions for clean samples
        fig.text(0, -0.5, ground_truth_str[i], transform=axs[0][i * num_images_per_sample].transAxes)

        title0 = utils.top_probs_as_string(ys[i])
        # axs[1][i].text(0, 1.5, title0, transform=axs[1][i].transAxes, fontsize=8)
        fig.text(0, -1, title0, transform=axs[1+num_explanation_methods][i * num_images_per_sample].transAxes)

        title1 = utils.top_probs_as_string(ys_man[i])
        fig.text(0, -1, title1, transform=axs[2 + (2*num_explanation_methods)][i * num_images_per_sample].transAxes)

        # Predictions for manipulated samples
        for man_id in range(run.num_of_attacks):
            fig.text(0, -0.5, ground_truth_str[i], transform=axs[0][i * num_images_per_sample + man_id + 1].transAxes)

            title0 = utils.top_probs_as_string(trg_ys[0][man_id,i])
            # axs[1][i].text(0, 1.5, title0, transform=axs[1][i].transAxes, fontsize=8)
            fig.text(0, -1, title0, transform=axs[1+num_explanation_methods][i * num_images_per_sample + man_id + 1].transAxes)

            title1 = utils.top_probs_as_string(trg_ys_man[0][man_id,i])
            fig.text(0, -1, title1, transform=axs[2 + (2*num_explanation_methods)][i * num_images_per_sample + man_id + 1].transAxes) # +1 as the first 'manipulator' is no manipulator but clean

    # Plot the actual explanations
    for explId in range(num_explanation_methods):
        rowClean = 2 + explId
        rowMan = 3 + num_explanation_methods + explId
        for i in range(num_samples):
            # Plot explanations for clean samples
            plot_single_sample(expls[explId][i].cpu(), axs[rowClean, i * num_images_per_sample], cmap='plasma')
            plot_single_sample(samples[i].cpu(), axs[rowClean, i * num_images_per_sample], normalize=True, bw=True, alpha=alpha)
            # Different visualization: Input times Relevance
            #plot_single_sample((expls[i].repeat(3, 1, 1) * samples[i]).cpu(), axs[2, i * num_images_per_sample], cmap='plasma')

            plot_single_sample(expls_man[explId][i].cpu(), axs[rowMan, i * num_images_per_sample], cmap='plasma')
            plot_single_sample(samples[i].cpu(), axs[rowMan, i * num_images_per_sample], normalize=True, bw=True, alpha=alpha)

            # Plot explanations for trigger samples
            for man_id in range(run.num_of_attacks):
                plot_single_sample(trg_expls[explId][man_id,i].cpu(), axs[rowClean, i * num_images_per_sample + man_id + 1], cmap='plasma')
                plot_single_sample(trg_samples[man_id,i].cpu(), axs[rowClean, i * num_images_per_sample + man_id + 1], normalize=True, bw=True, alpha=alpha)

                plot_single_sample(trg_expls_man[explId][man_id,i].cpu(), axs[rowMan, i * num_images_per_sample + man_id + 1], cmap='plasma')
                plot_single_sample(trg_samples[man_id,i].cpu(), axs[rowMan, i * num_images_per_sample + man_id + 1], normalize=True, bw=True, alpha=alpha)

    rownames = ["Input"]
    for explId in range(num_explanation_methods):
        rownames.append( f'Orig. M. \n{run.explanation_methodStrs[explId]}')
    for explId in range(num_explanation_methods):
        rownames.append(f'Man. M. \n{run.explanation_methodStrs[explId]}')
    add_left_text(axs, rownames)
    plt.suptitle(run.get_params_str_row() + f' epoch {epoch}')

    # fig.tight_layout(pad=0, h_pad=0.5)
    if save:
        utils.save_multiple_formats(fig, outdir / f'plot_{epoch:03d}')
    if show:
        fig.show()
    if save:
        plt.close(fig)
    else:
        return fig

def plot_explanation_to_ax(relevances, sample, ax):
    """
    Return the figure of an explanation plotted on the sample.
    """
    ax.imshow(relevances.permute(1, 2, 0).detach().clone().cpu(), cmap='viridis', interpolation=None)
    ax.imshow(utils.unnormalize_images(sample).mean(dim=0, keepdim=True).permute(1, 2, 0).detach().clone().cpu(), cmap='gray', interpolation=None, alpha=0.4)
    ax.axis('off')

def plot_explanation(relevances, sample, ax=None):
    """
    Return the figure of an explanation plotted on the sample.
    """

    plt.figure(figsize=(10, 10))
    fig, ax = plt.subplots(1, 1)
    fig.tight_layout()
    plt.tight_layout()
    plot_explanation_to_ax(relevances, sample, ax)
    return fig

def plot_sample(sample):
    """
    Plots a samples only
    """
    print(sample.shape)
    plt.figure(figsize=(10, 10))
    fig, ax = plt.subplots(1, 1)
    fig.tight_layout()
    plt.tight_layout()
    ax.set_axis_off()
    ax.imshow(sample.permute(1, 2, 0).detach().clone().cpu())
    ax.axis('off')
    return fig

def plot_single_sample(sample, ax, normalize=False, bw=False, cmap='gray', alpha=1.0):
    if normalize:
        sample = utils.unnormalize_images(sample.unsqueeze(0))[0]
    if bw:
        sample = sample.mean(dim=0,keepdim=True)

    ax.axis('off')
    ax.imshow(sample.permute(1, 2, 0), interpolation='none', cmap=cmap, alpha=alpha)

