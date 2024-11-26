import torch.nn as nn
import torch
import argparse
import wandb

from tqdm import tqdm
import numpy as np
import os

import json

from metrics import eval_and_save_epoch_end, compute_metrics_retrieval
from model.evaluate_rac import retrieve_evaluate_RAC_, final_evaluation
from model.classifier import classifier_hateClipper
from model.loss import compute_loss
from rac_dataloader import CLIP2Dataloader
from dataset import load_feats_from_CLIP

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def parse_args():

    arg_parser = argparse.ArgumentParser()

    # <----------------- Data Configs ----------------->
    arg_parser.add_argument("--path", type=str, default="./data/")
    arg_parser.add_argument("--output_path", type=str, default="./logging/")
    arg_parser.add_argument("--model", type=str, default="")
    arg_parser.add_argument("--dataset", type=str, default="HarMeme")
    # The threshold for the similarity score for RAC
    arg_parser.add_argument("--similarity_threshold", type=float, default=-1.)
    arg_parser.add_argument("--fusion_mode", type=str, default="concat")
    arg_parser.add_argument("--topk", type=int, default=5, help="Retrieve at most k pairs for validation")
    arg_parser.add_argument("--majority_voting", type=str, default="mean", help="Choose the majority voting method, options are mean, arithmetic, geometric, learned")

    # ----------------- Loss Configs -----------------
    # The loss function for the model is a combination of two parts:
    # Metric class and loss class, both need to be specified
    arg_parser.add_argument("--metric", type=str, default="cos",help="Choose the metric for similarity score, options are cos, ip, l2")
    """
    cos: cosine similarity
    ip: inner product
    l2: l2 distance
    if we use a certain type of metric, we will also use the same criterion for dense retrieval
    """
    arg_parser.add_argument("--loss", type=str, default="naive", help="Choose to use which loss function, options are naive, triplet, contrastive")
    arg_parser.add_argument("--triplet_margin", type=float, default=0.1, help="The margin for triplet loss, epsilon")
    arg_parser.add_argument("--norm_feats_loss", type=lambda x: (str(x).lower() == "true"), default=False, help="Whether to normalize the feature fpr computing loss ")
    # Do sqrt for L2
    arg_parser.add_argument("--l2_sqrt", type=lambda x: (str(x).lower() == "true"), default=False, help="Whether to do square root for L2 loss ")
    arg_parser.add_argument("--hybrid_loss", type=lambda x: (str(x).lower() == "true"), default=False, help="Whether to use logistic loss for the model")
    arg_parser.add_argument("--ce_weight", type=float, default=0.5, help="The weight for the cross entropy loss")
    arg_parser.add_argument("--pos_weight_value", type=float, default=None, help="The weight for the positive samples in the cross entropy loss")

    # <----------------- Model Configs ----------------->
    arg_parser.add_argument("--num_layers", type=int, default=3)
    # MLP dimension for general
    arg_parser.add_argument("--proj_dim", type=int, default=1024)
    # For hateclipper
    # the pre-modality fusion feature projection dimension
    arg_parser.add_argument("--map_dim", type=int, default=1024)
    arg_parser.add_argument("--dropout", type=float, nargs=3, default=[0.1, 0.4, 0.2], help="Set drop probabilities for map, fusion, pre_output")
    arg_parser.add_argument("--batch_norm", type=lambda x: (str(x).lower() == "true"), default=False, help="Whether to use batch norm for Mapping Network")
    arg_parser.add_argument("--last_layer", type=str, default="none", help="Choose the last layer for the model, options are none, sigmoid, tanh")

    # ----------------- Training Configs -----------------
    arg_parser.add_argument("--epochs", type=int, default=5)
    # batch size also sets the number of in_batch positive and in_batch negative
    # we can set limit to the size of in_batch samples
    arg_parser.add_argument("--batch_size", type=int, default=4)
    arg_parser.add_argument("--lr", type=float, default=0.0001)
    arg_parser.add_argument("--weight_decay", type=float, default=0.0001)
    arg_parser.add_argument("--lr_scheduler", type=lambda x: (str(x).lower() == "true"), default=False, help="Using LR scheduler or not",)
    arg_parser.add_argument("--num_workers", type=int, default=4)
    # default set to zero to match the number of in_batch samples
    arg_parser.add_argument("--grad_clip", type=float, default=0.1, help="Gradient clipping")

    # <----------------- Psuedo Gold Positive Configs ----------------->
    arg_parser.add_argument("--no_pseudo_gold_positives", type=int, default=1)

    # <----------------- Hard Negative Configs ----------------->
    # we need to experiment with different settings here:
    # set a limit ot the number of hard negatives to be retrieved
    # set a threshold for the hard negatives,
    # use single threshold or both of the above threhsolding
    arg_parser.add_argument("--in_batch_loss", type=lambda x: (str(x).lower() == "true"), default=True, help="Using in batch loss for model training",) 
    arg_parser.add_argument("--hard_negatives_loss", type=lambda x: (str(x).lower() == "true"), default=False, help="Using hard negative loss for model training",)
    arg_parser.add_argument("--no_hard_negatives", type=int, default=1)
    arg_parser.add_argument("--no_hard_positives", type=int, default=0)
    arg_parser.add_argument("--hard_negatives_multiple", type=int, default=12, help="The value times the no_hard_negatives is the\ number of most similar retrieved pairs hard negatives to be retrieved for each sample",)
    arg_parser.add_argument("--Faiss_GPU", type=lambda x: (str(x).lower() == "true"), default=False, help="Whether to use GPU for Faiss")
    arg_parser.add_argument("--reindex_every_step", type=lambda x: (str(x).lower() == "true"), default=False, help="Whether to reindex the faiss index every step for dense retrieval")
    # For sparse hard negative
    # If the sparse dictionary file is not None, we will use sparse retrieval,
    # otherwise, dense retrieval is used as default when the dictioary file is None
    arg_parser.add_argument("--sparse_dictionary", type=str, default=None, help="The name of the file of the sparse retrieval dictionary",)
    arg_parser.add_argument("--use_attribute", default=True, type=lambda x: (str(x).lower() == "true"), help="Whether to use attribute for object detection in sparse data",)
    arg_parser.add_argument("--sparse_topk", type=int, default=None, help="The number of topk retrieved samples for sparse retrieval",)    
    arg_parser.add_argument("--eval_retrieval", default=True, type=lambda x: (str(x).lower() == "true"), help="Using retrieval evaluation",)

    # <----------------- Logging Configs ----------------->
    arg_parser.add_argument("--log_interval", type=int, default=10)
    arg_parser.add_argument("--final_eval", type=lambda x: (str(x).lower() == "true"), default=False, help="Doing the final eval or not",)
    arg_parser.add_argument("--exp_comment", type=str, default="", help="Optional comment for the experiment")
    arg_parser.add_argument("--group_name", type=str, default="RAC_TEST", help=" Name for the wandb group")
    arg_parser.add_argument("--seed", type=int, default=0)
    arg_parser.add_argument("--device", type=str, default="cuda")
    arg_parser.add_argument("--visualise_embed", type=bool, default=False)
    arg_parser.add_argument("--force", type=lambda x: (str(x).lower() == "true"), default=False, help="Whether to force the run or not")
    arg_parser.add_argument("--save_embed", type=lambda x: (str(x).lower() == "true"), default=False, help="Save the embedding or not",)
    
    args = arg_parser.parse_args()
    return args

def model_pass(
    train_dl,
    evaluate_dl,
    test_seen_dl,
    model,
    epochs=0,
    log_interval=10,
    args=None,
    artifacts=None,
    train_set=None,
    sparse_dict=None,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    num_training_steps = args.epochs * len(train_dl)

    # Learning rate scheduler
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,  # Maximum iterations
            eta_min=1e-5,  # Minimum learning rate
        )

    global_step = -1
    best_acc = 0.0
    best_epoch_path = None

    for epoch in tqdm(range(epochs), desc="Training Progress"):
        train_feats = None
        train_labels = None

        for step, batch in enumerate(train_dl):
            if args.reindex_every_step:
                train_feats = None
                train_labels = None

            global_step += 1

            (
                total_loss,
                in_batch_loss,
                hard_loss,
                pseudo_gold_loss,
                cross_entropy,
                train_feats,
                train_labels,
            ) = compute_loss(
                batch,
                train_dl,
                model,
                args,
                train_set=train_set,
                sparse_retrieval_dictionary=sparse_dict,
                train_feats=train_feats,
                train_labels=train_labels,
            )

            # Detach gradients when using dense retrieval without sparse dictionary
            if not (
                (args.no_hard_negatives == 0 and args.no_pseudo_gold_positives == 0)
                or args.sparse_dictionary is not None
            ):
                train_feats = train_feats.detach()
                train_labels = train_labels.detach()

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            if args.lr_scheduler:
                scheduler.step()

            # if step % log_interval == 0:
            #     print(
            #         f"Train Epoch: {epoch} [{step}/{len(train_dl)} "
            #         f"({100.0 * step / len(train_dl):.0f}%)]\tLoss: {total_loss.item():.6f}"
            #     )

    # for epoch in range(1):
        # Evaluate retrieval metrics
        if args.eval_retrieval:
            # best_model = "/home/dhruv/Documents/Courses/EECE7205/Project/Code/RGCL-main_MyCopy/logging/Retrieval/HarMeme/RAC/RAC_lr0.0001_Bz4_Ep30_cosSim_triplet_drop[0.2, 0.4, 0.1]_topK20__PseudoGold_positive_1_hard_negative_1_seed0_hybrid_loss/ckpt/best_model_12_0.7684.pt"
            # model.load_state_dict(torch.load(best_model, weights_only=True))
            logging_dict, evaluate_labels = retrieve_evaluate_RAC_(
                train_dl,
                evaluate_dl,
                model,
                largest_retrieval=args.topk,
                threshold=args.similarity_threshold,
                args=args,
                eval_name="dev",
                epoch=epoch,
            )

            acc, roc, pre, recall, f1, _, _ = compute_metrics_retrieval(
                logging_dict, evaluate_labels, majority_voting=args.majority_voting, topk=args.topk, use_sim=True
            )

            logging_dict_test, test_labels = retrieve_evaluate_RAC_(
                train_dl,
                test_seen_dl,
                model,
                largest_retrieval=args.topk,
                threshold=args.similarity_threshold,
                args=args,
                eval_name="test",
                epoch=epoch,
            )

            acc_test, roc_test, pre_test, recall_test, f1_test, _, _ = compute_metrics_retrieval(
                logging_dict_test, test_labels, majority_voting=args.majority_voting, topk=args.topk, use_sim=True
            )
        else:
            acc, roc, pre, recall, f1 = (0, 0, 0, 0, 0)
            acc_test, roc_test, pre_test, recall_test, f1_test = (0, 0, 0, 0, 0)

        # Log hybrid loss metrics
        if args.hybrid_loss:
            eval_and_save_epoch_end(
                args, artifacts, train_dl, evaluate_dl, test_seen_dl, model, epoch
            )

        # Print epoch summary
        print(
            f"Val Retrieval Epoch {epoch}: acc: {acc:.4f}, roc: {roc:.4f}, "
            f"pre: {pre:.4f}, recall: {recall:.4f}, f1: {f1:.4f}"
        )
        print(
            f"Test Retrieval Epoch {epoch}: acc: {acc_test:.4f}, roc: {roc_test:.4f}, "
            f"pre: {pre_test:.4f}, recall: {recall_test:.4f}, f1: {f1_test:.4f}"
        )

        # Save model
        epoch_path = f"{args.output_path}/ckpt/model_{epoch}_{acc:.4f}.pt"
        torch.save(model.state_dict(), epoch_path)

        # # Save the best model
        # if acc > best_acc:
        #     best_acc = acc
        #     best_epoch_path = f"{args.output_path}/ckpt/best_model_{epoch}_{acc:.4f}.pt"
        #     print(f"Saving best model for epoch {epoch} with accuracy: {best_acc:.4f}")
        #     torch.save(model.state_dict(), best_epoch_path)

        # # Save the last model
        # if epoch == args.epochs - 1:
        #     last_model_path = f"{args.output_path}/ckpt/last_model_{epoch}_{acc:.4f}.pt"
        #     print("Saving the last model...")
        #     torch.save(model.state_dict(), last_model_path)

    return model

def main(args):
    # ----------------- Name the Model ----------------- #
    # Define the loss string based on the metric and loss type
    if args.metric == "cos":
        loss_str = "cosSim"
    elif args.metric == "ip":
        loss_str = "innerProduct"
    elif args.metric == "l2":
        loss_str = "L2"

    if args.loss == "naive":
        loss_str += "_naive"
    elif args.loss == "triplet":
        loss_str += "_triplet"
    elif args.loss == "contrastive":
        loss_str += "_contrastive"

    # Define naming for hard negatives and positives
    hard_negative_name = f"_hard_negative_{args.no_hard_negatives}"

    if args.no_pseudo_gold_positives != 0 and args.no_hard_positives != 0:
        positive_name = f"_PseudoGold_positive_{args.no_pseudo_gold_positives}_hard_positive_{args.no_hard_positives}"
    elif args.no_pseudo_gold_positives != 0:
        positive_name = f"_PseudoGold_positive_{args.no_pseudo_gold_positives}"
    elif args.no_hard_positives != 0:
        positive_name = f"_hard_positive_{args.no_hard_positives}"
    else:
        positive_name = "inbatch_positive"

    # Use group name from args
    group_name = args.group_name

    # Experiment naming convention
    exp_name = (
        f"RAC_lr{args.lr}_Bz{args.batch_size}_Ep{args.epochs}_{loss_str}"
        f"_drop{args.dropout}_topK{args.topk}_{positive_name}{hard_negative_name}"
        f"_seed{args.seed}{'_hybrid_loss' if args.hybrid_loss else ''}{args.exp_comment}"
        f"{'_' + args.sparse_dictionary if args.sparse_dictionary is not None else ''}"
    )

    # Construct output path
    args.output_path = os.path.join(
        args.output_path, "Retrieval", args.dataset, group_name, exp_name, ""
    )
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        os.makedirs(os.path.join(args.output_path, "ckpt/"))
    else:
        if not args.force:
            print(args.force)
            raise Exception("Output path already exists, aborting...")

    print(args)

    # ----------------- Load the Data ----------------- #
    # Load datasets based on the specified dataset type
    if args.dataset == "FB":
        train, dev, test_seen, test_unseen = load_feats_from_CLIP(
            os.path.join(args.path, "CLIP_Embedding"), "FB", args.model
        )
    else:
        train, dev, test_seen = load_feats_from_CLIP(
            os.path.join(args.path, "CLIP_Embedding"), args.dataset, args.model
        )

    # Prepare data loaders
    (train_dl, dev_dl, test_seen_dl), (train_set, _, _) = CLIP2Dataloader(
        train,
        dev,
        test_seen,
        batch_size=args.batch_size,
        return_dataset=True,
        normalize=False,
    )

    # The data loader contains:
    # - "ids": the ID of the sample
    # - "image_feats": image feature embeddings
    # - "text_feats": text feature embeddings
    # - "labels": labels for the samples

    # Load sparse retrieval dictionary if specified
    if args.sparse_dictionary is not None:
        sparse_dict = {}
        with open(
            os.path.join(
                args.path,
                "Sparse_Retrieval_Dict",
                args.dataset,
                f"{args.sparse_dictionary}.json",
            ),
            "r",
        ) as file:
            for line in file:
                subdict = json.loads(line)
                sparse_dict[subdict["id"]] = subdict
    else:
        sparse_dict = None

    # ----------------- Construct the Model ----------------- #
    # Get feature dimensions
    image_feat_dim = list(enumerate(train_dl))[0][1]["image_feats"].shape[1]
    text_feat_dim = list(enumerate(train_dl))[0][1]["text_feats"].shape[1]

    print("Image feature dimension:", image_feat_dim)
    print("Text feature dimension:", text_feat_dim)

    # Initialize the model
    model = classifier_hateClipper(
        image_feat_dim,
        text_feat_dim,
        args.num_layers,
        args.proj_dim,
        args.map_dim,
        args.fusion_mode,
        dropout=args.dropout,
        batch_norm=args.batch_norm,
        args=args,
    )
    model.to(args.device)
    print(model)

    # ----------------- Train the Model ----------------- #
    model = model_pass(
        train_dl,
        dev_dl,
        test_seen_dl,
        model,
        epochs=args.epochs,
        log_interval=args.log_interval,
        args=args,
        artifacts=None,
        train_set=train_set,
        sparse_dict=sparse_dict,
    )

if __name__ == "__main__":
    args = parse_args()

    # set the seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
