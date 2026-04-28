import argparse
import csv
import inspect
import json
import os
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing train.csv, val.csv, and test.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where model, metrics, and predictions will be saved",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="FacebookAI/roberta-base",
        help="Base model checkpoint",
    )
    parser.add_argument(
        "--text_col",
        type=str,
        default="sentence",
        help="Name of the input text column",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="cluster_name",
        help="Name of the label column",
    )
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def make_training_args(args):
    """
    Handles both newer and older transformers versions.

    Newer versions use eval_strategy.
    Older versions use evaluation_strategy.
    """
    kwargs = {
        "output_dir": args.output_dir,
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "logging_strategy": "steps",
        "logging_steps": 50,
        "save_total_limit": 2,
        "report_to": "none",
        "seed": args.seed,
    }

    signature = inspect.signature(TrainingArguments.__init__)

    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    return TrainingArguments(**kwargs)


def make_trainer(
    model,
    training_args,
    train_dataset,
    eval_dataset,
    tokenizer,
    data_collator,
    compute_metrics,
):
    """
    Handles both newer and older transformers versions.

    Newer versions prefer processing_class.
    Older versions use tokenizer.
    """
    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }

    signature = inspect.signature(Trainer.__init__)

    if "processing_class" in signature.parameters:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer

    return Trainer(**kwargs)


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_confusion_matrix_csv(path, cm, label_names):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_label \\ predicted_label"] + label_names)

        for label_name, row in zip(label_names, cm):
            writer.writerow([label_name] + list(row))


def save_predictions_csv(path, sentences, true_labels, pred_labels, pred_confidences):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sentence",
                "true_label",
                "pred_label",
                "pred_confidence",
            ]
        )

        for sentence, true_label, pred_label, confidence in zip(
            sentences,
            true_labels,
            pred_labels,
            pred_confidences,
        ):
            writer.writerow(
                [
                    sentence,
                    true_label,
                    pred_label,
                    confidence,
                ]
            )


def softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def main():
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_files = {
        "train": str(data_dir / "train.csv"),
        "validation": str(data_dir / "val.csv"),
        "test": str(data_dir / "test.csv"),
    }

    for split_name, file_path in data_files.items():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing {split_name} file: {file_path}")

    dataset = load_dataset("csv", data_files=data_files)

    text_col = args.text_col
    label_col = args.label_col

    for split_name in ["train", "validation", "test"]:
        columns = dataset[split_name].column_names

        if text_col not in columns:
            raise ValueError(
                f"Column '{text_col}' not found in {split_name}. "
                f"Available columns: {columns}"
            )

        if label_col not in columns:
            raise ValueError(
                f"Column '{label_col}' not found in {split_name}. "
                f"Available columns: {columns}"
            )

    # Build label mapping from train only.
    # Then check that val/test contain no unseen labels.
    train_labels = sorted(set(dataset["train"][label_col]))

    val_labels = set(dataset["validation"][label_col])
    test_labels_set = set(dataset["test"][label_col])
    train_label_set = set(train_labels)

    unseen_val_labels = sorted(val_labels - train_label_set)
    unseen_test_labels = sorted(test_labels_set - train_label_set)

    if unseen_val_labels:
        raise ValueError(
            "Validation set contains labels not present in train.csv: "
            f"{unseen_val_labels}"
        )

    if unseen_test_labels:
        raise ValueError(
            "Test set contains labels not present in train.csv: "
            f"{unseen_test_labels}"
        )

    label2id = {label: i for i, label in enumerate(train_labels)}
    id2label = {i: label for label, i in label2id.items()}

    print("\nLabels:")
    for label, idx in label2id.items():
        print(f"{idx}: {label}")

    save_json(
        output_dir / "labels.json",
        {
            "label2id": label2id,
            "id2label": {str(k): v for k, v in id2label.items()},
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def preprocess(batch):
        tokenized = tokenizer(
            batch[text_col],
            truncation=True,
            max_length=args.max_length,
        )
        tokenized["labels"] = [label2id[label] for label in batch[label_col]]
        return tokenized

    tokenized_dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            preds,
            average="macro",
            zero_division=0,
        )

        acc = accuracy_score(labels, preds)

        return {
            "accuracy": acc,
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1,
        }

    training_args = make_training_args(args)

    trainer = make_trainer(
        model=model,
        training_args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nStarting training...")
    train_result = trainer.train()

    print("\nSaving best validation-selected model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_state()

    train_metrics = train_result.metrics
    trainer.save_metrics("train", train_metrics)
    save_json(output_dir / "train_metrics.json", train_metrics)

    print("\nEvaluating best checkpoint on val.csv...")
    val_metrics = trainer.evaluate(
        eval_dataset=tokenized_dataset["validation"],
        metric_key_prefix="val",
    )

    print("\nValidation metrics:")
    print(val_metrics)

    trainer.save_metrics("val", val_metrics)
    save_json(output_dir / "val_metrics.json", val_metrics)

    print("\nEvaluating best checkpoint on test.csv...")
    test_output = trainer.predict(
        test_dataset=tokenized_dataset["test"],
        metric_key_prefix="test",
    )

    test_metrics = test_output.metrics

    print("\nTest metrics:")
    print(test_metrics)

    trainer.save_metrics("test", test_metrics)
    save_json(output_dir / "test_metrics.json", test_metrics)

    test_logits = test_output.predictions
    test_true_ids = test_output.label_ids
    test_pred_ids = np.argmax(test_logits, axis=-1)

    probabilities = softmax(test_logits)
    pred_confidences = np.max(probabilities, axis=1)

    label_names = [id2label[i] for i in range(len(id2label))]

    report_text = classification_report(
        test_true_ids,
        test_pred_ids,
        target_names=label_names,
        zero_division=0,
    )

    report_dict = classification_report(
        test_true_ids,
        test_pred_ids,
        target_names=label_names,
        zero_division=0,
        output_dict=True,
    )

    print("\nClassification report on test.csv:")
    print(report_text)

    report_txt_path = output_dir / "test_classification_report.txt"
    report_json_path = output_dir / "test_classification_report.json"

    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    save_json(report_json_path, report_dict)

    cm = confusion_matrix(test_true_ids, test_pred_ids)

    print("\nConfusion matrix on test.csv:")
    print(cm)

    save_confusion_matrix_csv(
        output_dir / "test_confusion_matrix.csv",
        cm,
        label_names,
    )

    test_sentences = dataset["test"][text_col]
    test_true_labels = dataset["test"][label_col]
    test_pred_labels = [id2label[int(i)] for i in test_pred_ids]

    save_predictions_csv(
        output_dir / "test_predictions.csv",
        test_sentences,
        test_true_labels,
        test_pred_labels,
        pred_confidences,
    )

    print("\nSaved files:")
    print(f"Best model:                     {output_dir}")
    print(f"Labels:                         {output_dir / 'labels.json'}")
    print(f"Train metrics:                  {output_dir / 'train_metrics.json'}")
    print(f"Validation metrics:             {output_dir / 'val_metrics.json'}")
    print(f"Test metrics:                   {output_dir / 'test_metrics.json'}")
    print(f"Test predictions:               {output_dir / 'test_predictions.csv'}")
    print(f"Test classification report TXT: {report_txt_path}")
    print(f"Test classification report JSON:{report_json_path}")
    print(f"Test confusion matrix:          {output_dir / 'test_confusion_matrix.csv'}")


if __name__ == "__main__":
    main()
