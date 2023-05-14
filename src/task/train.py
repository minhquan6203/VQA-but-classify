import os
import shutil
from typing import Dict
import transformers
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, logging
import torch
from transformers.optimization import get_scheduler


def setTrainingArgs(config: Dict, device) -> TrainingArguments:
    training_args = config["train"]
    if device.type == 'cuda':
        training_args["fp16"] = True

    # Add early stopping callback
    training_args["load_best_model_at_end"] = True
    training_args["metric_for_best_model"] = "eval_accuracy"
    training_args["greater_is_better"] = True
    early_stopping_patience = config["early_stoping"]["early_stopping_patience"]
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience)

    return TrainingArguments(**training_args), early_stopping_callback


def trainMultimodalModelForVQA(config, device, dataset, collator, model, compute_metrics):
    training_args, early_stopping_callback = setTrainingArgs(config, device)
    training_args.output_dir = os.path.join(training_args.output_dir, config["model"]["name"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)
    scheduler = get_scheduler(
        training_args.lr_scheduler_type,
        optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=training_args.num_train_epochs * len(dataset['train'])
    )

    # Load last saved model if exists
    if os.path.exists(training_args.output_dir):
        if len(os.listdir(training_args.output_dir)) != 0:
            checkpoint_folder=max(os.listdir(training_args.output_dir), key=lambda x: int(x.split('-')[1]))
            model_checkpoint = os.path.join(training_args.output_dir, checkpoint_folder)
            print(f"continue training at {checkpoint_folder}")
        else:
            model_checkpoint=None
    else:
        model_checkpoint=None
        print("frist time training")

    optimizers = (optimizer, scheduler)
    multi_trainer = Trainer(
        model,
        training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        optimizers=optimizers,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback],
    )

    train_multi_metrics = multi_trainer.train(resume_from_checkpoint=model_checkpoint)
    eval_multi_metrics = multi_trainer.evaluate()

    return train_multi_metrics, eval_multi_metrics
