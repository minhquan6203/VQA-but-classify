import os
import shutil
from typing import Dict
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, logging

def setTrainingArgs(config: Dict, device) -> TrainingArguments:
    training_args = config["train"]
    if device.type == 'cuda':
        training_args["fp16"] = True

    # Add early stopping callback
    training_args["load_best_model_at_end"] = True
    training_args["metric_for_best_model"] = "eval_accuracy"
    training_args["greater_is_better"] = False
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=training_args["early_stopping_patience"],
                                                     early_stopping_threshold=training_args["early_stopping_threshold"])

    return TrainingArguments(**training_args), early_stopping_callback

def trainMultimodalModelForVQA(config, device, dataset, collator, model, compute_metrics):
    training_args, early_stopping_callback = setTrainingArgs(config, device)
    training_args.output_dir = os.path.join(training_args.output_dir, config["model"]["name"])
    ckpt=sorted(os.listdir(training_args.output_dir))
    # Load last saved model if exists
    if os.path.exists(training_args.output_dir):
        model_checkpoint = os.path.join(training_args.output_dir, ckpt[-1])
        if os.path.isfile(model_checkpoint):
            model = model.from_pretrained(model_checkpoint)

    multi_trainer = Trainer(
        model,
        training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback]
    )

    train_multi_metrics = multi_trainer.train()
    eval_multi_metrics = multi_trainer.evaluate()

    return train_multi_metrics, eval_multi_metrics
