import torch
from torch.utils.data import DataLoader
from torch.nn import NLLLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from generate_model import ViTmBERTGeneration
from vocab import MultiModalVocab
from logging_utils import setup_logger
from data_utils.utils import collate_fn
from loaddata import RawQuestionImageDataset
import evaluation
from evaluation import Cider

import os
import numpy as np
import pickle
import random
from tqdm import tqdm
import itertools
from shutil import copyfile
import json

logger = setup_logger()

class BaseTask:
    def __init__(self, config):

        self.checkpoint_path = os.path.join(config.TRAINING.CHECKPOINT_PATH, config.MODEL.NAME)
        if not os.path.isdir(self.checkpoint_path):
            logger.info("Creating checkpoint path")
            os.makedirs(self.checkpoint_path)

        if not os.path.isfile(os.path.join(self.checkpoint_path, "vocab.bin")):
            logger.info("Creating vocab")
            self.vocab = self.load_vocab(config)
            logger.info("Saving vocab to %s" % os.path.join(self.checkpoint_path, "vocab.bin"))
            pickle.dump(self.vocab, open(os.path.join(self.checkpoint_path, "vocab.bin"), "wb"))
        else:
            logger.info("Loading vocab from %s" % os.path.join(self.checkpoint_path, "vocab.bin"))
            self.vocab = pickle.load(open(os.path.join(self.checkpoint_path, "vocab.bin"), "rb"))

        logger.info("Loading data")
        self.load_datasets(config)
        self.create_dataloaders(config)

        logger.info("Building model")
        self.model = ViTmBERTGeneration(config, self.vocab)
        self.config = config
        self.device = torch.device(config.DEVICE)

        logger.info("Defining optimizer and objective function")
        self.configuring_hyperparameters(config)
        self.optim = Adam(self.model.parameters(), lr=config.TRAINING.LEARNING_RATE, betas=(0.9, 0.98))
        self.scheduler = LambdaLR(self.optim, self.lambda_lr)
        self.loss_fn = NLLLoss(ignore_index=self.vocab.padding_idx)

    def configuring_hyperparameters(self, config):
        raise NotImplementedError

    def load_vocab(self, config):
        vocab = MultiModalVocab(config)

        return vocab
    
    def load_datasets(self, config):
        raise NotImplementedError

    def create_dataloaders(self, config):
        raise NotImplementedError

    def evaluate_loss(self, dataloader: DataLoader):
        raise NotImplementedError

    def evaluate_metrics(self, dataloader: DataLoader):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def lambda_lr(self, step):
        warm_up = self.warmup
        step += 1
        return (self.model.d_model ** -.5) * min(step ** -.5, step * warm_up ** -1.5)

    def load_checkpoint(self, fname) -> dict:
        if not os.path.exists(fname):
            return None

        logger.info("Loading checkpoint from %s", fname)

        checkpoint = torch.load(fname)

        torch.set_rng_state(checkpoint['torch_rng_state'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
        np.random.set_state(checkpoint['numpy_rng_state'])
        random.setstate(checkpoint['random_rng_state'])

        self.model.load_state_dict(checkpoint['state_dict'], strict=False)

        logger.info("Resuming from epoch %s", checkpoint['epoch'])

        return checkpoint

    def save_checkpoint(self, dict_for_updating: dict) -> None:
        dict_for_saving = {
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': self.epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optim.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }

        for key, value in dict_for_updating.items():
            dict_for_saving[key] = value

        torch.save(dict_for_saving, os.path.join(self.checkpoint_path, "last_model.pth"))

    def start(self):
        raise NotImplementedError

    def get_predictions(self, dataset, get_scores=True):
        raise NotImplementedError

class OpenEndedTask(BaseTask):
    def __init__(self, config):
        super().__init__(config)

    def load_feature_datasets(self, config):
        train_dataset = RawQuestionImageDataset(config.DATASET.TRAIN, self.vocab, config)
        dev_dataset = RawQuestionImageDataset(config.DATASET.DEV, self.vocab, config)
        test_dataset = RawQuestionImageDataset(config.DATASET.TEST, self.vocab, config)

        return train_dataset, dev_dataset, test_dataset


    def load_datasets(self, config):
        self.train_dataset, self.dev_dataset, self.test_dataset = self.load_feature_datasets(config)


    def create_feature_dataloaders(self, config):
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=config.DATASET.BATCH_SIZE,
            shuffle=True,
            num_workers=config.DATASET.WORKERS,
            collate_fn=collate_fn
        )
        self.dev_dataloader = DataLoader(
            dataset=self.dev_dataset,
            batch_size=config.DATASET.BATCH_SIZE,
            shuffle=True,
            num_workers=config.DATASET.WORKERS,
            collate_fn=collate_fn
        )
        self.test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=config.DATASET.WORKERS,
            collate_fn=collate_fn
        )

    def create_dataloaders(self, config):
        self.create_feature_dataloaders(config)
  

    def configuring_hyperparameters(self, config):
        self.epoch = 0
        self.warmup = config.TRAINING.WARMUP
        self.score = config.TRAINING.SCORE
        self.learning_rate = config.TRAINING.LEARNING_RATE
        self.rl_learning_rate = config.TRAINING.RL_LEARNING_RATE
        self.training_beam_size = config.TRAINING.TRAINING_BEAM_SIZE
        self.evaluating_beam_size = config.TRAINING.EVALUATING_BEAM_SIZE
        self.patience = config.TRAINING.PATIENCE
        self.train_cider = Cider({f"{idx}": answer for idx, answer in enumerate(self.train_dataset.answers)})

    def evaluate_loss(self, dataloader):
        self.model.eval()
        running_loss = .0
        with tqdm(desc='Epoch %d - Validation' % self.epoch, unit='it', total=len(dataloader)) as pbar:
            with torch.no_grad():
                for it, items in enumerate(dataloader):
                    items = items.to(self.device)
                    with torch.no_grad():
                        out = self.model(items).contiguous()
                    
                    shifted_right_answer_tokens = items.shifted_right_answer_tokens
                    loss = self.loss_fn(out.view(-1, out.shape[-1]), shifted_right_answer_tokens.view(-1))
                    this_loss = loss.item()
                    running_loss += this_loss

                    pbar.set_postfix(loss=running_loss / (it + 1))
                    pbar.update()

        val_loss = running_loss / len(dataloader)

        return val_loss

    def evaluate_metrics(self, dataloader):
        self.model.eval()
        gens = {}
        gts = {}
        with tqdm(desc='Epoch %d - Evaluation' % self.epoch, unit='it', total=len(dataloader)) as pbar:
            for it, items in enumerate(dataloader):
                items = items.to(self.device)
                with torch.no_grad():
                    outs, _ = self.model.beam_search(items, batch_size=items.batch_size, beam_size=self.evaluating_beam_size, out_size=1)

                answers_gt = items.answers
                answers_gen = self.vocab.decode_answer(outs.contiguous().view(-1, self.vocab.max_answer_length), join_words=False)
                for i, (gts_i, gen_i) in enumerate(zip(answers_gt, answers_gen)):
                    gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                    gens['%d_%d' % (it, i)] = [gen_i, ]
                    gts['%d_%d' % (it, i)] = gts_i
                pbar.update()

        scores, _ = evaluation.compute_scores(gts, gens)

        return scores

    def train(self):
        self.model.train()

        running_loss = .0
        with tqdm(desc='Epoch %d - Training with cross-entropy loss' % self.epoch, unit='it', total=len(self.train_dataloader)) as pbar:
            for it, items in enumerate(self.train_dataloader):
                items = items.to(self.device)
                out = self.model(items).contiguous()
                shifted_right_answer_tokens = items.shifted_right_answer_tokens
                self.optim.zero_grad()
                loss = self.loss_fn(out.view(-1, out.shape[-1]), shifted_right_answer_tokens.view(-1))
                loss.backward()

                self.optim.step()
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()
                self.scheduler.step()


    def start(self):
        if os.path.isfile(os.path.join(self.checkpoint_path, "last_model.pth")):
            checkpoint = self.load_checkpoint(os.path.join(self.checkpoint_path, "last_model.pth"))
            best_val_score = checkpoint["best_val_score"]
            patience = checkpoint["patience"]
            self.epoch = checkpoint["epoch"] + 1
            self.optim.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            best_val_score = 0.
            patience = 0

        while True:

            self.train()

            self.evaluate_loss(self.dev_dataloader)

            # val scores
            scores = self.evaluate_metrics(self.dev_dict_dataloader)
            logger.info("Validation scores %s", scores)
            val_score = scores[self.score]

            # Prepare for next epoch
            best = False
            if val_score > best_val_score:
                best_val_score = val_score
                patience = 0
                best = True
            else:
                patience += 1

            exit_train = False

            if patience == self.patience:
                logger.info('patience reached.')
                exit_train = True


            self.save_checkpoint({
                'best_val_score': best_val_score,
                'patience': patience,
            })

            if best:
                copyfile(os.path.join(self.checkpoint_path, "last_model.pth"), 
                        os.path.join(self.checkpoint_path, "best_model.pth"))

            if exit_train:
                break

            self.epoch += 1

    def get_predictions(self):
        if not os.path.isfile(os.path.join(self.checkpoint_path, 'best_model.pth')):
            logger.error("Prediction require the model must be trained. There is no weights to load for model prediction!")
            raise FileNotFoundError("Make sure your checkpoint path is correct or the best_model.pth is available in your checkpoint path")

        self.load_checkpoint(os.path.join(self.checkpoint_path, "best_model.pth"))

        self.model.eval()
        results = []
        overall_gens = {}
        overall_gts = {}
        with tqdm(desc='Getting predictions: ', unit='it', total=len(self.test_dict_dataloader)) as pbar:
            for it, items in enumerate(self.test_dict_dataloader):
                items = items.to(self.device)
                with torch.no_grad():
                    outs, _ = self.model.beam_search(items, batch_size=items.batch_size, beam_size=self.evaluating_beam_size, out_size=1)

                answers_gt = items.answers
                answers_gen = self.vocab.decode_answer(outs.contiguous().view(-1, self.vocab.max_answer_length), join_words=False)
                gts = {}
                gens = {}
                for i, (gts_i, gen_i) in enumerate(zip(answers_gt, answers_gen)):
                    gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                    gens['%d_%d' % (it, i)] = gen_i
                    gts['%d_%d' % (it, i)] = gts_i
                    overall_gens['%d_%d' % (it, i)] = [gen_i, ]
                    overall_gts['%d_%d' % (it, i)] = gts_i
                pbar.update()

                results.append({
                    "id": items.question_id,
                    "image_id": items.image_id,
                    "filename": items.filename,
                    "gens": gens,
                    "gts": gts
                })

                pbar.update()

        scores, _ = evaluation.compute_scores(overall_gts, overall_gens)
        logger.info("Evaluation scores on test: %s", scores)

        json.dump({
            "results": results,
            **scores,
        }, open(os.path.join(self.checkpoint_path, "test_results.json"), "w+"), ensure_ascii=False)