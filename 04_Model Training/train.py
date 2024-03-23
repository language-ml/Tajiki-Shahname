from pathlib import Path

import torch
import wandb
from tqdm import tqdm
import transformers

from _trainer.auto_save import AutoSave
from _trainer.run_loops import train_loop, valid_loop
from _trainer.best_finder import BestFinder
from _datasets import get_tokenizer, load_dataset, load_dict_dataset, map_dataset, generate_dataloader, compute_metrics, generate_output_preprocess
from _config import load_config
from _utils import print_system_info, silent_logs
from _model import get_model


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_project_name(config):
    name_stack = [config.source, config.target]
    if config.project_name_prefix is not None:
        name_stack = [config.project_name_prefix] + name_stack
    return '_'.join(name_stack)

def get_run_name(config, model):
    name_stack = [model.name_or_path]
    if config.experiment_name_suffix is not None:
        name_stack.append(config.experiment_name_suffix)
    return '_'.join(name_stack)

def run_config(config):
    dataset = load_dict_dataset(config.data_path)
    tokenizer = get_tokenizer()
    
    dataset = map_dataset(dataset, tokenizer, config.source, config.target)
    
    model = get_model(tokenizer, config.model_params.to_dict())
    model.to(DEVICE)

    best_finder = BestFinder(config.best_finder.higher_better)
    
    project_name = get_project_name(config)
    run_name = get_run_name(config, model)

    wandb.init(
        name=run_name,
        project=project_name,
        config=config.to_dict(),
    )

    saver = AutoSave(
        model=model,
        path=Path(config.base_save_path) / project_name / run_name
    )

    train_loader, valid_loader = generate_dataloader(
        tokenizer,
        dataset['train'],
        dataset['valid'],
        train_bs=config.train_batch_size,
        valid_bs=config.valid_batch_size
    )
    
    output_preprocess = generate_output_preprocess(tokenizer)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate
    )

    saver.save('first')

    epochs_range = range(config.num_epochs)
    if config.use_tqdm:
        epochs_range = tqdm(epochs_range, position=1, desc="EPOCHS", leave=False)

    for epoch in epochs_range:
        epoch_results = {}

        epoch_results.update(
            train_loop(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                gradient_clipping=config.gradient_clipping,
                use_tqdm=config.use_tqdm
            )
        )

        epoch_results.update(
            valid_loop(
                model=model,
                loader=valid_loader,
                use_tqdm=config.use_tqdm,
                compute_metrics=compute_metrics,
                output_preprocess=output_preprocess
            )
        )
        wandb.log(epoch_results)

        if best_finder.is_better(epoch_results[config.best_finder.metric]):
            saver.save('best')

        saver.save('last')

    wandb.finish()

    
if __name__ == '__main__':
    silent_logs()
    print_system_info()
    
    all_configs = load_config('./config.yaml')
    
    for config in tqdm(all_configs.run_configs, position=0, desc="Experiment"):
        torch.cuda.empty_cache()
        run_config(config)
