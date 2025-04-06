import os, torch, time, argparse
import yaml
from trainer_ffa_net import FFANetTrainer
from configs import Config



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="trainer")
    parser.add_argument('--config_file', type=str, default='configs/config.yaml', help="path to YAML config")
    parser.add_argument('--output_dir', type=str, default=None, help="path to output directory (optional); defaults to outputs/model_name")
    args = parser.parse_args()

    # Load YAML configuration
    with open(args.config_file, 'r') as file:
        config_dict = yaml.safe_load(file)
        config = Config(config_dict=config_dict)

    if config.network.model.lower() == 'kjrd_net':
        net_trainer = KJRDTrainer(config=config, output_dir=args.output_dir)
    else:
        raise ValueError(f"{config.network.model} not supported.") 
    
    net_trainer.train()