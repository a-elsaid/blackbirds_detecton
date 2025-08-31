from dataclasses import dataclass, field
import yaml
from pathlib import Path
from typing import List, Optional

@dataclass
class Config:
    output_dir: str = "runs"
    model_dir: str = f"{output_dir}/models"
    log_dir: str = f"{output_dir}/logs"
    tiles_dir: str = f"{output_dir}/tiles"
    batch_size: int = 8
    data_path: Optional[str] = None
    labels_path: Optional[str] = None
    d_x: int = 180
    d_y: int = 240
    use_tiles: bool = False
    data_shuffle: bool = False
    read_from_files: bool = False
    create_tiles: bool = False
    use_4chn: bool = False
    iou_threshold: float = 0.5


    def __post_init__(self):
        self.model_dir = Path(self.model_dir)
        self.tiles_dir = Path(self.tiles_dir)
        self.log_dir = Path(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainConfig(Config):
    num_epochs: int = 10000
    lr: float = 1e-4
    lr_scheduler: bool = False
    use_heavy_aug: bool = False
    
    

    def __post_init__(self):
        super().__post_init__()  
        self.checkpts_dir: str = f"{self.output_dir}/checkpoints"
        self.data_path = Path(self.data_path)
        self.labels_path = Path(self.labels_path)
        self.checkpts_dir = Path(self.checkpts_dir)

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.tiles_dir.mkdir(parents=True, exist_ok=True)
        self.checkpts_dir.mkdir(parents=True, exist_ok=True)

@dataclass
class DetectionConfig(Config):
    models_names: List[str] = None
    burn_results: str | None = None


    def __post_init__(self):
        super().__post_init__()  
        self.results_dir: str = f"{self.output_dir}/detections"
        self.results_dir = Path(self.results_dir)
        # Turn model names into list of Paths
        if self.models_names is None:
            self.models_names = []
        else:
            self.models_names = [self.model_dir / name for name in self.models_names]

        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        if self.burn_results is not None:
            self.burn_results = self.results_dir / self.burn_results
            self.burn_results.mkdir(parents=True, exist_ok=True)


def load_config(file_path: str) -> "Config":
    with open(file_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    return config_dict