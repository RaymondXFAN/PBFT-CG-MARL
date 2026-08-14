"""
训练日志工具

- Logger类：记录训练指标
- 支持console输出和文件输出
- 支持WandB（可选）
- 关键指标：episode_reward, episode_length, value_loss, policy_loss, entropy, consensus_rate, lr
"""

import os
import time
import json
import logging
import numpy as np
import torch
from typing import Dict, Optional, Any, List


class NumpyEncoder(json.JSONEncoder):
    """支持 numpy/torch 类型的 JSON encoder"""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        return super().default(obj)


class Logger:
    """
    训练日志记录器
    
    支持多种输出方式：console、文件、WandB
    
    Args:
        log_dir: 日志保存目录
        use_wandb: 是否使用WandB
        wandb_project: WandB项目名称
        wandb_config: WandB配置
        console_level: 控制台日志级别
        file_level: 文件日志级别
    """

    def __init__(
        self,
        log_dir: str = "./logs",
        use_wandb: bool = False,
        wandb_project: str = "PBFT-CG-MARL",
        wandb_config: Optional[Dict[str, Any]] = None,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
    ):
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project

        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)

        # 设置Python logger
        self._logger = logging.getLogger("PBFT-CG-MARL")
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers = []  # 清除已有handler

        # 控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_format = logging.Formatter(
            "[%(asctime)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_format)
        self._logger.addHandler(console_handler)

        # 文件输出
        log_file = os.path.join(log_dir, "training.log")
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(file_level)
        file_format = logging.Formatter(
            "[%(asctime)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        self._logger.addHandler(file_handler)

        # 训练指标存储
        self._metrics: Dict[str, List[float]] = {}
        self._current_episode = 0
        self._start_time = time.time()

        # WandB初始化
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=wandb_project,
                    config=wandb_config or {},
                    dir=log_dir,
                )
                self._wandb = wandb
            except ImportError:
                self._logger.warning("WandB未安装，将跳过WandB日志记录")
                self.use_wandb = False

    def info(self, message: str) -> None:
        """记录INFO级别日志"""
        self._logger.info(message)

    def debug(self, message: str) -> None:
        """记录DEBUG级别日志"""
        self._logger.debug(message)

    def warning(self, message: str) -> None:
        """记录WARNING级别日志"""
        self._logger.warning(message)

    def error(self, message: str) -> None:
        """记录ERROR级别日志"""
        self._logger.error(message)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = "",
    ) -> None:
        """
        记录训练指标
        
        Args:
            metrics: 指标字典
            step: 当前步数
            prefix: 指标前缀
        """
        # 存储指标
        for key, value in metrics.items():
            full_key = f"{prefix}/{key}" if prefix else key
            if full_key not in self._metrics:
                self._metrics[full_key] = []
            self._metrics[full_key].append(value)

        # 控制台输出
        metrics_str = " | ".join(
            [f"{k}: {v:.4f}" for k, v in metrics.items()]
        )
        self._logger.info(f"Step {step} | {prefix} | {metrics_str}")

        # WandB记录
        if self.use_wandb:
            logged_metrics = {}
            for key, value in metrics.items():
                full_key = f"{prefix}/{key}" if prefix else key
                logged_metrics[full_key] = value
            self._wandb.log(logged_metrics, step=step)

    def log_episode(
        self,
        episode: int,
        episode_reward: float,
        episode_length: int,
        consensus_rate: float = 0.0,
        extra_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        记录Episode信息
        
        Args:
            episode: Episode编号
            episode_reward: Episode总奖励
            episode_length: Episode长度
            consensus_rate: 共识达成率
            extra_metrics: 额外指标
        """
        self._current_episode = episode
        elapsed_time = time.time() - self._start_time

        metrics = {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "consensus_rate": consensus_rate,
            "elapsed_time": elapsed_time,
        }
        if extra_metrics:
            metrics.update(extra_metrics)

        self.log_metrics(metrics, step=episode, prefix="episode")

    def log_training(
        self,
        step: int,
        value_loss: float = 0.0,
        policy_loss: float = 0.0,
        entropy: float = 0.0,
        lr: float = 0.0,
        consensus_rate: float = 0.0,
        extra_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        记录训练指标
        
        Args:
            step: 训练步数
            value_loss: 价值损失
            policy_loss: 策略损失
            entropy: 策略熵
            lr: 学习率
            consensus_rate: 共识率
            extra_metrics: 额外指标
        """
        metrics = {
            "value_loss": value_loss,
            "policy_loss": policy_loss,
            "entropy": entropy,
            "lr": lr,
            "consensus_rate": consensus_rate,
        }
        if extra_metrics:
            metrics.update(extra_metrics)

        self.log_metrics(metrics, step=step, prefix="train")

    def get_metrics(self, key: Optional[str] = None) -> Any:
        """
        获取记录的指标
        
        Args:
            key: 指标名称，None则返回所有指标
            
        Returns:
            指标值或指标字典
        """
        if key is not None:
            return self._metrics.get(key, [])
        return self._metrics

    def save_metrics(self, filename: str = "metrics.json") -> None:
        """
        保存所有指标到JSON文件
        
        Args:
            filename: 文件名
        """
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, "w") as f:
            json.dump(self._metrics, f, indent=2, cls=NumpyEncoder)
        self._logger.info(f"指标已保存到 {filepath}")

    def close(self) -> None:
        """关闭日志记录器"""
        if self.use_wandb:
            self._wandb.finish()
        self.save_metrics()
