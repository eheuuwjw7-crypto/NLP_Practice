import time
from dataclasses import dataclass

import torch.cuda
from datasets import tqdm
from sklearn.metrics import accuracy_score, f1_score
from torch import GradScaler
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader
from src.configuration.config import *
from src.preprocess.dataset import get_dataset


# 训练配置
@dataclass
class TrainConfig:
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    save_step: int = SAVE_STEP
    log_dir: str = LOG_DIR
    output_dir: str = MODELS_DIR
    early_stop_metric: str = 'loss'
    early_stop_patience: int = 5
    use_amp: bool = True

class Trainer:
    def __init__(self, model, train_dataset, valid_dataset, collate_fn, compute_metrics, device, train_config=TrainConfig()):
        # 训练参数配置
        self.train_config = train_config
        # 模型和设备
        self.model = model.to(device)
        self.device = device
        # 数据集和整理函数
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.collate_fn = collate_fn
        # 评价函数
        self.compute_metrics = compute_metrics
        # 优化器
        self.optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
        self.step = 1
        self.writer = SummaryWriter(log_dir=str(Path(self.train_config.log_dir) / time.strftime("%Y-%m-%d_%H:%M:%S")))
        self.early_stop_best_score = -float('inf')
        self.early_stop_patience = 0
        self.scaler = GradScaler(device=self.device.type, enabled=self.train_config.use_amp)
        self.checkpoint_path = Path(self.train_config.output_dir) / 'last' / 'checkpoint.pt'
    # 获取数据加载器
    def _get_dataloader(self, dataset):
        dataset.set_format("torch")
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=self.collate_fn
        )
        return dataloader
    # 训练
    def train(self):
        # 加载模型
        self._load_checkpoint()
        self.model.train()

        train_loader = self._get_dataloader(self.train_dataset)

        for epoch in range(self.train_config.epochs):
            for batch in tqdm(train_loader, desc=f'Epoch: {epoch + 1}'):
                loss = self._train_one_batch(batch)
                if self.step % self.train_config.save_step == 0:
                    self.writer.add_scalar('loss', loss, self.step)
                    metrics = self.evaluate()
                    metrics_str = '|'.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
                    tqdm.write(f'Evaluation:{metrics_str}')
                    if self._early_stopping(metrics):
                        tqdm.write('Early stopping')
                        return
                    # 保存模型
                    self._save_checkpoint()
                    # if loss < self.min_loss:
                    #     self.min_loss = loss
                    #     tqdm.write(f'Saved model with loss {loss:.4f}')
                    #     self.model.save_pretrained(self.train_config.output_dir)
                self.step += 1
    # 训练一个批次
    def _train_one_batch(self, batch):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.autocast(
                self.device.type,
                torch.float16,
                enabled=self.train_config.use_amp
        ):
            output = self.model(**batch)
            loss = output.loss
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        return loss.item()

    # 评估，返回字典，记录不同评价指标：训练集的准确率、召回率、F1分数和损失函数值。
    def evaluate(self) -> dict:
        self.model.eval()
        valid_loader = self._get_dataloader(self.valid_dataset)
        total_loss = 0.0
        all_labels = []
        all_preds = []
        for batch in tqdm(valid_loader, desc='Evaluation'):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            output = self.model(**batch)
            loss = output.loss
            total_loss += loss.item()
            logits = output.logits
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.tolist())
            labels = batch['labels']
            all_labels.extend(labels.tolist())
        loss = total_loss / len(valid_loader)
        metrics = self.compute_metrics(all_labels, all_preds)
        return {
            'loss': loss,
            **metrics
        }
    # 提前停止
    def _early_stopping(self, metrics):
        metrics = metrics[self.train_config.early_stop_metric]
        score = -metrics if self.train_config.early_stop_metric == 'loss' else metrics
        if score > self.early_stop_best_score:
            self.early_stop_best_score = score
            self.early_stop_patience = 0
            tqdm.write(f'Saved model with {self.train_config.early_stop_metric} {score:.4f}')
            self.model.save_pretrained(str(Path(self.train_config.output_dir) / 'best'))
            return False
        else:
            self.early_stop_patience += 1
            if self.early_stop_patience >= self.train_config.early_stop_patience:
                return True
            else:
                return False

    def _save_checkpoint(self):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'step': self.step,
            'early_stop_best_score': self.early_stop_best_score,
            'early_stop_patience': self.early_stop_patience,
        }
        torch.save(checkpoint, self.checkpoint_path)

    def _load_checkpoint(self):
        if self.checkpoint_path.exists():
            tqdm.write(f'Loaded checkpoint from {self.checkpoint_path}')
            checkpoint = torch.load(self.checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            self.step = checkpoint['step']
            self.early_stop_best_score = checkpoint['early_stop_best_score']
            self.early_stop_patience = checkpoint['early_stop_patience']
        else:
            tqdm.write(f'No checkpoint found at {self.checkpoint_path}')

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    # 从持久化的标签文件读取类别数，避免和预处理阶段的标签映射不一致。
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = f.read().splitlines()
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    train_dataset = get_dataset("train")
    valid_dataset = get_dataset("valid")
    collate_fn = DataCollatorWithPadding(
        tokenizer,
        padding=True,
        return_tensors='pt'
    )

    def compute_metrics(labels, preds) -> dict:
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='weighted')
        return {
            'accuracy': acc,
            'f1': f1
        }

    # 训练配置
    train_config = TrainConfig(
        batch_size=BATCH_SIZE,
        output_dir=MODELS_DIR,
        log_dir=LOG_DIR,
    )
    # 定义训练器
    trainer = Trainer(
        model,
        train_dataset,
        valid_dataset,
        collate_fn,
        compute_metrics,
        device,
        train_config
    )
    trainer.train()

if __name__ == '__main__':
    train()