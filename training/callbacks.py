from pytorch_lightning.callbacks import TQDMProgressBar

class DetailedProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__()
        self.enable = True

    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        if trainer.training:
            if 'loss' in trainer.callback_metrics:
                items["loss"] = f"{trainer.callback_metrics['loss']:.3f}"
            if 'lr' in trainer.callback_metrics:
                items["lr"] = f"{trainer.callback_metrics['lr']:.2e}"
        elif trainer.validating:
            if 'val_loss' in trainer.callback_metrics:
                items["val_loss"] = f"{trainer.callback_metrics['val_loss']:.3f}"
        return items

    def on_train_epoch_start(self, trainer, *args, **kwargs):
        super().on_train_epoch_start(trainer, *args, **kwargs)
        print(f"\nEpoch {trainer.current_epoch}")
