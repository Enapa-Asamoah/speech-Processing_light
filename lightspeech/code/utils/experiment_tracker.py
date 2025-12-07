import os

class ExperimentTracker:
    def __init__(self, cfg):
        self.cfg = cfg.get('tracking', {})
        self.wandb = None
        self.mlflow = None
        if self.cfg.get('use_wandb', False):
            try:
                import wandb
                self.wandb = wandb
                wandb.init(project=self.cfg.get('project_name','lightspeech'),
                           name=self.cfg.get('experiment_name','run'))
            except Exception as e:
                print("W&B init failed:", e)
        if self.cfg.get('use_mlflow', False):
            try:
                import mlflow
                self.mlflow = mlflow
                mlflow.set_experiment(self.cfg.get('experiment_name','lightspeech_mlflow'))
                mlflow.start_run()
            except Exception as e:
                print("MLflow init failed:", e)

    def log_metrics(self, metrics, step=None):
        if self.wandb:
            self.wandb.log(metrics, step=step)
        if self.mlflow:
            for k,v in metrics.items():
                self.mlflow.log_metric(k, v, step=step)

    def finish(self):
        if self.wandb:
            self.wandb.finish()
        if self.mlflow:
            self.mlflow.end_run()
