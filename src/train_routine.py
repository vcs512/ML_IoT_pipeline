# Holdout training routine.

from Trainer import Trainer

# separate datasets.
trainer = Trainer()
[train_set, val_set] = trainer.train_val_split()

# train floating point model.
trainer.build_fp_model()
trainer.training_loop()
trainer.load_model_trained()

# get confusion matrix and wrong inferences.
trainer.init_metrics_handler()
for name, set in zip(["Train", "Validation"], [train_set, val_set]):
    trainer.get_confusion_matrix(set, name)
    trainer.get_errors(set, name)

trainer.end_run()