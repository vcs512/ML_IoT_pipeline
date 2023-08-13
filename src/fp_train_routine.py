# Holdout training routine.

from Trainer import Trainer

# separate datasets.
trainer = Trainer()
[train_set, val_set] = trainer.train_val_split(augment=True)

# train floating point model.
trainer.build_fp_model()
trainer.training_loop()
trainer.load_model_trained()

# turn off data augmentation to reproducible results.
[train_set, val_set] = trainer.train_val_split(augment=False)

# get confusion matrix and wrong inferences.
trainer.init_metrics_handler()
for name, set in zip(["Train", "Validation"], [train_set, val_set]):
    trainer.get_confusion_matrix(set, name)
    trainer.get_errors(set, name, draw_errors=True)

trainer.end_run()