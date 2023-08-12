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
for name, dataset in zip(["Train_fp", "Validation_fp"], [train_set, val_set]):
    trainer.get_confusion_matrix(dataset, name, "fp")
    trainer.get_errors(dataset, name, "fp", draw_errors=True)

# build qt model.
trainer.build_qt_model()

# compare qt and fp models.
trainer.quantization_error(train_set)

# qt metrics.
for name, dataset in zip(["Train_qt", "Validation_qt"], [train_set, val_set]):
    trainer.get_confusion_matrix(dataset, name, "qt")
    trainer.get_errors(dataset, name, "qt", draw_errors=True)


trainer.end_run()
