# Holdout training routine.

from Trainer import Trainer


DRAW_WRONGS = False
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
for name, dataset in zip(["Train", "Validation"], [train_set, val_set]):
    trainer.get_confusion_matrix(dataset, name, "fp")
    trainer.get_errors(dataset, name, "fp", draw_errors=DRAW_WRONGS)

# build qt model.
trainer.build_qt_model()

# compare qt and fp models.
trainer.quantization_error(train_set)

# qt metrics.
for name, dataset in zip(["Train", "Validation"], [train_set, val_set]):
    trainer.get_confusion_matrix(dataset, name, "qt")
    trainer.get_errors(dataset, name, "qt", draw_errors=DRAW_WRONGS)

# test results.
test_set = trainer.test_set_gen()
models_type = ["fp", "qt"]
for model_type in models_type:
    trainer.get_confusion_matrix(test_set, "Test", model_type)
    trainer.get_errors(test_set, "Test", model_type, draw_errors=DRAW_WRONGS)


trainer.end_run()
