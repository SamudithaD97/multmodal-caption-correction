from .dataloaders import build_dataloaders

train_loader, val_loader, test_loader, tokenizer = build_dataloaders(
    data_dir="coco_correction_data",
    batch_size=4,
    num_workers=0,
)

batch = next(iter(train_loader))
print(batch["images"].shape)
print(batch["input_ids"].shape)
print(batch["target_ids"].shape)
print(batch["labels"])