@echo off
REM NeMo 训练脚本 - 保存输出用于精度对齐 (Windows版本)
REM 如果 GPU 检测失败，可以手动修改下面的 accelerator 为 "cpu" 或 "cuda"

python scripts/masked_token_pred_pretrain_with_saver.py ^
    --config-path=NeMo/examples/asr/conf/ssl/nest ^
    --config-name=nest_fast-conformer ^
    model.train_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/train_manifest.json ^
    model.validation_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/val_manifest.json ^
    model.train_ds.batch_size=2 ^
    model.validation_ds.batch_size=2 ^
    model.train_ds.num_workers=0 ^
    model.validation_ds.num_workers=0 ^
    model.train_ds.drop_last=false ^
    model.train_ds.shuffle=false ^
    +model.train_ds.seed=42 ^
    model.train_ds.batch_augmentor.prob=0.0 ^
    model.preprocessor.dither=0.0 ^
    model.encoder.dropout=0.0 ^
    model.encoder.dropout_pre_encoder=0.0 ^
    model.encoder.dropout_emb=0.0 ^
    model.encoder.dropout_att=0.0 ^
    trainer.devices=1 ^
    trainer.accelerator=auto ^
    trainer.max_epochs=1 ^
    +trainer.limit_train_batches=1 ^
    trainer.num_sanity_val_steps=0 ^
    exp_manager.resume_if_exists=false ^
    exp_manager.resume_ignore_no_checkpoint=true ^
    +output_dir=../nest_ssl_project/saved_nemo_outputs ^
    +seed=42 ^
    +save_steps="0"

pause

