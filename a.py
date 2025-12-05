def get_lr_scheduler(opt, hps):
    """
    Create learning rate scheduler based on Optimizer hyperparameters.
    
    Args:
        opt: Optimizer instance
        hps: Hyperparams config with Optimizer settings
    
    Returns:
        LambdaLR scheduler
    """
    def _get_cosine_schedule_with_warmup_lr_lambda(
        current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
    ):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if current_step > num_training_steps:
            return hps.Optimizer.lr_min_scale
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(hps.Optimizer.lr_min_scale, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    def lr_lambda(step):
        if hps.Optimizer.lr_use_linear_decay:
            lr_scale = hps.Optimizer.lr_scale * min(1.0, step / hps.Optimizer.lr_warmup)
            decay = max(hps.Optimizer.lr_min_scale, 1.0 - max(0.0, step - hps.Optimizer.lr_start_linear_decay) / hps.Optimizer.lr_decay)
            if decay == 0.0:
                print("Reached end of training")
            return lr_scale * decay
        
        elif hps.Optimizer.lr_use_cosine_decay:
            return _get_cosine_schedule_with_warmup_lr_lambda(
                step, 
                num_warmup_steps=hps.Optimizer.lr_warmup,
                num_training_steps=hps.Optimizer.lr_decay,
                num_cycles=0.5
            )
        
        elif hps.Optimizer.lr_use_constant:
            return 1.0
        
        elif hps.Optimizer.lr_use_noam:
            # Noam scheduler: lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
            # Get d_model from model config
            d_model = hps.model.encoder.d_model if hasattr(hps, 'model') and hasattr(hps.model, 'encoder') else 512
            warmup_steps = hps.Optimizer.lr_warmup
            min_lr_scale = hps.Optimizer.lr_min_scale if hps.Optimizer.lr_min_scale > 0 else 1e-6 / hps.Optimizer.lr
            
            step = max(step, 1)  # Avoid division by zero
            scale = d_model ** (-0.5)
            lr_scale = min(step ** (-0.5), step * warmup_steps ** (-1.5))
            return max(min_lr_scale, scale * lr_scale)
        
        else:
            # Default: exponential decay with warmup
            return hps.Optimizer.lr_scale * (hps.Optimizer.lr_gamma ** (step // hps.Optimizer.lr_decay)) * min(1.0, step / hps.Optimizer.lr_warmup)
    
    shd = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return shd