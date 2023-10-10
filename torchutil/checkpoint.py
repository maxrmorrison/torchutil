import torch


###############################################################################
# Checkpoint utilities
###############################################################################


def latest_path(
        directory,
        regex='*.pt',
        latest_fn=lambda path: int(''.join(filter(str.isdigit, path.stem)))):
    """Retrieve the path to the most recent checkpoint"""
    # Retrieve checkpoint filenames
    files = list(directory.glob(regex))

    # If no matching checkpoint files, no training has occurred
    if not files:
        return

    # Retrieve latest checkpoint
    files.sort(key=latest_fn)
    return files[-1]


def load(file, model, optimizer=None, map_location='cpu'):
    """Load model checkpoint"""
    checkpoint = torch.load(file, map_location=map_location)

    # Restore model
    model.load_state_dict(checkpoint['model'])
    del checkpoint['model']

    # Restore optimizer
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    del checkpoint['optimizer']

    return model, optimizer, checkpoint


def save(file, model, optimizer, accelerator=None, **kwargs):
    """Save training checkpoint to disk"""
    if accelerator is None:
        save_fn = torch.save
    else:
        save_fn = accelerator.save
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}
    save_fn(checkpoint | kwargs, file)
