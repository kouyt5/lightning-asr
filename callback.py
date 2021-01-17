import pytorch_lightning as pl

class MyCallback(pl.Callback):
    pass

class CheckCallback(pl.callbacks.ModelCheckpoint):
    pass