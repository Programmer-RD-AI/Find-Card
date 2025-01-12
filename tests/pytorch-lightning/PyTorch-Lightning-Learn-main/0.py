class Model(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.criterion = MSELoss()
        # all_of_the_neurons_activation_etc

    @staticmethod
    def forward(X):
        # iter over the neurons_activation_etc
        preds = X
        return preds

    def training_step(self, batch, batch_idx):
        images, labels = batch
        preds = self(images)
        loss = self.criterion(preds, labels)
        wandb.log({"Loss": loss.item()})
        return {"train_loss": loss.item()}

    def valid_step(self, batch, batch_idx):
        images, labels = batch
        preds = self(images)
        loss = self.criterion(preds, labels)
        wandb.log({"Val Loss": loss.item()})
        return {"valid_loss": loss.item()}

    @staticmethod
    def valid_dataloaders():
        dataset = Dataset()
        dataloader = DataLoader(dataset)
        return dataloader

    @staticmethod
    def train_dataloaders():
        dataset = Dataset()
        dataloader = DataLoader(dataset)
        return dataloader


wandb.init(project=PROJECT_NAME, name="baseline")
trainer = DefaultTrainer()
trainer.train(Model())
wandb.finish()
