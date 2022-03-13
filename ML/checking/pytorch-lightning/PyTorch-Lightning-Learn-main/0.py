class Model(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.criterion = MSELoss()
        # all_of_the_neurons_activation_etc

    def forward(self, X):
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

    def valid_dataloaders(self):
        dataset = Dataset()
        dataloader = DataLoader(dataset)
        return dataloader

    def train_dataloaders(self):
        dataset = Dataset()
        dataloader = DataLoader(dataset)
        return dataloader


wandb.init(project=PROJECT_NAME, name="baseline")
trainer = DefaultTrainer()
trainer.train(Model())
wandb.finish()
