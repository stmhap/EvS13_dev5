import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.memory import garbage_collection_cuda
from yolo3 import YOLOv3
from dataset import YOLODataset
from loss import YoloLossCumulative
from torch import optim
from torchmetrics import MeanMetric
from torch.utils.data import DataLoader
from torchinfo import summary

import config
from utils import ResizeDataLoader, check_class_accuracy, plot_couple_examples, save_checkpoint, mean_average_precision, get_evaluation_bboxes


class Yolo3_PL_Model(LightningModule):
    def __init__(self, in_channels=3, nclasses=config.NUM_CLASSES, batch_size=config.BATCH_SIZE,
                 learning_rate=config.LEARNING_RATE, collect_garbage='batch', 
                 nepochs=config.NUM_EPOCHS, lambda_noobj=10, lambda_box=10):
        super(Yolo3_PL_Model, self).__init__()
        self.network_architecture = YOLOv3(in_channels, nclasses)
        self.loss_criterion = YoloLossCumulative(config.SCALED_ANCHORS, lambda_noobj=lambda_noobj, lambda_box=lambda_box)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.collect_garbage = collect_garbage
        self.nepochs = nepochs
        self.scaler = torch.cuda.amp.GradScaler()

        # self.scaled_anchors = config.SCALED_ANCHORS
        # we want the scaled anchors to be saved and restored in the state_dict, 
        # but not trained by the optimizer, so we register them as buffers
        # self.register_buffer("scaled_anchors", config.SCALED_ANCHORS)

        self.scaled_anchors = (
                torch.tensor(config.ANCHORS)
                * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
            ).to(config.DEVICE)
        self.train_step_outputs = []
        self.val_step_outputs = []
        self.model_train_loss = MeanMetric()
        self.model_val_loss = MeanMetric()        

    def forward(self, x):
        return self.network_architecture(x)

    def common_step(self, batch, metric):
        x, y = batch
        out = self.forward(x)
        loss = self.loss_criterion(out, y)
        metric.update(loss, x.shape[0])
        del x, y, out
        return loss

    def training_step(self, batch, batch_idx):
        #loss = self.common_step(batch)
        # x, y = batch
        # out = self.forward(x)
        # loss = self.loss_criterion(out, y)
        # self.model_train_loss.update(loss, x.shape[0])
        # del out, x, y
        loss = self.common_step(batch, self.model_train_loss)
        #self.log(f"train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        # Logging the training loss for visualization
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)  
        self.train_step_outputs.append(loss)

        return loss

    def validation_step(self, batch, batch_idx):
        #loss = self.common_step(batch)
        # x, y = batch
        # out = self.forward(x)
        # loss = self.loss_criterion(out, y)
        # self.model_val_loss.update(loss, x.shape[0])
        # del out, x, y
        loss = self.common_step(batch, self.model_val_loss)
        self.log(f"val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.val_step_outputs.append(loss)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, (tuple, list)):
            x, _ = batch
        else:
            x = batch
        return self.forward(x)

    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate/100, weight_decay=config.WEIGHT_DECAY)
        #self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate/100, momentum=0.9)
        scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            steps_per_epoch=len(self.train_dataloader()),
            epochs=self.trainer.max_epochs, #nepochs,
            pct_start=0.2,
            div_factor=100,
            three_phase=False,
            final_div_factor=100,
            anneal_strategy='linear'
        )
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': {
                "scheduler": scheduler,
                "interval": "step",
            }
        }

    def train_dataloader(self):
        train_dataset = YOLODataset(
            config.DATASET + '/train.csv',
            transform=config.train_transforms,
            img_dir=config.IMG_DIR,
            label_dir=config.LABEL_DIR,
            anchors=config.ANCHORS,
            mosaic=0.75
        )

        train_loader = ResizeDataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            shuffle=True,
            resolutions=config.MULTIRES,
            cum_weights=config.CUM_PROBS
        )

        return train_loader

    def val_dataloader(self):
        train_eval_dataset = YOLODataset(
            config.DATASET + '/test.csv',
            transform=config.test_transforms,
            img_dir=config.IMG_DIR,
            label_dir=config.LABEL_DIR,
            anchors=config.ANCHORS,
            mosaic=0
        )

        train_eval_loader = DataLoader(
            dataset=train_eval_dataset,
            batch_size=self.batch_size,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            shuffle=False
        )

        return train_eval_loader

    def predict_dataloader(self):
        return self.val_dataloader()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Clean up Cuda after batch for effective memory management
        if self.collect_garbage == 'batch':
            garbage_collection_cuda()

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        # Clean up Cuda after batch for effective memory management
        if self.collect_garbage == 'batch':
            garbage_collection_cuda()

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        # Clean up Cuda after batch for effective memory management
        if self.collect_garbage == 'batch':
            garbage_collection_cuda()

    def on_validation_epoch_end(self):
        val_epoch_average = torch.stack(self.val_step_outputs).mean()
        self.val_step_outputs.clear()
        print(f"Val loss {val_epoch_average}")

        print(f"Epoch: {self.trainer.current_epoch + 1}, Val Loss: {self.model_val_loss.compute():.2f}")
        self.model_val_loss.reset()        

        if self.collect_garbage == 'epoch':
            garbage_collection_cuda()         

    def on_train_epoch_end(self):
        # Clean up Cuda after batch for effective memory management

        if config.SAVE_MODEL:
            save_checkpoint(self.network_architecture, self.optimizer, filename=config.CHECKPOINT_FILE)
        
        epoch = self.trainer.current_epoch + 1
        #print("Epoch: ", epoch)
		#print(f"Epoch: {epoch}, Global Steps: {self.global_step}, Train Loss: {self.model_train_loss.compute():.2f}")
        print(f"Epoch: {epoch}, Train Loss: {self.model_train_loss.compute():.2f}")
        self.model_train_loss.reset()
        train_epoch_average = torch.stack(self.train_step_outputs).mean()
        self.train_step_outputs.clear()
        print(f"Train loss {train_epoch_average}")

        if epoch > 1 and epoch % 10 == 0:
            plot_couple_examples(self.network_architecture, self.val_dataloader(), 0.6, 0.5, self.scaled_anchors)
        # print(f"\nCurrently epoch {self.current_epoch}")

            # print("On Train Eval loader:")
            print("On Train loader:")
            # class_accuracy, no_obj_accuracy, obj_accuracy = check_class_accuracy(self.network_architecture, self.train_dataloader(),
            #                                                                      threshold=config.CONF_THRESHOLD)

            check_class_accuracy(self.network_architecture, self.train_dataloader(),
                                 threshold=config.CONF_THRESHOLD) 
        if epoch > 30 and epoch % 8 == 0:    
            print("On Train Eval loader:")
            check_class_accuracy(self.network_architecture, self.val_dataloader(),
                                 threshold=config.CONF_THRESHOLD)

            pred_boxes, true_boxes = get_evaluation_bboxes(
                                        self.val_dataloader(),
                                        self.network_architecture,
                                        iou_threshold=config.NMS_IOU_THRESH,
                                        anchors=config.ANCHORS,
                                        threshold=config.CONF_THRESHOLD,
                                    )
            mapval = mean_average_precision(
                                        pred_boxes,
                                        true_boxes,
                                        iou_threshold=config.MAP_IOU_THRESH,
                                        box_format="midpoint",
                                        num_classes= config.NUM_CLASSES,
                                    )
            print(f"MAP:{mapval.item()}")
            self.network_architecture.train()                                                                                                          
            # self.log("class_accuracy", class_accuracy, on_epoch=True, prog_bar=True, logger=True)
            # self.log("no_obj_accuracy", no_obj_accuracy, on_epoch=True, prog_bar=True, logger=True)
            # self.log("obj_accuracy", obj_accuracy, on_epoch=True, prog_bar=True, logger=True)

        if self.collect_garbage == 'epoch':
            garbage_collection_cuda()


def main():
    num_classes = 20
    IMAGE_SIZE = 416
    INPUT_SIZE = IMAGE_SIZE * 2
    yolo3_model = Yolo3_PL_Model(nclasses=num_classes)
    print(summary(yolo3_model, input_size=(2, 3, INPUT_SIZE, INPUT_SIZE)))
    inp = torch.randn((2, 3, INPUT_SIZE, INPUT_SIZE))
    out = yolo3_model(inp)
    assert out[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert out[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert out[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success!")


if __name__ == "__main__":
    main()
