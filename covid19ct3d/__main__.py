# MIT License
# 
# Copyright (c) 2022 Resha Dwika Hefni Al-Fahsi
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================


import typer
import torch
import logging
import os
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.utils.data import random_split
from .arch import COVID19CT3D
from .train import train_epoch
from .utils import plot_data, AvgMeter
from .dataloader import get_loader, get_dataset, predict_dataset


app = typer.Typer()


@app.command()
def train(epoch: int = typer.Option(50, "--epoch", help='epoch number'),
          lr: float = typer.Option(1e-4, "--lr", help='learning rate'),
          augmentation: bool = typer.Option(False, "--augmentation", help='choose to do data augmentation'),
          batchsize: int = typer.Option(2, "--batchsize", help='training batch size'),
          trainsize: str = typer.Option("128 128 64", "--trainsize", help='training dataset size'),
          train_path: str = typer.Option('./dataset/', "--train_path", help='path to train dataset'),
          train_save: str = typer.Option('./model/', "--train_save", help='path to saved model'),
          step_lr: int = typer.Option(2, "--step_lr", help='set the epoch to drop learning rate')
          ):
    trainsize = [int(x) for x in trainsize.split(" ")]

    logging.basicConfig(filename='train_log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    model = COVID19CT3D()
    weight_pth = os.path.join(train_save, "COVID19CT3D.pth")
    if os.path.exists(weight_pth):
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(weight_pth))
        else:
            model.load_state_dict(torch.load(weight_pth, map_location=torch.device('cpu')))
    
    if torch.cuda.is_available():
        model.cuda()

    params = model.parameters()

    optimizer = torch.optim.Adadelta(params,
                                     lr=lr,
                                     weight_decay=1e-8)

    logging.info(optimizer)
    print(optimizer)

    criterion = nn.BCEWithLogitsLoss()
    if torch.cuda.is_available(): criterion.cuda()

    dataset = get_dataset(train_path, trainsize=trainsize, augmentation=augmentation)
    n_train = int(len(dataset) * 0.9)
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    train_loader = get_loader(train_set, batchsize)
    val_loader = get_loader(val_set, batchsize)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=step_lr)

    total_step = len(train_loader)

    logging.info("#################### Start Training ####################")
    print("#" * 20, "Start Training", "#" * 20)

    best = 99999999

    for n in range(1, epoch + 1):
        train_epoch(train_loader, model, optimizer, n, epoch,
                    batchsize, total_step, logging, criterion
                    )

        val_record = AvgMeter()
        for i, pack in enumerate(val_loader, start=1):
            model.eval()

            images, labels = pack
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()

            with torch.inference_mode():
                pred = model(images)
                loss = criterion(pred, labels)
                val_record.update(loss.data, batchsize)

        scheduler.step(val_record.show())

        if not os.path.exists(train_save):
            os.makedirs(train_save)

        print("Best: ", best)
        print("Val: ", float(val_record.show()))

        logging.info("Best: {}".format(best))
        logging.info("Val: {}".format(float(val_record.show())))

        if val_record.show() < best:
           best = float(val_record.show())
           torch.save(model.state_dict(), os.path.join(train_save, 'COVID19CT3D_best.pth'))

        torch.save(model.state_dict(), os.path.join(train_save, 'COVID19CT3D.pth'))


@app.command()
def predict(predict_size: str = typer.Option("128 128 64", "--predict_size", help='predict size'),
            pth_path: str = typer.Option('./model/COVID19CT3D_best.pth', "--pth_path", help='path to the trained model'),
            data_path: str = typer.Option('./dataset/CT-0/study_0100.nii.gz', "--data_path", help='path to the dataset')
            ):
    predict_size = [int(x) for x in predict_size.split(" ")]

    model = COVID19CT3D()

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(pth_path))
        model.cuda()
    else:
        model.load_state_dict(torch.load(pth_path, map_location=torch.device('cpu')))
    model.eval()

    predict_loader = predict_dataset(data_path, predict_size)
    image = Variable(predict_loader.load_data())
    if torch.cuda.is_available():
        image = image.cuda()

    res = model(image).sigmoid().data.cpu().numpy().squeeze()
    res[res >= 0.5] = 1
    res[res < 0.5] = 0

    image = image.squeeze(0).permute(0,2,3,1).data.cpu().numpy().squeeze(0)
    plot_data(image)

    if int(res):
        print("Suspected COVID-19!!!")
    else:
        print("Normal and Healthy Lung")


@app.command()
def info():
    if torch.cuda.is_available():
        model = COVID19CT3D().cuda()
    else:
        model = COVID19CT3D()

    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model, (1, 64, 128, 128), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
