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


import torch
from torch.autograd import Variable
from datetime import datetime

from .utils import AvgMeter
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import logging


def train_epoch(train_loader, model, optimizer, n, epoch,
                batchsize, total_step, logging, criterion
                ):
    model.train()

    loss_record = AvgMeter()

    for i, pack in enumerate(train_loader, start=1):

        optimizer.zero_grad()

        # ---- data prepare ----
        images, labels = pack
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        # ---- forward ---- 
        pred = model(images)

        # ---- loss function ----
        loss = criterion(pred, labels)

        # ---- backward ----
        loss.backward()

        optimizer.step()

        # ---- recording loss ----
        loss_record.update(loss.data, batchsize)

        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{}/{}], Step [{}/{}], '
                  '[loss: {}'
                  ']'.
                  format(datetime.now(), n, epoch, i, total_step, loss_record.show(),
                         ))
            logging.info('Epoch [{}/{}], Step [{}/{}], '
                        '[loss: {}'
                        ']'.
                        format(n, epoch, i, total_step, loss_record.show(),
                               ))
