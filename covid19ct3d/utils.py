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
#
# Mainly modified from:
# - https://github.com/hasibzunair/3D-image-classification-tutorial/blob/master/3D_image_classification.ipynb
# ==============================================================================


import torch
import numpy as np
import matplotlib.pyplot as plt


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))


def plot_data(data_raw):

    data = np.copy(data_raw)

    height = data.shape[0]
    width = data.shape[1]
    depth = data.shape[2]

    depth = depth - (depth % 5)
    data = data[:, :, :depth]

    num_rows = 5
    num_columns = depth // 5

    data = np.rot90(np.array(data))
    data = np.transpose(data)
    data = np.reshape(data, (num_rows, num_columns, width, height))

    rows_data, columns_data = data.shape[0], data.shape[1]

    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]

    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)

    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )

    for i in range(rows_data):
        for j in range(columns_data):
            axarr[i, j].imshow(data[i][j], cmap="gray")
            axarr[i, j].axis("off")

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.savefig('data.png')

    from moviepy.editor import *

    data = data_raw * 255.0
    data = [np.dstack([data[:, :, idx].astype(np.uint8)]*3) for idx in range(depth)]
    clip = ImageSequenceClip(data, fps=30)
    clip.write_videofile('data.mp4')

    print("Saved in data.png and data.mp4")
