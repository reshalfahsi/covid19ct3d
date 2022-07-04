<h1 align="center">
  COVID-19 Classification from 3D CT Images
</h1>

<h4 align="center">
  Tutorial on COVID-19 Classification from 3D CT Images.
</h4>


<div align="center">
<img src="https://raw.githubusercontent.com/reshalfahsi/covid19ct3d/master/assets/thorax.gif" width="400">

An example of COVID-19 infected lung 3D CT image.
</div>


## <div align="center">Install</div>

<details closed>
<summary>Install covid19ct3d using pip `(for Python >=3.6)`</summary>

```console
git clone https://github.com/reshalfahsi/covid19ct3d
cd covid19ct3d
pip install .
```

</details>


## <div align="center">Use from Python</div>


<details open>
<summary>Training</summary>

```python
import covid19ct3d
  
# set training parameters
epoch = 30  # epoch number
lr = 1e-2  # learning rate
augmentation = True # choose to do data augmentation
batchsize = 2 # training batch size
trainsize = "128 128 64" # training dataset size
train_path = "./dataset" # path to train dataset
train_save = "./model" # path to saved model
step_lr = 2 # set the epoch to drop learning rate

# perform training
covid19ct3d.train(epoch=epoch,
                  lr=lr,
                  augmentation=augmentation,
                  batchsize=batchsize,
                  trainsize=trainsize,
                  train_path=train_path,
                  train_save=train_save,
                  step_lr=step_lr
                  )


```

</details>

<details closed>
<summary>Prediction</summary>

```python
import covid19ct3d
  
# set prediction parameters
predict_size = "128 128 64" # predict size
pth_path = './model/COVID19CT3D.pth' # path to the trained model
data_path = './dataset/CT-0/study_0100.nii.gz' # path to the dataset

# perform prediction
covid19ct3d.predict(predict_size=predict_size,
                    pth_path=pth_path,
                    data_path=data_path
                    )


```

</details>

<details closed>
<summary>Model Information</summary>

```python
import covid19ct3d
  
covid19ct3d.info()

```

</details>


## <div align="center">Use from CLI</div>


<details open>
<summary>Training</summary>

```bash
covid19ct3d train \
--epoch 30 \
--lr 1e-2 \
--batchsize 2 \
--trainsize "128 128 64" \
--train_path "./dataset" \
--train_save "./model" \
--augmentation


```

</details>

<details closed>
<summary>Prediction</summary>

```bash
covid19ct3d predict \
--predict_size "128 128 64" \
--pth_path "./model/COVID19CT3D.pth" \
--data_path "./dataset/CT-23/study_0982.nii.gz"


```

</details>

<details closed>
<summary>Model Information</summary>

```bash
covid19ct3d info

```

</details>


## <div align="center">Credits</div>

<ul>
  <li><a href="https://github.com/hasibzunair/3D-image-classification-tutorial">A tutorial notebook on 3D image classification</a></li>
</ul>
