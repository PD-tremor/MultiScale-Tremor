# MultiScale-Tremor
Official implementation of "Multi-Scale Motion Representation Learning for Video-based Hand Tremor Assessment"
## Requirements

The code is built with the following libraries:

* PyTorch
* Torchvision
* TensorboardX
* NumPy
* Matplotlib
* scikit-learn
* Pillow
* Python 3.x
---

## Training

To train the model(s) described in the paper, run:

```bash

python main.py tremor Flow \
    --arch vit \
    --num_segments 18 \
    --epochs 30 \
    -b 32 \
    -j 4 \
    --lr 0.0001 \
    --lr_type cos \
    --dropout 0.5 \
    --warmup-epochs 5 \
    --min-lr 1e-6 \
    --eval-freq 1 \
    --clip-gradient 0.5 \
    --npb
````
The results of the training are saved as a `.pth` file format.

---

## Testing

For example, to test the trained model on PD tremor videos, run:

```bash
python test_models.py tremor \
    --weights <pth_path> \
    --test_segments 18 \
    --test_list <txt_path> \
    --batch_size 32 -j 2 \
    --test_crops 1 \
    --modality Flow

```

