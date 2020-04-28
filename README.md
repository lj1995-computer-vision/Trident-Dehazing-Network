# Trident Dehazing Network
NTIRE 2020 NonHomogeneous Dehazing Challenge (CVPR Workshop 2020)  **1st** Solution.

### Environment:

- Ubuntu16.04
- Python3.6
- NVIDIA GPU+CUDA8

### Dependencies:

- pretrainedmodels==0.7.4
- torchvision==0.2.1
- torch==0.4.1
- tqdm

### Test

Compile the DCN module fisrt. If your environment is the same as ours, compile was done. If your pytorch version is >=1.0.0, use DCNv2_pytorch1.

Check the hazy images path (test.py line 14), the model path (test.py line 13) and the output path (test.py line 15)

```
python test.py
```

### Pretrained model

https://pan.baidu.com/s/1l0-hOnIAAbFzmauUmFaRjw  password: 22so

https://drive.google.com/file/d/1LcSsCWGLkjmq5o08yhMbSU6DjCGugmRw

### Another solution (**2nd** Solution, Knowledge Transfer Dehazing Network) of our team: 

https://github.com/GlassyWu/KTDN
