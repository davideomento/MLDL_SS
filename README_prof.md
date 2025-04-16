## Model initialization

In the following link you can find the pretrained weights for DeepLab.

**DeepLab petrained weights**: https://drive.google.com/file/d/1ZX0UCXvJwqd2uBGCX7LI2n-DfMg3t74v/view?usp=sharing


## Datasets

To download the dataset use the following download links.

**Cityscapes**: https://drive.google.com/file/d/1Qb4UrNsjvlU-wEsR9d7rckB0YS_LXgb2/view?usp=sharing

**GTA5**: https://drive.google.com/file/d/1xYxlcMR2WFCpayNrW2-Rb7N-950vvl23/view?usp=sharing

## GTA5: label color mapping

Plese refer to this link to convert GTA5 labels in the same format of Cityscapes: https://github.com/sarrrrry/PyTorchDL_GTA5/blob/master/pytorchdl_gta5/labels.py

## FLOPs and parameters

First install fvcore with this command:
```bash
!pip install -U fvcore
```

To calculate the FLOPs and number of parameters please use this code:
```python
from fvcore.nn import FlopCountAnalysis, flop_count_table

# -----------------------------
# Initizialize your model here
# -----------------------------

height = ...
width = ...
image = torch.zeros((3, height, width))

flops = FlopCountAnalysis(model, image)
print(flop_count_table(flops))
```
Reference: https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md

## Latency and FPS

Please refer to this pseudo-code for latency and FPS calculation.

> $\texttt{image} \gets \texttt{random(3, height, width)}$\
$\texttt{iterations} \gets 1000$\
$\texttt{latency} \gets \texttt{[]}$\
$\texttt{FPS} \gets \texttt{[]}$ \
repeat $\texttt{iterations}$ times \
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\texttt{start = time.time()}$\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\texttt{output = model(image)}$\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\texttt{end = time.time()}$\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\texttt{latency}_i \texttt{ = end - start} $\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\texttt{latency.append(latency}_i \texttt{}) $\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\texttt{FPS}_i = \frac{\texttt{1}}{\texttt{latency}_i}$\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\texttt{FPS.append(FPS}_i \texttt{})$    
end      
> $\texttt{meanLatency}  \gets \texttt{mean(latency)*1000}$\
$\texttt{stdLatency} \gets \texttt{std(latency)*1000}$\
$\texttt{meanFPS} \gets \texttt{mean(FPS)}$\
$\texttt{stdFPS} \gets \texttt{std(FPS)}$
