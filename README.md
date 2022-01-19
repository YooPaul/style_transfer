# PyTorch Implementation of Perceptual Loss for Image Style Transfer
<!---
[![Open NeRF in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YooPaul/NeRF/blob/master/NeRF.ipynb)<br>
-->

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Content Target</th>
<th valign="bottom">Style Target</th>
<th valign="bottom">Model Output</th>
<!-- TABLE BODY -->
<tr>
<td align="center"><img src="imgs/000000000139.jpg" width="250"/></td>
<td align="center"><img src="imgs/starry_night.jpeg" width="250"/></td>
<td align="center"><img src="imgs/stylized2.png" width="250"/></td>
</tr>
</tbody></table>

## Dataset
MS-COCO dataset was used to train the model. In addition to the COCO dataset, pick a style target image you would like
to transfer the style from.

The dataset can be downloaded by running the following commands
```
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
```

## Test Model

Run
```
python3 test.py model image
```
where ```model``` is a path to your pre-trained model.pt checkpoint and ```image``` being the path to an image.
Output will be saved in the current directory by default.

## References

[1] Johnson, Justin, Alexandre Alahi, and Li Fei-Fei. "Perceptual losses for real-time style transfer and super-resolution." In European conference on computer vision, pp. 694-711. Springer, Cham, 2016.



