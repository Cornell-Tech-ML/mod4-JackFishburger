# MiniTorch Module 4

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module4.html

This module requires `fast_ops.py`, `cuda_ops.py`, `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 3.


Additionally you will need to install and download the MNist library.

(On Mac, this may require installing the `wget` command)

```
pip install python-mnist
mnist_get_data.sh
```


* Tests:

```
python run_tests.py
```

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py minitorch/tensor_ops.py minitorch/fast_ops.py minitorch/cuda_ops.py project/parallel_check.py tests/test_tensor_general.py

## Task 4.5

### Links to Training Logs

- [Sentiment Analysis Logs (run_sentiment_log.txt)](run_sentiment_log.txt)
- [Digit Classification Logs (run_mnist_multiclass_log.txt)](run_mnist_multiclass_log.txt)

### RAW TRAINING LOGS:
Traning Log From run_sentiment.py:
Epoch 1, loss 31.525269850720562, train accuracy: 50.00%
Validation accuracy: 48.00%
Best Valid accuracy: 48.00%
Epoch 2, loss 31.29667086335143, train accuracy: 49.56%
Validation accuracy: 57.00%
Best Valid accuracy: 57.00%
Epoch 3, loss 31.146369022584256, train accuracy: 50.44%
Validation accuracy: 50.00%
Best Valid accuracy: 57.00%
Epoch 4, loss 30.918722079006674, train accuracy: 53.11%
Validation accuracy: 54.00%
Best Valid accuracy: 57.00%
Epoch 5, loss 30.80945680458696, train accuracy: 58.44%
Validation accuracy: 66.00%
Best Valid accuracy: 66.00%
Epoch 6, loss 30.59799906844649, train accuracy: 60.00%
Validation accuracy: 57.00%
Best Valid accuracy: 66.00%
Epoch 7, loss 30.31452117228922, train accuracy: 60.44%
Validation accuracy: 62.00%
Best Valid accuracy: 66.00%
Epoch 8, loss 30.133532809065006, train accuracy: 58.00%
Validation accuracy: 70.00%
Best Valid accuracy: 70.00%
Epoch 9, loss 29.891828723057422, train accuracy: 63.11%
Validation accuracy: 56.00%
Best Valid accuracy: 70.00%
Epoch 10, loss 29.53863261655168, train accuracy: 64.00%
Validation accuracy: 71.00%
Best Valid accuracy: 71.00%
Epoch 11, loss 29.0051747943035, train accuracy: 68.44%
Validation accuracy: 64.00%
Best Valid accuracy: 71.00%
Epoch 12, loss 28.998465354152977, train accuracy: 67.33%
Validation accuracy: 61.00%
Best Valid accuracy: 71.00%
Epoch 13, loss 28.244355524413727, train accuracy: 70.67%
Validation accuracy: 58.00%
Best Valid accuracy: 71.00%
Epoch 14, loss 28.0480378184645, train accuracy: 69.33%
Validation accuracy: 72.00%
Best Valid accuracy: 72.00%
Epoch 15, loss 27.19768249822752, train accuracy: 73.11%
Validation accuracy: 72.00%
Best Valid accuracy: 72.00%
Epoch 16, loss 26.692592466127607, train accuracy: 72.44%
Validation accuracy: 75.00%
Best Valid accuracy: 75.00%
Epoch 17, loss 26.040762595072312, train accuracy: 74.44%
Validation accuracy: 72.00%
Best Valid accuracy: 75.00%
Epoch 18, loss 25.8794418710864, train accuracy: 73.11%
Validation accuracy: 71.00%
Best Valid accuracy: 75.00%
Epoch 19, loss 24.928092957320814, train accuracy: 74.44%
Validation accuracy: 76.00%
Best Valid accuracy: 76.00%
Epoch 20, loss 24.220319097751542, train accuracy: 76.00%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 21, loss 23.588465293575812, train accuracy: 75.78%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 22, loss 23.110086302816555, train accuracy: 75.33%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 23, loss 22.174016628340365, train accuracy: 76.67%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 24, loss 21.458996568834923, train accuracy: 77.33%
Validation accuracy: 67.00%
Best Valid accuracy: 76.00%
Epoch 25, loss 21.138225340043444, train accuracy: 78.89%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 26, loss 20.89457667539737, train accuracy: 80.89%
Validation accuracy: 68.00%
Best Valid accuracy: 76.00%
Epoch 27, loss 20.324225693956326, train accuracy: 78.00%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 28, loss 20.0654353167021, train accuracy: 77.33%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 29, loss 18.683500885792427, train accuracy: 81.78%
Validation accuracy: 75.00%
Best Valid accuracy: 76.00%
Epoch 30, loss 18.94023739149321, train accuracy: 78.44%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 31, loss 17.919485087193003, train accuracy: 82.44%
Validation accuracy: 69.00%
Best Valid accuracy: 76.00%
Epoch 32, loss 16.827253411107122, train accuracy: 82.67%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 33, loss 17.362626418184732, train accuracy: 84.22%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 34, loss 16.772870413791097, train accuracy: 82.67%
Validation accuracy: 75.00%
Best Valid accuracy: 76.00%
Epoch 35, loss 16.066476220368436, train accuracy: 85.11%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 36, loss 16.110967324621097, train accuracy: 84.67%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 37, loss 16.374662121939203, train accuracy: 82.89%
Validation accuracy: 68.00%
Best Valid accuracy: 76.00%
Epoch 38, loss 15.244840696500464, train accuracy: 84.00%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 39, loss 14.753064958912063, train accuracy: 86.67%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 40, loss 14.327225221851075, train accuracy: 86.44%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 41, loss 14.326757459292986, train accuracy: 83.11%
Validation accuracy: 67.00%
Best Valid accuracy: 76.00%
Epoch 42, loss 13.47045831615443, train accuracy: 85.56%
Validation accuracy: 75.00%
Best Valid accuracy: 76.00%
Epoch 43, loss 13.073983989952916, train accuracy: 87.33%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 44, loss 14.062923549856732, train accuracy: 83.56%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 45, loss 12.859867677025546, train accuracy: 86.22%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 46, loss 13.012359422670658, train accuracy: 85.33%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 47, loss 12.922291047809203, train accuracy: 86.22%
Validation accuracy: 65.00%
Best Valid accuracy: 76.00%
Epoch 48, loss 12.728889142656275, train accuracy: 86.89%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 49, loss 12.637620668693744, train accuracy: 86.00%
Validation accuracy: 67.00%
Best Valid accuracy: 76.00%
Epoch 50, loss 12.533376772294115, train accuracy: 85.11%
Validation accuracy: 66.00%
Best Valid accuracy: 76.00%
Epoch 51, loss 12.332219566301072, train accuracy: 85.78%
Validation accuracy: 68.00%
Best Valid accuracy: 76.00%
Epoch 52, loss 12.619389071816983, train accuracy: 84.00%
Validation accuracy: 65.00%
Best Valid accuracy: 76.00%
Epoch 53, loss 12.372025934051678, train accuracy: 84.67%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 54, loss 11.246712974264346, train accuracy: 85.11%
Validation accuracy: 69.00%
Best Valid accuracy: 76.00%
Epoch 55, loss 11.61079039142934, train accuracy: 86.67%
Validation accuracy: 66.00%
Best Valid accuracy: 76.00%
Epoch 56, loss 12.067162769472626, train accuracy: 86.22%
Validation accuracy: 65.00%
Best Valid accuracy: 76.00%
Epoch 57, loss 12.140814592480693, train accuracy: 85.56%
Validation accuracy: 65.00%
Best Valid accuracy: 76.00%
Epoch 58, loss 11.863649024124474, train accuracy: 85.56%
Validation accuracy: 65.00%
Best Valid accuracy: 76.00%
Epoch 59, loss 10.625475905444635, train accuracy: 85.78%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 60, loss 11.496793475847086, train accuracy: 84.22%
Validation accuracy: 68.00%
Best Valid accuracy: 76.00%
Epoch 61, loss 10.378498254987996, train accuracy: 87.11%
Validation accuracy: 69.00%
Best Valid accuracy: 76.00%
Epoch 62, loss 10.810794737631442, train accuracy: 87.11%
Validation accuracy: 66.00%
Best Valid accuracy: 76.00%
Epoch 63, loss 10.012707539624794, train accuracy: 86.89%
Validation accuracy: 68.00%
Best Valid accuracy: 76.00%
Epoch 64, loss 10.716122033834523, train accuracy: 86.00%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 65, loss 10.688244151002014, train accuracy: 84.44%
Validation accuracy: 66.00%
Best Valid accuracy: 76.00%
Epoch 66, loss 10.241536700433434, train accuracy: 86.44%
Validation accuracy: 67.00%
Best Valid accuracy: 76.00%
Epoch 67, loss 10.896106020218562, train accuracy: 85.33%
Validation accuracy: 69.00%
Best Valid accuracy: 76.00%
Epoch 68, loss 11.329766605604515, train accuracy: 84.00%
Validation accuracy: 67.00%
Best Valid accuracy: 76.00%
Epoch 69, loss 10.24858442445815, train accuracy: 88.44%
Validation accuracy: 68.00%
Best Valid accuracy: 76.00%
Epoch 70, loss 10.004775836553542, train accuracy: 87.56%
Validation accuracy: 66.00%
Best Valid accuracy: 76.00%
Epoch 71, loss 10.919951354086617, train accuracy: 83.78%
Validation accuracy: 68.00%
Best Valid accuracy: 76.00%
Epoch 72, loss 10.422084440585865, train accuracy: 86.89%
Validation accuracy: 67.00%
Best Valid accuracy: 76.00%
Epoch 73, loss 9.859525018433299, train accuracy: 87.11%
Validation accuracy: 67.00%
Best Valid accuracy: 76.00%
Epoch 74, loss 10.52167051865009, train accuracy: 86.44%
Validation accuracy: 68.00%
Best Valid accuracy: 76.00%
Epoch 75, loss 8.93856710005833, train accuracy: 87.56%
Validation accuracy: 69.00%
Best Valid accuracy: 76.00%
Epoch 76, loss 10.326346007042144, train accuracy: 84.89%
Validation accuracy: 68.00%
Best Valid accuracy: 76.00%
Epoch 77, loss 10.259186552730027, train accuracy: 85.56%
Validation accuracy: 67.00%
Best Valid accuracy: 76.00%
Epoch 78, loss 9.428452809695793, train accuracy: 87.11%
Validation accuracy: 68.00%
Best Valid accuracy: 76.00%
Epoch 79, loss 8.63957413381683, train accuracy: 89.11%
Validation accuracy: 68.00%
Best Valid accuracy: 76.00%
Epoch 80, loss 9.17826904008398, train accuracy: 85.78%
Validation accuracy: 69.00%
Best Valid accuracy: 76.00%
Epoch 81, loss 9.210334226721958, train accuracy: 87.56%
Validation accuracy: 69.00%
Best Valid accuracy: 76.00%
Epoch 82, loss 10.082738283514301, train accuracy: 83.78%
Validation accuracy: 68.00%
Best Valid accuracy: 76.00%
Epoch 83, loss 9.658020116095866, train accuracy: 88.00%
Validation accuracy: 67.00%
Best Valid accuracy: 76.00%
Epoch 84, loss 9.95781122615198, train accuracy: 85.11%
Validation accuracy: 66.00%
Best Valid accuracy: 76.00%
Epoch 85, loss 9.719262108275936, train accuracy: 84.00%
Validation accuracy: 75.00%
Best Valid accuracy: 76.00%
Epoch 86, loss 9.173324923315832, train accuracy: 86.89%
Validation accuracy: 67.00%
Best Valid accuracy: 76.00%
Epoch 87, loss 9.032074359717855, train accuracy: 87.33%
Validation accuracy: 67.00%
Best Valid accuracy: 76.00%
Epoch 88, loss 9.748012148987858, train accuracy: 86.22%
Validation accuracy: 67.00%
Best Valid accuracy: 76.00%
Epoch 89, loss 8.126303103082781, train accuracy: 88.67%
Validation accuracy: 69.00%
Best Valid accuracy: 76.00%
Epoch 90, loss 9.020805737097243, train accuracy: 87.11%
Validation accuracy: 67.00%
Best Valid accuracy: 76.00%
Epoch 91, loss 9.593493560839846, train accuracy: 86.00%
Validation accuracy: 67.00%
Best Valid accuracy: 76.00%
Epoch 92, loss 8.241866878499515, train accuracy: 88.67%
Validation accuracy: 69.00%
Best Valid accuracy: 76.00%
Epoch 93, loss 7.70034998411028, train accuracy: 89.11%
Validation accuracy: 69.00%
Best Valid accuracy: 76.00%
Epoch 94, loss 9.897429774557862, train accuracy: 86.67%
Validation accuracy: 68.00%
Best Valid accuracy: 76.00%
Epoch 95, loss 8.483636879415394, train accuracy: 88.67%
Validation accuracy: 68.00%
Best Valid accuracy: 76.00%
Epoch 96, loss 8.937757142791037, train accuracy: 87.56%
Validation accuracy: 69.00%
Best Valid accuracy: 76.00%
Epoch 97, loss 9.898090592190869, train accuracy: 86.00%
Validation accuracy: 66.00%
Best Valid accuracy: 76.00%
Epoch 98, loss 8.393036716722044, train accuracy: 87.33%
Validation accuracy: 69.00%
Best Valid accuracy: 76.00%
Epoch 99, loss 9.313492813766777, train accuracy: 86.00%
Validation accuracy: 66.00%
Best Valid accuracy: 76.00%
Epoch 100, loss 8.383080504366342, train accuracy: 88.22%
Validation accuracy: 68.00%
Best Valid accuracy: 76.00%
Epoch 101, loss 9.85398962223512, train accuracy: 86.22%
Validation accuracy: 66.00%
Best Valid accuracy: 76.00%
Epoch 102, loss 8.695020133633776, train accuracy: 86.44%
Validation accuracy: 68.00%
Best Valid accuracy: 76.00%
Epoch 103, loss 9.656749527593178, train accuracy: 85.56%
Validation accuracy: 68.00%
Best Valid accuracy: 76.00%
Epoch 104, loss 8.800678194499113, train accuracy: 86.00%
Validation accuracy: 67.00%
Best Valid accuracy: 76.00%
Epoch 105, loss 9.02568476927611, train accuracy: 86.44%
Validation accuracy: 67.00%
Best Valid accuracy: 76.00%
Epoch 106, loss 8.660997232138186, train accuracy: 86.89%
Validation accuracy: 68.00%
Best Valid accuracy: 76.00%
Epoch 107, loss 8.631422138456763, train accuracy: 86.89%
Validation accuracy: 68.00%
Best Valid accuracy: 76.00%
Epoch 108, loss 8.237924591583425, train accuracy: 90.00%
Validation accuracy: 68.00%
Best Valid accuracy: 76.00%
Epoch 109, loss 8.136290721647974, train accuracy: 88.22%
Validation accuracy: 67.00%
Best Valid accuracy: 76.00%
Epoch 110, loss 9.502117970164049, train accuracy: 85.11%
Validation accuracy: 68.00%
Best Valid accuracy: 76.00%
Epoch 111, loss 9.236100029384884, train accuracy: 85.56%
Validation accuracy: 66.00%
Best Valid accuracy: 76.00%
Epoch 112, loss 8.926645127811183, train accuracy: 86.00%
Validation accuracy: 67.00%
Best Valid accuracy: 76.00%
Epoch 113, loss 9.36729573259794, train accuracy: 84.67%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 114, loss 9.187683561436344, train accuracy: 86.22%
Validation accuracy: 67.00%
Best Valid accuracy: 76.00%
Epoch 115, loss 7.910988530153885, train accuracy: 88.67%
Validation accuracy: 68.00%
Best Valid accuracy: 76.00%
Epoch 116, loss 9.085305474938703, train accuracy: 85.11%
Validation accuracy: 68.00%
Best Valid accuracy: 76.00%
Epoch 117, loss 7.854621069662906, train accuracy: 88.89%
Validation accuracy: 67.00%
Best Valid accuracy: 76.00%
Epoch 118, loss 8.726494981279227, train accuracy: 87.11%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 119, loss 8.792438879375798, train accuracy: 85.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 120, loss 8.726822613072034, train accuracy: 86.00%
Validation accuracy: 67.00%
Best Valid accuracy: 76.00%
Epoch 121, loss 8.9876415375874, train accuracy: 86.67%
Validation accuracy: 68.00%
Best Valid accuracy: 76.00%
Epoch 122, loss 8.610002574385149, train accuracy: 86.22%
Validation accuracy: 68.00%
Best Valid accuracy: 76.00%
Epoch 123, loss 9.203885531099967, train accuracy: 86.00%
Validation accuracy: 66.00%
Best Valid accuracy: 76.00%
Epoch 124, loss 8.61117240611412, train accuracy: 87.11%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 125, loss 8.171671281389843, train accuracy: 88.89%
Validation accuracy: 67.00%
Best Valid accuracy: 76.00%
Epoch 126, loss 9.552238084782182, train accuracy: 84.89%
Validation accuracy: 69.00%
Best Valid accuracy: 76.00%
Epoch 127, loss 8.628016838513469, train accuracy: 87.56%
Validation accuracy: 66.00%
Best Valid accuracy: 76.00%
Epoch 128, loss 8.748390921939055, train accuracy: 86.22%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 129, loss 9.280771518226226, train accuracy: 86.44%
Validation accuracy: 66.00%
Best Valid accuracy: 76.00%
Epoch 130, loss 8.30101172397642, train accuracy: 86.67%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 131, loss 9.059612373200249, train accuracy: 86.22%
Validation accuracy: 68.00%
Best Valid accuracy: 76.00%
Epoch 132, loss 9.250474076768015, train accuracy: 85.78%
Validation accuracy: 66.00%
Best Valid accuracy: 76.00%
Epoch 133, loss 7.406354922363853, train accuracy: 88.67%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 134, loss 7.6241152657839555, train accuracy: 90.44%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 135, loss 7.7703145767958395, train accuracy: 91.56%
Validation accuracy: 66.00%
Best Valid accuracy: 76.00%
Epoch 136, loss 9.52489819448545, train accuracy: 88.44%
Validation accuracy: 66.00%
Best Valid accuracy: 76.00%
Epoch 137, loss 8.522853179033552, train accuracy: 86.44%
Validation accuracy: 66.00%
Best Valid accuracy: 76.00%
Epoch 138, loss 7.731751102104093, train accuracy: 87.11%
Validation accuracy: 68.00%
Best Valid accuracy: 76.00%
Epoch 139, loss 9.021180803398419, train accuracy: 85.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 140, loss 8.602190841811375, train accuracy: 87.56%
Validation accuracy: 66.00%
Best Valid accuracy: 76.00%
Epoch 141, loss 8.822300642440487, train accuracy: 86.89%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 142, loss 9.090710535807931, train accuracy: 86.00%
Validation accuracy: 66.00%
Best Valid accuracy: 76.00%
Epoch 143, loss 7.85810192763061, train accuracy: 88.22%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 144, loss 8.326582606916721, train accuracy: 86.00%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 145, loss 7.70742873557066, train accuracy: 89.78%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 146, loss 8.914687856303033, train accuracy: 87.56%
Validation accuracy: 68.00%
Best Valid accuracy: 76.00%
Epoch 147, loss 8.882557343183874, train accuracy: 88.67%
Validation accuracy: 66.00%
Best Valid accuracy: 76.00%
Epoch 148, loss 8.566472224051484, train accuracy: 87.11%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 149, loss 8.343130291264346, train accuracy: 88.00%
Validation accuracy: 66.00%
Best Valid accuracy: 76.00%
Epoch 150, loss 8.779065014002141, train accuracy: 85.33%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 151, loss 9.29285818367465, train accuracy: 86.89%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 152, loss 8.305001948048027, train accuracy: 85.11%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 153, loss 7.94973229342513, train accuracy: 87.78%
Validation accuracy: 66.00%
Best Valid accuracy: 76.00%
Epoch 154, loss 8.226966289658298, train accuracy: 86.00%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 155, loss 8.366437543954639, train accuracy: 85.56%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 156, loss 8.216417986171212, train accuracy: 88.00%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 157, loss 9.086634491998208, train accuracy: 84.67%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 158, loss 7.634496363091751, train accuracy: 87.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 159, loss 9.678938536471332, train accuracy: 84.89%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 160, loss 8.718462385691957, train accuracy: 86.00%
Validation accuracy: 69.00%
Best Valid accuracy: 76.00%
Epoch 161, loss 8.23250744689503, train accuracy: 86.22%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 162, loss 8.508845813199478, train accuracy: 86.89%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 163, loss 8.090063087571707, train accuracy: 86.22%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 164, loss 7.614916252769144, train accuracy: 88.67%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 165, loss 8.707610021520987, train accuracy: 86.00%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 166, loss 9.08931372534713, train accuracy: 85.56%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 167, loss 7.443464273652367, train accuracy: 87.56%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 168, loss 8.536195998936448, train accuracy: 87.33%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 169, loss 8.133141238382093, train accuracy: 87.11%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 170, loss 9.42267703683204, train accuracy: 85.11%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 171, loss 8.443915946889, train accuracy: 87.56%
Validation accuracy: 67.00%
Best Valid accuracy: 76.00%
Epoch 172, loss 8.449379983926688, train accuracy: 86.67%
Validation accuracy: 66.00%
Best Valid accuracy: 76.00%
Epoch 173, loss 8.370416577485088, train accuracy: 87.56%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 174, loss 8.56761521071156, train accuracy: 84.89%
Validation accuracy: 69.00%
Best Valid accuracy: 76.00%
Epoch 175, loss 8.115328448856118, train accuracy: 87.33%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 176, loss 8.73216157400534, train accuracy: 85.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 177, loss 8.357785089489914, train accuracy: 85.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 178, loss 8.408917651251103, train accuracy: 88.00%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 179, loss 7.8716766877794875, train accuracy: 88.22%
Validation accuracy: 69.00%
Best Valid accuracy: 76.00%
Epoch 180, loss 9.008356776956663, train accuracy: 86.22%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 181, loss 7.392858145957847, train accuracy: 87.33%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 182, loss 8.276066411156375, train accuracy: 86.22%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 183, loss 7.755250440295735, train accuracy: 86.44%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 184, loss 8.225197414535362, train accuracy: 87.56%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 185, loss 6.8090454413957655, train accuracy: 90.22%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 186, loss 8.120676950001835, train accuracy: 86.00%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 187, loss 8.994159391369982, train accuracy: 84.44%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 188, loss 7.8755753677519955, train accuracy: 87.11%
Validation accuracy: 69.00%
Best Valid accuracy: 76.00%
Epoch 189, loss 8.085637640133452, train accuracy: 87.56%
Validation accuracy: 68.00%
Best Valid accuracy: 76.00%
Epoch 190, loss 8.113984318425286, train accuracy: 88.00%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 191, loss 9.005452412636007, train accuracy: 86.67%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 192, loss 8.983288164266645, train accuracy: 86.00%
Validation accuracy: 69.00%
Best Valid accuracy: 76.00%
Epoch 193, loss 8.181336025865114, train accuracy: 86.22%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 194, loss 8.208815692117447, train accuracy: 87.11%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 195, loss 9.225824988992736, train accuracy: 86.67%
Validation accuracy: 68.00%
Best Valid accuracy: 76.00%
Epoch 196, loss 8.660773753123388, train accuracy: 86.00%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 197, loss 7.956758310071669, train accuracy: 88.00%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 198, loss 8.604325186491959, train accuracy: 88.89%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 199, loss 8.264602121812414, train accuracy: 83.78%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 200, loss 7.616480150628547, train accuracy: 87.11%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 201, loss 8.015315372725238, train accuracy: 88.22%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 202, loss 7.883151887722117, train accuracy: 87.11%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 203, loss 8.09798553944069, train accuracy: 88.00%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 204, loss 8.989859725324049, train accuracy: 86.00%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 205, loss 8.390686809220597, train accuracy: 86.67%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 206, loss 7.964152853479189, train accuracy: 87.11%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 207, loss 8.825261351246668, train accuracy: 85.11%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 208, loss 7.083033925917781, train accuracy: 88.44%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 209, loss 8.21647004519276, train accuracy: 87.11%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 210, loss 7.519739526477831, train accuracy: 88.89%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 211, loss 8.975976250835155, train accuracy: 84.89%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 212, loss 7.16200088331482, train accuracy: 88.89%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 213, loss 7.993737703318842, train accuracy: 87.33%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 214, loss 7.444784565203474, train accuracy: 88.22%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 215, loss 8.831698507177435, train accuracy: 86.22%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 216, loss 8.829642311908792, train accuracy: 85.11%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 217, loss 7.439463652871382, train accuracy: 87.11%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 218, loss 7.895036544558739, train accuracy: 88.22%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 219, loss 8.693037765751379, train accuracy: 84.67%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 220, loss 8.330372391098111, train accuracy: 87.11%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 221, loss 8.220501898315781, train accuracy: 86.22%
Validation accuracy: 68.00%
Best Valid accuracy: 76.00%
Epoch 222, loss 8.514225837809189, train accuracy: 87.11%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 223, loss 8.570432695954086, train accuracy: 84.67%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 224, loss 8.079296303755331, train accuracy: 86.89%
Validation accuracy: 68.00%
Best Valid accuracy: 76.00%
Epoch 225, loss 7.990187968237928, train accuracy: 89.11%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 226, loss 8.070200939378653, train accuracy: 86.22%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 227, loss 7.246439849329673, train accuracy: 88.89%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 228, loss 9.127921633393214, train accuracy: 85.78%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 229, loss 8.134597409178712, train accuracy: 86.22%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 230, loss 8.122304139278429, train accuracy: 85.33%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 231, loss 9.130855712854641, train accuracy: 84.89%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 232, loss 7.212921095313189, train accuracy: 88.22%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 233, loss 7.983898129196848, train accuracy: 87.33%
Validation accuracy: 68.00%
Best Valid accuracy: 76.00%
Epoch 234, loss 7.701166170885796, train accuracy: 86.00%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 235, loss 8.181501670197193, train accuracy: 86.89%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 236, loss 8.385006080589683, train accuracy: 86.22%
Validation accuracy: 68.00%
Best Valid accuracy: 76.00%
Epoch 237, loss 8.17808198130804, train accuracy: 86.89%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 238, loss 8.658417963821964, train accuracy: 86.89%
Validation accuracy: 69.00%
Best Valid accuracy: 76.00%
Epoch 239, loss 8.901967949390261, train accuracy: 84.00%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 240, loss 7.801286738398882, train accuracy: 86.22%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 241, loss 8.17253588751373, train accuracy: 87.33%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 242, loss 8.577723005096917, train accuracy: 85.33%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 243, loss 7.939723220084221, train accuracy: 85.33%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 244, loss 8.258984898465496, train accuracy: 87.78%
Validation accuracy: 68.00%
Best Valid accuracy: 76.00%
Epoch 245, loss 7.394518715328796, train accuracy: 85.78%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 246, loss 8.675606444092173, train accuracy: 84.89%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 247, loss 7.819642912412293, train accuracy: 86.44%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 248, loss 8.406307168413846, train accuracy: 89.33%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 249, loss 7.8591127831937975, train accuracy: 86.67%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 250, loss 8.503913265650104, train accuracy: 85.33%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%


Traning Log From run_mnist_multiclass.py:
Epoch 6 loss 0.4839910971424419 valid acc 16/16
Epoch 6 loss 0.1266203642072194 valid acc 16/16
Epoch 6 loss 0.961158371515314 valid acc 16/16
Epoch 6 loss 0.9660101006317042 valid acc 16/16
Epoch 6 loss 0.36923372141433924 valid acc 16/16
Epoch 6 loss 1.1102488727453634 valid acc 15/16
Epoch 6 loss 0.35828605538104097 valid acc 16/16
Epoch 6 loss 0.37993230600201316 valid acc 16/16
Epoch 6 loss 1.4413644786420126 valid acc 16/16
Epoch 6 loss 1.0095802723412688 valid acc 16/16
Epoch 6 loss 0.31824076418718744 valid acc 16/16
Epoch 6 loss 0.9862755317958476 valid acc 16/16
Epoch 6 loss 0.9151126955285771 valid acc 16/16
Epoch 6 loss 0.36818800343850144 valid acc 16/16
Epoch 6 loss 0.17036216184228903 valid acc 16/16
Epoch 6 loss 0.3822508136472828 valid acc 16/16
Epoch 6 loss 0.5233497449312985 valid acc 16/16
Epoch 6 loss 0.7870163314299012 valid acc 16/16
Epoch 6 loss 0.33544163867721105 valid acc 16/16
Epoch 6 loss 1.4092517834629086 valid acc 16/16
Epoch 6 loss 0.6220382350997955 valid acc 16/16
Epoch 6 loss 0.5438144359785722 valid acc 15/16
Epoch 6 loss 0.4977089983202098 valid acc 16/16
Epoch 6 loss 0.6355435922714939 valid acc 16/16
Epoch 7 loss 0.0721056631191821 valid acc 16/16
Epoch 7 loss 0.5732041468357363 valid acc 15/16
Epoch 7 loss 1.3581871222896023 valid acc 15/16
Epoch 7 loss 0.7743220985096579 valid acc 15/16
Epoch 7 loss 0.257953155612347 valid acc 15/16
Epoch 7 loss 0.4073452206441817 valid acc 16/16
Epoch 7 loss 0.46007851290860724 valid acc 15/16
Epoch 7 loss 0.7228725713488077 valid acc 16/16
Epoch 7 loss 0.9786524145561639 valid acc 14/16
Epoch 7 loss 0.6222388802600038 valid acc 16/16
Epoch 7 loss 0.49202561490696484 valid acc 16/16
Epoch 7 loss 0.7238006786548519 valid acc 15/16
Epoch 7 loss 1.366262655906171 valid acc 14/16
Epoch 7 loss 0.7731976214021619 valid acc 16/16
Epoch 7 loss 1.0842136246684788 valid acc 15/16
Epoch 7 loss 0.7015338257637461 valid acc 13/16
Epoch 7 loss 1.0037921904905729 valid acc 16/16
Epoch 7 loss 0.43508724105660634 valid acc 16/16
Epoch 7 loss 0.6949468851242511 valid acc 15/16
Epoch 7 loss 0.5023200462331987 valid acc 15/16
Epoch 7 loss 0.7131845176564273 valid acc 15/16
Epoch 7 loss 0.7839547153147055 valid acc 16/16
Epoch 7 loss 0.3337584618251716 valid acc 16/16
Epoch 7 loss 0.651230881753563 valid acc 15/16
Epoch 7 loss 0.6666394739235888 valid acc 15/16
Epoch 7 loss 0.43470070072697986 valid acc 16/16
Epoch 7 loss 0.41418793963849804 valid acc 15/16
Epoch 7 loss 0.2135422931712447 valid acc 14/16
Epoch 7 loss 0.6359158664699538 valid acc 15/16
Epoch 7 loss 0.3223445707966289 valid acc 16/16
Epoch 7 loss 0.5688655578117378 valid acc 16/16
Epoch 7 loss 0.5799908508196472 valid acc 16/16
Epoch 7 loss 0.5284998035751723 valid acc 16/16
Epoch 7 loss 0.6710091427510912 valid acc 16/16
Epoch 7 loss 0.9819984855883626 valid acc 16/16
Epoch 7 loss 0.8622686478598154 valid acc 15/16
Epoch 7 loss 0.7385976829881602 valid acc 15/16
Epoch 7 loss 0.5359448657644142 valid acc 16/16
Epoch 7 loss 0.6564259807827427 valid acc 16/16
Epoch 7 loss 0.8040039046678598 valid acc 15/16
Epoch 7 loss 0.24747619967573142 valid acc 15/16
Epoch 7 loss 0.4301538142500883 valid acc 16/16
Epoch 7 loss 0.2801749156510621 valid acc 15/16
Epoch 7 loss 0.641933059282872 valid acc 15/16
Epoch 7 loss 0.8077894123136687 valid acc 15/16
Epoch 7 loss 0.2516667732541906 valid acc 15/16
Epoch 7 loss 0.560342395640156 valid acc 16/16
Epoch 7 loss 1.4692251967398176 valid acc 16/16
Epoch 7 loss 0.5940886601734677 valid acc 15/16
Epoch 7 loss 0.541244910828365 valid acc 15/16
Epoch 7 loss 0.7748304728595016 valid acc 15/16
Epoch 7 loss 0.4162985076981839 valid acc 15/16
Epoch 7 loss 0.4458757520491381 valid acc 15/16
Epoch 7 loss 0.6141981570699047 valid acc 16/16
Epoch 7 loss 0.2794715509768262 valid acc 16/16
Epoch 7 loss 0.18697349355258186 valid acc 16/16
Epoch 7 loss 0.5687632089336492 valid acc 15/16
Epoch 7 loss 0.4656245450263385 valid acc 15/16
Epoch 7 loss 1.7377955470951085 valid acc 14/16
Epoch 7 loss 0.9700244369039964 valid acc 16/16
Epoch 7 loss 0.5601644829327512 valid acc 15/16
Epoch 7 loss 0.340474470233627 valid acc 16/16
Epoch 7 loss 0.4013259211013612 valid acc 16/16
Epoch 8 loss 0.017696849534666548 valid acc 16/16
Epoch 8 loss 0.27123756810095123 valid acc 16/16
Epoch 8 loss 0.6161317686078 valid acc 16/16
Epoch 8 loss 0.29878973633841266 valid acc 14/16
Epoch 8 loss 0.5366907906659175 valid acc 14/16
Epoch 8 loss 0.359914143699276 valid acc 16/16
Epoch 8 loss 0.35558714381090106 valid acc 15/16
Epoch 8 loss 1.0858990659907126 valid acc 16/16
Epoch 8 loss 0.529978395670859 valid acc 15/16
Epoch 8 loss 0.2060540744426178 valid acc 15/16
Epoch 8 loss 0.8874919585931145 valid acc 16/16
Epoch 8 loss 0.5154297307653697 valid acc 15/16
Epoch 8 loss 1.0614259932195615 valid acc 15/16
Epoch 8 loss 0.9566697716005506 valid acc 15/16
Epoch 8 loss 0.7009892345414616 valid acc 16/16
Epoch 8 loss 0.5243698421647649 valid acc 15/16
Epoch 8 loss 0.8689566805211061 valid acc 15/16
Epoch 8 loss 0.9950721064904472 valid acc 15/16
Epoch 8 loss 0.9350725026357636 valid acc 15/16
Epoch 8 loss 0.18902437432711383 valid acc 16/16
Epoch 8 loss 0.7405846397908322 valid acc 15/16
Epoch 8 loss 0.18811832748525792 valid acc 14/16
Epoch 8 loss 0.33990412635579026 valid acc 16/16
Epoch 8 loss 0.47422100993636823 valid acc 15/16
Epoch 8 loss 0.2861876265992601 valid acc 15/16
Epoch 8 loss 0.8402383732071432 valid acc 15/16
Epoch 8 loss 0.30543430113777686 valid acc 15/16
Epoch 8 loss 0.3819023437873968 valid acc 13/16
Epoch 8 loss 0.25571030160968633 valid acc 14/16
Epoch 8 loss 0.3272871819571548 valid acc 16/16
Epoch 8 loss 0.23564686453668332 valid acc 16/16
Epoch 8 loss 0.7521748595966062 valid acc 16/16
Epoch 8 loss 0.44415400526710797 valid acc 16/16
Epoch 8 loss 0.27153255598946413 valid acc 16/16
Epoch 8 loss 0.7810746201907108 valid acc 16/16
Epoch 8 loss 0.7108562933104418 valid acc 16/16
Epoch 8 loss 0.3581074183548703 valid acc 16/16
Epoch 8 loss 0.3426694035086331 valid acc 16/16
Epoch 8 loss 0.29400181418755456 valid acc 16/16
Epoch 8 loss 0.1437756214161954 valid acc 16/16
Epoch 8 loss 0.1964130428895339 valid acc 16/16
Epoch 8 loss 0.5606888457519164 valid acc 16/16
Epoch 8 loss 0.6951368168760268 valid acc 15/16
Epoch 8 loss 0.647483979310924 valid acc 16/16
Epoch 8 loss 1.0380202588530723 valid acc 16/16
Epoch 8 loss 0.15675026172848172 valid acc 16/16
Epoch 8 loss 0.7077967269777652 valid acc 16/16
Epoch 8 loss 1.7297824249667657 valid acc 15/16
Epoch 8 loss 0.42680312655201064 valid acc 15/16
Epoch 8 loss 0.6789464046924701 valid acc 16/16
Epoch 8 loss 0.762871408027644 valid acc 15/16
Epoch 8 loss 0.36100197500622544 valid acc 16/16
Epoch 8 loss 0.45117161119261945 valid acc 15/16
Epoch 8 loss 0.39190661531061377 valid acc 15/16
Epoch 8 loss 0.3202862744719964 valid acc 15/16
Epoch 8 loss 0.10174403414589023 valid acc 15/16
Epoch 8 loss 0.7258255214835871 valid acc 15/16
Epoch 8 loss 0.5505800369605041 valid acc 15/16
Epoch 8 loss 0.42383015239378574 valid acc 16/16
Epoch 8 loss 0.23877621826704892 valid acc 16/16
Epoch 8 loss 0.6361664973853063 valid acc 16/16
Epoch 8 loss 0.5146081230800117 valid acc 16/16
Epoch 8 loss 0.6699950999005552 valid acc 16/16
Epoch 9 loss 0.054693464630474364 valid acc 16/16
Epoch 9 loss 0.7801638580155265 valid acc 16/16
Epoch 9 loss 0.7661505167761371 valid acc 16/16
Epoch 9 loss 0.2983185555202463 valid acc 16/16
Epoch 9 loss 0.3821839561979966 valid acc 16/16
Epoch 9 loss 0.3253359991180914 valid acc 16/16
Epoch 9 loss 0.8652964124034366 valid acc 16/16
Epoch 9 loss 0.9249706547602327 valid acc 16/16
Epoch 9 loss 0.4606615471613755 valid acc 16/16
Epoch 9 loss 0.4016721870280123 valid acc 16/16
Epoch 9 loss 0.25345297650212734 valid acc 15/16
Epoch 9 loss 0.3950142367377379 valid acc 16/16
Epoch 9 loss 0.4092357380395063 valid acc 16/16
Epoch 9 loss 0.41625968235076716 valid acc 15/16
Epoch 9 loss 0.43498023421536675 valid acc 16/16
Epoch 9 loss 0.6683097102612539 valid acc 16/16
Epoch 9 loss 1.0300020334238056 valid acc 15/16
Epoch 9 loss 0.4765710645199555 valid acc 16/16
Epoch 9 loss 0.275266323331063 valid acc 16/16
Epoch 9 loss 0.427423171279113 valid acc 16/16
Epoch 9 loss 0.49290545310252515 valid acc 15/16
Epoch 9 loss 0.22197495727564265 valid acc 16/16
Epoch 9 loss 0.5261635699092039 valid acc 16/16
Epoch 9 loss 0.45602523718623 valid acc 16/16
Epoch 9 loss 0.21260245740416095 valid acc 15/16
Epoch 9 loss 0.26812783602080437 valid acc 16/16
Epoch 9 loss 0.45754821056451256 valid acc 16/16
Epoch 9 loss 0.242201096107367 valid acc 15/16
Epoch 9 loss 0.5271106617229155 valid acc 16/16
Epoch 9 loss 0.1467079530981019 valid acc 16/16
Epoch 9 loss 0.7919262058838545 valid acc 15/16
Epoch 9 loss 0.6221887225644795 valid acc 16/16
Epoch 9 loss 0.1203785104849791 valid acc 16/16
Epoch 9 loss 0.4799106393423062 valid acc 16/16
Epoch 9 loss 1.0577652301247031 valid acc 16/16
Epoch 9 loss 0.3890647407921667 valid acc 16/16
Epoch 9 loss 0.32376765166065496 valid acc 16/16
Epoch 9 loss 0.4900491823119044 valid acc 16/16
Epoch 9 loss 0.2962595854763146 valid acc 16/16
Epoch 9 loss 0.4605854752550887 valid acc 16/16
Epoch 9 loss 0.21504799818978498 valid acc 16/16
Epoch 9 loss 0.3854903031929371 valid acc 16/16
Epoch 9 loss 0.3591464223801836 valid acc 16/16
Epoch 9 loss 0.3125407869853011 valid acc 16/16
Epoch 9 loss 0.7213564861690543 valid acc 16/16
Epoch 9 loss 0.3834507951081905 valid acc 16/16
Epoch 9 loss 0.2786721924765214 valid acc 16/16
Epoch 9 loss 0.9792708471066992 valid acc 15/16
Epoch 9 loss 0.20853647646071466 valid acc 16/16
Epoch 9 loss 0.08772682659792819 valid acc 16/16
Epoch 9 loss 0.155489294478582 valid acc 16/16
Epoch 9 loss 0.5212895971346276 valid acc 16/16
Epoch 9 loss 0.7627216647906035 valid acc 16/16
Epoch 9 loss 0.34662178479273104 valid acc 16/16
Epoch 9 loss 0.47441225032018525 valid acc 16/16
Epoch 9 loss 0.2385141288446085 valid acc 16/16
Epoch 9 loss 0.5922226373155959 valid acc 16/16
Epoch 9 loss 0.4018110048036616 valid acc 15/16
Epoch 9 loss 0.6474480250121146 valid acc 15/16
Epoch 9 loss 0.40001841205058924 valid acc 16/16
Epoch 9 loss 0.16612068716025197 valid acc 16/16
Epoch 9 loss 0.3398894385154077 valid acc 16/16
Epoch 9 loss 0.4570335819395718 valid acc 16/16
Epoch 10 loss 0.0012628634593458976 valid acc 16/16
Epoch 10 loss 0.3945844928930181 valid acc 15/16
Epoch 10 loss 0.9465152465869704 valid acc 15/16
Epoch 10 loss 0.2966467863407012 valid acc 14/16
Epoch 10 loss 0.145639859264872 valid acc 15/16
Epoch 10 loss 0.38053871524909405 valid acc 16/16
Epoch 10 loss 0.1688150541439306 valid acc 15/16
Epoch 10 loss 0.13281355000913003 valid acc 15/16
Epoch 10 loss 0.34530459623194154 valid acc 16/16
Epoch 10 loss 0.24051779945854065 valid acc 15/16
Epoch 10 loss 0.18266265941291554 valid acc 15/16
Epoch 10 loss 0.277214287235996 valid acc 15/16
Epoch 10 loss 0.7156247192630613 valid acc 16/16
Epoch 10 loss 0.8808421864081056 valid acc 14/16
Epoch 10 loss 0.8010434125382959 valid acc 16/16
Epoch 10 loss 0.4065428502819355 valid acc 16/16
Epoch 10 loss 0.6972877996141296 valid acc 16/16
Epoch 10 loss 0.20054267629396455 valid acc 16/16
Epoch 10 loss 0.22102337192990879 valid acc 16/16
Epoch 10 loss 0.14252295049480296 valid acc 16/16
Epoch 10 loss 0.995757772271519 valid acc 16/16
Epoch 10 loss 0.2341128463948487 valid acc 16/16
Epoch 10 loss 0.3014050240530838 valid acc 16/16
Epoch 10 loss 0.3870448639046864 valid acc 15/16
Epoch 10 loss 0.07412513205810056 valid acc 15/16
Epoch 10 loss 0.20308940807531656 valid acc 15/16
Epoch 10 loss 0.29275030758971693 valid acc 15/16
Epoch 10 loss 0.45727269032809337 valid acc 15/16
Epoch 10 loss 0.5932715100595237 valid acc 15/16
Epoch 10 loss 0.19303489481093117 valid acc 15/16
Epoch 10 loss 0.4310814424791492 valid acc 15/16
Epoch 10 loss 0.34429854961465706 valid acc 16/16
Epoch 10 loss 0.44169564690346286 valid acc 16/16
Epoch 10 loss 0.26983074362159826 valid acc 16/16
Epoch 10 loss 1.3186568211945646 valid acc 16/16
Epoch 10 loss 0.9400531826370192 valid acc 16/16
Epoch 10 loss 0.16727141490630282 valid acc 16/16
Epoch 10 loss 0.16082630182157168 valid acc 16/16
Epoch 10 loss 0.2691142942618976 valid acc 16/16
Epoch 10 loss 0.25865655331792164 valid acc 16/16
Epoch 10 loss 0.10479244795042894 valid acc 16/16
Epoch 10 loss 0.1960963505532468 valid acc 16/16
Epoch 10 loss 0.3027157914328965 valid acc 16/16
Epoch 10 loss 0.20137391498554408 valid acc 16/16
Epoch 10 loss 0.5389829951880407 valid acc 16/16
Epoch 10 loss 0.23502083165026588 valid acc 16/16
Epoch 10 loss 0.23406482976523713 valid acc 16/16
Epoch 10 loss 1.1932251231353204 valid acc 16/16
Epoch 10 loss 0.161519955647215 valid acc 15/16
Epoch 10 loss 0.2948666167145823 valid acc 15/16
Epoch 10 loss 0.6088646564353308 valid acc 16/16
Epoch 10 loss 0.2167790549740544 valid acc 16/16
Epoch 10 loss 0.49187084960974053 valid acc 16/16
Epoch 10 loss 0.11708999922976848 valid acc 16/16
Epoch 10 loss 0.20126815145166027 valid acc 16/16
Epoch 10 loss 0.35132775109564157 valid acc 16/16
Epoch 10 loss 0.2139293885112159 valid acc 16/16
Epoch 10 loss 0.7357145595076453 valid acc 16/16
Epoch 10 loss 1.1922699946412694 valid acc 16/16
Epoch 10 loss 0.33237797790210105 valid acc 16/16
Epoch 10 loss 0.15232648856329029 valid acc 16/16
Epoch 10 loss 0.25966874037951837 valid acc 16/16
Epoch 10 loss 0.3524053786726491 valid acc 15/16
Epoch 11 loss 0.010217292076280304 valid acc 15/16
Epoch 11 loss 0.24957290388804487 valid acc 16/16
Epoch 11 loss 0.45551465274842884 valid acc 16/16
Epoch 11 loss 0.42807213219495055 valid acc 16/16
Epoch 11 loss 0.11684342740556419 valid acc 15/16
Epoch 11 loss 0.5233121298757559 valid acc 15/16
Epoch 11 loss 0.6211468984814683 valid acc 15/16
Epoch 11 loss 0.3645170437523457 valid acc 16/16
Epoch 11 loss 0.3169457611421738 valid acc 16/16
Epoch 11 loss 0.7377943734973491 valid acc 16/16
Epoch 11 loss 0.45891220592607557 valid acc 16/16
Epoch 11 loss 0.4112901256819904 valid acc 16/16
Epoch 11 loss 0.935908815577236 valid acc 16/16
Epoch 11 loss 0.32454235700080786 valid acc 16/16
Epoch 11 loss 0.159934478207379 valid acc 16/16
Epoch 11 loss 0.14296704492481302 valid acc 15/16
Epoch 11 loss 0.34491592244916347 valid acc 16/16
Epoch 11 loss 0.22734227410656277 valid acc 16/16
Epoch 11 loss 0.3814672131083623 valid acc 16/16
Epoch 11 loss 0.4886237679655474 valid acc 14/16
Epoch 11 loss 0.5025650190593872 valid acc 15/16
Epoch 11 loss 0.15097267467025968 valid acc 16/16
Epoch 11 loss 0.14512905835605794 valid acc 16/16
Epoch 11 loss 0.33730446021946564 valid acc 16/16
Epoch 11 loss 0.2786992794537627 valid acc 16/16
Epoch 11 loss 0.44363890204363765 valid acc 16/16
Epoch 11 loss 0.214366760862492 valid acc 16/16
Epoch 11 loss 0.35897977425097766 valid acc 15/16
Epoch 11 loss 0.21517783727617268 valid acc 15/16
Epoch 11 loss 0.41603490249091213 valid acc 15/16
Epoch 11 loss 0.047318990252289306 valid acc 16/16
Epoch 11 loss 0.15277881314153402 valid acc 16/16
Epoch 11 loss 0.08712661479038558 valid acc 16/16
Epoch 11 loss 0.11651525033497354 valid acc 16/16
Epoch 11 loss 0.7623860194238568 valid acc 16/16
Epoch 11 loss 0.35554466543994956 valid acc 16/16
Epoch 11 loss 0.17402449455627222 valid acc 16/16
Epoch 11 loss 0.41366778760312295 valid acc 16/16
Epoch 11 loss 0.4566960438444348 valid acc 16/16
Epoch 11 loss 0.460973593087589 valid acc 16/16
Epoch 11 loss 0.3662974365985422 valid acc 16/16
Epoch 11 loss 0.5300763240808245 valid acc 15/16
Epoch 11 loss 0.4161888995995905 valid acc 16/16
Epoch 11 loss 0.3515875607236903 valid acc 16/16
Epoch 11 loss 0.7123480117064978 valid acc 14/16
Epoch 11 loss 0.4807483176832566 valid acc 16/16
Epoch 11 loss 0.5454873256291088 valid acc 16/16
Epoch 11 loss 0.6319799661668551 valid acc 15/16
Epoch 11 loss 0.09174463609359729 valid acc 16/16
Epoch 11 loss 0.028145561073030012 valid acc 16/16
Epoch 11 loss 0.13637098899159733 valid acc 16/16
Epoch 11 loss 0.5411527558311753 valid acc 16/16
Epoch 11 loss 0.7915897664929814 valid acc 15/16
Epoch 11 loss 0.7210496937313056 valid acc 15/16
Epoch 11 loss 0.4674543772612372 valid acc 15/16
Epoch 11 loss 0.1672075473989202 valid acc 15/16
Epoch 11 loss 0.8186559044068051 valid acc 14/16
Epoch 11 loss 0.10036352134089382 valid acc 15/16
Epoch 11 loss 1.0837342758638793 valid acc 15/16
Epoch 11 loss 0.7703371107617091 valid acc 16/16
Epoch 11 loss 0.3204924911433852 valid acc 15/16
Epoch 11 loss 0.28425583657817455 valid acc 16/16
Epoch 11 loss 0.2870341695194667 valid acc 16/16
Epoch 12 loss 0.000724658431599412 valid acc 16/16
Epoch 12 loss 0.25309683421614915 valid acc 16/16
Epoch 12 loss 0.4872580592099716 valid acc 15/16
Epoch 12 loss 0.19887974144365717 valid acc 16/16
Epoch 12 loss 0.1306091745693389 valid acc 16/16
Epoch 12 loss 0.2595082532032105 valid acc 15/16
Epoch 12 loss 0.3866639848334634 valid acc 14/16
Epoch 12 loss 0.40719189935166505 valid acc 16/16
Epoch 12 loss 0.18131663053102948 valid acc 16/16
Epoch 12 loss 0.3106813554561385 valid acc 15/16
Epoch 12 loss 0.5561561141259435 valid acc 16/16
Epoch 12 loss 0.5823116524190598 valid acc 16/16
Epoch 12 loss 0.6713094798582075 valid acc 16/16
Epoch 12 loss 0.5221374710440523 valid acc 15/16
Epoch 12 loss 0.6438442271567101 valid acc 16/16
Epoch 12 loss 0.27100134325643654 valid acc 15/16
Epoch 12 loss 0.9015150522462095 valid acc 16/16
Epoch 12 loss 0.6110079672343712 valid acc 16/16
Epoch 12 loss 0.3001560359878658 valid acc 15/16
Epoch 12 loss 0.3668784262115334 valid acc 16/16
Epoch 12 loss 0.7230460166645691 valid acc 15/16
Epoch 12 loss 0.4162469602365926 valid acc 15/16
Epoch 12 loss 0.1744664370870994 valid acc 16/16
Epoch 12 loss 0.18657957936651504 valid acc 16/16
Epoch 12 loss 0.27219215679945835 valid acc 15/16
Epoch 12 loss 0.14856819969340207 valid acc 15/16
Epoch 12 loss 0.10936079281650726 valid acc 14/16
Epoch 12 loss 0.29809275554585546 valid acc 14/16
Epoch 12 loss 0.12642671720324272 valid acc 14/16
Epoch 12 loss 0.35544554859684885 valid acc 15/16
Epoch 12 loss 1.1239030085585267 valid acc 16/16
Epoch 12 loss 0.08375500101543476 valid acc 16/16
Epoch 12 loss 0.3264626698872792 valid acc 16/16
Epoch 12 loss 0.2832468862189164 valid acc 16/16
Epoch 12 loss 0.42022288065019503 valid acc 16/16
Epoch 12 loss 0.13986793893902771 valid acc 16/16
Epoch 12 loss 0.33419036082159476 valid acc 16/16
Epoch 12 loss 0.09867459668766343 valid acc 16/16
Epoch 12 loss 0.1683482926833045 valid acc 16/16
Epoch 12 loss 0.23381159575769914 valid acc 16/16
Epoch 12 loss 0.10608509632822277 valid acc 16/16
Epoch 12 loss 0.8086486196282698 valid acc 16/16
Epoch 12 loss 0.3049354748730571 valid acc 16/16
Epoch 12 loss 0.8866650113429867 valid acc 16/16
Epoch 12 loss 0.586002058511712 valid acc 16/16
Epoch 12 loss 0.4257360639187809 valid acc 16/16
Epoch 12 loss 0.35586138172908877 valid acc 15/16
Epoch 12 loss 1.2882687837200093 valid acc 16/16
Epoch 12 loss 0.3091656193253245 valid acc 16/16
Epoch 12 loss 0.4660864178083601 valid acc 16/16
Epoch 12 loss 0.18904879301995625 valid acc 16/16
Epoch 12 loss 0.4802920161149475 valid acc 16/16
Epoch 12 loss 0.7756271880638942 valid acc 16/16
Epoch 12 loss 0.23198963333099687 valid acc 16/16
Epoch 12 loss 0.20834292915581987 valid acc 16/16
Epoch 12 loss 0.10738298937419472 valid acc 16/16
Epoch 12 loss 1.3423725571902823 valid acc 15/16
Epoch 12 loss 0.20779378988748087 valid acc 16/16
Epoch 12 loss 0.3132671630487544 valid acc 15/16
Epoch 12 loss 0.29374301556913984 valid acc 15/16
Epoch 12 loss 0.27068955064031186 valid acc 16/16
Epoch 12 loss 0.09439939109927595 valid acc 16/16
Epoch 12 loss 0.18687956109307446 valid acc 16/16
Epoch 13 loss 0.0356382808680103 valid acc 16/16
Epoch 13 loss 0.3184407313014704 valid acc 14/16
Epoch 13 loss 0.2749815437922771 valid acc 14/16
Epoch 13 loss 0.10516513984606471 valid acc 15/16
Epoch 13 loss 0.13606479444877995 valid acc 15/16
Epoch 13 loss 0.4554502863127036 valid acc 16/16
Epoch 13 loss 0.513248725582063 valid acc 16/16
Epoch 13 loss 0.6981919158095149 valid acc 15/16
Epoch 13 loss 0.728560625316965 valid acc 14/16
Epoch 13 loss 0.3247590704410982 valid acc 15/16
Epoch 13 loss 0.3767092319461133 valid acc 15/16
Epoch 13 loss 0.19017147616898944 valid acc 16/16
Epoch 13 loss 0.5942418537999359 valid acc 16/16
Epoch 13 loss 0.6170343723361055 valid acc 15/16
Epoch 13 loss 0.42941154033112844 valid acc 16/16
Epoch 13 loss 0.5862007907602144 valid acc 15/16
Epoch 13 loss 0.17368322208702136 valid acc 16/16
Epoch 13 loss 0.1823713140507559 valid acc 16/16
Epoch 13 loss 0.6194907230073132 valid acc 16/16
Epoch 13 loss 0.17457084387511562 valid acc 16/16
Epoch 13 loss 0.4029611104539519 valid acc 16/16
Epoch 13 loss 0.1160151154643313 valid acc 16/16
Epoch 13 loss 0.049387270608409195 valid acc 16/16
Epoch 13 loss 0.10266069704764508 valid acc 16/16
Epoch 13 loss 0.12781123144160983 valid acc 15/16
Epoch 13 loss 0.16380994290445464 valid acc 15/16
Epoch 13 loss 0.07651290806484867 valid acc 15/16
Epoch 13 loss 0.1429203900409401 valid acc 15/16
Epoch 13 loss 0.2203719015451246 valid acc 15/16
Epoch 13 loss 0.05844973059268699 valid acc 16/16
Epoch 13 loss 0.13762069298702873 valid acc 16/16
Epoch 13 loss 0.5553470803284339 valid acc 15/16
Epoch 13 loss 0.2335927580386734 valid acc 15/16
Epoch 13 loss 0.8005127108084666 valid acc 16/16
Epoch 13 loss 1.1377832182413061 valid acc 15/16
Epoch 13 loss 0.10329145607108592 valid acc 16/16
Epoch 13 loss 0.6656053479701579 valid acc 15/16
Epoch 13 loss 0.24535268340042787 valid acc 16/16
Epoch 13 loss 0.5817529970731554 valid acc 16/16
Epoch 13 loss 0.5338130498882155 valid acc 16/16
Epoch 13 loss 0.047129923079023806 valid acc 16/16
Epoch 13 loss 0.27918799147554485 valid acc 16/16
Epoch 13 loss 0.22597500041835933 valid acc 16/16
Epoch 13 loss 0.29557599429580705 valid acc 15/16
Epoch 13 loss 0.3830271243086411 valid acc 15/16
Epoch 13 loss 0.12331041921875321 valid acc 15/16
Epoch 13 loss 0.23089049216578167 valid acc 15/16
Epoch 13 loss 0.5763432058024839 valid acc 15/16
Epoch 13 loss 0.5383524961920609 valid acc 15/16
Epoch 13 loss 0.28325272599605167 valid acc 15/16
Epoch 13 loss 0.2865714434184799 valid acc 15/16
Epoch 13 loss 0.17350550936672604 valid acc 15/16
Epoch 13 loss 0.3974920306396268 valid acc 15/16
Epoch 13 loss 0.19293754456883727 valid acc 15/16
Epoch 13 loss 0.3627038258989635 valid acc 15/16
Epoch 13 loss 0.2996458942886851 valid acc 15/16
Epoch 13 loss 0.1334511806653837 valid acc 14/16
Epoch 13 loss 0.1017975215440749 valid acc 16/16
Epoch 13 loss 0.251298465310386 valid acc 14/16
Epoch 13 loss 0.4546525238908239 valid acc 14/16
Epoch 13 loss 0.06143655020869193 valid acc 16/16
Epoch 13 loss 0.36732062401458837 valid acc 15/16
Epoch 13 loss 0.11312433630198848 valid acc 16/16
Epoch 14 loss 0.001378461231263839 valid acc 15/16
Epoch 14 loss 0.2567211969856051 valid acc 16/16
Epoch 14 loss 0.8070528560298839 valid acc 15/16
Epoch 14 loss 0.28189468865902295 valid acc 15/16
Epoch 14 loss 0.17285579003523893 valid acc 15/16
Epoch 14 loss 0.3795146742329437 valid acc 16/16
Epoch 14 loss 0.2218702471724811 valid acc 15/16
Epoch 14 loss 1.0651865203806328 valid acc 15/16
Epoch 14 loss 0.3305367283647882 valid acc 15/16
Epoch 14 loss 0.12845330738727895 valid acc 15/16
Epoch 14 loss 0.16755801673195247 valid acc 15/16
Epoch 14 loss 0.30637098157379716 valid acc 15/16
Epoch 14 loss 0.6872224961423226 valid acc 15/16
Epoch 14 loss 0.37059089063342004 valid acc 15/16
Epoch 14 loss 0.22133332693745073 valid acc 15/16
Epoch 14 loss 0.3197631409508713 valid acc 15/16
Epoch 14 loss 0.7637095228519939 valid acc 15/16
Epoch 14 loss 0.3515363394979233 valid acc 15/16
Epoch 14 loss 0.8861351722583606 valid acc 15/16
Epoch 14 loss 0.3565412387635204 valid acc 15/16
Epoch 14 loss 0.20987171497422963 valid acc 15/16
Epoch 14 loss 0.3587246763689737 valid acc 15/16
Epoch 14 loss 0.03928009926713716 valid acc 16/16
Epoch 14 loss 0.3552175160836352 valid acc 16/16
Epoch 14 loss 0.07560600336239698 valid acc 15/16
Epoch 14 loss 0.7394989650606043 valid acc 15/16
Epoch 14 loss 0.22018223953035257 valid acc 15/16
Epoch 14 loss 0.173765310370378 valid acc 15/16
Epoch 14 loss 0.06597103381808728 valid acc 15/16
Epoch 14 loss 0.02111985661360366 valid acc 15/16
Epoch 14 loss 0.37106868919754543 valid acc 16/16
Epoch 14 loss 0.21707243189519176 valid acc 16/16
Epoch 14 loss 0.48994996911599276 valid acc 16/16
Epoch 14 loss 0.23416569866204875 valid acc 16/16
Epoch 14 loss 0.6850844807575514 valid acc 16/16
Epoch 14 loss 0.3798185605207546 valid acc 15/16
Epoch 14 loss 0.27047180230563683 valid acc 15/16
Epoch 14 loss 0.17436295665238355 valid acc 16/16
Epoch 14 loss 0.06396590767479993 valid acc 15/16
Epoch 14 loss 0.06570034992569446 valid acc 16/16
Epoch 14 loss 0.0636026284461203 valid acc 16/16
Epoch 14 loss 0.13653976806411505 valid acc 16/16
Epoch 14 loss 0.47076563847870084 valid acc 15/16
Epoch 14 loss 0.14648962119564093 valid acc 16/16
Epoch 14 loss 0.4999697535843231 valid acc 16/16
Epoch 14 loss 0.09727434309770638 valid acc 16/16
Epoch 14 loss 0.507167652274715 valid acc 16/16
Epoch 14 loss 0.7345889100140924 valid acc 16/16
Epoch 14 loss 0.1325527299552964 valid acc 16/16
Epoch 14 loss 0.1552222320340628 valid acc 16/16
Epoch 14 loss 0.03946573797503383 valid acc 16/16
Epoch 14 loss 0.12635445539748447 valid acc 16/16
Epoch 14 loss 0.3718438715223755 valid acc 16/16
Epoch 14 loss 0.2617926880066974 valid acc 16/16
Epoch 14 loss 0.3268305193532751 valid acc 16/16
Epoch 14 loss 0.03518198748594048 valid acc 16/16
Epoch 14 loss 0.11122554559217729 valid acc 16/16
Epoch 14 loss 0.07169576311474271 valid acc 16/16
Epoch 14 loss 0.09271361480415935 valid acc 16/16
Epoch 14 loss 0.2466012868716403 valid acc 16/16
Epoch 14 loss 0.4936229088574784 valid acc 16/16
Epoch 14 loss 0.5368539765189448 valid acc 16/16
Epoch 14 loss 0.2218385097036073 valid acc 16/16
Epoch 15 loss 0.005098279049094656 valid acc 16/16
Epoch 15 loss 0.38514361769731614 valid acc 15/16
Epoch 15 loss 0.31979582291712083 valid acc 14/16
Epoch 15 loss 0.5183269317405341 valid acc 14/16
Epoch 15 loss 0.2205560073094613 valid acc 14/16
Epoch 15 loss 0.2064564213557447 valid acc 15/16
Epoch 15 loss 0.18437218135546798 valid acc 16/16
Epoch 15 loss 0.09092384971203293 valid acc 16/16
Epoch 15 loss 0.035304646961472896 valid acc 16/16
Epoch 15 loss 0.16086869603781145 valid acc 16/16
Epoch 15 loss 0.2890048283619564 valid acc 16/16
Epoch 15 loss 0.25434307530108946 valid acc 16/16
Epoch 15 loss 0.8257458135264935 valid acc 15/16
Epoch 15 loss 0.5383654053944644 valid acc 16/16
Epoch 15 loss 0.5709561685851603 valid acc 16/16
Epoch 15 loss 0.2664684346600815 valid acc 16/16
Epoch 15 loss 0.21615804753252305 valid acc 15/16
Epoch 15 loss 0.11970294552239291 valid acc 15/16
Epoch 15 loss 0.1527181987224791 valid acc 15/16
Epoch 15 loss 0.35782458393404293 valid acc 16/16
Epoch 15 loss 0.34832351936616657 valid acc 15/16
Epoch 15 loss 0.2115185670900382 valid acc 15/16
Epoch 15 loss 0.05622013187988961 valid acc 16/16
Epoch 15 loss 0.3271279157521569 valid acc 15/16
Epoch 15 loss 0.3787081403107291 valid acc 15/16
Epoch 15 loss 0.6125253182768191 valid acc 15/16
Epoch 15 loss 0.18481000633114586 valid acc 15/16
Epoch 15 loss 0.1487746175607642 valid acc 15/16
Epoch 15 loss 0.04266967903912533 valid acc 15/16
Epoch 15 loss 0.13022377551808803 valid acc 15/16
Epoch 15 loss 0.44180968311360036 valid acc 16/16
Epoch 15 loss 0.16332309967818182 valid acc 15/16
Epoch 15 loss 0.13018780979470676 valid acc 15/16
Epoch 15 loss 0.040158880259825946 valid acc 15/16
Epoch 15 loss 0.11670545199542592 valid acc 15/16
Epoch 15 loss 0.4243182658880159 valid acc 15/16
Epoch 15 loss 0.6252549851751377 valid acc 15/16
Epoch 15 loss 0.06079034801716754 valid acc 15/16
Epoch 15 loss 0.16364177401433377 valid acc 15/16
Epoch 15 loss 0.19895303353545946 valid acc 16/16
Epoch 15 loss 0.03859134807578035 valid acc 16/16
Epoch 15 loss 0.1799709379118745 valid acc 16/16
Epoch 15 loss 0.22470275745564455 valid acc 16/16
Epoch 15 loss 0.059371562562301305 valid acc 16/16
Epoch 15 loss 0.5378073258315119 valid acc 16/16
Epoch 15 loss 0.3792348535096606 valid acc 16/16
Epoch 15 loss 0.43305807392897194 valid acc 16/16
Epoch 15 loss 0.6747025988797515 valid acc 15/16
Epoch 15 loss 0.3671583489181045 valid acc 16/16
Epoch 15 loss 0.01977604931616328 valid acc 16/16
Epoch 15 loss 0.20721913902625072 valid acc 16/16
Epoch 15 loss 0.1292775699169949 valid acc 16/16
Epoch 15 loss 0.37934796536046217 valid acc 16/16
Epoch 15 loss 0.5669638902194449 valid acc 16/16
Epoch 15 loss 0.23894755925698777 valid acc 16/16
Epoch 15 loss 0.027635715741838485 valid acc 16/16
Epoch 15 loss 0.2792533460376153 valid acc 15/16
Epoch 15 loss 0.436153482635303 valid acc 16/16
Epoch 15 loss 1.259815687253276 valid acc 16/16
Epoch 15 loss 0.2870619233536588 valid acc 15/16
Epoch 15 loss 0.26902096335603304 valid acc 16/16
Epoch 15 loss 0.23352602741113118 valid acc 16/16
Epoch 15 loss 0.24975624317930767 valid acc 16/16
Epoch 16 loss 0.002682380194679701 valid acc 16/16
Epoch 16 loss 0.35842787966426465 valid acc 16/16
Epoch 16 loss 0.15799175073951982 valid acc 16/16
Epoch 16 loss 0.3890302738468275 valid acc 14/16
Epoch 16 loss 0.20151271786247144 valid acc 16/16
Epoch 16 loss 0.15740609912394532 valid acc 16/16
Epoch 16 loss 0.3763614666952956 valid acc 15/16
Epoch 16 loss 0.5810368804813162 valid acc 15/16
Epoch 16 loss 0.44615857765322836 valid acc 15/16
Epoch 16 loss 0.39392429581820143 valid acc 16/16
Epoch 16 loss 0.16915813385571507 valid acc 16/16
Epoch 16 loss 0.42687795166927145 valid acc 15/16
Epoch 16 loss 0.6724343561505803 valid acc 15/16
Epoch 16 loss 0.37034476067341293 valid acc 14/16
Epoch 16 loss 0.346700405258375 valid acc 14/16
Epoch 16 loss 0.3573569551324555 valid acc 14/16
Epoch 16 loss 0.3220849588818222 valid acc 16/16
Epoch 16 loss 0.25521549685675676 valid acc 16/16
Epoch 16 loss 0.27480144324458383 valid acc 15/16
Epoch 16 loss 0.28633399739356635 valid acc 14/16
Epoch 16 loss 0.28892172839586183 valid acc 15/16
Epoch 16 loss 0.026748698343727284 valid acc 15/16
Epoch 16 loss 0.05886501193155441 valid acc 16/16
Epoch 16 loss 0.22613511064992337 valid acc 15/16
Epoch 16 loss 0.1783862470372653 valid acc 16/16
Epoch 16 loss 0.2205765133256815 valid acc 16/16
Epoch 16 loss 0.06918727361272053 valid acc 16/16
Epoch 16 loss 0.046041358800126 valid acc 16/16
Epoch 16 loss 0.17743981765758066 valid acc 16/16
Epoch 16 loss 0.21889475026895 valid acc 16/16
Epoch 16 loss 0.31614396562476443 valid acc 15/16
Epoch 16 loss 1.167095318172608 valid acc 15/16
Epoch 16 loss 0.7526211107912222 valid acc 16/16
Epoch 16 loss 0.19057155336443782 valid acc 16/16
Epoch 16 loss 0.5733689054885311 valid acc 16/16
Epoch 16 loss 0.25777612540319106 valid acc 16/16
Epoch 16 loss 0.1029208012819881 valid acc 16/16
Epoch 16 loss 0.13638205143019144 valid acc 16/16
Epoch 16 loss 0.43348567712497854 valid acc 16/16
Epoch 16 loss 0.21382207367659234 valid acc 16/16
Epoch 16 loss 0.1398592415332402 valid acc 16/16
Epoch 16 loss 0.15241686016239253 valid acc 16/16
Epoch 16 loss 0.158595544672335 valid acc 15/16
Epoch 16 loss 0.06947477869268309 valid acc 16/16
Epoch 16 loss 0.15414052882298468 valid acc 16/16
Epoch 16 loss 0.06365113050104465 valid acc 16/16
Epoch 16 loss 0.07732261046185418 valid acc 16/16
Epoch 16 loss 0.9524647828985211 valid acc 15/16
Epoch 16 loss 0.09702848817171228 valid acc 15/16
Epoch 16 loss 0.10228587612584472 valid acc 16/16
Epoch 16 loss 0.46247544465904716 valid acc 16/16
Epoch 16 loss 0.19139093415035568 valid acc 16/16
Epoch 16 loss 0.18024707947309174 valid acc 16/16
Epoch 16 loss 0.42891433944280616 valid acc 16/16
Epoch 16 loss 0.5742619689188885 valid acc 16/16
Epoch 16 loss 0.5458718403019829 valid acc 16/16
Epoch 16 loss 0.7223031380306624 valid acc 16/16
Epoch 16 loss 0.4995900254022717 valid acc 16/16
Epoch 16 loss 0.1479422323684413 valid acc 16/16
Epoch 16 loss 0.2224607130898245 valid acc 16/16
Epoch 16 loss 0.24370608182575276 valid acc 16/16
Epoch 16 loss 0.054161363816217056 valid acc 16/16
Epoch 16 loss 0.4245409302294862 valid acc 16/16
Epoch 17 loss 0.003817021295339229 valid acc 16/16
Epoch 17 loss 0.32167967106051515 valid acc 16/16
Epoch 17 loss 0.0403273678278937 valid acc 16/16
Epoch 17 loss 0.6136775827106755 valid acc 16/16
Epoch 17 loss 0.6465868621979316 valid acc 14/16
Epoch 17 loss 0.06121419717886395 valid acc 15/16
Epoch 17 loss 0.5594131841899246 valid acc 15/16
Epoch 17 loss 0.12353351540141883 valid acc 15/16
Epoch 17 loss 0.567374517130276 valid acc 15/16
Epoch 17 loss 0.10288864295670597 valid acc 16/16
Epoch 17 loss 0.26507822539959575 valid acc 16/16
Epoch 17 loss 0.140632173634483 valid acc 16/16
Epoch 17 loss 0.6786941822973482 valid acc 16/16
Epoch 17 loss 0.7538958194581491 valid acc 16/16
Epoch 17 loss 0.7047776546637277 valid acc 16/16
Epoch 17 loss 0.359406686735752 valid acc 16/16
Epoch 17 loss 0.345419548861967 valid acc 15/16
Epoch 17 loss 0.09938708546013225 valid acc 16/16
Epoch 17 loss 0.3365302217324681 valid acc 16/16
Epoch 17 loss 0.17044511860332506 valid acc 16/16
Epoch 17 loss 0.7159308684903234 valid acc 15/16
Epoch 17 loss 0.792553044734747 valid acc 16/16
Epoch 17 loss 0.036791931694032665 valid acc 16/16
Epoch 17 loss 0.0428043820092861 valid acc 16/16
Epoch 17 loss 0.10648270482525554 valid acc 16/16
Epoch 17 loss 0.290698936222648 valid acc 15/16
Epoch 17 loss 0.25907519376404986 valid acc 15/16
Epoch 17 loss 0.23188350237328714 valid acc 15/16
Epoch 17 loss 0.7541517114503746 valid acc 15/16
Epoch 17 loss 0.082974398509847 valid acc 15/16
Epoch 17 loss 0.13618733104930886 valid acc 16/16
Epoch 17 loss 0.16931406302290564 valid acc 16/16
Epoch 17 loss 0.11154064457287127 valid acc 16/16
Epoch 17 loss 0.14070902976115418 valid acc 16/16
Epoch 17 loss 0.7933747391403376 valid acc 15/16
Epoch 17 loss 0.9244811640996242 valid acc 16/16
Epoch 17 loss 0.22828219649809106 valid acc 16/16
Epoch 17 loss 0.43459278743218793 valid acc 16/16
Epoch 17 loss 0.10032209147066146 valid acc 16/16
Epoch 17 loss 0.24030090427709944 valid acc 16/16
Epoch 17 loss 0.17730894479202075 valid acc 16/16
Epoch 17 loss 0.05984749021578112 valid acc 16/16
Epoch 17 loss 0.10679818285063554 valid acc 16/16
Epoch 17 loss 0.06219230663906694 valid acc 16/16
Epoch 17 loss 0.6609552423580503 valid acc 16/16
Epoch 17 loss 0.047843342513821996 valid acc 16/16
Epoch 17 loss 0.26476771564111967 valid acc 16/16
Epoch 17 loss 0.3329896695964442 valid acc 15/16
Epoch 17 loss 0.12441720055983874 valid acc 16/16
Epoch 17 loss 0.19826914714615412 valid acc 15/16
Epoch 17 loss 0.19268193005615464 valid acc 16/16
Epoch 17 loss 0.13644371700438002 valid acc 16/16
Epoch 17 loss 0.45818175201970773 valid acc 16/16
Epoch 17 loss 0.10586468486553807 valid acc 16/16
Epoch 17 loss 0.1995359220418529 valid acc 16/16
Epoch 17 loss 0.29063744759815424 valid acc 16/16
Epoch 17 loss 0.2529987012497483 valid acc 16/16
Epoch 17 loss 0.07517737414897244 valid acc 16/16
Epoch 17 loss 0.2976591204197918 valid acc 15/16
Epoch 17 loss 0.1322845968337476 valid acc 15/16
Epoch 17 loss 0.22724486136689676 valid acc 16/16
Epoch 17 loss 0.11188385634603959 valid acc 16/16
Epoch 17 loss 0.08314074547267702 valid acc 16/16
Epoch 18 loss 0.00034523599307978436 valid acc 16/16
Epoch 18 loss 0.29397503718262646 valid acc 16/16
Epoch 18 loss 0.6025722291682991 valid acc 16/16
Epoch 18 loss 0.11631850081758321 valid acc 16/16
Epoch 18 loss 0.2444083370090897 valid acc 16/16
Epoch 18 loss 0.15114840713627203 valid acc 16/16
Epoch 18 loss 0.4085025154490032 valid acc 16/16
Epoch 18 loss 0.06137265513725981 valid acc 16/16
Epoch 18 loss 0.9069979977501819 valid acc 16/16
Epoch 18 loss 0.25985775840814185 valid acc 16/16
Epoch 18 loss 0.17439095706093155 valid acc 16/16
Epoch 18 loss 0.1484078903807923 valid acc 16/16
Epoch 18 loss 0.2143366940884811 valid acc 16/16
Epoch 18 loss 0.7314462949463105 valid acc 16/16
Epoch 18 loss 0.27043552778552526 valid acc 16/16
Epoch 18 loss 0.28559258272687993 valid acc 16/16
Epoch 18 loss 0.41337958679172815 valid acc 16/16
Epoch 18 loss 0.3367400126347251 valid acc 16/16
Epoch 18 loss 0.21171543068612947 valid acc 16/16
Epoch 18 loss 0.04882515693365058 valid acc 16/16
Epoch 18 loss 0.07428346895665322 valid acc 16/16
Epoch 18 loss 0.051403252386768905 valid acc 16/16
Epoch 18 loss 0.02791378556799562 valid acc 16/16
Epoch 18 loss 0.14192720956579646 valid acc 16/16
Epoch 18 loss 0.12041366823794658 valid acc 16/16
Epoch 18 loss 0.060476679409731116 valid acc 16/16
Epoch 18 loss 0.22576511056435036 valid acc 16/16
Epoch 18 loss 0.2454352437226135 valid acc 16/16
Epoch 18 loss 0.5488192273507853 valid acc 16/16
Epoch 18 loss 0.10194471901931251 valid acc 16/16
Epoch 18 loss 0.062382984347461645 valid acc 16/16
Epoch 18 loss 0.2611423736387463 valid acc 16/16
Epoch 18 loss 0.32061762514828274 valid acc 16/16
Epoch 18 loss 0.2894310377762712 valid acc 16/16
Epoch 18 loss 0.6744553140124127 valid acc 16/16
Epoch 18 loss 0.5337557191379172 valid acc 16/16
Epoch 18 loss 0.638172824903176 valid acc 16/16
Epoch 18 loss 0.5562477466608087 valid acc 16/16
Epoch 18 loss 0.17969189862903118 valid acc 16/16
Epoch 18 loss 0.07304478152341559 valid acc 16/16
Epoch 18 loss 0.07081988729646788 valid acc 16/16
Epoch 18 loss 0.45289862763291217 valid acc 16/16
Epoch 18 loss 0.1191593021761454 valid acc 16/16
Epoch 18 loss 0.17878014534884834 valid acc 16/16
Epoch 18 loss 0.40106638599088673 valid acc 15/16
Epoch 18 loss 0.12202648769138014 valid acc 16/16
Epoch 18 loss 0.23587701368557545 valid acc 16/16
Epoch 18 loss 0.27927018575637297 valid acc 16/16
Epoch 18 loss 0.05356733536259406 valid acc 16/16
Epoch 18 loss 0.024623344974996242 valid acc 16/16
Epoch 18 loss 0.5100804622331777 valid acc 16/16
Epoch 18 loss 0.2681573695673202 valid acc 16/16
Epoch 18 loss 0.614344732425919 valid acc 15/16
Epoch 18 loss 0.2829445324923938 valid acc 16/16
Epoch 18 loss 0.32954771267496613 valid acc 16/16
Epoch 18 loss 0.13917206576755398 valid acc 16/16
Epoch 18 loss 0.18384953367773987 valid acc 16/16
Epoch 18 loss 0.22677060907252827 valid acc 16/16
Epoch 18 loss 0.20934061277771565 valid acc 16/16
Epoch 18 loss 0.05995914083887788 valid acc 16/16
Epoch 18 loss 0.03568983289551142 valid acc 16/16
Epoch 18 loss 0.04901300830176075 valid acc 16/16
Epoch 18 loss 0.6728772527869076 valid acc 16/16
Epoch 19 loss 0.0010023374693862719 valid acc 16/16
Epoch 19 loss 0.319006769436097 valid acc 16/16
Epoch 19 loss 0.0921590400148139 valid acc 16/16
Epoch 19 loss 0.47887372389021365 valid acc 16/16
Epoch 19 loss 0.2862069596724191 valid acc 16/16
Epoch 19 loss 0.12850072773173185 valid acc 16/16
Epoch 19 loss 0.23840616148938 valid acc 16/16
Epoch 19 loss 0.23272978370337716 valid acc 16/16
Epoch 19 loss 0.2445062918988541 valid acc 16/16
Epoch 19 loss 0.37725833627278305 valid acc 16/16
Epoch 19 loss 0.13549704218103575 valid acc 16/16
Epoch 19 loss 0.35988389962026435 valid acc 16/16
Epoch 19 loss 0.4438990634422905 valid acc 16/16
Epoch 19 loss 0.37501906154902637 valid acc 15/16
Epoch 19 loss 0.14083758048676126 valid acc 16/16
Epoch 19 loss 0.12007057595303905 valid acc 15/16
Epoch 19 loss 0.8395173590873024 valid acc 15/16
Epoch 19 loss 0.5981618392126465 valid acc 16/16
Epoch 19 loss 0.11608159827586079 valid acc 16/16
Epoch 19 loss 0.1191411731846802 valid acc 16/16
Epoch 19 loss 0.40483856290168097 valid acc 15/16
Epoch 19 loss 0.270315896814613 valid acc 16/16
Epoch 19 loss 0.10791573357702344 valid acc 16/16
Epoch 19 loss 0.21215432963769604 valid acc 16/16
Epoch 19 loss 0.16722331239511048 valid acc 16/16
Epoch 19 loss 0.7089757089725657 valid acc 15/16
Epoch 19 loss 0.0701429953194233 valid acc 15/16
Epoch 19 loss 0.2088110874576268 valid acc 16/16
Epoch 19 loss 0.38204551500699024 valid acc 16/16
Epoch 19 loss 0.32172695291926184 valid acc 16/16
Epoch 19 loss 0.3977495572474432 valid acc 16/16
Epoch 19 loss 0.39083923048815056 valid acc 16/16
Epoch 19 loss 0.11406210029789304 valid acc 16/16
Epoch 19 loss 0.1200270231189064 valid acc 16/16
Epoch 19 loss 0.3619061060561212 valid acc 16/16
Epoch 19 loss 0.1719676388475017 valid acc 16/16
Epoch 19 loss 0.11432457389240672 valid acc 16/16
Epoch 19 loss 0.14249300955958277 valid acc 16/16
Epoch 19 loss 0.04953752554304924 valid acc 16/16
Epoch 19 loss 0.031425404612994146 valid acc 16/16
Epoch 19 loss 0.06505333951793302 valid acc 16/16
Epoch 19 loss 0.17032918985745837 valid acc 16/16
Epoch 19 loss 0.18108204295240296 valid acc 16/16
Epoch 19 loss 0.16147595687227664 valid acc 16/16
Epoch 19 loss 0.5173322159122773 valid acc 16/16
Epoch 19 loss 0.02501406151807678 valid acc 16/16
Epoch 19 loss 0.17086775932308695 valid acc 16/16
Epoch 19 loss 0.5803084447690847 valid acc 16/16
Epoch 19 loss 0.08659535538190105 valid acc 16/16
Epoch 19 loss 0.17663558954839442 valid acc 16/16
Epoch 19 loss 0.23749917322893133 valid acc 16/16
Epoch 19 loss 0.01150982448721427 valid acc 16/16
Epoch 19 loss 0.15828126815655796 valid acc 16/16
Epoch 19 loss 0.1445198820269189 valid acc 16/16
Epoch 19 loss 0.03347654590036098 valid acc 16/16
Epoch 19 loss 0.3240080336531899 valid acc 16/16
Epoch 19 loss 0.48435122443680023 valid acc 15/16
Epoch 19 loss 0.053302466806950666 valid acc 16/16
Epoch 19 loss 0.4989083496444352 valid acc 16/16
Epoch 19 loss 0.07871671294995375 valid acc 16/16
Epoch 19 loss 0.11196208051371942 valid acc 16/16
Epoch 19 loss 0.1052306184794708 valid acc 16/16
Epoch 19 loss 0.2519896098135794 valid acc 16/16
Epoch 20 loss 0.0016830712231211686 valid acc 16/16
Epoch 20 loss 0.0730400318627985 valid acc 16/16
Epoch 20 loss 0.15010872048443163 valid acc 16/16
Epoch 20 loss 0.12730148651980683 valid acc 16/16
Epoch 20 loss 0.47049671539016014 valid acc 15/16
Epoch 20 loss 0.6278871923483482 valid acc 16/16
Epoch 20 loss 0.32071174878458697 valid acc 16/16
Epoch 20 loss 0.12991684474975562 valid acc 16/16
Epoch 20 loss 0.15339138875367386 valid acc 16/16
Epoch 20 loss 0.11675916521582619 valid acc 16/16
Epoch 20 loss 0.10050970699210848 valid acc 16/16
Epoch 20 loss 0.05660997297802134 valid acc 16/16
Epoch 20 loss 0.2726695086042022 valid acc 16/16
Epoch 20 loss 0.028892427204353144 valid acc 16/16
Epoch 20 loss 0.1278945408996267 valid acc 16/16
Epoch 20 loss 0.39978540879158775 valid acc 16/16
Epoch 20 loss 0.1045849855356108 valid acc 16/16
Epoch 20 loss 0.0796568242335004 valid acc 16/16
Epoch 20 loss 0.30287805119612177 valid acc 16/16
Epoch 20 loss 0.025833566133214025 valid acc 16/16
Epoch 20 loss 0.36462862096746074 valid acc 16/16
Epoch 20 loss 0.10969991344515057 valid acc 16/16
Epoch 20 loss 0.08377912548363087 valid acc 15/16
Epoch 20 loss 0.026008347700486212 valid acc 16/16
Epoch 20 loss 0.030349318968679373 valid acc 16/16
Epoch 20 loss 0.7583333194455697 valid acc 16/16
Epoch 20 loss 0.10970760440316385 valid acc 15/16
Epoch 20 loss 0.07066414474106764 valid acc 16/16
Epoch 20 loss 0.343075552940507 valid acc 15/16
Epoch 20 loss 0.18492811009622762 valid acc 16/16
Epoch 20 loss 0.06876182880001785 valid acc 16/16
Epoch 20 loss 0.4774799497836406 valid acc 16/16
Epoch 20 loss 0.11384406761750937 valid acc 16/16
Epoch 20 loss 0.12406776914232726 valid acc 16/16
Epoch 20 loss 0.10172115994244535 valid acc 16/16
Epoch 20 loss 0.24886352441415244 valid acc 16/16
Epoch 20 loss 0.038490386987552416 valid acc 16/16
Epoch 20 loss 0.3107966213053406 valid acc 16/16
Epoch 20 loss 0.40886784908274215 valid acc 16/16
Epoch 20 loss 0.5101210613752416 valid acc 16/16
Epoch 20 loss 0.022381599341228076 valid acc 16/16
Epoch 20 loss 0.23718792618945306 valid acc 16/16
Epoch 20 loss 0.049982687243855684 valid acc 16/16
Epoch 20 loss 0.5410332104721399 valid acc 16/16
Epoch 20 loss 0.0970105114511427 valid acc 16/16
Epoch 20 loss 0.1532253135182728 valid acc 16/16
Epoch 20 loss 0.20360565173082512 valid acc 16/16
Epoch 20 loss 0.753664198061401 valid acc 15/16
Epoch 20 loss 0.39282746695329857 valid acc 15/16
Epoch 20 loss 0.14302860238570803 valid acc 16/16
Epoch 20 loss 0.06008765874175215 valid acc 16/16
Epoch 20 loss 0.38645041392763835 valid acc 16/16
Epoch 20 loss 0.5595982955043877 valid acc 15/16
Epoch 20 loss 0.14794436733979932 valid acc 16/16
Epoch 20 loss 0.32887328664357196 valid acc 16/16
Epoch 20 loss 0.17644529324539643 valid acc 16/16
Epoch 20 loss 0.07181462425408092 valid acc 16/16
Epoch 20 loss 0.12687192963337046 valid acc 16/16
Epoch 20 loss 0.4808706685066238 valid acc 16/16
Epoch 20 loss 0.3357225514246618 valid acc 16/16
Epoch 20 loss 0.1858959455596214 valid acc 16/16
Epoch 20 loss 0.12203148291734925 valid acc 16/16
Epoch 20 loss 0.045501263682765414 valid acc 16/16
