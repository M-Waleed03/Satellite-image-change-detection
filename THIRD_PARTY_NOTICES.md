# Third-party notices

## SNUNet-CD implementation

The separate SNUNet-CD inference implementation in `src/snunet_model.py` is based on the SNUNet-CD implementation by likyoo:

- Source: <https://github.com/likyoo/Siam-NestedUNet>
- License: MIT
- Copyright: Copyright (c) 2020 likyoo

The following MIT license notice applies to that adapted implementation:

> MIT License
>
> Copyright (c) 2020 likyoo
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

## Pretrained checkpoint

`models/pretrained/snunet_c16_256x256_40k_levircd.pth` was downloaded from the [Open-CD model zoo](https://huggingface.co/likyoo/Open-CD_Model_Zoo/tree/main/LEVIR-CD). The model card declares Apache-2.0. The checkpoint is ignored by Git; do not redistribute it without retaining its applicable license and attribution.

This notice only covers the third-party implementation and checkpoint listed above. It does not grant a license to the original LevirCD repository, which has no LICENSE file.
