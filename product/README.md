# Torch C++ Demo

This demo show you how to use libtorch, please reference https://pytorch.org/tutorials/advanced/cpp_export.html

## Requirements

- PyTorch (1.6.0)

  Only be used to make trace model

- Libtorch(1.6.0)

  Suppose be installed at /usr/local/libtorch, if not, please modify TORCH_INSTALL_DIR in Makefile


- OpenCV(3.2.0)
`sudo apt-get install libopencv-dev`

  

## Build

```
python model.py
make

```

## Test

![](./images/dog.jpg)

```
$ ./classifier resnet50.pt label.txt images/dog.jpg 
Class Result:
  beagle: 99.1227%
  Walker hound, Walker foxhound: 0.469355%
  English foxhound: 0.110916%

```
![](./images/shark.jpg)
```
$ ./classifier resnet50.pt label.txt images/shark.jpg 
Class Result:
  tiger shark, Galeocerdo cuvieri: 92.2599%
  great white shark, white shark, man-eater, man-eating shark: 5.9425%
  hammerhead, hammerhead shark: 1.77417%
  
```
