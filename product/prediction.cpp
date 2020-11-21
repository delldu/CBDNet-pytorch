/************************************************************************************
***
*** Copyright 2019 Dell(18588220928@163.com), All Rights Reserved.
***
*** File Author: Dell, 2019-09-28 00:37:40
***
************************************************************************************/

// One-stop header.
#include <torch/script.h>
#include <torch/csrc/api/include/torch/cuda.h>

// headers for image, from /usr/local/include/nimage/nimage.h
#include <nimage/image.h>

#define CHANNELS 3
// https://pytorch.org/cppdocs/notes/tensor_basics.html
// https://pytorch.org/cppdocs/notes/tensor_creation.html

int image_totensor(IMAGE *image, torch::Tensor *tensor)
{
	int i, j;

	if (! image_valid(image)) {
		syslog_error("Invalid image.");
		return RET_ERROR;
	}

	// Suppose tensor with BxCxHxW (== 1x3xHxW) dimension
	if (tensor->size(0) != 1 || tensor->size(1) != 3 || tensor->size(2) != image->height || tensor->size(3) != image->width) {
		syslog_error("Size of image and tensor does not match.");
		return RET_ERROR;
	}

	auto a = tensor->accessor<float, 4>();
	for (i = 0; i < image->height; i++) {
		for (j = 0; j < image->width; j++) {
			a[0][0][i][j] = image->ie[i][j].r;
			a[0][1][i][j] = image->ie[i][j].g;
			a[0][2][i][j] = image->ie[i][j].b;
		}
	}
	tensor->div_(255.0);

	return RET_OK;
}

IMAGE *image_fromtensor(torch::Tensor *tensor)
{
	int i, j;
	IMAGE *image;

	// Suppose tensor with BxCxHxW (== 1x3xHxW) dimension
	if (tensor->size(0) != 1 || tensor->size(1) != 3 || tensor->size(2) < 1 || tensor->size(3) < 1) {
		syslog_error("Size of tensor is not valid.");
		return NULL;
	}

	image = image_create(tensor->size(2), tensor->size(3)); CHECK_IMAGE(image);
	tensor->mul_(255.0);
	auto a = tensor->accessor<float, 4>();
	for (i = 0; i < image->height; i++) {
		for (j = 0; j < image->width; j++) {
			image->ie[i][j].r = (BYTE)a[0][0][i][j];
			image->ie[i][j].g = (BYTE)a[0][1][i][j];
			image->ie[i][j].b = (BYTE)a[0][2][i][j];
		}
	}

	return image;
}

int cuda_available()
{
	return torch::cuda::is_available();	
}

int main(int argc, const char *argv[])
{
	IMAGE *image;

	if (argc != 2) {
		std::cerr << "Usage: " << argv[0] << " image" << std::endl;
		return -1;
	}

	torch::jit::script::Module model;
	try {
		model = torch::jit::load("image_clean.pt");
	}
	catch(const c10::Error &e) {
		std::cerr << "Loading model error." << std::endl;
		return -1;
	}

	if (cuda_available())
		model.to(torch::kCUDA);

	// Reduce GPU memory !!!
   	torch::NoGradGuard no_grad;

	image = image_load((char *)argv[1]);
	if (! image_valid(image)) {
		std::cerr << "Loading image error." << std::endl;
	}

	std::cout << "Start cleaning " << argv[1] << " ... " << std::endl;
	// std::vector<int64_t>{1, 3, h, w});
	std::vector<int64_t> input_size;
	input_size.push_back(1);
	input_size.push_back(3);
	input_size.push_back(image->height);
	input_size.push_back(image->width);
	torch::Tensor input_tensor = torch::zeros(input_size);

	if (image_totensor(image, &input_tensor) == RET_OK) {
	    std::vector<torch::jit::IValue> inputs;

		if (cuda_available())
			input_tensor = input_tensor.to(torch::kCUDA);

	    inputs.push_back(input_tensor);

	    // Test performance ...
		time_reset();
		for (int i = 0; i < 10; i++) {
			std::cout << i << " " << std::endl; 
			// model.forward( {input_tensor} ).toTensor();
			model.forward(inputs);
		}
		time_spend((char *)"Image cleaning 10 times");

		auto outputs = model.forward(inputs).toTuple();
		torch::Tensor noise_tensor = outputs->elements()[0].toTensor();
		torch::Tensor clean_tensor = outputs->elements()[1].toTensor();

		if (cuda_available())
			clean_tensor = clean_tensor.to(torch::kCPU);

		image = image_fromtensor(&clean_tensor);check_image(image);
		image_save(image, "result.jpg");
		image_destroy(image);
	}
	else {
		std::cerr << "Convert image to tensor error." << std::endl;
		return -1;
	}

	return 0;
}
