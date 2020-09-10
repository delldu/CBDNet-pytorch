/************************************************************************************
***
*** Copyright 2019 Dell(18588220928@163.com), All Rights Reserved.
***
*** File Author: Dell, 2019-09-28 00:37:40
***
************************************************************************************/


// One-stop header.
#include <torch/script.h>

// headers for image
#include "image.h"

#define IMAGE_SIZE 224
#define CHANNELS 3

int main(int argc, const char *argv[])
{
	if (argc != 2) {
		std::cerr << "Usage: " << argv[0] << " image" << std::endl;
		return -1;
	}

	torch::jit::script::Module module;
	try {
		module = torch::jit::load("imageclean.onnx");
	}
	catch(const c10::Error & e) {
		std::cerr << "Loading model error." << std::endl;
		return -1;
	}

	// to GPU
	module.to(at::kCUDA);

	IMAGE *image = image_load((char *)argv[0]);
	if (! image_valid(image)) {
		std::cerr << "Loading image error." << std::endl;
	}

	auto input_tensor = torch::from_blob(image_blob(image), { 1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS });
	input_tensor = input_tensor.div_(255.0);

	// to GPU
	input_tensor = input_tensor.to(at::kCUDA);

	// Test speed ...
	for (int i = 0; i < 1000; i++) {
		if (i % 100 == 0) {
			std::cout << i << " ... " << std::endl;
		}
		module.forward( {input_tensor} ).toTensor();
	}

	torch::Tensor out_tensor;
	out_tensor = module.forward( {input_tensor} ).toTensor();

	// Save out_tensor to image file

	image_destroy(image);
}
