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
#include <cuda_runtime_api.h>

// headers for image, from /usr/local/include/nimage/nimage.h
#include <nimage/image.h>

#include <getopt.h>

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

int cuda_memory_log(const char *checkpoint)
{
    int id, gpu_id, num_gpus;
    size_t free, total;
    static size_t lastfree = 0;
    double delta;

    cudaGetDeviceCount(&num_gpus);

    std::cout.setf(std::ios::fixed);
    for (gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        cudaSetDevice( gpu_id );
        cudaGetDevice( &id );
        cudaMemGetInfo( &free, &total );
        std::cout << checkpoint << std::endl;

        if (lastfree > 0) {
        	delta = lastfree;
        	delta -= free;
        	delta /= (1024 * 1024);
        } else {
        	delta = 0.0;
        }
        std::cout << "    GPU " << id \
        	<< " memory: free=" << std::setprecision(2) << (float)free/(1024.0*1024.0) \
        	<< ", total=" << std::setprecision(2) << total/(1024.0*1024.0) \
	    	<< ", delta=" << std::setprecision(2) << delta << " M" << std::endl;

        lastfree = free;
    }
    return 0;
}	

void help(const char *cmd)
{
	printf("Usage: %s [option]\n", cmd);
	printf("    -h, --help                   Display this help.\n");
	printf("    -m, --model <model.pt>       Model file.\n");
	printf("    -i, --input <image file>     Input image.\n");

	exit(1);
}

int main(int argc, char *argv[])
{
	int optc;
	int option_index = 0;
	char *input_file = NULL;
	char *model_file = NULL;
	IMAGE *image;

	const struct option long_opts[] = {
		{ "help", 0, 0, 'h'},
		{ "model", 1, 0, 'm'},
		{ "input", 1, 0, 'i'},
		{ 0, 0, 0, 0}
	};

	if (argc <= 1)
		help(argv[0]);
	
	while ((optc = getopt_long(argc, argv, "h m: i:", long_opts, &option_index)) != EOF) {
		switch (optc) {
		case 'm':
			model_file = optarg;
			break;
		case 'i':
			input_file = optarg;
			break;
		case 'h':	// help
		default:
			help(argv[0]);
			break;
	    }
	}

	if (! model_file || ! input_file)
		help(argv[0]);

	cuda_memory_log("Program start ...");

	torch::jit::script::Module model;
	try {
		model = torch::jit::load(model_file);
	}
	catch(const c10::Error &e) {
		std::cerr << "Loading model error." << std::endl;
		return -1;
	}

	if (cuda_available())
		model.to(torch::kCUDA);
	cuda_memory_log("Model loading end ...");

	// Reduce GPU memory !!!
   	torch::NoGradGuard no_grad;

	image = image_load(input_file);
	if (! image_valid(image)) {
		std::cerr << "Loading image error." << std::endl;
	}

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
		time_spend((char *)"Model forward 10 times");

#if 1
		auto outputs = model.forward(inputs).toTuple();
		auto elements = outputs->elements();
		torch::Tensor noise_tensor = elements[0].toTensor();
		torch::Tensor output_tensor = elements[elements.size() - 1].toTensor();
		cuda_memory_log("Model forward end ...");

		if (cuda_available())
			output_tensor = output_tensor.to(torch::kCPU);

		output_tensor = output_tensor.clamp(0.0, 1.0);
		image = image_fromtensor(&output_tensor);check_image(image);
#else
		torch::Tensor output_tensor = model.forward(inputs).toTensor();
		cuda_memory_log("Model forward end ...");

		if (cuda_available())
			output_tensor = output_tensor.to(torch::kCPU);

		output_tensor = output_tensor.clamp(0.0, 1.0);
		image = image_fromtensor(&output_tensor); check_image(image);
#endif		
		image_save(image, "result.jpg");
		image_destroy(image);
	}
	else {
		std::cerr << "Convert image to tensor error." << std::endl;
		return -1;
	}

	cuda_memory_log("Program stop.");

	return 0;
}
