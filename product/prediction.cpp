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

torch::Tensor rgb2xyz(torch::Tensor &rgb)
{
	// input: BxCxHxW, where C==3 (RGB)

	torch::Tensor mask1 = rgb.gt(0.04045).to(torch::kFloat32);;
	torch::Tensor mask2 = rgb.le(0.04045).to(torch::kFloat32);;
	torch::Tensor rgb1 = rgb.add(0.055).div(1.055).pow(2.4);
	torch::Tensor rgb2 = rgb.div(12.92);
	torch::Tensor rgbtemp = rgb1.mul(mask1) + rgb2.mul(mask2);

	// slice(/*dim =*/1, start =0, /*stop =*/1, /*step =*/1)
	torch::Tensor R = rgbtemp.slice(1, 0, 1, 1).squeeze(1); // [:, 0, :, :]
	torch::Tensor G = rgbtemp.slice(1, 1, 2, 1).squeeze(1); // [:, 1, :, :]
	torch::Tensor B = rgbtemp.slice(1, 2, 3, 1).squeeze(1); // [:, 2, :, :]

	// Now R, G, B sizes() is BxHxW
	torch::Tensor X = .412453*R + .357580*G + .180423*B;
	torch::Tensor Y = .212671*R + .715160*G + .072169*B;
	torch::Tensor Z = .019334*R + .119193*G + .950227*B;
	X.unsqueeze_(1); Y.unsqueeze_(1); Z.unsqueeze_(1);

	return torch::cat({X, Y, Z}, /*dim =*/1);
}

torch::Tensor xyz2rgb(torch::Tensor &xyz)
{
	// input: BxCxHxW, where C==3 (XYZ)

	// slice(/*dim =*/1, start =0, /*stop =*/1, /*step =*/1)
	torch::Tensor X = xyz.slice(1, 0, 1, 1).squeeze(1); // [:, 0, :, :]
	torch::Tensor Y = xyz.slice(1, 1, 2, 1).squeeze(1); // [:, 1, :, :]
	torch::Tensor Z = xyz.slice(1, 2, 3, 1).squeeze(1); // [:, 2, :, :]

	// Now X, Y, Z sizes() is BxHxW
	torch::Tensor R =  3.24048134*X - 1.53715152*Y - 0.49853633*Z;
	torch::Tensor G = -0.96925495*X + 1.87599*Y + .04155593*Z;
	torch::Tensor B = .05564664*X - .20404134*Y + 1.05731107*Z;
	R.unsqueeze_(1); G.unsqueeze_(1); B.unsqueeze_(1);
	torch::Tensor rgb = torch::cat({R, G, B}, /*dim = */1);

	// Some times reaches a small negative number, which causes NaNs	
	rgb = torch::max(rgb, torch::zeros_like(rgb));

	torch::Tensor mask1 = rgb.gt(.0031308).to(torch::kFloat32);;
	torch::Tensor mask2 = rgb.le(.0031308).to(torch::kFloat32);;

	torch::Tensor rgb1 = rgb.pow(1./2.4).mul(1.055).sub(0.055);
	torch::Tensor rgb2 = rgb.mul(12.92);

	return rgb1.mul(mask1) + rgb2.mul(mask2);
}

torch::Tensor lab2xyz(torch::Tensor &lab)
{
	// input: BxCxHxW, where C==3 (Lab)
	// slice(/*dim =*/1, start =0, /*stop =*/1, /*step =*/1)
	torch::Tensor Y = lab.slice(1, 0, 1, 1).squeeze(1); // [:, 0, :, :]
	torch::Tensor X = lab.slice(1, 1, 2, 1).squeeze(1); // [:, 1, :, :]
	torch::Tensor Z = lab.slice(1, 2, 3, 1).squeeze(1); // [:, 2, :, :]

	// Now X, Y, Z sizes is BxHxW !!!
	Y.add_(16.0).div_(116.0);
	X = X.div(500.0) + Y;
	Z = Y - Z.div(200.0);
	Z = torch::max(Z, torch::zeros_like(Z));
	X.unsqueeze_(1); Y.unsqueeze_(1); Z.unsqueeze_(1);
	torch::Tensor XYZ = torch::cat({X, Y, Z}, /*dim = */1);

	torch::Tensor mask1 = XYZ.gt(.2068966).to(torch::kFloat32);;
	torch::Tensor mask2 = XYZ.le(.2068966).to(torch::kFloat32);;
	torch::Tensor xyz1 = XYZ.pow(3.0);
	torch::Tensor xyz2 = XYZ.sub(16.0/116.0).div(7.787);
	torch::Tensor xyz = xyz1.mul(mask1) + xyz2.mul(mask2);

	float sc_float[3] = {0.95047, 1., 1.08883};
	torch::Tensor sc = torch::from_blob(sc_float, {1, 3, 1, 1});

	return xyz * sc;
}

torch::Tensor xyz2lab(torch::Tensor &xyz)
{
	// input: BxCxHxW, where C==3 (xyz)
	// slice(/*dim =*/1, start =0, /*stop =*/1, /*step =*/1)
	float sc_float[3] = {0.95047, 1., 1.08883};
	torch::Tensor sc = torch::from_blob(sc_float, {1, 3, 1, 1});
	torch::Tensor xyz_scale = xyz/sc;
	torch::Tensor mask1 = xyz_scale.gt(.008856).to(torch::kFloat32);;
	torch::Tensor mask2 = xyz_scale.le(.008856).to(torch::kFloat32);;

	torch::Tensor xyz_int1 = xyz_scale.pow(1/3.);
	torch::Tensor xyz_int2 = xyz_scale.mul(7.787).add(16./116.);
	torch::Tensor xyz_int = xyz_int1.mul(mask1) + xyz_int2.mul(mask2);

	// L, A, B
	torch::Tensor L = xyz_int.slice(1, 1, 2, 1).squeeze(1); // [:, 1, :, :]
	torch::Tensor A = xyz_int.slice(1, 0, 1, 1).squeeze(1); // [:, 0, :, :]
	torch::Tensor B = xyz_int.slice(1, 2, 3, 1).squeeze(1); // [:, 2, :, :]
	A = A.mul(500.0) - L;
	B = L.mul(200.0) - B;
	L = L.mul(116.0).sub(16.0);
	// Now X, Y, Z sizes is BxHxW !!!

	L.unsqueeze_(1); A.unsqueeze_(1); B.unsqueeze_(1);
	return torch::cat({L, A, B}, /*dim = */1);
}

// def rgb2lab(rgb):
//     lab = xyz2lab(rgb2xyz(rgb))
//     # xyz2lab(rgb2xyz(rgb)) parameters:
//     # input: rgb in [0, 1.0]
//     # output: l in [0, 100], ab in [-110, 110]

//     l_rs = lab[:, [0], :, :]/100.0
//     ab_rs = (lab[:, 1:, :, :] + 110.0)/220.0
//     out = torch.cat((l_rs, ab_rs), dim=1)
//     # return: tensor space: [0.0, 1.0]
//     return out
torch::Tensor rgb2lab(torch::Tensor &rgb)
{
	// input rgb: BxCxHxW, [0, 1.0]
	// output Lab: L in [0, 100], ab in [-110, 110]
	torch::Tensor xyz = rgb2xyz(rgb);
	torch::Tensor lab = xyz2lab(xyz);
	return lab;	
}

torch::Tensor lab2rgb(torch::Tensor &lab)
{
	// input Lab: L in [0, 100], ab in [-110, 110]
	// output rgb: BxCxHxW, [0, 1.0]
	torch::Tensor xyz = lab2xyz(lab);
	torch::Tensor rgb = xyz2rgb(xyz);
	return rgb;
}

torch::Tensor image_labencode(torch::Tensor &lab)
{
	// input: l in [0, 100], ab in [-110, 110]
    // l_rs = (lab[:, [0], :, :] - 50.0)/100.0
    // ab_rs = lab[:, 1:, :, :]/110.0
	torch::Tensor L = xyz_int.slice(1, 0, 1, 1).squeeze(1); // [:, 0, :, :]
	torch::Tensor AB = xyz_int.slice(1, 1, 3, 1).squeeze(1); // [:, 1:3, :, :]
	L.sub_(50.0).div_(100.0);
	AB.div_(110.0);
	L.unsqueeze_(1); AB.unsqueeze_(1);
	return torch::cat({L, AB}, /* dim = */1);
}

torch::Tensor image_labdecode(torch::Tensor &lab)
{
    // l = lab_rs[:, [0], :, :] * 100.0 + 50.0
    // ab = lab_rs[:, 1:, :, :] * 110.0

	torch::Tensor L = xyz_int.slice(1, 0, 1, 1).squeeze(1); // [:, 0, :, :]
	torch::Tensor AB = xyz_int.slice(1, 1, 3, 1).squeeze(1); // [:, 1:3, :, :]
	L.mul_(100.0).add_(50.0);
	AB.mul_(110.0);
	L.unsqueeze_(1); AB.unsqueeze_(1);

	return torch::cat({L, AB}, /* dim = */1);
}

torch::Tensor video_labencode(torch::Tensor &lab)
{
	// input: l in [0, 100], ab in [-110, 110]
	// l_rs = lab[:, [0], :, :]/100.0
	// ab_rs = (lab[:, 1:, :, :] + 110.0)/220.0
	// output = torch.cat((l_rs, ab_rs), dim=1)
	torch::Tensor L = xyz_int.slice(1, 0, 1, 1).squeeze(1); // [:, 0, :, :]
	torch::Tensor AB = xyz_int.slice(1, 1, 3, 1).squeeze(1); // [:, 1:3, :, :]
	L.div_(100.0);
	AB.add_(110.0).div_(220.0);
	L.unsqueeze_(1); AB.unsqueeze_(1);

	return torch::cat({L, AB}, /* dim = */1);
}

torch::Tensor video_labdecode(torch::Tensor &lab)
{
	// l = lab_rs[:, [0], :, :] * 100.0
	// ab = (lab_rs[:, 1:, :, :]) * 220.0 - 110.0
	// lab = torch.cat((l, ab), dim=1)
	// # lab range: l->[0, 100], ab in [-110, 110] ==> rgb: [0, 1.0]
	torch::Tensor L = xyz_int.slice(1, 0, 1, 1).squeeze(1); // [:, 0, :, :]
	torch::Tensor AB = xyz_int.slice(1, 1, 3, 1).squeeze(1); // [:, 1:3, :, :]
	L.mul_(100.0);
	AB.mul_(220.0).sub_(110.0);
	L.unsqueeze_(1); AB.unsqueeze_(1);

	return torch::cat({L, AB}, /* dim = */1);
}


int image_totensor(IMAGE *image, torch::Tensor &tensor)
{
	int i, j;

	if (! image_valid(image)) {
		syslog_error("Invalid image.");
		return RET_ERROR;
	}

	// Suppose tensor with BxCxHxW (== 1x3xHxW) dimension
	if (tensor.size(0) != 1 || tensor.size(1) != 3 || tensor.size(2) != image->height || tensor.size(3) != image->width) {
		syslog_error("Size of image and tensor does not match.");
		return RET_ERROR;
	}

	auto a = tensor.accessor<float, 4>();
	for (i = 0; i < image->height; i++) {
		for (j = 0; j < image->width; j++) {
			a[0][0][i][j] = image->ie[i][j].r;
			a[0][1][i][j] = image->ie[i][j].g;
			a[0][2][i][j] = image->ie[i][j].b;
		}
	}
	tensor.div_(255.0);

	return RET_OK;
}

IMAGE *image_fromtensor(torch::Tensor &tensor)
{
	int i, j;
	IMAGE *image;

	// Suppose tensor with BxCxHxW (== 1x3xHxW) dimension
	if (tensor.size(0) != 1 || tensor.size(1) != 3 || tensor.size(2) < 1 || tensor.size(3) < 1) {
		syslog_error("Size of tensor is not valid.");
		return NULL;
	}

	image = image_create(tensor.size(2), tensor.size(3)); CHECK_IMAGE(image);
	tensor.mul_(255.0);
	auto a = tensor.accessor<float, 4>();
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

	double sc[3] = {0.95047, 1., 1.08883};
	torch::Tensor data = torch::from_blob(sc, {1, 3, 1, 1});
	std::cout << "Data:" << std::endl;
	std::cout << data << std::endl;

	return 0;


	torch::Tensor rgb = torch::ones({32, 3, 16, 16});
	rgb.fill_(0.5);
	torch::Tensor xyz = rgb2xyz(rgb);
	torch::Tensor y_rgb = xyz2rgb(xyz);
	std::cout << "y_rgb:" << y_rgb << std::endl;
	std::cout << "y_rgb-size:" << y_rgb.sizes() << std::endl;

	// tensor.sizes() == torch::IntArrayRef{3, 4, 5}

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

	if (image_totensor(image, input_tensor) == RET_OK) {
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
		image = image_fromtensor(output_tensor);check_image(image);
#else
		torch::Tensor output_tensor = model.forward(inputs).toTensor();
		cuda_memory_log("Model forward end ...");

		if (cuda_available())
			output_tensor = output_tensor.to(torch::kCPU);

		output_tensor = output_tensor.clamp(0.0, 1.0);
		image = image_fromtensor(output_tensor); check_image(image);
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
