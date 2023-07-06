import tensorrt as trt
import os 

TRT_LOGGER = trt.Logger()

def build_engine(onnx_model_path, tensorrt_engine_path, engine_precision, img_size, batch_size):
    
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    
    if engine_precision == 'FP16':
        config.set_flag(trt.BuilderFlag.FP16)
    
    parser = trt.OnnxParser(network, logger)

    if not os.path.exists(onnx_model_path):
        print("Failed finding ONNX file!")
        exit()
    print("Succeeded finding ONNX file!")

    with open(onnx_model_path, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed parsing .onnx file!")
            for error in range(parser.num_errors):
            	print(parser.get_error(error))
            exit()
        print("Succeeded parsing .onnx file!")
    
    
    inputTensor = network.get_input(0) 
    print('inputTensor.name:', inputTensor.name)
    
    profile.set_shape(inputTensor.name, (batch_size, img_size[0], img_size[1], img_size[2]), \
        (batch_size, img_size[0], img_size[1], img_size[2]), \
        (batch_size, img_size[0], img_size[1], img_size[2]))

    config.add_optimization_profile(profile)
    
    
    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")
    with open(tensorrt_engine_path, "wb") as f:
        f.write(engineString)