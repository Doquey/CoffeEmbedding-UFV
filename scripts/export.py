import tensorflow as tf
from tensorflow.keras.applications import ResNet50
import argparse
import os

DEFAULT_OUTPUT_PATH = "./resources/weights/"
os.makedirs(DEFAULT_OUTPUT_PATH, exist_ok=True)

def figure_out_model_format(model_path:str) -> str:
    model_name = os.path.basename(model_path)
    model_extension = model_name.split('.')[-1]
    return model_extension    
    
def main(args):

    if args.input_model_path:
        base_architecture = ResNet50()
        extension = figure_out_model_format(args.input_model_path)
        if extension == "h5":
            Converter = tf.lite.TFLiteConverter.from_keras_model(base_architecture.load_weights(args.input_model_path)) if args.weights_only else tf.lite.TFLiteConverter.from_keras_model(base_architecture.load_model(args.input_model_path))
        else:
            Converter = tf.lite.TFLiteConverter.from_saved_model(args.input_model_path)
    else:
        Model = ResNet50('imagenet')
        Converter = tf.lite.TFLiteConverter.from_keras_model(Model)
    if args.int8:
        Converter.optimizations = [tf.lite.Optimize.DEFAULT]
        Converter.target_spec.supported_types = [tf.int8]
        Converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        Converter.inference_input_type = tf.uint8
        Converter.inference_output_type = tf.uint8
        def representative_data_gen():
            for _ in range(100):
                yield [tf.random.normal([1, 224, 224, 3])]
        Converter.representative_dataset = representative_data_gen
    else:
        Converter.optimizations = [tf.lite.Optimize.DEFAULT]
        Converter.target_spec.supported_types = [tf.float16]

    tflite_model = Converter.convert()
    model_name = "CaffeNet50_int8" if args.int8 else "CaffeNet50_f16"

    tflite_model_path = os.path.join(DEFAULT_OUTPUT_PATH, f"{model_name}.tflite")
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    print(f"Model saved to: {tflite_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model_path", type=str, help="Path to your pre-trained model")
    parser.add_argument("--weights_only", action="store_true", help="Pass this argument if you exported only the weights of your model")
    parser.add_argument("--int8", action="store_true", help="Enable int8 quantization")
    args = parser.parse_args()
    main(args)
