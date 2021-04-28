import tflite_runtime.interpreter as tflite
import time
import os
import numpy as np
from tablutconfig import TablutConfig

if __name__ == '__main__':
    config = TablutConfig()
    folder = config.folder
    filename = config.tflite_model
    model_path = os.path.join(folder, filename)

    interpreter = tflite.Interpreter(model_path, num_threads=4)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    n = 100
    sum = 0

    for i in range(n):

        startTime = time.time()
        interpreter.invoke()
        deltaTime = time.time()-startTime
        sum += deltaTime

    print("Inference time: {0} ms ({3} invokes), Input details: {1}, Output details: {2}".format((sum/n)*1000, input_details, output_details, n))