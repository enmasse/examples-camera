#    Copyright 2019 Google LLC
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        https://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""A demo to run the detector in a Pygame camera stream."""
import argparse
import os
import io
import time
import re
from collections import deque
import numpy as np
import pygame
import pygame.camera
from pygame.locals import *

#from edgetpu.detection.engine import DetectionEngine
from tflite_runtime.interpreter import Interpreter

def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def detect_in_image(interpreter, image):
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()
  output = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
  label = np.squeeze(interpreter.get_tensor(output_details[1]['index']))
  score = np.squeeze(interpreter.get_tensor(output_details[2]['index']))
 
  return [(label[i], score[i], output[i]) for i in range(len(label))]

def load_labels(path):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(path, 'r', encoding='utf-8') as f:
       lines = (p.match(line).groups() for line in f.readlines())
       return {int(num): text.strip() for num, text in lines}

def main():
    cam_w, cam_h = 640, 480
    default_model_dir = "../../all_models"
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess.tflite'
    default_labels = 'coco_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=1,
                        help='number of classes with highest score to display')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='class score threshold')
    args = parser.parse_args()

    with open(args.labels, 'r') as f:
        pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
        labels = dict((int(k), v) for k, v in pairs)

    print("Loading %s with %s labels."%(args.model, args.labels))
    interpreter = Interpreter(args.model)
    interpreter.allocate_tensors()
    labels = load_labels(args.labels)

    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont("Arial", 20)

    pygame.camera.init()
    camlist = pygame.camera.list_cameras()

    _, w, h, c = interpreter.get_input_details()[0]['shape']
    camera = pygame.camera.Camera(camlist[0], (cam_w, cam_h)) 
    display = pygame.display.set_mode((cam_w, cam_h), 0)

    red = pygame.Color(255, 0, 0)

    camera.start()
    try:
        last_time = time.monotonic()
        while True:
            mysurface = camera.get_image()
            imagen = pygame.transform.scale(mysurface, (w, h))
            input = np.frombuffer(imagen.get_buffer(), dtype=np.uint8)
            input = np.reshape(input, [w, h, c])
            start_time = time.monotonic()
            results = detect_in_image(interpreter, input)
            stop_time = time.monotonic()
            inference_ms = (stop_time - start_time)*1000.0
            fps_ms = 1.0 / (stop_time - last_time)
            last_time = stop_time
            annotate_text = "Inference: %5.2fms FPS: %3.1f" % (inference_ms, fps_ms)
            for (label, score, output) in results: 
               if score < args.threshold:
                  continue
               y0, x0, y1, x1 = output
               rect = pygame.Rect(x0 * cam_w, y0 * cam_h, (x1 - x0) * cam_w, (y1 - y0) * cam_h)
               pygame.draw.rect(mysurface, red, rect, 1)
               label = "%.0f%% %s" % (100*score, labels[label])
               text = font.render(label, True, red)
               mysurface.blit(text, (x0 * cam_w , y0 * cam_h))
            text = font.render(annotate_text, True, red)
            mysurface.blit(text, (0, 0))
            display.blit(mysurface, (0, 0))
            pygame.display.flip()
    finally:
        camera.stop()


if __name__ == '__main__':
    main()
