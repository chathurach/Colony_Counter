import 'package:yolo_tflite/recognition.dart';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';
import 'package:image/image.dart' as img;
import 'package:yolo_tflite/recognition.dart';
import 'dart:math';
import 'package:yolo_tflite/box_widget.dart';
import 'package:yolo_tflite/camera_view_singleton.dart';
import 'package:flutter/services.dart';
import 'package:loading_overlay/loading_overlay.dart';
import 'package:yolo_tflite/nms.dart';

Future<List<Recognition>> prediction(img.Image image) async {
  List<Recognition> results;
  tfl.Interpreter interpreter;
  List<String> labels;
  TensorImage _inputImage = TensorImage(tfl.TfLiteType.float32);
  List<int> _inputShape;
  List<List<int>> _outputShape;
  List<tfl.TfLiteType> _outputType;
  ImageProcessor returnProcess;
  // File _image;
  // double _imageWidth;
  // double _imageHeight;
  // Image _imageWidget;
  // img.Image imageInput;
  // bool _busy = false;
  // int inputSize = 1024; //input image size for the tflite model
  // double normalMean = 127.5;
  // double normalStd = 127.5;

  //Loading Model

  

  try {
    //interpreter = interpreter2;
    //final String lableLink = 'assets/labels.txt';
    // ignore: unnecessary_statements
    labels = await FileUtil.loadLabels('assets/labels.txt');

    interpreter = await tfl.Interpreter?.fromAsset(
      'yolov4-416.tflite',
      options: tfl.InterpreterOptions()..threads = 1,
    );

    _inputShape = interpreter.getInputTensor(0).shape;
    //_inputType = interpreter.getInputTensor(0).type;

    var _outputTensors = interpreter.getOutputTensors();
    _outputShape = [];
    _outputType = [];
    _outputTensors.forEach((tfl.Tensor tensor) {
      _outputShape.add(tensor.shape);
      _outputType.add(tensor.type);
    });

    //print('load model sucess!');
  } on Exception catch (e) {
    print('Error while loading the model: $e');
  }
//end of Model Loading

try {
    _inputImage.loadImage(image);
    returnProcess = await imageProcess(_inputShape);
    _inputImage = returnProcess.process(_inputImage);
  } on Exception catch (e) {
    print('failed to resize the image: $e');
  }

// Image resizing

  //end of Resizing Operation

  TensorBuffer outputLocations = TensorBufferFloat(_outputShape[0]);

  List<List<List<double>>> outputClass = new List.generate(
      _outputShape[1][0],
      (_) => new List.generate(
          _outputShape[1][1], (_) => new List.filled(_outputShape[1][2], 0.0),
          growable: false),
      growable: false);

  Map<int, Object> outputs = {
    0: outputLocations.buffer,
    1: outputClass,
  };
  List<Object> inputs = [_inputImage.buffer];

  interpreter.runForMultipleInputs(inputs, outputs);

  List<Rect> locations = BoundingBoxUtils.convert(
    tensor: outputLocations,
    boundingBoxAxis: 2,
    boundingBoxType: BoundingBoxType.CENTER,
    coordinateType: CoordinateType.PIXEL,
    // height: inputSize,
    // width: inputSize,
    height: 1024,
    width: 1024,
  );

  List<Recognition> recognitions = [];

  var gridWidth = _outputShape[0][1];

  for (int i = 0; i < gridWidth; i++) {
    // Since we are given a list of scores for each class for
    // each detected Object, we are interested in finding the class
    // with the highest output score
    var maxClassScore = 0.00;
    var labelIndex = -1;

    for (int c = 0; c < labels.length; c++) {
      // output[0][i][c] is the confidence score of c class
      if (outputClass[0][i][c] > maxClassScore) {
        labelIndex = c;
        maxClassScore = outputClass[0][i][c];
      }
    }
    // Prediction score
    var score = maxClassScore;

    var label;
    if (labelIndex != -1) {
      // Label string
      label = labels.elementAt(labelIndex);
      //print(label);
    } else {
      label = null;
    }
    // Makes sure the confidence is above the
    // minimum threshold score for each object.
    if (score > 0.4) {
      // inverse of rect
      // [locations] corresponds to the inputSize
      // inverseTransformRect transforms it our [inputImage]

      Rect rectAti = Rect.fromLTRB(
          max(0, locations[i].left),
          max(0, locations[i].top),
          // min(inputSize + 0.0, locations[i].right),
          // min(inputSize + 0.0, locations[i].bottom));
          min(1024 + 0.0, locations[i].right),
          min(1024 + 0.0, locations[i].bottom));

      // Gets the coordinates based on the original image if anything was done to it.
      Rect transformedRect = returnProcess.inverseTransformRect(
        rectAti,
        1024,
        1024,
        // _imageHeight.toInt(),
        // _imageWidth.toInt(),
      );

      recognitions.add(
        Recognition(i, label, score, transformedRect),
      );
    }
  }
  // End of for loop and added all recognitions
  interpreter.close();
  return nms(recognitions, labels);
}

Future<ImageProcessor> imageProcess(List<int> _inputShape) async {
  ImageProcessor imageProcessor;
  //int cropSize = min(_imageHeight.toInt(), _imageWidth.toInt());
  int cropSize = min(1024, 1024);
  imageProcessor = ImageProcessorBuilder()
      .add(ResizeWithCropOrPadOp(
        cropSize,
        cropSize,
      ))
      .add(ResizeOp(_inputShape[1], _inputShape[2], ResizeMethod.BILINEAR))
      .add(NormalizeOp(127.5, 127.5))
      .build();
  //inputTensorImage = imageProcessor.process(inputTensorImage);
  return imageProcessor;
}
