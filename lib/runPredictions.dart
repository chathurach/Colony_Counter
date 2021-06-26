import 'package:yolo_tflite/recognition.dart';

import 'package:flutter/material.dart';

import 'dart:math';

import 'package:yolo_tflite/nms.dart';

List<Recognition> getRecognitions(Map map) {
  List<Recognition> recognitions = [];

  final labels = map['val1'];
  final gridWidth = map['val2'];
  final outputClass = map['val3'];
  final locations = map['val4'];
  final imageProcessor = map['val5'];
  final _imageHeight = map['val6'];
  final _imageWidth = map['val7'];
  final inputSize = map['val8'];

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
          min(inputSize + 0.0, locations[i].right),
          min(inputSize + 0.0, locations[i].bottom));

      // Gets the coordinates based on the original image if anything was done to it.

      Rect transformedRect = imageProcessor.inverseTransformRect(
        rectAti,
        _imageWidth.toInt(),
        _imageHeight.toInt(),
      );

      recognitions.add(
        Recognition(i, label, score, transformedRect),
      );
    }
  }
  List<Recognition> nmsCalc = nms(recognitions, labels);
  return nmsCalc;
}
