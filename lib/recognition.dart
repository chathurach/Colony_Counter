import 'dart:math';

import 'package:flutter/cupertino.dart';
import 'package:yolo_tflite/camera_view_singleton.dart';

/// Represents the recognition output from the model
class Recognition implements Comparable<Recognition> {
  /// Index of the result
  int _id;

  /// Label of the result
  String _label;

  /// Confidence [0.0, 1.0]
  double _score;

  /// Location of bounding box rect
  ///
  /// The rectangle corresponds to the raw input image
  /// passed for inference
  Rect _location;

  Recognition(this._id, this._label, this._score, [this._location]);

  int get id => _id;

  String get label => _label;

  double get score => _score;

  Rect get location => _location;

  /// Returns bounding box rectangle corresponding to the
  /// displayed image on screen
  ///
  /// This is the actual location where rectangle is rendered on
  /// the screen
  Rect get renderLocation {
    // ratioX = screenWidth / imageInputWidth
    // ratioY = ratioX if image fits screenWidth with aspectRatio = constant

    double ratioX = CameraViewSingleton.ratioX;
    double ratioY = ratioX;

    double transLeft = max(0.1,
        CameraViewSingleton.actualPreviewSize.width - location.bottom * ratioX);
    double transTop = max(0.1, (location.right * ratioY) - 17.0);
    double transWidth = min(
        location.width * ratioX, CameraViewSingleton.actualPreviewSize.width);
    double transHeight = min(
        location.height * ratioY, CameraViewSingleton.actualPreviewSize.height);
    // double transWidth = 832;
    // double transHeight = 832;

    Rect transformedRect =
        Rect.fromLTWH(transLeft, transTop, transWidth, transHeight);
    // print(transformedRect);
    // print(
    //     CameraViewSingleton.actualPreviewSize.width - location.bottom * ratioX);
    return transformedRect;
  }

  @override
  String toString() {
    return 'Recognition(id: $id, label: $label, score: ${(score * 100).toStringAsPrecision(3)}, location: $location)';
  }

  @override
  int compareTo(Recognition other) {
    if (this.score == other.score) {
      return 0;
    } else if (this.score > other.score) {
      return -1;
    } else {
      return 1;
    }
  }
}
