import 'dart:async';

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

import 'package:yolo_tflite/runPredictions.dart';
import 'package:loading_animations/loading_animations.dart';

void main() => runApp(MaterialApp(
      home: Home(),
      debugShowCheckedModeBanner: false,
    ));

class Home extends StatefulWidget {
  @override
  _HomeState createState() => _HomeState();
}

class _HomeState extends State<Home> {
  final picker = ImagePicker();
  List<Recognition> results;
  tfl.Interpreter interpreter;
  List<String> labels;
  TensorImage _inputImage = TensorImage(tfl.TfLiteType.float32);
  List<int> _inputShape;
  List<List<int>> _outputShape;
  List<tfl.TfLiteType> _outputType;
  File _image;
  double _imageWidth;
  double _imageHeight;
  Image _imageWidget;
  img.Image imageInput;
  bool _busy = false;
  int inputSize = 1024; //input image size for the tflite model
  double normalMean = 127.5;
  double normalStd = 127.5;
  ImageProcessor imageProcessor;

  Future getImage(ImageSource key) async {
    var pickedImage = await picker.getImage(
      source: key,
      maxWidth: 1500,
      maxHeight: 1500,
    );

    if (pickedImage != null) {
      _image = File(pickedImage.path);
      _imageWidget = Image.file(_image);
      imageInput = img.decodeImage(_image.readAsBytesSync());
      _imageWidth = imageInput.height.toDouble();
      _imageHeight = imageInput.width.toDouble();

      Size imageSize = Size(_imageWidth.toDouble(), _imageHeight.toDouble());
      CameraViewSingleton.inputImageSize = imageSize;
      double ratio = imageSize.height / imageSize.width;
      double scWidth = MediaQuery.of(context).size.width;

      double scHeigth = scWidth * ratio;

      Size screenSize = Size(scWidth, scHeigth);
      CameraViewSingleton.screenSize = screenSize;
      CameraViewSingleton.ratioX = screenSize.width / imageSize.width;

      setState(() {
        _busy = true;
      });

// give some time to load the loading screen
      await Future.delayed(Duration(microseconds: 200), () {
        predict(imageInput).then((value) => {
              setState(() {
                results = value;
                _busy = false;
              })
            });
      });
    }
  }

  Future imageResize(img.Image image) async {
    try {
      _inputImage.loadImage(image);
      _inputImage = imageProcess(_inputImage);
    } on Exception catch (e) {
      print('failed to resize the image: $e');
    }
//resize operation
  }

  TensorImage imageProcess(TensorImage inputTensorImage) {
    int cropSize = min(_imageHeight.toInt(), _imageWidth.toInt());
    imageProcessor = ImageProcessorBuilder()
        .add(ResizeWithCropOrPadOp(
          cropSize,
          cropSize,
        ))
        .add(ResizeOp(_inputShape[1], _inputShape[2], ResizeMethod.BILINEAR))
        .add(NormalizeOp(normalMean, normalStd))
        .build();
    inputTensorImage = imageProcessor.process(inputTensorImage);
    return inputTensorImage;
  }

  Future loadModel() async {
    try {
      interpreter = await tfl.Interpreter.fromAsset(
        'yolov4-416.tflite',
        options: tfl.InterpreterOptions()..threads = 4,
      );
      labels = await FileUtil.loadLabels('assets/labels.txt');

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
  }

// Main function starts to load the model and start the prediction process

  Future<List<Recognition>> predict(img.Image image) async {
    await loadModel();
    await imageResize(image);
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
      valueIndex: [0, 1, 3, 2],
      boundingBoxAxis: 2,
      boundingBoxType: BoundingBoxType.CENTER,
      coordinateType: CoordinateType.PIXEL,
      height: inputSize,
      width: inputSize,
    );

    var gridWidth = _outputShape[0][1];
    // Put variables in to a map to feed to the isolate
    Map map = Map();
    map['val1'] = labels;
    map['val2'] = gridWidth;
    map['val3'] = outputClass;
    map['val4'] = locations;
    map['val5'] = imageProcessor;
    map['val6'] = _imageHeight;
    map['val7'] = _imageWidth;
    map['val8'] = inputSize;

    //Spin ups a isolate to calculate bounding boxes
    var fromCompute = await compute(getRecognitions, map);

    // End of for loop and added all recognitions
    interpreter.close();
    return fromCompute;
  }

  @override
  Widget build(BuildContext context) {
    SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
    return Scaffold(
      appBar: AppBar(
        title: Center(
          child: Text('Colony Count'),
        ),
      ),
      body: imageShow(context),
    );
  }

  /// Returns Stack of bounding boxes
  Widget boundingBoxes(List<Recognition> results) {
    if (results == null) {
      return Container();
    }
    return Stack(
      alignment: Alignment.topLeft,
      children: results
          .map((e) => BoxWidget(
                result: e,
              ))
          .toList(),
    );
  }

  Widget imageShow(BuildContext context) {
    Widget child;
    if (_imageWidget != null && !_busy) {
      child = Container(
        height: MediaQuery.of(context).size.height,
        child: Column(
          children: [
            Container(
              alignment: Alignment.topLeft,
              width: MediaQuery.of(context).size.width,
              //height: _imageHeight,
              height: _imageHeight *
                  MediaQuery.of(context).size.width /
                  _imageWidth,
              child: Stack(
                alignment: Alignment.topLeft,
                children: <Widget>[
                  _imageWidget,
                  boundingBoxes(results),
                ],
              ),
            ),
            Expanded(
              child: Column(
                children: [
                  Text(
                    "Colony Count - " + results.length.toString(),
                    textScaleFactor: 1.5,
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Center(
                    child: FloatingActionButton(
                      onPressed: () {
                        setState(() {
                          _imageWidget = null;
                          results = null;
                          //_busy = false;
                        });
                      },
                      tooltip: 'Back',
                      child: Icon(Icons.arrow_back_rounded),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      );
    } else if (!_busy && _imageWidget == null) {
      child = Column(
        children: [
          Expanded(
            child: Center(
              child: Text(
                'Take a picture!',
                textScaleFactor: 1.5,
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ),
          Padding(
            padding: EdgeInsets.all(25.0),
            child: Row(
              children: [
                Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: FloatingActionButton(
                    onPressed: () {
                      getImage(ImageSource.gallery);
                    },
                    tooltip: 'Pick Image Using Gallery',
                    child: Icon(Icons.add_photo_alternate),
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: FloatingActionButton(
                    onPressed: () {
                      getImage(ImageSource.camera);
                    },
                    tooltip: 'Pick Image Using Camera',
                    child: Icon(Icons.add_a_photo),
                  ),
                ),
              ],
            ),
          )
        ],
      );
    } else {
      child = Center(
        child: LoadingBouncingGrid.circle(),
      );
    }

    return new Container(child: child);
  }
}
