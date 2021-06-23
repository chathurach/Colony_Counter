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
import 'package:loading_overlay/loading_overlay.dart';
import 'package:yolo_tflite/nms.dart';
import 'package:yolo_tflite/runPredictions.dart';

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

  Future getImage() async {
    var pickedImage = await picker.getImage(
      source: ImageSource.gallery,
      maxWidth: 1500,
      maxHeight: 1500,
    );

    _image = File(pickedImage.path);
    _imageWidget = Image.file(_image);
    imageInput = img.decodeImage(_image.readAsBytesSync());
    _imageWidth = imageInput.width.toDouble();
    _imageHeight = imageInput.height.toDouble();
    Size imageSize = Size(_imageWidth.toDouble(), _imageHeight.toDouble());
    CameraViewSingleton.inputImageSize = imageSize;
    double ratio = imageSize.width / imageSize.height;
    double scWidth = MediaQuery.of(context).size.width;
    double scHeigth = scWidth * ratio;
    Size screenSize = Size(scWidth, scHeigth);
    CameraViewSingleton.screenSize = screenSize;
    CameraViewSingleton.ratioY = screenSize.width / imageSize.height;
    CameraViewSingleton.ratioX = screenSize.height / imageSize.width;
    //results = await compute(predict(), imageInput);

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
      boundingBoxAxis: 2,
      boundingBoxType: BoundingBoxType.CENTER,
      coordinateType: CoordinateType.PIXEL,
      height: inputSize,
      width: inputSize,
    );

    //List<Recognition> recognitions = [];

    var gridWidth = _outputShape[0][1];

    Map map = Map();
    map['val1'] = labels;
    map['val2'] = gridWidth;
    map['val3'] = outputClass;
    map['val4'] = locations;
    map['val5'] = imageProcessor;
    map['val6'] = _imageHeight;
    map['val7'] = _imageWidth;
    map['val8'] = inputSize;

    var fromCompute = await compute(getRecognitions, map);

    // for (int i = 0; i < gridWidth; i++) {
    //   // Since we are given a list of scores for each class for
    //   // each detected Object, we are interested in finding the class
    //   // with the highest output score
    //   var maxClassScore = 0.00;
    //   var labelIndex = -1;

    //   for (int c = 0; c < labels.length; c++) {
    //     // output[0][i][c] is the confidence score of c class
    //     if (outputClass[0][i][c] > maxClassScore) {
    //       labelIndex = c;
    //       maxClassScore = outputClass[0][i][c];
    //     }
    //   }
    //   // Prediction score
    //   var score = maxClassScore;

    //   var label;
    //   if (labelIndex != -1) {
    //     // Label string
    //     label = labels.elementAt(labelIndex);
    //     //print(label);
    //   } else {
    //     label = null;
    //   }
    //   // Makes sure the confidence is above the
    //   // minimum threshold score for each object.
    //   if (score > 0.4) {
    //     // inverse of rect
    //     // [locations] corresponds to the inputSize
    //     // inverseTransformRect transforms it our [inputImage]

    //     Rect rectAti = Rect.fromLTRB(
    //         max(0, locations[i].left),
    //         max(0, locations[i].top),
    //         min(inputSize + 0.0, locations[i].right),
    //         min(inputSize + 0.0, locations[i].bottom));

    //     // Gets the coordinates based on the original image if anything was done to it.
    //     Rect transformedRect = imageProcessor.inverseTransformRect(
    //       rectAti,
    //       _imageHeight.toInt(),
    //       _imageWidth.toInt(),
    //     );

    //     recognitions.add(
    //       //await compute<[int, String, double, Rect],Rect>(Recognition,[i,label,score,transformedRect]);
    //       Recognition(i, label, score, transformedRect),
    //     );
    //   }
    //}
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
      body: LoadingOverlay(
        isLoading: _busy,
        opacity: 1.0,
        color: Colors.white,
        child: imageShow(context),
        progressIndicator: CircularProgressIndicator(),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          setState(() {
            _busy = false;
          });
          getImage();
        },
        tooltip: 'Pick Image',
        child: Icon(Icons.add_a_photo),
      ),
    );
  }

  /// Returns Stack of bounding boxes
  Widget boundingBoxes(List<Recognition> results) {
    if (results == null) {
      return Container();
    }
    return Stack(
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
      child = Stack(
        children: <Widget>[
          _imageWidget,
          boundingBoxes(results),
          Align(
            alignment: Alignment.bottomCenter,
            child: Container(
              height: MediaQuery.of(context).size.height / 4,
              width: MediaQuery.of(context).size.width,
              child: Center(
                child: Text(
                  "Colony Count - " + results.length.toString(),
                  textScaleFactor: 1.5,
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
              color: Colors.white,
            ),
          ),
        ],
      );
    } else {
      child = Center(
          child: Text(
        'Take a picture!',
        textScaleFactor: 1.5,
        style: TextStyle(
          fontWeight: FontWeight.bold,
        ),
      ));
    }

    return new Container(child: child);
  }
}
