//import 'dart:html';

import 'dart:ffi';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
//import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';
import 'package:image/image.dart' as img;
//import 'dart:math';
//import 'package:collection/collection.dart';
import 'package:yolo_tflite/recognition.dart';
import 'dart:math';
import 'package:yolo_tflite/box_widget.dart';
import 'package:yolo_tflite/camera_view_singleton.dart';
import 'package:collection/collection.dart';

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
  //tfl.TfLiteType _inputType;
  File _image;
  double _imageWidth;
  double _imageHeight;
  Image _imageWidget;
  img.Image imageInput;

  //TensorBuffer _outputBuffer;

  Future getImage() async {
    var pickedImage = await picker.getImage(
      source: ImageSource.gallery,
      imageQuality: 100,
    );
    //final File file = File(pickedImage.path);

    _image = File(pickedImage.path);
    _imageWidget = Image.file(_image);
    imageInput = img.decodeImage(_image.readAsBytesSync());
    _imageWidth = imageInput.height.toDouble();
    _imageHeight = imageInput.width.toDouble();
    Size imageSize = Size(_imageWidth.toDouble(), _imageHeight.toDouble());
    CameraViewSingleton.inputImageSize = imageSize;
    double ratio = imageSize.width / imageSize.height;
    double scWidth = MediaQuery.of(context).size.width;
    double scHeigth = scWidth / ratio;
    Size screenSize = Size(scWidth, scHeigth);
    CameraViewSingleton.screenSize = screenSize;
    CameraViewSingleton.ratio = screenSize.width / imageSize.width;
    print(_imageWidth);
    print(_imageHeight);
    print(scWidth);
    print(scHeigth);
    print(ratio);

    //final File file = File(pickedImage.path);
    // setState(() {
    //   _busy = true;
    // });
    predict(imageInput);
  }

  Future imageResize(img.Image image) async {
    try {
      int cropSize = min(_imageHeight.toInt(), _imageWidth.toInt());
      print(cropSize);
      //_inputImage = TensorImage.fromImage(image);
      ImageProcessor imageProcessor = ImageProcessorBuilder()
          .add(ResizeWithCropOrPadOp(
            cropSize,
            cropSize,
          ))
          .add(ResizeOp(_inputShape[1], _inputShape[2], ResizeMethod.BILINEAR))
          .add(NormalizeOp(127, 127))
          .build();
      _inputImage.loadImage(image);
      //cropSize = min(_inputImage.width, _inputImage.height);
      _inputImage = imageProcessor.process(_inputImage);
    } on Exception catch (e) {
      print('failed to resize the image: $e');
    }
//resize operation
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

      print('load model sucess!');
    } on Exception catch (e) {
      print('Error while loading the model: $e');
    }
  }

  List<Recognition> nms(
      List<Recognition> list) // Turned from Java's ArrayList to Dart's List.
  {
    List<Recognition> nmsList = new List<Recognition>();

    for (int k = 0; k < labels.length; k++) {
      // 1.find max confidence per class
      PriorityQueue<Recognition> pq = new HeapPriorityQueue<Recognition>();
      for (int i = 0; i < list.length; ++i) {
        if (list[i].label == labels[k]) {
          // Changed from comparing #th class to class to string to string
          pq.add(list[i]);
        }
      }

      // 2.do non maximum suppression
      while (pq.length > 0) {
        // insert detection with max confidence
        List<Recognition> detections = pq.toList(); //In Java: pq.toArray(a)
        Recognition max = detections[0];
        nmsList.add(max);
        pq.clear();
        for (int j = 1; j < detections.length; j++) {
          Recognition detection = detections[j];
          Rect b = detection.location;
          if (boxIou(max.location, b) < 0.6) {
            pq.add(detection);
          }
        }
      }
    }

    return nmsList;
  }

  double boxIou(Rect a, Rect b) {
    return boxIntersection(a, b) / boxUnion(a, b);
  }

  double boxIntersection(Rect a, Rect b) {
    double w = overlap((a.left + a.right) / 2, a.right - a.left,
        (b.left + b.right) / 2, b.right - b.left);
    double h = overlap((a.top + a.bottom) / 2, a.bottom - a.top,
        (b.top + b.bottom) / 2, b.bottom - b.top);
    if ((w < 0) || (h < 0)) {
      return 0;
    }
    double area = (w * h);
    return area;
  }

  double boxUnion(Rect a, Rect b) {
    double i = boxIntersection(a, b);
    double u = ((((a.right - a.left) * (a.bottom - a.top)) +
            ((b.right - b.left) * (b.bottom - b.top))) -
        i);
    return u;
  }

  double overlap(double x1, double w1, double x2, double w2) {
    double l1 = (x1 - (w1 / 2));
    double l2 = (x2 - (w2 / 2));
    double left = ((l1 > l2) ? l1 : l2);
    double r1 = (x1 + (w1 / 2));
    double r2 = (x2 + (w2 / 2));
    double right = ((r1 < r2) ? r1 : r2);
    return right - left;
  }

  Future predict(img.Image image) async {
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
    print('model ran');

    List<Rect> locations = BoundingBoxUtils.convert(
      tensor: outputLocations,
      //valueIndex: [1, 0, 3, 2],
      boundingBoxAxis: 2,
      boundingBoxType: BoundingBoxType.CENTER,
      coordinateType: CoordinateType.PIXEL,
      height: 832,
      width: 832,
    );

    List<Recognition> recognitions = [];

    var gridWidth = _outputShape[0][1];
    //print("gridWidth = $gridWidth");

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
      if (score > 0.3) {
        // inverse of rect
        // [locations] corresponds to the image size 300 X 300
        // inverseTransformRect transforms it our [inputImage]

        Rect rectAti = Rect.fromLTRB(
            max(0, locations[i].left),
            max(0, locations[i].top),
            min(832 + 0.0, locations[i].right),
            min(832 + 0.0, locations[i].bottom));

        int cropSize = min(_imageHeight.toInt(), _imageWidth.toInt());
        //print(cropSize);
        //_inputImage = TensorImage.fromImage(image);
        ImageProcessor imageProcessor = ImageProcessorBuilder()
            .add(ResizeWithCropOrPadOp(
              cropSize,
              cropSize,
            ))
            .add(
                ResizeOp(_inputShape[1], _inputShape[2], ResizeMethod.BILINEAR))
            .add(NormalizeOp(127, 127))
            .build();

        // Gets the coordinates based on the original image if anything was done to it.
        Rect transformedRect = imageProcessor.inverseTransformRect(
          rectAti,
          _imageHeight.toInt(),
          _imageWidth.toInt(),
        );

        recognitions.add(
          Recognition(i, label, score, transformedRect),
        );

        setState(() {
          results = nms(recognitions);
        });
        //print(recognitions);
      }
    } // End of for loop and added all recognitions
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Center(
          child: Text('Colony Count'),
        ),
      ),
      body: imageShow(context),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
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
    if (_imageWidget != null) {
      child = Stack(
        children: <Widget>[
          _imageWidget,
          boundingBoxes(results),
        ],
      );
    } else {
      child = Text('No Image!');
    }
    return new Container(child: child);
  }
}
