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
  //bool _busy = false;
  tfl.Interpreter interpreter;
  List<String> labels;
  //List<int> _imageShape = [1, 832, 832, 3];
  TensorImage _inputImage = TensorImage(tfl.TfLiteType.float32);
  List<int> _inputShape;
  List<List<int>> _outputShape;
  List<tfl.TfLiteType> _outputType;
  tfl.TfLiteType _inputType;
  img.Image _image;
  //TensorBuffer _outputBuffer;

  Future getImage() async {
    var pickedImage = await picker.getImage(
      source: ImageSource.gallery,
    );
    //final File file = File(pickedImage.path);
    _image = img.decodeImage(File(pickedImage.path).readAsBytesSync());

    //final File file = File(pickedImage.path);
    // setState(() {
    //   _busy = true;
    // });
    predict(_image);
  }

  Future imageResize(var image) async {
    try {
      //int cropSize;
      ImageProcessor imageProcessor = ImageProcessorBuilder()
          .add(ResizeWithCropOrPadOp(
            4000,
            4000,
          ))
          .add(ResizeOp(
              _inputShape[1], _inputShape[2], ResizeMethod.NEAREST_NEIGHBOUR))
          .build();
      _inputImage.loadImage(image);
      //cropSize = min(_inputImage.width, _inputImage.height);
      _inputImage = imageProcessor.process(_inputImage);
      print(_inputImage.height);
      print(_inputImage.width);
    } on Exception catch (e) {
      print('failed to resize the image: $e');
    }
//resize operation
  }

  Future loadModel() async {
    try {
      interpreter = await tfl.Interpreter.fromAsset('yolov4-416.tflite');
      labels = await FileUtil.loadLabels('assets/labels.txt');
      print(labels);
      _inputShape = interpreter.getInputTensor(0).shape;
      _inputType = interpreter.getInputTensor(0).type;
      print(_inputType);
      // _outputShape = interpreter.getOutputTensor(0).shape;
      // _outputType = interpreter.getOutputTensor(0).type;
      var _outputTensors = interpreter.getOutputTensors();
      _outputShape = [];
      _outputType = [];
      _outputTensors.forEach((element) {
        _outputShape.add(element.shape);
        _outputType.add(element.type);
      });
      //_outputBuffer = TensorBuffer.createFixedSize(_outputShape, _outputType);
      print(_inputShape);
      print(_outputShape[0][2]);
      print('load model sucess!');
    } on Exception catch (e) {
      print('Error while loading the model: $e');
    }
  }

  Future predict(var image) async {
    await loadModel();
    await imageResize(image);
    TensorBuffer outputLocations = TensorBufferFloat(_outputShape[0]);

    List<List<List<double>>> outputClassScores = new List.generate(
        _outputShape[1][0],
        (_) => new List.generate(
            _outputShape[1][1], (_) => new List.filled(_outputShape[1][2], 0.0),
            growable: false),
        growable: false);

    Map<int, Object> outputs = {
      0: outputLocations.buffer,
      1: outputClassScores,
    };
    List<Object> inputs = [_inputImage.buffer];
    // print(_inputImage.height);
    // print(_inputImage.width);

    interpreter.runForMultipleInputs(inputs, outputs);
    print('model ran');

    List<Rect> locations = BoundingBoxUtils.convert(
      tensor: outputLocations,
      //valueIndex: [1, 0, 3, 2], Commented out because default order is needed.
      boundingBoxAxis: 2,
      boundingBoxType: BoundingBoxType.CENTER,
      coordinateType: CoordinateType.PIXEL,
      height: 832,
      width: 832,
    );
    //print(_outputShape[0][1]);
    // print(locations[1]);
    //print(outputClassScores[0][1][0]);
    // print(locations[2]);
    // print(outputClassScores[2]);
    // print(locations[3]);
    // print(outputClassScores[3]);
    // print(locations[4]);
    // print(outputClassScores[4]);
    // print(locations[5]);
    // print(outputClassScores[5]);
    // print(locations[6]);
    // print(outputClassScores[6]);
    // print(locations[7]);
    // print(outputClassScores[7]);
    // print(locations[8]);
    // print(outputClassScores[8]);
    // print(locations[9]);
    // print(outputClassScores[9]);
    // print(locations[10]);
    // print(outputClassScores[10]);

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
        if (outputClassScores[0][i][c] > maxClassScore) {
          labelIndex = c;
          maxClassScore = outputClassScores[0][i][c];
          print(maxClassScore);
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

        ImageProcessor imageProcessor = ImageProcessorBuilder()
            .add(ResizeWithCropOrPadOp(1000, 1000))
            .add(ResizeOp(832, 832, ResizeMethod.BILINEAR))
            .build();

        // Gets the coordinates based on the original image if anything was done to it.
        Rect transformedRect = imageProcessor.inverseTransformRect(
            rectAti, image.height, image.width);

        recognitions.add(
          Recognition(i, label, score, transformedRect),
        );
        print(recognitions);
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
      body: SingleChildScrollView(
        child: Column(
          children: <Widget>[
            Container(
              padding: EdgeInsets.all(20.0),
              // child: _image == null
              //     ? Text('No image to show!')
              //     : Image.file(_image),
            ),
            new RaisedButton(
              onPressed: () {},
              child: Text('Save'),
            )
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          getImage();
        },
        tooltip: 'Pick Image',
        child: Icon(Icons.add_a_photo),
      ),
    );
  }
}
