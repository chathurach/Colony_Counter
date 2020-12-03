import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
//import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';
import 'package:image/image.dart' as img;
//import 'dart:math';
//import 'package:collection/collection.dart';

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
  //TensorBuffer _outputBuffer;

  Future getImage() async {
    var pickedImage = await picker.getImage(
      source: ImageSource.gallery,
    );
    //final File file = File(pickedImage.path);
    img.Image _image =
        img.decodeImage(File(pickedImage.path).readAsBytesSync());
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
            1000,
            1000,
          ))
          .add(ResizeOp(_inputShape[1], _inputShape[2], ResizeMethod.BILINEAR))
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
      print(_outputShape);
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
    //List<Object> inputs = [_inputImage.buffer];
    // print(_inputImage.height);
    // print(_inputImage.width);

    interpreter.run(_inputImage.getBuffer(), outputs);
    print('model ran');
    print(outputs);
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
