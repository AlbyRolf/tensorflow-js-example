let webcamElement = document.getElementById('camera');
let labels = [];
let xs = null;
let ys = null;
let mobilenet;
let model;
let array = Array.from(Array(10), () => 0);
let isPredicting = false;
const optimizer = tf.train.adam(0.0001);

function capture() {
    return tf.tidy(() => {
        const webcamImage = tf.browser.fromPixels(webcamElement);
        const reversedImage = webcamImage.reverse(1);
        const croppedImage = cropImage(reversedImage);
        const batchedImage = croppedImage.expandDims(0);
        return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
    });
}

function cropImage(img) {
    const size = Math.min(img.shape[0], img.shape[1]);
    const centerHeight = img.shape[0] / 2;
    const beginHeight = centerHeight - (size / 2);
    const centerWidth = img.shape[1] / 2;
    const beginWidth = centerWidth - (size / 2);
    return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
}

function adjustVideoSize(width, height) {
    const aspectRatio = width / height;
    if (width >= height) {
        webcamElement.width = aspectRatio * webcamElement.height;
    } else if (width < height) {
        webcamElement.height = webcamElement.width / aspectRatio;
    }
}

async function setup() {
    return new Promise((resolve, reject) => {
        navigator.getUserMedia = navigator.getUserMedia ||
            navigator.webkitGetUserMedia || navigator.mozGetUserMedia ||
            navigator.msGetUserMedia;
        if (navigator.getUserMedia) {
            navigator.getUserMedia(
                {video: {width: 224, height: 224}},
                stream => {
                    webcamElement.srcObject = stream;
                    webcamElement.addEventListener('loadeddata', async () => {
                        adjustVideoSize(
                            webcamElement.videoWidth,
                            webcamElement.videoHeight);
                        resolve();
                    }, false);
                },
                error => {
                    reject(error);
                });
        } else {
            reject();
        }
    });
}

function addExample(example, label) {
    if (xs == null) {
        xs = tf.keep(example);
        labels.push(label);
    } else {
        const oldX = xs;
        xs = tf.keep(oldX.concat(example, 0));
        labels.push(label);
        oldX.dispose();
    }
}

function encodeLabels(numClasses) {
    for (let i = 0; i < labels.length; i++) {
        if (ys == null) {
            ys = tf.keep(tf.tidy(
                () => {
                    return tf.oneHot(
                        tf.tensor1d([labels[i]]).toInt(), numClasses)
                }));
        } else {
            const y = tf.tidy(
                () => {
                    return tf.oneHot(
                        tf.tensor1d([labels[i]]).toInt(), numClasses)
                });
            const oldY = ys;
            ys = tf.keep(oldY.concat(y, 0));
            oldY.dispose();
            y.dispose();
        }
    }
}

async function loadMobilenet() {
    const mobileNetModel = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json');
    const layer = mobileNetModel.getLayer('conv_pw_13_relu');
    mobilenet = tf.model({inputs: mobileNetModel.inputs, outputs: layer.output});
}

async function train() {
    ys = null;
    encodeLabels(10);
    model = tf.sequential({
        layers: [
            tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
            tf.layers.dense({units: 100, activation: 'relu'}),
            tf.layers.dense({units: 10, activation: 'softmax'})
        ]
    });

    model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});
    let loss = 0;
    model.fit(xs, ys, {
        epochs: 10,
        callbacks: {
            onBatchEnd: async (batch, logs) => {
                loss = logs.loss.toFixed(5);
                console.log('LOSS: ' + loss);
            }
        }
    });
}


function handleButton(elem) {
    let label = parseInt(elem.id);
    array[label]++;
    document.getElementById("samples_" + elem.id).innerText = elem.id + ": " + array[label];
    const img = capture();
    addExample(mobilenet.predict(img), label);
}

async function predict() {
    while (isPredicting) {
        const predictedClass = tf.tidy(() => {
            const img = capture();
            const activation = mobilenet.predict(img);
            const predictions = model.predict(activation);
            return predictions.as1D().argMax();
        });
        const classId = (await predictedClass.data())[0];
        document.getElementById("prediction").innerText = "I see " + classId;
        predictedClass.dispose();
        await tf.nextFrame();
    }
}


function doTraining() {
    train();
    alert("Training Done!")
}

function startPredicting() {
    isPredicting = true;
    predict();
}

function stopPredicting() {
    isPredicting = false;
    predict();
}

function saveModel() {
    model.save('downloads://my_model');
}

async function init() {
    await setup();
    await loadMobilenet();
    tf.tidy(() => mobilenet.predict(capture()));
}

init();