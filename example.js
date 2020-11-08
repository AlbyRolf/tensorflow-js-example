let webcamElement = document.getElementById('camera');
let labels = [];
let xs;
let ys;
let mobilenet;
let model;
let array = Array.from(Array(10), () => 0);
let isPredicting = false;
const optimizer = tf.train.adam(0.0001);

//================================= Web camera and MobileNet initialization =================================

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
        if (navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia(
                {video: {width: 224, height: 224}}).then(stream => {
                webcamElement.srcObject = stream;
                webcamElement.addEventListener('loadeddata', async () => {
                    adjustVideoSize(
                        webcamElement.videoWidth,
                        webcamElement.videoHeight);
                    resolve();
                }, false);
            }).catch(error => {
                reject(error);
            });
        } else {
            reject();
        }
    });
}

async function loadMobilenet() {
    const mobileNetModel = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json');
    const layer = mobileNetModel.getLayer('conv_pw_13_relu');
    mobilenet = tf.model({inputs: mobileNetModel.inputs, outputs: layer.output});
}

//================================= Frame capturing =========================================================

function cropImage(img) {
    const size = Math.min(img.shape[0], img.shape[1]);
    const centerHeight = img.shape[0] / 2;
    const beginHeight = centerHeight - (size / 2);
    const centerWidth = img.shape[1] / 2;
    const beginWidth = centerWidth - (size / 2);
    return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
}

function capture() {
    return tf.tidy(() => {
        const webcamImage = tf.browser.fromPixels(webcamElement);
        const reversedImage = webcamImage.reverse(1);
        const croppedImage = cropImage(reversedImage);
        const batchedImage = croppedImage.expandDims(0);
        return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
    });
}

//================================= Training model ==========================================================

function encodeLabels(numClasses) {
    for (let i = 0; i < labels.length; i++) {
        const y = tf.tidy(
            () => {
                return tf.oneHot(tf.tensor1d([labels[i]]).toInt(), numClasses)
            });
        if (ys == null) {
            ys = tf.keep(y);
        } else {
            const oldY = ys;
            ys = tf.keep(oldY.concat(y, 0));
            oldY.dispose();
            y.dispose();
        }
    }
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

function doTraining() {
    train();
    alert("Training Done!")
}

//================================= Single data sample addition =============================================

function addExample(example, label) {
    if (xs == null) {
        xs = tf.keep(example);
    } else {
        const oldX = xs;
        xs = tf.keep(oldX.concat(example, 0));
        oldX.dispose();
    }
    labels.push(label);
}

function handleButton(elem) {
    let label = parseInt(elem.id);
    array[label]++;
    document.getElementById("samples_" + elem.id).innerText = "" + array[label];
    const img = capture();
    addExample(mobilenet.predict(img), label);
}

//================================= Inference process =======================================================

async function predict() {
    while (isPredicting) {
        const predictedClass = tf.tidy(() => {
            const img = capture();
            const activation = mobilenet.predict(img);
            const predictions = model.predict(activation);
            return predictions.as1D().argMax();
        });
        document.getElementById("prediction").innerText = (await predictedClass.data())[0];
        predictedClass.dispose();
        await tf.nextFrame();
    }
}

function setPredicting(predicting) {
    isPredicting = predicting;
    predict();
}

//================================= Aux functions ===========================================================

function saveModel() {
    model.save('downloads://my_model');
}

async function init() {
    await setup();
    await loadMobilenet();
}

init();