import * as tf from '@tensorflow/tfjs';
import {generateData} from './data';
import {plotData, plotDataAndPredictions, renderCoefficients} from './ui';

/*
Step 1
*/
const a = tf.variable(tf.scalar(Math.random()));
const b = tf.variable(tf.scalar(Math.random()));
const c = tf.variable(tf.scalar(Math.random()));
const d = tf.variable(tf.scalar(Math.random()));

/*
Step 2
*/
function predict(x) {
    // y = a * x ^ 3 + b * x ^ 2 + c * x + d
    return tf.tidy(() => {
      return a.mul(x.pow(tf.scalar(3))) // a * x^3
        .add(b.mul(x.square())) // + b * x ^ 2
        .add(c.mul(x)) // + c * x
        .add(d); // + d
    });
  }

/* 
Step 3
*/

const numIterations = 75;
const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);

function loss(predictions, labels) {
  // Subtract our labels (actual values) from predictions, square the results,
  // and take the mean.
  const meanSquareError = predictions.sub(labels).square().mean();
  return meanSquareError;
}

function train(xs, ys, numIterations = 75) {

  const learningRate = 0.5;
  const optimizer = tf.train.sgd(learningRate);

  for (let iter = 0; iter < numIterations; iter++) {
    optimizer.minimize(() => {
      const predsYs = predict(xs);
      return loss(predsYs, ys);
    });
  }
}






/* 
        DON'T TOUCH.
        plots the graphs of each step.
*/
async function learnCoefficients() {
  const trueCoefficients = {a: -.8, b: -.2, c: .9, d: .5};
  const trainingData = generateData(100, trueCoefficients);

  // Plot original data
  renderCoefficients('#data .coeff', trueCoefficients);
  await plotData('#data .plot', trainingData.xs, trainingData.ys)

  // See what the predictions look like with random coefficients
  renderCoefficients('#random .coeff', {
    a: a.dataSync()[0],
    b: b.dataSync()[0],
    c: c.dataSync()[0],
    d: d.dataSync()[0],
  });
  const predictionsBefore = predict(trainingData.xs);
  await plotDataAndPredictions(
      '#random .plot', trainingData.xs, trainingData.ys, predictionsBefore);

  // Train the model!
  await train(trainingData.xs, trainingData.ys, numIterations);

  // See what the final results predictions are after training.
  renderCoefficients('#trained .coeff', {
    a: a.dataSync()[0],
    b: b.dataSync()[0],
    c: c.dataSync()[0],
    d: d.dataSync()[0],
  });
  const predictionsAfter = predict(trainingData.xs);
  await plotDataAndPredictions(
      '#trained .plot', trainingData.xs, trainingData.ys, predictionsAfter);

  predictionsBefore.dispose();
  predictionsAfter.dispose();
}


learnCoefficients();
  


