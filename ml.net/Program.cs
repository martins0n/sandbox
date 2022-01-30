using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;

using System.IO;

namespace Iris
{

    public enum IrisLabel
    {
        setosa,
        versicolor,
        virginica
    }
    public class IrisData
    {
        [LoadColumn(0)]
        public float SepalLength;

        [LoadColumn(1)]
        public float SepalWidth;

        [LoadColumn(2)]
        public float PetalLength;

        [LoadColumn(3)]
        public float PetalWidth;

        [LoadColumn(4), ColumnName("Label")]
        public string Label;
    }

    public class IrisPrediction
    {
        [ColumnName("Label")]
        public uint Label;

        [ColumnName("Score")]
        public float[] Score;
    }
}

namespace mlnet
{
    class Program
    {
        static void Main(string[] args)
        {
            string _dataPath = Path.Combine(Environment.CurrentDirectory, "data", "iris.csv");

            var mlContext = new MLContext(seed: 0);

            IDataView dataView = mlContext.Data.LoadFromTextFile<Iris.IrisData>(_dataPath, hasHeader: false, separatorChar: ',');

            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            var model = mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression());
            var estimator = mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(
                    mlContext.Transforms.Concatenate("Features", new[] { "SepalLength", "SepalWidth", "PetalLength", "PetalWidth" })
                ).Append(model);

            var fittedPipe = estimator.Fit(splitDataView.TrainSet);

            var trainPredictions = fittedPipe.Transform(splitDataView.TrainSet);
            var testPredictions = fittedPipe.Transform(splitDataView.TestSet);

            var trainMetrics = mlContext.MulticlassClassification.Evaluate(trainPredictions);
            var testMetrics = mlContext.MulticlassClassification.Evaluate(testPredictions);
            Console.WriteLine($"Train Accuracy: {trainMetrics.MicroAccuracy}");
            Console.WriteLine($"Test Accuracy: {testMetrics.MicroAccuracy}");

        }
    }
}
