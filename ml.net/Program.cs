using System;
using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;


namespace TsForecast
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            var trainData = context
                .Data
                .LoadFromTextFile<SolarData>("./data/solar.csv", hasHeader: true, separatorChar: ',');

            var dataPipeline = context.Transforms.Concatenate("f");

            var forecastingPipeline = context.Forecasting.ForecastBySsa(
                outputColumnName: "Mean",
                inputColumnName: "Mean",
                windowSize: 7,
                seriesLength: 30,
                trainSize: 365,
                horizon: 7,
                confidenceLevel: 0.95f
            );


        }
    }
}
