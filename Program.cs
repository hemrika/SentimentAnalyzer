using System;
using Microsoft.ML;

namespace SentimentAnalyzer
{
    class Program
    {
        static void Main(string[] args)
        {
            var sampleData = new SentimentData { SentimentText = "" };
            var mlContext = new MLContext();

            // Train or load model
            var model = ModelBuilder.BuildAndTrainModel(mlContext);

            Console.WriteLine("Enter text to analyze sentiment (type 'exit' to quit):");
            while (true)
            {
                var input = Console.ReadLine();
                if (input?.ToLower() == "exit") break;
                sampleData.SentimentText = input ?? "";

                var prediction = ModelBuilder.Predict(mlContext, model, sampleData);
                Console.WriteLine($"Prediction: {(prediction.Prediction ? "Positive" : "Negative")}, Probability: {prediction.Probability:P2}");
            }
        }
    }
}
