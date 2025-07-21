using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace SentimentAnalyzer
{
    public static class ModelBuilder
    {
        private static readonly string ModelPath = "sentiment-model.zip";
        private static readonly string DataPath = "sentiment-data.tsv";

        // Sample dataset for demo
        private static readonly string[] DefaultData = new[]
        {
            "I love this product!\ttrue",
            "This is the worst experience ever.\tfalse",
            "Absolutely fantastic service.\ttrue",
            "I hate it.\tfalse",
            "Not bad.\ttrue"
        };

        public static ITransformer BuildAndTrainModel(MLContext mlContext)
        {
            // Ensure data exists
            if (!File.Exists(DataPath))
                File.WriteAllLines(DataPath, DefaultData);

            var dataView = mlContext.Data.LoadFromTextFile<SentimentData>(
                path: DataPath, hasHeader: false, separatorChar: '\t');

            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(SentimentData.SentimentText))
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Sentiment", featureColumnName: "Features"));

            ITransformer model;
            if (File.Exists(ModelPath))
                model = mlContext.Model.Load(ModelPath, out _);
            else
            {
                model = pipeline.Fit(dataView);
                mlContext.Model.Save(model, dataView.Schema, ModelPath);
            }
            return model;
        }

        public static SentimentPrediction Predict(MLContext mlContext, ITransformer model, SentimentData input)
        {
            var predictionEngine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
            return predictionEngine.Predict(input);
        }
    }
}
