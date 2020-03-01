 
using Microsoft.ML.Data;

namespace TsForecast
{
    public class MeanPrediction
    {
        [ColumnName("Mean")]
        public float Mean { get; set; }
    }
}