using System;
using Microsoft.ML.Data;

namespace TsForecast
{
    public class SolarData
    {
        [LoadColumn(0), ColumnName("Source")]
        public string Source;

        [LoadColumn(1), ColumnName("Date")]
        public DateTime Time;

        [LoadColumn(2), ColumnName("Mean")]

        public  string Mean;
    }
}