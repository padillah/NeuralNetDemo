using System;

namespace FeedForwardDemo
{
    public class NeuralNetwork
    {
        private readonly VectorLite _hiddenBiasVector;
        private readonly VectorLite _hiddenResultsVector;
        private VectorLite _inputsVector;
        private readonly VectorLite _outputBiasVector;
        private VectorLite _outputResultsVector;

        private readonly MatrixLite _hiddenToOutputMatrix;
        private readonly MatrixLite _inputToHiddenMatrix;

        private readonly int _numHidden;
        private readonly int _numInput;
        private readonly int _numOutput;

        public NeuralNetwork(int numInput, int numHidden, int numOutput)
        {
            _numInput = numInput;
            _numHidden = numHidden;
            _numOutput = numOutput;

            _inputsVector = new VectorLite(numInput);

            _inputToHiddenMatrix = new MatrixLite(numInput, numHidden);

            _hiddenBiasVector = new VectorLite(numHidden);

            _hiddenResultsVector = new VectorLite(numHidden);

            _hiddenToOutputMatrix = new MatrixLite(numHidden, numOutput);

            _outputBiasVector = new VectorLite(1, numOutput);

            _outputResultsVector = new VectorLite(numOutput);
        }

        public void SetWeights(VectorLite weights)
        {
            int numWeights = _numInput * _numHidden + _numHidden + _numHidden * _numOutput + _numOutput;
            if (weights.Length != numWeights)
            {
                throw new Exception("Bad weights array");
            }

            Console.WriteLine("Set the Input to Hidden weight");
            _inputToHiddenMatrix.SetValues(weights, 0);

            Console.WriteLine("Set the Hiddewn Biases");
            _hiddenBiasVector.SetValues(weights, _numInput * _numHidden);

            Console.WriteLine("Set Hidden to Output weights");
            _hiddenToOutputMatrix.SetValues(weights, _numInput * _numHidden + _numHidden);

            Console.WriteLine("Set output biases");
            _outputBiasVector.SetValues(weights, _numInput * _numHidden * _numOutput);

        }

        public VectorLite ComputeOutputs(VectorLite xValues)
        {
            if (xValues.Length != _numInput)
            {
                throw new Exception("Bad xValues array");
            }

            _inputsVector = new VectorLite(xValues);

            // ex: hSum[ 0] = (in[ 0] * ihW[[ 0][ 0]) + (in[ 1] * ihW[ 1][ 0]) + (in[ 2] * ihW[ 2][ 0]) + . . 
            // hSum[ 1] = (in[ 0] * ihW[[ 0][ 1]) + (in[ 1] * ihW[ 1][ 1]) + (in[ 2] * ihW[ 2][ 1]) + . . // . . .

            VectorLite hiddenSumVector = _inputsVector * _inputToHiddenMatrix;

            hiddenSumVector = hiddenSumVector + _hiddenBiasVector;

            Console.WriteLine("Pre-activation hidden sums:");
            FeedForwardProgram.ShowVector(hiddenSumVector, 4, true);

            //HiddenResultsVector = new VectorLite(hiddenSumVector.Length);
            for (int index = 0; index < _hiddenResultsVector.Length; index++ )
            {
                _hiddenResultsVector[index] = HyperTan(hiddenSumVector[index]);
            }

            Console.WriteLine("Hidden outputs:");
            FeedForwardProgram.ShowVector(_hiddenResultsVector, 4, true);

            VectorLite outputSumVector = _hiddenResultsVector * _hiddenToOutputMatrix;

            _outputResultsVector = outputSumVector + _outputBiasVector;

            Console.WriteLine("Pre-activation output sums:");
            FeedForwardProgram.ShowVector(_outputResultsVector, 4, true);

            VectorLite softOutMatrix = Softmax(_outputResultsVector);

            Console.WriteLine("Final output sums:");
            FeedForwardProgram.ShowVector(softOutMatrix, 4, true);

            return softOutMatrix;
        }

        private static double HyperTan(double v)
        {
            if (v < -20.0)
            {
                return -1.0;
            }
            if (v > 20.0)
            {
                return 1.0;
            }
            return Math.Tanh(v);
        }

        public static VectorLite Softmax(VectorLite oSums)
        {
            // Does all output nodes at once. // Determine max oSum. 
            double max = oSums[0];
            for (int i = 0; i < oSums.Length; ++i)
            {
                if (oSums[i] > max)
                {
                    max = oSums[i]; // Determine scaling factor -- sum of exp( each val - max).
                }
            }
            double scale = 0.0;
            for (int i = 0; i < oSums.Length; ++i)
            {
                scale += Math.Exp(oSums[i] - max);
            }

            VectorLite result = new VectorLite(oSums.Length);
            for (int i = 0; i < oSums.Length; ++i)
            {
                result[i] = Math.Exp(oSums[i] - max) / scale;
            }
            return result; // Now scaled so that xi sums to 1.0. 
        }

        public static MatrixLite Softmax(MatrixLite oSums)
        {
            //Make sure this is a one-dimensional matrix
            if (oSums.RowCount != 1)
            {
                throw new InvalidOperationException("Sofmax requires a one-dimentional matrix.");
            }

            // Does all output nodes at once. // Determine max oSum. 
            double max = oSums[0];
            for (int i = 0; i < oSums.ColumnCount; ++i)
            {
                if (oSums[i] > max)
                {
                    max = oSums[i]; // Determine scaling factor -- sum of exp( each val - max).
                }
            }

            double scale = 0.0;
            for (int i = 0; i < oSums.ColumnCount; ++i)
            {
                scale += Math.Exp(oSums[i] - max);
            }

            MatrixLite result = new MatrixLite(1, oSums.ColumnCount);
            for (int i = 0; i < oSums.ColumnCount; ++i)
            {
                result[i] = Math.Exp(oSums[i] - max) / scale;
            }
            return result; // Now scaled so that xi sums to 1.0. 
        }
    }
}