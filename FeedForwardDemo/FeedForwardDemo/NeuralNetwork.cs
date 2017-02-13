using System;

namespace FeedForwardDemo
{
    public class NeuralNetwork
    {
        private readonly VectorLite HiddenBiasVector;
        private VectorLite HiddenResultsVector;
        private VectorLite inputsVector;
        private readonly VectorLite OutputBiasVector;
        private VectorLite outputResultsVector;

        private readonly MatrixLite HiddenToOutputMatrix;
        private readonly MatrixLite inputToHiddenMatrix;

        private readonly int numHidden;
        private readonly int numInput;
        private readonly int numOutput;

        public NeuralNetwork(int numInput, int numHidden, int numOutput)
        {
            this.numInput = numInput;
            this.numHidden = numHidden;
            this.numOutput = numOutput;

            inputsVector = new VectorLite(numInput);

            inputToHiddenMatrix = new MatrixLite(numInput, numHidden);

            HiddenBiasVector = new VectorLite(numHidden);

            HiddenResultsVector = new VectorLite(numHidden);

            HiddenToOutputMatrix = new MatrixLite(numHidden, numOutput);

            OutputBiasVector = new VectorLite(1, numOutput);

            outputResultsVector = new VectorLite(numOutput);
        }

        private static double[][] MakeMatrix(int rows, int cols)
        {
            double[][] result = new double[rows][];
            for (int i = 0; i < rows; ++i)
            {
                result[i] = new double[cols];
            }
            return result;
        }

        public void SetWeights(VectorLite weights)
        {
            int numWeights = numInput * numHidden + numHidden + numHidden * numOutput + numOutput;
            if (weights.Length != numWeights)
            {
                throw new Exception("Bad weights array");
            }

            int weightIndex = 0; // Pointer into weights.

            Console.WriteLine("Set the Input to Hidden weight");
            inputToHiddenMatrix.SetValues(weights, 0);

            Console.WriteLine("Set the Hiddewn Biases");
            HiddenBiasVector.SetValues(weights, numInput * numHidden);

            Console.WriteLine("Set Hidden to Output weights");
            HiddenToOutputMatrix.SetValues(weights, numInput * numHidden + numHidden);

            Console.WriteLine("Set output biases");
            OutputBiasVector.SetValues(weights, numInput * numHidden * numOutput);

        }

        public VectorLite ComputeOutputs(VectorLite xValues)
        {
            if (xValues.Length != numInput)
            {
                throw new Exception("Bad xValues array");
            }

            inputsVector = new VectorLite(xValues);

            // ex: hSum[ 0] = (in[ 0] * ihW[[ 0][ 0]) + (in[ 1] * ihW[ 1][ 0]) + (in[ 2] * ihW[ 2][ 0]) + . . 
            // hSum[ 1] = (in[ 0] * ihW[[ 0][ 1]) + (in[ 1] * ihW[ 1][ 1]) + (in[ 2] * ihW[ 2][ 1]) + . . // . . .

            VectorLite hiddenSumVector = inputsVector * inputToHiddenMatrix;

            hiddenSumVector = hiddenSumVector + HiddenBiasVector;

            Console.WriteLine("Pre-activation hidden sums:");
            FeedForwardProgram.ShowVector(hiddenSumVector, 4, true);

            //HiddenResultsVector = new VectorLite(hiddenSumVector.Length);
            for (int index = 0; index < HiddenResultsVector.Length; index++ )
            {
                HiddenResultsVector[index] = HyperTan(hiddenSumVector[index]);
            }

            Console.WriteLine("Hidden outputs:");
            FeedForwardProgram.ShowVector(HiddenResultsVector, 4, true);

            VectorLite outputSumVector = HiddenResultsVector * HiddenToOutputMatrix;

            outputResultsVector = outputSumVector + OutputBiasVector;

            Console.WriteLine("Pre-activation output sums:");
            FeedForwardProgram.ShowVector(outputResultsVector, 4, true);

            VectorLite softOutMatrix = Softmax(outputResultsVector);

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