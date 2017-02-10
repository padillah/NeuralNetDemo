using System;

namespace FeedForwardDemo
{
    public class NeuralNetwork
    {
        private readonly double[] hBiases;
        private readonly MatrixLite HiddenBiasMatrix;
        private VectorLite HiddenResultsMatrix;
        private readonly MatrixLite HiddenToOutputMatrix;
        private readonly double[][] hoWeights;
        private readonly double[] hResults;
        private readonly double[][] ihWeights;
        private readonly double[] inputs;

        private VectorLite inputsMatrix;
        private readonly MatrixLite inputToHiddenMatrix;
        private readonly int numHidden;
        private readonly int numInput;
        private readonly int numOutput;
        private readonly double[] oBiases;
        private readonly MatrixLite OutputBiasMatrix;
        private MatrixLite outputResultsMatrix;
        private readonly double[] outputs;

        public NeuralNetwork(int numInput, int numHidden, int numOutput)
        {
            this.numInput = numInput;
            this.numHidden = numHidden;
            this.numOutput = numOutput;

            inputs = new double[numInput];
            inputsMatrix = new MatrixLite(1, numInput);

            ihWeights = MakeMatrix(numInput, numHidden);
            inputToHiddenMatrix = new MatrixLite(numInput, numHidden);

            hBiases = new double[numHidden];
            HiddenBiasMatrix = new MatrixLite(1, numHidden);

            hResults = new double[numHidden];
            HiddenResultsMatrix = new VectorLite(numHidden);

            hoWeights = MakeMatrix(numHidden, numOutput);
            HiddenToOutputMatrix = new MatrixLite(numHidden, numOutput);

            oBiases = new double[numOutput];
            OutputBiasMatrix = new MatrixLite(1, numOutput);

            outputs = new double[numOutput];
            outputResultsMatrix = new MatrixLite(1, numOutput);
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
            for (int inputIndex = 0; inputIndex < numInput; ++inputIndex)
            {
                for (int hiddenIndex = 0; hiddenIndex < numHidden; ++hiddenIndex)
                {
                    //Set the Input to Hidden weight
                    Console.WriteLine($"Input {inputIndex} to hidden {hiddenIndex} = {weights[weightIndex]}");
                    ihWeights[inputIndex][hiddenIndex] = weights[weightIndex++];
                }
            }

            Console.WriteLine("Set the Hiddewn Biases");
            HiddenBiasMatrix.SetValues(weights, numInput * numHidden);
            for (int hiddenIndex = 0; hiddenIndex < numHidden; ++hiddenIndex)
            {
                //Set the Hiddewn Biases
                Console.WriteLine($"Hidden bias {hiddenIndex} = {weights[weightIndex]}");
                hBiases[hiddenIndex] = weights[weightIndex++];
            }

            Console.WriteLine("Set Hidden to Output weights");
            HiddenToOutputMatrix.SetValues(weights, numInput * numHidden + numHidden);
            for (int hiddenIndex = 0; hiddenIndex < numHidden; ++hiddenIndex)
            {
                for (int outputIndex = 0; outputIndex < numOutput; ++outputIndex)
                {
                    Console.WriteLine($"Hidden {hiddenIndex} to Output {outputIndex} = {weights[weightIndex]}");
                    hoWeights[hiddenIndex][outputIndex] = weights[weightIndex++];
                }
            }

            Console.WriteLine("Set output biases");
            OutputBiasMatrix.SetValues(weights, numInput * numHidden * numOutput);
            for (int outputBias = 0; outputBias < numOutput; ++outputBias)
            {
                Console.WriteLine($"Output bias {outputBias} = {weights[weightIndex]}");
                oBiases[outputBias] = weights[weightIndex++];
            }
        }

        public VectorLite ComputeOutputs(VectorLite xValues)
        {
            if (xValues.Length != numInput)
            {
                throw new Exception("Bad xValues array");
            }

            VectorLite oSums = new VectorLite(numOutput);

            inputsMatrix = new VectorLite(xValues);
            for (int i = 0; i < xValues.Length; ++i)
            {
                inputs[i] = xValues[i];
            }

            // ex: hSum[ 0] = (in[ 0] * ihW[[ 0][ 0]) + (in[ 1] * ihW[ 1][ 0]) + (in[ 2] * ihW[ 2][ 0]) + . . 
            // hSum[ 1] = (in[ 0] * ihW[[ 0][ 1]) + (in[ 1] * ihW[ 1][ 1]) + (in[ 2] * ihW[ 2][ 1]) + . . // . . .

            MatrixLite hiddenSumMatrix = inputsMatrix * inputToHiddenMatrix;

            hiddenSumMatrix = hiddenSumMatrix + HiddenBiasMatrix;

            Console.WriteLine("Pre-activation hidden sums:");
            FeedForwardProgram.ShowVector(hiddenSumMatrix, 4, true);

            HiddenResultsMatrix = new VectorLite(hiddenSumMatrix.Length);
            foreach (double currentValue in hiddenSumMatrix)
            {
                HiddenResultsMatrix.Add(HyperTan(currentValue));
            }

            Console.WriteLine("Hidden outputs:");
            FeedForwardProgram.ShowVector(HiddenResultsMatrix, 4, true);

            MatrixLite outputSumMatrix = HiddenResultsMatrix * HiddenToOutputMatrix;

            outputResultsMatrix = outputSumMatrix + OutputBiasMatrix;

            for (int i = 0; i < numOutput; ++i)
            {
                oSums[i] += oBiases[i];
            }

            Console.WriteLine("Pre-activation output sums:");
            FeedForwardProgram.ShowVector(outputResultsMatrix, 4, true);


            VectorLite softOut = Softmax(oSums);
            MatrixLite softOutMatrix = Softmax(outputResultsMatrix);
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