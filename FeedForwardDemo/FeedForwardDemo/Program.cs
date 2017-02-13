using System;

namespace FeedForwardDemo
{
    public class FeedForwardProgram
    {
        private static void Main()
        {
            Console.WriteLine("Begin feed-forward demo");
            const int numInput = 3;
            const int numHidden = 4;
            const int numOutput = 2;

            Console.WriteLine("Creating a 3-4-2 tanh-softmax neural network");
            NeuralNetwork nn = new NeuralNetwork(numInput, numHidden, numOutput);
            VectorLite weights = new VectorLite()
            {
                0.01, 0.02, 0.03,
                0.04, 0.05, 0.06,
                0.07, 0.08, 0.09,
                0.10, 0.11, 0.12,
                0.13, 0.14, 0.15,
                0.16, 0.17, 0.18,
                0.19, 0.20, 0.21,
                0.22, 0.23, 0.24,
                0.25, 0.26
            };

            Console.WriteLine("Setting dummy weights and biases:");
            ShowVector(weights, 2, true);
            nn.SetWeights(weights);

            VectorLite xValues = new VectorLite()
            { 1.0, 2.0, 3.0 };

            Console.WriteLine("Inputs are:");
            ShowVector(xValues, 1, true);

            Console.WriteLine("Computing outputs");
            VectorLite yValues = nn.ComputeOutputs(xValues);

            Console.WriteLine("Outputs computed");

            Console.WriteLine("Outputs are:");
            ShowVector(yValues, 4, true);

            Console.WriteLine("End feed-forward demo");
            Console.ReadLine();
        } // Main

        public static void ShowVector(VectorLite vectors, int precision, bool newLine)
        {
            foreach (double currentVector in vectors)
            {
                Console.Write(currentVector.ToString($"F{precision}").PadLeft(precision + 4) + " ");
            }

            if (newLine)
            {
                Console.WriteLine("");
            }
        }
    } // Program

} // ns