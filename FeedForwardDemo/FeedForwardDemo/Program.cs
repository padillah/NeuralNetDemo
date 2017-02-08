﻿using System;
using System.Collections;
using System.Data.SqlTypes;

namespace FeedForwardDemo
{
    class FeedForwardProgram


    {
        static void Main(string[] args)
        {
            Console.WriteLine("Begin feed-forward demo");
            int numInput = 3;
            int numHidden = 4;
            int numOutput = 2;

            Console.WriteLine("Creating a 3-4-2 tanh-softmax neural network");
            NeuralNetwork nn = new NeuralNetwork(numInput, numHidden, numOutput);
            double[] weights = new double[]
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
            ShowVector(weights, 8, 2, true);
            nn.SetWeights(weights);

            double[] xValues = new double[] { 1.0, 2.0, 3.0 };
            var inputMatrix = new MatrixLite(xValues, 1, 3, 0);

            Console.WriteLine("Inputs are:");
            ShowVector(xValues, 3, 1, true);

            Console.WriteLine("Computing outputs");
            double[] yValues = nn.ComputeOutputs(xValues, inputMatrix);

            Console.WriteLine("Outputs computed");

            Console.WriteLine("Outputs are:");
            ShowVector(yValues, 2, 4, true);

            Console.WriteLine("End feed-forward demo");
            Console.ReadLine();
        } // Main

        public static void ShowVector(double[] vector, int valsPerRow, int decimals, bool newLine)
        {
            for (int i = 0; i < vector.Length; ++i)
            {
                if (i % valsPerRow == 0) Console.WriteLine("");
                Console.Write(vector[i].ToString($"F{decimals}").PadLeft(decimals + 4) + " ");
            }
            if (newLine == true) Console.WriteLine("");
        }
    } // Program

    public class NeuralNetwork
    {
        private int numInput;
        private int numHidden;
        private int numOutput;
        private double[] inputs;
        private double[][] ihWeights;
        private double[] hBiases;
        private double[] hOutputs;
        private double[][] hoWeights;
        private double[] oBiases;
        private double[] outputs;

        private MatrixLite inputToHiddenMatrix;
        private MatrixLite HiddenBiasMatrix;
        private MatrixLite HiddenToOutputMatrix;
        private MatrixLite OutputMatrix;


        public NeuralNetwork(int numInput, int numHidden, int numOutput)
        {
            this.numInput = numInput;
            this.numHidden = numHidden;
            this.numOutput = numOutput;
            this.inputs = new double[numInput];
            this.ihWeights = MakeMatrix(numInput, numHidden);
            this.hBiases = new double[numHidden];
            this.hOutputs = new double[numHidden];
            this.hoWeights = MakeMatrix(numHidden, numOutput);
            this.oBiases = new double[numOutput];
            this.outputs = new double[numOutput];

            inputToHiddenMatrix = new MatrixLite(numInput, numHidden);
            HiddenBiasMatrix = new MatrixLite(1, numHidden);
            HiddenToOutputMatrix = new MatrixLite(numHidden, numOutput);
            OutputMatrix = new MatrixLite(1, numOutput);
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

        public void SetWeights(double[] weights)
        {
            int numWeights = (numInput * numHidden) + numHidden + (numHidden * numOutput) + numOutput;
            if (weights.Length != numWeights)
                throw new Exception("Bad weights array");

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
            HiddenBiasMatrix.SetValues(weights, (numInput * numHidden));
            for (int hiddenIndex = 0; hiddenIndex < numHidden; ++hiddenIndex)
            {
                //Set the Hiddewn Biases
                Console.WriteLine($"Hidden bias {hiddenIndex} = {weights[weightIndex]}");
                hBiases[hiddenIndex] = weights[weightIndex++];
            }

            Console.WriteLine("Set Hidden to Output weights");
            HiddenToOutputMatrix.SetValues(weights, (numInput * numHidden) + numHidden);
            for (int hiddenIndex = 0; hiddenIndex < numHidden; ++hiddenIndex)
            {
                for (int outputIndex = 0; outputIndex < numOutput; ++outputIndex)
                {
                    Console.WriteLine($"Hidden {hiddenIndex} to Output {outputIndex} = {weights[weightIndex]}");
                    hoWeights[hiddenIndex][outputIndex] = weights[weightIndex++];
                }
            }

            Console.WriteLine("Set output biases");
            OutputMatrix.SetValues(weights, (numInput * numHidden * numOutput));
            for (int outputBias = 0; outputBias < numOutput; ++outputBias)
            {
                Console.WriteLine($"Output bias {outputBias} = {weights[weightIndex]}");
                oBiases[outputBias] = weights[weightIndex++];
            }
        }

        public double[] ComputeOutputs(double[] xValues, MatrixLite inputMatrix)
        {
            if (xValues.Length != numInput)
                throw new Exception("Bad xValues array");

            double[] hSums = new double[numHidden];
            double[] oSums = new double[numOutput];
            for (int i = 0; i < xValues.Length; ++i)
            {
                inputs[i] = xValues[i];
            }
            // ex: hSum[ 0] = (in[ 0] * ihW[[ 0][ 0]) + (in[ 1] * ihW[ 1][ 0]) + (in[ 2] * ihW[ 2][ 0]) + . . 
            // hSum[ 1] = (in[ 0] * ihW[[ 0][ 1]) + (in[ 1] * ihW[ 1][ 1]) + (in[ 2] * ihW[ 2][ 1]) + . . // . . .

            var sumMatrix = inputMatrix * inputToHiddenMatrix;

            for (int j = 0; j < numHidden; ++j)
            {
                for (int i = 0; i < numInput; ++i)
                {
                    hSums[j] += inputs[i] * ihWeights[i][j];
                }
            }
            for (int i = 0; i < numHidden; ++i)
            {
                hSums[i] += hBiases[i];
            }
            Console.WriteLine("Pre-activation hidden sums:");
            FeedForwardProgram.ShowVector(hSums, 4, 4, true);
            for (int i = 0; i < numHidden; ++i)
            {
                hOutputs[i] = HyperTan(hSums[i]);
            }
            Console.WriteLine("Hidden outputs:");
            FeedForwardProgram.ShowVector(hOutputs, 4, 4, true);
            for (int j = 0; j < numOutput; ++j)
            {
                for (int i = 0; i < numHidden; ++i)
                {
                    oSums[j] += hOutputs[i] * hoWeights[i][j];
                }
            }
            for (int i = 0; i < numOutput; ++i)
            {
                oSums[i] += oBiases[i];
            }
            Console.WriteLine("Pre-activation output sums:");
            FeedForwardProgram.ShowVector(oSums, 2, 4, true);
            double[] softOut = Softmax(oSums);
            // Softmax does all outputs at once. 
            for (int i = 0; i < outputs.Length; ++i)
            {
                outputs[i] = softOut[i];
            }
            double[] result = new double[numOutput];
            for (int i = 0; i < outputs.Length; ++i)
            {
                result[i] = outputs[i];
            }
            return result;
        }

        private static double HyperTan(double v)
        {
            if (v < -20.0) return -1.0;
            else if (v > 20.0) return 1.0;
            else return Math.Tanh(v);
        }

        public static double[] Softmax(double[] oSums)
        {
            // Does all output nodes at once. // Determine max oSum. 
            double max = oSums[0];
            for (int i = 0; i < oSums.Length; ++i)
            {
                if (oSums[i] > max) max = oSums[i]; // Determine scaling factor -- sum of exp( each val - max).
            }
            double scale = 0.0;
            for (int i = 0; i < oSums.Length; ++i)
            {
                scale += Math.Exp(oSums[i] - max);
            }
            double[] result = new double[oSums.Length];
            for (int i = 0; i < oSums.Length; ++i)
            {
                result[i] =
                    Math.Exp(oSums[i] - max) / scale;
            }
            return result; // Now scaled so that xi sums to 1.0. 
        }

        public static double[] SoftmaxNaive(double[] oSums)
        {
            double denom = 0.0;
            for (int i = 0; i < oSums.Length; ++i)
            {
                denom += Math.Exp(oSums[i]);
            }
            double[] result = new double[oSums.Length];
            for (int i = 0; i < oSums.Length; ++i)
            {
                result[i] = Math.Exp(oSums[i]) / denom;
            }
            return result;
        }
    } // NeuralNetwork

    public class MatrixLite
    {
        private double[,] matrixValues;

        public MatrixLite(int rows, int cols)
        {
            Console.WriteLine($"Matrix created: {rows} x {cols}");

            RowCount = rows;
            ColumnCount = cols;

            matrixValues = new double[rows, cols];
        }

        //Need ctor, Multiply, and Coalesce
        public MatrixLite(double[] weights, int rows, int cols, int offset)
            : this(rows, cols)
        {
            SetValues(weights, offset);
        }

        public static MatrixLite operator *(MatrixLite operandOne, MatrixLite operandTwo)
        {
            MatrixLite firstOperand;
            MatrixLite secondOperand;
            MatrixLite responseMatrix;
            double intermediateValue;

            //This checks the two matrices have compatible dimentions
            if (operandOne.ColumnCount != operandTwo.RowCount)
            {
                if (operandTwo.ColumnCount != operandOne.RowCount)
                {
                    throw new ArithmeticException($"Incompatible matrices. Cannot multiply {operandOne.RowCount} x {operandOne.ColumnCount} matrix and {operandTwo.RowCount} x {operandTwo.ColumnCount} matix.");
                }

                firstOperand = operandTwo;
                secondOperand = operandOne;
            }
            else
            {
                firstOperand = operandOne;
                secondOperand = operandTwo;
            }

            responseMatrix = new MatrixLite(firstOperand.RowCount, secondOperand.ColumnCount);

            for (int i = 1; i <= firstOperand.RowCount; i++)
            {
                for (int j = 1; j <= secondOperand.ColumnCount; j++)
                {
                    responseMatrix[i, j] = Dot(firstOperand.Row(i), secondOperand.Column(j));
                }
            }

            return null;
        }

        private void SetValue(int row, int column, double value)
        {
            matrixValues[row, column] = value;
        }

        public int ColumnCount { get; }

        public int RowCount { get; }

        public virtual double this[int rowIndex]
        {
            set
            {
                if (RowCount == 1)
                {
                    // row vector

                    //TODO: Implement dynamic expansion
                    // dynamically extend vector if necessary
                    //if (i > ColumnCount)
                    //{
                    //    // dynamically add j-Cols columns to each row
                    //    for (int t = 0; t < i - ColumnCount; t++)
                    //        ((ArrayList)Values[0]).Add(Complex.Zero);

                    //    columnCount = i;
                    //}
                    
                    matrixValues[0, rowIndex-1] = value;
                }
                else if (ColumnCount == 1)
                {
                    // column vector

                    //TODO: Implement dynamic expansion
                    //if (i > RowCount)
                    //{
                    //    // dynamically add i-Rows new rows...
                    //    for (int k = 0; k < i - rowCount; k++)
                    //    {
                    //        this.Values.Add(new ArrayList(columnCount));

                    //        // ...with one column each
                    //        ((ArrayList)Values[rowCount + k]).Add(Complex.Zero);
                    //    }

                    //    rowCount = i; // ha!
                    //}

                    matrixValues[rowIndex - 1, 0] = value;
                }
                else
                    throw new InvalidOperationException("Cannot access multidimensional matrix via single index.");
            }
            get
            {
                if (this.RowCount == 1) // row vector
                {
                    return matrixValues[0,rowIndex - 1];
                }

                if (this.ColumnCount == 1) // coumn vector
                {
                    return matrixValues[rowIndex - 1,0];
                }

                throw new InvalidOperationException("General matrix acces requires double indexing.");
            }
        }

        public virtual double this[int rowIndex, int columnIndex]
        {
            set
            {
                if (rowIndex <= 0 || columnIndex <= 0)
                    throw new ArgumentOutOfRangeException("Indices must be real positive.");

                //TODO: Implement dynamic sizing
                //if (rowIndex > RowCount)
                //{
                //    // dynamically add i-Rows new rows...
                //    for (int k = 0; k < rowIndex - rowCount; k++)
                //    {
                //        this.Values.Add(new ArrayList(columnCount));

                //        // ...with Cols columns
                //        for (int t = 0; t < columnCount; t++)
                //        {
                //            ((ArrayList)Values[rowCount + k]).Add(Complex.Zero);
                //        }
                //    }

                //    rowCount = rowIndex; // ha!
                //}

                //TODO: Implements dynamic sizing
                //if (columnIndex > ColumnCount)
                //{
                //    // dynamically add j-Cols columns to each row
                //    for (int k = 0; k < rowCount; k++)
                //    {
                //        for (int t = 0; t < columnIndex - columnCount; t++)
                //        {
                //            ((ArrayList)Values[k]).Add(Complex.Zero);
                //        }
                //    }

                //    columnCount = columnIndex;
                //}

                matrixValues[rowIndex - 1, columnIndex - 1] = value;

            }
            get
            {
                if (rowIndex > 0 && rowIndex <= RowCount && columnIndex > 0 && columnIndex <= ColumnCount)
                {
                    return matrixValues[rowIndex - 1, columnIndex - 1];
                }
                else
                    throw new ArgumentOutOfRangeException("Indices must not exceed size of matrix.");
            }
        }

        /// <summary>
        /// Retrieves column with one-based index j.
        /// </summary>
        /// <param name="j"></param>
        /// <returns>j-th column...</returns>
        public MatrixLite Column(int j)
        {
            MatrixLite buf = new MatrixLite(this.RowCount, 1);

            for (int i = 1; i <= this.RowCount; i++)
            {
                buf[i] = matrixValues[i, j];
            }

            return buf;
        }

        /// <summary>
        /// Retrieves row with one-based index i.
        /// </summary>
        /// <param name="i"></param>
        /// <returns>i-th row...</returns>
        public MatrixLite Row(int i)
        {
            if (i <= 0 || i > RowCount)
                throw new ArgumentException("Index exceed matrix dimension.");

            //return (new Matrix((Complex[])((ArrayList)Values[i - 1]).ToArray(typeof(Complex)))).Transpose();

            MatrixLite buf = new MatrixLite(ColumnCount, 1);

            for (int j = 1; j <= this.ColumnCount; j++)
            {
                buf[j] = matrixValues[i, j];
            }

            return buf;
        }

        public void SetValues(double[] weights, int offset)
        {
            int weightIndex = offset; // Pointer into array.

            for (int rowIndex = 0; rowIndex < matrixValues.GetLength(0); rowIndex++)
            {
                for (int colIndex = 0; colIndex < matrixValues.GetLength(1); ++colIndex)
                {
                    //Set the Input to Hidden weight
                    Console.WriteLine($"Row {rowIndex} Column {colIndex} = {weights[weightIndex]}");
                    matrixValues[rowIndex, colIndex] = weights[weightIndex++];
                }
            }
        }

        /// <summary>
        /// Checks if matrix is n by one or one by n.
        /// </summary>
        /// <returns>Length, if vector; zero else.</returns>
        public int VectorLength()
        {
            if (ColumnCount > 1 && RowCount > 1)
                return 0;
            else return Math.Max(ColumnCount, RowCount);
        }

        public static double Dot(MatrixLite operandOne, MatrixLite operandTwo)
        {
            int m = operandOne.VectorLength();
            int n = operandTwo.VectorLength();

            if (m == 0 || n == 0)
                throw new ArgumentException("Arguments need to be vectors.");
            else if (m != n)
                throw new ArgumentException("Vectors must be of the same length.");

            double buf = 0;

            for (int i = 1; i <= m; i++)
            {
                buf += operandOne[i] * operandTwo[i];
            }

            return buf;
        }
    }
} // ns