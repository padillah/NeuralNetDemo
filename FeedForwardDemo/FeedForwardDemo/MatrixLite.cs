using System;
using System.Collections;
using System.Collections.Generic;

namespace FeedForwardDemo
{
    public class MatrixLite
    {
        private readonly double[,] _matrixValues;

        public MatrixLite(int rows, int cols)
        {
            RowCount = rows;
            ColumnCount = cols;

            _matrixValues = new double[rows, cols];
        }

        //Need ctor, Multiply, and Coalesce
        public MatrixLite(double[] weights, int rows, int cols, int offset)
            : this(rows, cols)
        {
            var newWeights = new VectorLite(weights);
            SetValues(newWeights, offset);
        }

        public MatrixLite(double[] weights)
            : this(weights, 1, weights.Length, 0)
        {
        }

        public MatrixLite(VectorLite weights)
            : this(weights, 1, weights.Length, 0)
        {
        }

        public int ColumnCount { get; }

        public int RowCount { get; }

        public VectorLite this[int rowIndex]
        {
            set
            {
                for (int index = 0; index < value.Length; index++)
                {
                    _matrixValues[rowIndex, index] = value[index];
                }
            }
            get
            {
                VectorLite result = new VectorLite(_matrixValues.GetLength(rowIndex));
                for (int index = 0; index < result.Length; index++)
                {
                    result[index] = _matrixValues[rowIndex, index];
                }
                return result;
            }
        }

        public double this[int rowIndex, int columnIndex]
        {
            set
            {
                if (rowIndex < 0)
                {
                    throw new ArgumentOutOfRangeException(nameof(rowIndex), "Indices must be real positive.");
                }
                if (columnIndex < 0)
                {
                    throw new ArgumentOutOfRangeException(nameof(columnIndex), "Indices must be real positive.");
                }

                //TODO: Implement dynamic sizing
                //if (rowIndex > RowCount)
                //{
                //    // dynamically add rowIndex-Rows new rows...
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
                //    // dynamically add columnIndex-Cols columns to each row
                //    for (int k = 0; k < rowCount; k++)
                //    {
                //        for (int t = 0; t < columnIndex - columnCount; t++)
                //        {
                //            ((ArrayList)Values[k]).Add(Complex.Zero);
                //        }
                //    }

                //    columnCount = columnIndex;
                //}

                _matrixValues[rowIndex, columnIndex] = value;
            }

            get
            {
                if ((rowIndex >= 0) && (rowIndex < RowCount))
                {
                    if ((columnIndex >= 0) && (columnIndex < ColumnCount))
                    {
                        return _matrixValues[rowIndex, columnIndex];
                    }
                    else
                    {
                        throw new ArgumentOutOfRangeException(nameof(columnIndex), "Indices must not exceed size of matrix.");
                    }
                }
                else
                {
                    throw new ArgumentOutOfRangeException(nameof(rowIndex), "Indices must not exceed size of matrix.");
                }
            }
        }

        public static MatrixLite operator *(MatrixLite firstOperand, MatrixLite secondOperand)
        {
            if (firstOperand.ColumnCount != secondOperand.RowCount)
            {
                throw new ArgumentException("The columns of the first matrix/vector must equal the rows of the second matrix/vector.", nameof(firstOperand));
            }

            MatrixLite responseMatrix = new MatrixLite(firstOperand.RowCount, secondOperand.ColumnCount);

            for (int i = 0; i < firstOperand.RowCount; i++)
            {
                for (int j = 0; j < secondOperand.ColumnCount; j++)
                {
                    responseMatrix[i, j] = Dot(firstOperand.Row(i), secondOperand.Column(j));
                }
            }

            return responseMatrix;
        }

        public static VectorLite operator *(VectorLite operandOne, MatrixLite operandTwo)
        {
            var newOperand = new MatrixLite(operandOne);
            return (newOperand * operandTwo).Row(0);
        }

        public static MatrixLite operator +(MatrixLite firstOperand, MatrixLite secondOperand)
        {
            if ((firstOperand.RowCount != secondOperand.RowCount) || (firstOperand.ColumnCount != secondOperand.ColumnCount))
            {
                throw new ArgumentException("Matrices must be of the same dimension.");
            }

            for (int i = 0; i < firstOperand.RowCount; i++)
            {
                for (int j = 0; j < firstOperand.ColumnCount; j++)
                {
                    firstOperand[i, j] += secondOperand[i, j];
                }
            }

            return firstOperand;
        }

        public static implicit operator double[](MatrixLite operatorOne)
        {
            if (operatorOne.RowCount > 1)
            {
                throw new InvalidOperationException("Cannot implicitly cast multidimmesional MatrixLite. MatrixLite has more than one row.");
            }

            double[] buf = new double[operatorOne.ColumnCount];

            for (int j = 0; j < operatorOne.ColumnCount; j++)
            {
                buf[j] = operatorOne._matrixValues[0, j];
            }

            return buf;
        }

        /// <summary>
        ///     Retrieves column with zero-based index columnIndex.
        /// </summary>
        /// <param name="columnIndex"></param>
        /// <returns>columnIndex-th column...</returns>
        public VectorLite Column(int columnIndex)
        {
            VectorLite buf = new VectorLite(RowCount);

            for (int i = 0; i < RowCount; i++)
            {
                buf[i] = _matrixValues[i, columnIndex];
            }

            return buf;
        }

        /// <summary>
        ///     Retrieves row with one-based index rowIndex.
        /// </summary>
        /// <param name="rowIndex"></param>
        /// <returns>rowIndex-th row...</returns>
        public VectorLite Row(int rowIndex)
        {
            if ((rowIndex < 0) || (rowIndex > RowCount))
            {
                throw new ArgumentException("Index exceed matrix dimension.");
            }

            VectorLite buf = new VectorLite(ColumnCount);

            for (int j = 0; j < ColumnCount; j++)
            {
                buf[j] = _matrixValues[rowIndex, j];
            }

            return buf;
        }

        public void SetValues(VectorLite weights, int offset)
        {
            int weightIndex = offset; // Pointer into array.

            for (int rowIndex = 0; rowIndex < _matrixValues.GetLength(0); rowIndex++)
            {
                for (int colIndex = 0; colIndex < _matrixValues.GetLength(1); ++colIndex)
                {
                    //Set the Input to Hidden weight
                    Console.WriteLine($"Row {rowIndex} Column {colIndex} = {weights[weightIndex]}");
                    _matrixValues[rowIndex, colIndex] = weights[weightIndex++];
                }
            }
        }

        /// <summary>
        ///     Checks if matrix is n by one or one by n.
        /// </summary>
        /// <returns>Length, if vectors; zero else.</returns>
        public int VectorLength()
        {
            if ((ColumnCount > 1) && (RowCount > 1))
            {
                return 0;
            }
            return Math.Max(ColumnCount, RowCount);
        }

        public static double Dot(VectorLite operandOne, VectorLite operandTwo)
        {
            int m = operandOne.Length;
            int n = operandTwo.Length;

            if ((m == 0) || (n == 0))
            {
                throw new ArgumentException("Arguments need to be vectors.");
            }
            if (m != n)
            {
                throw new ArgumentException("Vectors must be of the same length.");
            }

            double buf = 0;

            for (int i = 0; i < m; i++)
            {
                buf += operandOne[i] * operandTwo[i];
            }

            return buf;
        }
    }

    public class VectorLite : IEnumerable<double>
    {
        private double[] _vectorValues;

        public VectorLite(params double[] data)
        {
            _vectorValues = data;
        }

        public VectorLite(int columnCount)
        {
            _vectorValues = new double[columnCount];
        }

        public static implicit operator double[](VectorLite firstOperator)
        {
            return firstOperator._vectorValues;
        }

        public double this[int index]
        {
            get { return _vectorValues[index]; }

            set { _vectorValues[index] = value; }
        }

        public void Add(double value)
        {
            double[] tempValues;
            if (_vectorValues.Length == 0)
            {
                _vectorValues = new double[1];
                tempValues = new double[1];
            }
            else
            {
                tempValues = new double[_vectorValues.Length + 1];
            }

            Array.Copy(_vectorValues, tempValues, _vectorValues.Length);

            tempValues[tempValues.Length - 1] = value;

            _vectorValues = tempValues;
        }

        public int Length => _vectorValues.Length;

        public IEnumerator<double> GetEnumerator()
        {
            return ((IEnumerable<double>)_vectorValues).GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public void SetValues(VectorLite weights, int offset)
        {
            int weightIndex = offset; // Pointer into array.

            for (int rowIndex = 0; rowIndex < _vectorValues.GetLength(0); rowIndex++)
            {
                //Set the Input to Hidden weight
                Console.WriteLine($"Row {rowIndex} = {weights[weightIndex]}");
                _vectorValues[rowIndex] = weights[weightIndex++];
            }
        }

        public static VectorLite operator +(VectorLite firstOperand, VectorLite secondOperand)
        {
            if (firstOperand.Length != secondOperand.Length)
            {
                throw new ArgumentException("Vectors must be of the same dimension.");
            }

            for (int j = 0; j < firstOperand.Length; j++)
            {
                firstOperand[j] += secondOperand[j];
            }

            return firstOperand;
        }
    }
}