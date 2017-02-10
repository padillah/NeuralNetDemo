using System;

namespace FeedForwardDemo
{
    public class MatrixLite
    {
        private readonly double[,] matrixValues;

        public MatrixLite(int rows, int cols)
        {
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

        public MatrixLite(double[] weights)
            : this(weights, 1, weights.Length, 0)
        {
        }

        public int ColumnCount { get; }

        public int RowCount { get; }

        public virtual double this[int rowIndex]
        {
            set
            {
                if (RowCount == 1)
                {
                    // row vectors

                    //TODO: Implement dynamic expansion
                    // dynamically extend vectors if necessary
                    //if (rowIndex > ColumnCount)
                    //{
                    //    // dynamically add columnIndex-Cols columns to each row
                    //    for (int t = 0; t < rowIndex - ColumnCount; t++)
                    //        ((ArrayList)Values[0]).Add(Complex.Zero);

                    //    columnCount = rowIndex;
                    //}

                    matrixValues[0, rowIndex] = value;
                }
                else if (ColumnCount == 1)
                {
                    // column vectors

                    //TODO: Implement dynamic expansion
                    //if (rowIndex > RowCount)
                    //{
                    //    // dynamically add rowIndex-Rows new rows...
                    //    for (int k = 0; k < rowIndex - rowCount; k++)
                    //    {
                    //        this.Values.Add(new ArrayList(columnCount));

                    //        // ...with one column each
                    //        ((ArrayList)Values[rowCount + k]).Add(Complex.Zero);
                    //    }

                    //    rowCount = rowIndex; // ha!
                    //}

                    matrixValues[rowIndex, 0] = value;
                }
                else
                {
                    throw new InvalidOperationException("Cannot access multidimensional matrix via single index.");
                }
            }
            get
            {
                if (RowCount == 1) // row vectors
                {
                    return matrixValues[0, rowIndex];
                }

                if (ColumnCount == 1) // coumn vectors
                {
                    return matrixValues[rowIndex, 0];
                }

                throw new InvalidOperationException("General matrix acces requires double indexing.");
            }
        }

        public virtual double this[int rowIndex, int columnIndex]
        {
            set
            {
                if ((rowIndex < 0) || (columnIndex < 0))
                {
                    throw new ArgumentOutOfRangeException("Indices must be real positive.");
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

                matrixValues[rowIndex, columnIndex] = value;
            }

            get
            {
                if ((rowIndex >= 0) && (rowIndex < RowCount) && (columnIndex >= 0) && (columnIndex < ColumnCount))
                {
                    return matrixValues[rowIndex, columnIndex];
                }

                throw new ArgumentOutOfRangeException("Indices must not exceed size of matrix.");
            }
        }

        public static MatrixLite operator *(MatrixLite operandOne, MatrixLite operandTwo)
        {
            MatrixLite firstOperand;
            MatrixLite secondOperand;
            MatrixLite responseMatrix;

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

            for (int i = 0; i < firstOperand.RowCount; i++)
            {
                for (int j = 0; j < secondOperand.ColumnCount; j++)
                {
                    responseMatrix[i, j] = Dot(firstOperand.Row(i), secondOperand.Column(j));
                }
            }

            return responseMatrix;
        }

        public static MatrixLite operator +(MatrixLite A, MatrixLite B)
        {
            if ((A.RowCount != B.RowCount) || (A.ColumnCount != B.ColumnCount))
            {
                throw new ArgumentException("Matrices must be of the same dimension.");
            }

            for (int i = 0; i < A.RowCount; i++)
            {
                for (int j = 0; j < A.ColumnCount; j++)
                {
                    A[i, j] += B[i, j];
                }
            }

            return A;
        }

        public static implicit operator double[] (MatrixLite operatorOne)
        {
            if (operatorOne.RowCount > 1)
            {
                throw new InvalidOperationException("Cannot implicitly cast multidimmesional MatrixLite. MatrixLite has more than one row.");
            }

            double[] buf = new double[operatorOne.ColumnCount];

            for (int j = 0; j < operatorOne.ColumnCount; j++)
            {
                buf[j] = operatorOne.matrixValues[0, j];
            }

            return buf;
        }

        private void SetValue(int row, int column, double value)
        {
            matrixValues[row, column] = value;
        }

        /// <summary>
        ///     Retrieves column with zero-based index columnIndex.
        /// </summary>
        /// <param name="columnIndex"></param>
        /// <returns>columnIndex-th column...</returns>
        public MatrixLite Column(int columnIndex)
        {
            MatrixLite buf = new MatrixLite(RowCount, 1);

            for (int i = 0; i < RowCount; i++)
            {
                buf[i] = matrixValues[i, columnIndex];
            }

            return buf;
        }

        /// <summary>
        ///     Retrieves row with one-based index rowIndex.
        /// </summary>
        /// <param name="rowIndex"></param>
        /// <returns>rowIndex-th row...</returns>
        public MatrixLite Row(int rowIndex)
        {
            if ((rowIndex < 0) || (rowIndex > RowCount))
            {
                throw new ArgumentException("Index exceed matrix dimension.");
            }

            MatrixLite buf = new MatrixLite(ColumnCount, 1);

            for (int j = 0; j < ColumnCount; j++)
            {
                buf[j] = matrixValues[rowIndex, j];
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

        public static double Dot(MatrixLite operandOne, MatrixLite operandTwo)
        {
            int m = operandOne.VectorLength();
            int n = operandTwo.VectorLength();

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
}