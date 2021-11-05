package model;

/**
 * @author artrayme
 * 11/5/21
 */

public class FastFloatMatrix implements FloatMatrix {

    private final int width;
    private final int height;
    private final float[][] matrix;

    public FastFloatMatrix(int width, int height) {
        if (width < 0)
            throw new IllegalArgumentException();
        if (height < 0)
            throw new IllegalArgumentException();
        this.width = width;
        this.height = height;
        matrix = new float[height][width];
    }

    public FastFloatMatrix(float[][] array) {
        this.height = array.length;
        this.width = array[0].length;
        this.matrix = new float[height][width];
        for (int i = 0; i < array.length; i++) {
            assert array[i].length == width;
            System.arraycopy(array[i], 0, matrix[i], 0, array[i].length);
        }
    }

    @Override
    public FloatMatrix mult(FloatMatrix otherMatrix) {
        if (this.width != otherMatrix.getHeight())
            throw new IllegalArgumentException();
        FastFloatMatrix result = new FastFloatMatrix(otherMatrix.getWidth(), this.height);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < otherMatrix.getWidth(); j++) {
                for (int k = 0; k < width; k++) {
                    result.matrix[i][j] += this.matrix[i][k] * otherMatrix.toArray()[k][j];
                }
            }
        }
        return result;
    }

    @Override
    public FloatMatrix sum(FloatMatrix otherMatrix) {
        if (this.width != otherMatrix.getWidth())
            throw new IllegalArgumentException();
        if (this.height != otherMatrix.getHeight())
            throw new IllegalArgumentException();
        FastFloatMatrix result = new FastFloatMatrix(width, height);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                result.matrix[i][j] = matrix[i][j] + otherMatrix.toArray()[i][j];
            }
        }

        return result;
    }

    @Override
    public FloatMatrix scale(float scale) {
        FastFloatMatrix result = new FastFloatMatrix(width, height);
        for (int i = 0; i < height; i++) {
            System.arraycopy(matrix[i], 0, result.matrix[i], 0, width);
        }

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                result.matrix[i][j] *= scale;
            }
        }
        return result;
    }

    @Override
    public FloatMatrix scaleThis(float scale) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                matrix[i][j] *= scale;
            }
        }
        return this;
    }

    @Override
    public FloatMatrix transpose() {
        FastFloatMatrix result = new FastFloatMatrix(height, width);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                result.matrix[j][i] = matrix[i][j];
            }
        }
        return result;
    }

    public int getHeight() {
        return height;
    }

    public int getWidth() {
        return width;
    }

    @Override
    public float[][] toArray() {
        return matrix;
    }

    @Override
    public String toString() {
        return "FastFloatMatrix{" +
                "matrix=\n" + matrixToString() +
                '}';
    }

    private String matrixToString() {
        StringBuilder result = new StringBuilder();
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                result.append(matrix[i][j]).append("  ");
            }
            result.append("\n");
        }
        return result.toString();
    }

}
