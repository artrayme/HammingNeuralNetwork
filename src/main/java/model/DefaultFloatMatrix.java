package model;

import java.util.List;

/**
 * @author artrayme
 * 11/5/21
 */
public class DefaultFloatMatrix implements FloatMatrix {
    public final int width;
    public final int height;
    private final float[][] matrix;

    public DefaultFloatMatrix(int width, int height) {
        if (width < 0)
            throw new IllegalArgumentException();
        if (height < 0)
            throw new IllegalArgumentException();
        this.width = width;
        this.height = height;
        matrix = new float[height][width];
    }

    public DefaultFloatMatrix(float[][] array) {
        this.height = array.length;
        this.width = array[0].length;
        this.matrix = new float[height][width];
        for (int i = 0; i < array.length; i++) {
            assert array[i].length == width;
            System.arraycopy(array[i], 0, matrix[i], 0, array[i].length);
        }
    }

    public DefaultFloatMatrix(Float[] vector) {
        this.height = 1;
        this.width = vector.length;
        this.matrix = new float[height][width];
        System.arraycopy(vector, 0, matrix[0], 0, vector.length);
    }

    public DefaultFloatMatrix(List<List<Float>> array) {
        this.height = array.size();
        this.width = array.get(0).size();
        this.matrix = new float[height][width];
        for (int i = 0; i < array.size(); i++) {
            for (int j = 0; j < array.get(i).size(); j++) {
                matrix[i][j] = array.get(i).get(j);
            }
        }
    }

    public DefaultFloatMatrix(List<Float> array, boolean b) {
        this.height = 1;
        this.width = array.size();
        this.matrix = new float[height][width];
        for (int i = 0; i < array.size(); i++) {
            matrix[0][i] = array.get(i);
        }
    }

    @Override
    public FloatMatrix mult(FloatMatrix otherMatrix) {
        if (this.width != otherMatrix.getHeight())
            throw new IllegalArgumentException();
        DefaultFloatMatrix result = new DefaultFloatMatrix(otherMatrix.getWidth(), this.height);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < otherMatrix.getWidth(); j++) {
                for (int k = 0; k < width; k++) {
                    result.toArray()[i][j] += this.matrix[i][k] * otherMatrix.toArray()[k][j];
                }
            }
        }
        return result;
    }

    @Override
    public FloatMatrix plus(FloatMatrix otherMatrix) {
        if (this.width != otherMatrix.getWidth())
            throw new IllegalArgumentException();
        if (this.height != otherMatrix.getHeight())
            throw new IllegalArgumentException();
        DefaultFloatMatrix result = new DefaultFloatMatrix(width, height);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                result.matrix[i][j] = matrix[i][j] + otherMatrix.toArray()[i][j];
            }
        }

        return result;
    }

    @Override
    public FloatMatrix minus(FloatMatrix otherMatrix) {
        if (this.width != otherMatrix.getWidth())
            throw new IllegalArgumentException();
        if (this.height != otherMatrix.getHeight())
            throw new IllegalArgumentException();
        DefaultFloatMatrix result = new DefaultFloatMatrix(width, height);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                result.matrix[i][j] = matrix[i][j] - otherMatrix.toArray()[i][j];
            }
        }

        return result;
    }

    @Override
    public FloatMatrix abs() {
        DefaultFloatMatrix result = new DefaultFloatMatrix(width, height);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                result.matrix[i][j] = Math.abs(matrix[i][j]);
            }
        }
        return result;
    }


    @Override
    public FloatMatrix absThis() {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                matrix[i][j] = Math.abs(matrix[i][j]);
            }
        }
        return this;
    }

    @Override
    public double sum() {
        double result = 0;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                result += matrix[i][j];
            }
        }
        return result;
    }

    @Override
    public FloatMatrix scale(float scale) {
        DefaultFloatMatrix result = new DefaultFloatMatrix(width, height);
        for (int i = 0; i < height; i++) {
            System.arraycopy(matrix[i], 0, result.toArray()[i], 0, width);
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
        DefaultFloatMatrix result = new DefaultFloatMatrix(height, width);
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
