package model;

/**
 * @author artrayme
 * 11/5/21
 */
public interface FloatMatrix {
    FloatMatrix mult(FloatMatrix otherMatrix);

    FloatMatrix plus(FloatMatrix otherMatrix);

    FloatMatrix minus(FloatMatrix otherMatrix);

    FloatMatrix abs();

    FloatMatrix absThis();

    double sum();

    FloatMatrix scale(float scale);

    FloatMatrix scaleThis(float scale);

    FloatMatrix transpose();

    int getHeight();

    int getWidth();

    float[][] toArray();
}
