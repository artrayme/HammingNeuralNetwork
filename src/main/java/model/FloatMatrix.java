package model;

/**
 * @author artrayme
 * 11/5/21
 */
public interface FloatMatrix {
    FloatMatrix mult(FloatMatrix otherMatrix);

    FloatMatrix sum(FloatMatrix otherMatrix);

    FloatMatrix scale(float scale);

    FloatMatrix scaleThis(float scale);

    FloatMatrix transpose();

    int getHeight();

    int getWidth();

    float[][] toArray();
}
