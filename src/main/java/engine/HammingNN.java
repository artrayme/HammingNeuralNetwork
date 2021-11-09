package engine;

import model.FloatMatrix;

import java.util.List;

/**
 * @author artrayme
 * 11/5/21
 */
public interface HammingNN {

    /**
     * Main image recognition method.
     *
     * @param image an is unknown image
     * @return index of the most similar known image
     */
    int getAnswerByImage(List<Float> image);

    /**
     * @return matrix of learned images
     */
    FloatMatrix getPatterns();

}
