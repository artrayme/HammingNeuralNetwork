package engine;

import model.FloatMatrix;

import java.util.List;

/**
 * @author artrayme
 * 11/5/21
 */
public interface HammingNN {

    int getAnswerByImage(List<Float> image);

    FloatMatrix getPatterns();


}
